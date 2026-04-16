"""
test_hot_score.py
------------------
验证 Step 4：Redis ZSet 热度分

分两组：
  A. 单元测试（mock Redis，不依赖真实服务）
  B. 集成测试（需要 Redis 在线，自动跳过）

运行：
  python -m pytest tests/test_hot_score.py -v
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _redis_available():
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False

requires_redis = pytest.mark.skipif(
    not _redis_available(), reason="Redis 未启动，跳过集成测试"
)


# ── A. 单元测试（mock Redis）────────────────────────────

class TestCalcScore:
    def test_all_zero(self):
        from hot_score import calc_score
        assert calc_score(0, 0, 0, 0.0) == 0.0

    def test_weights(self):
        from hot_score import calc_score
        # likes=1 → +3, comments=1 → +2, views=1 → +1, quality=1.0 → +10
        assert calc_score(1, 1, 1, 1.0) == 3 + 2 + 1 + 10

    def test_likes_weight_3(self):
        from hot_score import calc_score
        assert calc_score(5, 0, 0, 0.0) == 15

    def test_quality_weight_10(self):
        from hot_score import calc_score
        assert calc_score(0, 0, 0, 0.8) == pytest.approx(8.0)


class TestIncrements:
    """on_like / on_view / on_comment 应调用正确的 zincrby 增量"""

    def _mock_redis(self, monkeypatch):
        mock_r = MagicMock()
        import hot_score
        monkeypatch.setattr(hot_score, "_get_redis", lambda: mock_r)
        return mock_r

    def test_on_like_increments_3(self, monkeypatch):
        from hot_score import on_like, ZSET_KEY
        mock_r = self._mock_redis(monkeypatch)
        on_like("test.pdf")
        mock_r.zincrby.assert_called_once_with(ZSET_KEY, 3, "test.pdf")

    def test_on_view_increments_1(self, monkeypatch):
        from hot_score import on_view, ZSET_KEY
        mock_r = self._mock_redis(monkeypatch)
        on_view("test.pdf")
        mock_r.zincrby.assert_called_once_with(ZSET_KEY, 1, "test.pdf")

    def test_on_comment_increments_2(self, monkeypatch):
        from hot_score import on_comment, ZSET_KEY
        mock_r = self._mock_redis(monkeypatch)
        on_comment("test.pdf")
        mock_r.zincrby.assert_called_once_with(ZSET_KEY, 2, "test.pdf")

    def test_on_like_noop_when_redis_unavailable(self, monkeypatch):
        import hot_score
        monkeypatch.setattr(hot_score, "_get_redis", lambda: None)
        on_like = hot_score.on_like
        on_like("test.pdf")   # 不应抛异常

    def test_get_hot_ranking_returns_none_when_unavailable(self, monkeypatch):
        import hot_score
        monkeypatch.setattr(hot_score, "_get_redis", lambda: None)
        result = hot_score.get_hot_ranking(page=1, size=10)
        assert result is None

    def test_get_hot_ranking_with_mock(self, monkeypatch):
        from hot_score import get_hot_ranking, ZSET_KEY
        mock_r = self._mock_redis(monkeypatch)
        mock_r.zrevrange.return_value = [("paper_a.pdf", 55.0), ("paper_b.pdf", 32.0)]
        result = get_hot_ranking(page=1, size=10)
        assert result is not None
        assert result[0][0] == "paper_a.pdf"
        assert result[0][1] == 55.0
        mock_r.zrevrange.assert_called_once_with(ZSET_KEY, 0, 9, withscores=True)

    def test_get_hot_ranking_page2_offset(self, monkeypatch):
        """第2页 offset 应为 size"""
        from hot_score import get_hot_ranking, ZSET_KEY
        mock_r = self._mock_redis(monkeypatch)
        mock_r.zrevrange.return_value = []
        get_hot_ranking(page=2, size=10)
        mock_r.zrevrange.assert_called_once_with(ZSET_KEY, 10, 19, withscores=True)


# ── B. 集成测试（需要 Redis）────────────────────────────

class TestHotScoreIntegration:

    @requires_redis
    def test_on_like_real_redis(self):
        import redis, hot_score
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.delete(hot_score.ZSET_KEY)

        hot_score.on_like("integ_test.pdf")
        score = r.zscore(hot_score.ZSET_KEY, "integ_test.pdf")
        assert score == 3.0

    @requires_redis
    def test_multiple_events_accumulate(self):
        import redis, hot_score
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.delete(hot_score.ZSET_KEY)

        hot_score.on_like("integ_test.pdf")    # +3
        hot_score.on_view("integ_test.pdf")    # +1
        hot_score.on_comment("integ_test.pdf") # +2
        score = r.zscore(hot_score.ZSET_KEY, "integ_test.pdf")
        assert score == 6.0

    @requires_redis
    def test_ranking_order(self):
        import redis, hot_score
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.delete(hot_score.ZSET_KEY)

        r.zadd(hot_score.ZSET_KEY, {"paper_a.pdf": 100, "paper_b.pdf": 50})
        ranking = hot_score.get_hot_ranking(page=1, size=2)
        assert ranking[0][0] == "paper_a.pdf"   # 分高的在前
        assert ranking[1][0] == "paper_b.pdf"
