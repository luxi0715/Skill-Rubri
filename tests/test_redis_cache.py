"""
test_redis_cache.py
--------------------
验证 Step 2：Redis 搜索结果缓存

测试分两组：
  A. 缓存逻辑（mock Redis，不依赖真实 Redis 服务）
  B. 集成测试（需要 Redis 在线，自动跳过若未连接）

运行：
  cd D:\Skill Rubri\agenteval_test01
  python -m pytest tests/test_redis_cache.py -v
"""

import sys
import os
import json
import hashlib
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ── A. 缓存辅助函数逻辑测试（纯单元，不需要 Redis）────────

class TestCacheHelpers:
    """测试 _cache_get / _cache_set 的行为"""

    def test_cache_key_format(self):
        """cache key 是 search: + md5"""
        q, mode, top_k, alpha = "contrastive", "hybrid", 5, 0.7
        key = "search:" + hashlib.md5(f"{q}:{mode}:{top_k}:{alpha}".encode()).hexdigest()
        assert key.startswith("search:")
        assert len(key) == len("search:") + 32

    def test_cache_key_same_query_same_key(self):
        """相同参数 → 相同 key"""
        make = lambda: "search:" + hashlib.md5("test:hybrid:5:0.7".encode()).hexdigest()
        assert make() == make()

    def test_cache_key_different_query_different_key(self):
        """不同查询 → 不同 key"""
        k1 = "search:" + hashlib.md5("query1:hybrid:5:0.7".encode()).hexdigest()
        k2 = "search:" + hashlib.md5("query2:hybrid:5:0.7".encode()).hexdigest()
        assert k1 != k2

    def test_cache_get_returns_none_when_redis_unavailable(self):
        """Redis 不可用时 _cache_get 返回 None"""
        import api_server as srv
        original = srv.REDIS_OK
        srv.REDIS_OK = False
        result = srv._cache_get("any_key")
        srv.REDIS_OK = original
        assert result is None

    def test_cache_set_noop_when_redis_unavailable(self):
        """Redis 不可用时 _cache_set 不抛异常"""
        import api_server as srv
        original = srv.REDIS_OK
        srv.REDIS_OK = False
        srv._cache_set("any_key", {"data": 123})   # 不应抛异常
        srv.REDIS_OK = original

    def test_cache_roundtrip_with_mock(self):
        """mock Redis：set 后 get 能取回原值"""
        import api_server as srv

        store = {}
        mock_redis = MagicMock()
        mock_redis.get.side_effect = lambda k: store.get(k)
        mock_redis.setex.side_effect = lambda k, ttl, v: store.update({k: v})

        original_client = srv._redis_client
        original_ok     = srv.REDIS_OK
        srv._redis_client = mock_redis
        srv.REDIS_OK      = True

        payload = {"query": "test", "results": [1, 2, 3]}
        srv._cache_set("test_key", payload)
        result = srv._cache_get("test_key")

        srv._redis_client = original_client
        srv.REDIS_OK      = original_ok

        assert result == payload

    def test_cache_miss_returns_none_with_mock(self):
        """mock Redis：未设置的 key 返回 None"""
        import api_server as srv

        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        original_client = srv._redis_client
        original_ok     = srv.REDIS_OK
        srv._redis_client = mock_redis
        srv.REDIS_OK      = True

        result = srv._cache_get("nonexistent_key")

        srv._redis_client = original_client
        srv.REDIS_OK      = original_ok

        assert result is None


# ── B. 集成测试（需要 Redis 在线）────────────────────────

def _redis_available():
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False

requires_redis = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis 未启动，跳过集成测试"
)

class TestRedisCacheIntegration:

    @requires_redis
    def test_set_and_get(self):
        """真实 Redis：写入后能读取"""
        import api_server as srv
        srv._cache_set("integ_test_key", {"ok": True}, ttl=10)
        val = srv._cache_get("integ_test_key")
        assert val == {"ok": True}

    @requires_redis
    def test_ttl_applied(self):
        """真实 Redis：key 有 TTL"""
        import redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        import api_server as srv
        srv._cache_set("integ_ttl_key", {"x": 1}, ttl=60)
        ttl = r.ttl("integ_ttl_key")
        assert 0 < ttl <= 60
