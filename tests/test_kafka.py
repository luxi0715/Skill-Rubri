"""
test_kafka.py
--------------
验证 Step 5：Kafka 行为事件总线

分两组：
  A. 单元测试（mock KafkaProducer，不依赖真实 Kafka）
  B. 集成测试（需要 Kafka 在线，自动跳过）

运行：
  python -m pytest tests/test_kafka.py -v
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ── 判断 Kafka 是否可用 ─────────────────────────────────
def _kafka_available():
    try:
        from kafka import KafkaProducer
        p = KafkaProducer(
            bootstrap_servers="localhost:9092",
            request_timeout_ms=2000,
            api_version_auto_timeout_ms=2000,
        )
        p.close()
        return True
    except Exception:
        return False

requires_kafka = pytest.mark.skipif(
    not _kafka_available(), reason="Kafka 未启动，跳过集成测试"
)


# ── A. 单元测试 ──────────────────────────────────────────

class TestProducerMessages:

    def _make_mock_producer(self, monkeypatch):
        mock_p = MagicMock()
        mock_p.send.return_value = MagicMock()
        mock_p.flush.return_value = None
        import kafka_producer as kp
        kp._producer = mock_p
        return mock_p

    def teardown_method(self):
        import kafka_producer as kp
        kp._producer = None   # 每个测试后重置

    def test_produce_behavior_sends_to_correct_topic(self, monkeypatch):
        from kafka_producer import TOPIC_BEHAVIORS
        import kafka_producer as kp
        mock_p = self._make_mock_producer(monkeypatch)

        kp.produce_behavior("test.pdf", "view", user_id=1)

        mock_p.send.assert_called_once()
        call_args = mock_p.send.call_args
        assert call_args[0][0] == TOPIC_BEHAVIORS

    def test_produce_behavior_payload_fields(self, monkeypatch):
        import kafka_producer as kp
        mock_p = self._make_mock_producer(monkeypatch)

        kp.produce_behavior("test.pdf", "like", user_id=42)

        payload = mock_p.send.call_args[0][1]
        assert payload["paper"]   == "test.pdf"
        assert payload["action"]  == "like"
        assert payload["user_id"] == 42
        assert "created_at" in payload

    def test_produce_hot_delta_sends_to_correct_topic(self, monkeypatch):
        from kafka_producer import TOPIC_HOT_DELTA
        import kafka_producer as kp
        mock_p = self._make_mock_producer(monkeypatch)

        kp.produce_hot_delta("test.pdf", 3.0)

        call_args = mock_p.send.call_args
        assert call_args[0][0] == TOPIC_HOT_DELTA

    def test_produce_hot_delta_payload(self, monkeypatch):
        import kafka_producer as kp
        mock_p = self._make_mock_producer(monkeypatch)

        kp.produce_hot_delta("paper_a.pdf", 5.0)

        payload = mock_p.send.call_args[0][1]
        assert payload["paper"] == "paper_a.pdf"
        assert payload["delta"] == 5.0

    def test_produce_returns_false_when_kafka_unavailable(self):
        import kafka_producer as kp
        kp._producer = None
        # KafkaProducer 在函数内 lazy import，patch kafka 模块本身
        with patch("kafka.KafkaProducer", side_effect=Exception("no kafka")):
            result = kp.produce_behavior("test.pdf", "view")
        assert result == False

    def test_produce_noop_does_not_raise(self):
        """Kafka 不可用时不抛异常"""
        import kafka_producer as kp
        kp._producer = None
        with patch("kafka.KafkaProducer", side_effect=Exception("no kafka")):
            kp.produce_behavior("any.pdf", "view")   # 不应抛异常
            kp.produce_hot_delta("any.pdf", 3.0)


class TestConsumerHandlers:
    """Consumer handler 函数单元测试（不启动真实 Consumer 循环）"""

    def test_handle_behavior_calls_record_behavior(self, monkeypatch):
        import kafka_consumer as kc
        called = {}
        def mock_record(paper, action, user_id=None):
            called["paper"]  = paper
            called["action"] = action
        monkeypatch.setattr(kc, "record_behavior", mock_record, raising=False)
        # 直接 import db 内的函数被 monkeypatch
        import db
        monkeypatch.setattr(db, "record_behavior", mock_record)

        kc.handle_behavior({"paper": "a.pdf", "action": "view", "user_id": 1})
        assert called["paper"]  == "a.pdf"
        assert called["action"] == "view"

    def test_handle_hot_delta_calls_zincrby(self, monkeypatch):
        import kafka_consumer as kc
        mock_r = MagicMock()
        import hot_score
        monkeypatch.setattr(hot_score, "_get_redis", lambda: mock_r)

        kc.handle_hot_delta({"paper": "b.pdf", "delta": 3.0})
        mock_r.zincrby.assert_called_once_with(hot_score.ZSET_KEY, 3.0, "b.pdf")


# ── B. 集成测试 ─────────────────────────────────────────

class TestKafkaIntegration:

    @requires_kafka
    def test_produce_and_receive(self):
        """真实 Kafka：produce 后能 consume 到同一条消息"""
        from kafka import KafkaConsumer
        import kafka_producer as kp

        kp._producer = None   # 强制重连

        sent = kp.produce_behavior("integ_test.pdf", "view")
        assert sent == True

        consumer = KafkaConsumer(
            "user-behaviors",
            bootstrap_servers="localhost:9092",
            auto_offset_reset="latest",
            consumer_timeout_ms=3000,
            value_deserializer=lambda v: json.loads(v.decode()),
        )
        msgs = [m.value for m in consumer]
        consumer.close()
        papers = [m["paper"] for m in msgs]
        assert "integ_test.pdf" in papers
