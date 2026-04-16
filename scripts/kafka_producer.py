"""
kafka_producer.py
------------------
行为事件 Kafka Producer。

当用户浏览/点赞/评论时，把事件 produce 到 Kafka topic，
Consumer 再异步批量写 SQLite（将来是 PostgreSQL）。

Topic 规划：
  user-behaviors   浏览、点赞、搜索
  hot-score-delta  热度分增量（供 Consumer 更新 Redis ZSet）

Kafka 不可用时静默降级（事件直接同步写 DB，行为与现在一致）。

使用：
  from kafka_producer import produce_behavior, produce_hot_delta
"""

import json
import os
from datetime import datetime

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_BEHAVIORS = "user-behaviors"
TOPIC_HOT_DELTA = "hot-score-delta"

_producer = None


def _get_producer():
    global _producer
    if _producer is not None:
        return _producer
    try:
        from kafka import KafkaProducer
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            request_timeout_ms=2000,
            api_version_auto_timeout_ms=2000,
        )
        return _producer
    except Exception:
        return None


def _send(topic: str, payload: dict):
    """发送一条消息，失败静默（不影响主流程）"""
    p = _get_producer()
    if not p:
        return False
    try:
        p.send(topic, payload)
        p.flush(timeout=1)
        return True
    except Exception:
        return False


def produce_behavior(paper: str, action: str, user_id: int = None):
    """
    发送用户行为事件到 user-behaviors topic。
    action: "view" | "like" | "comment" | "search"
    """
    return _send(TOPIC_BEHAVIORS, {
        "paper":      paper,
        "action":     action,
        "user_id":    user_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def produce_hot_delta(paper: str, delta: float):
    """
    发送热度分增量事件到 hot-score-delta topic。
    Consumer 消费后执行 Redis ZINCRBY。
    """
    return _send(TOPIC_HOT_DELTA, {
        "paper":      paper,
        "delta":      delta,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def kafka_available() -> bool:
    return _get_producer() is not None
