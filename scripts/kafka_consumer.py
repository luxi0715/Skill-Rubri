"""
kafka_consumer.py
------------------
行为事件 Kafka Consumer（独立进程运行）。

消费 user-behaviors topic，批量写入 SQLite/PostgreSQL。
消费 hot-score-delta topic，更新 Redis ZSet。

运行：
  python kafka_consumer.py

设计：
  每条消息独立处理，失败打印日志继续消费（at-least-once）。
  未来可加 batch 提交提升吞吐量。
"""

import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPICS          = ["user-behaviors", "hot-score-delta"]


def handle_behavior(msg: dict):
    """写入 user_behaviors 表"""
    from db import record_behavior
    record_behavior(
        paper   = msg.get("paper", ""),
        action  = msg.get("action", ""),
        user_id = msg.get("user_id"),
    )


def handle_hot_delta(msg: dict):
    """更新 Redis ZSet"""
    from hot_score import _get_redis, ZSET_KEY
    r = _get_redis()
    if r:
        r.zincrby(ZSET_KEY, msg.get("delta", 0), msg.get("paper", ""))


def run():
    try:
        from kafka import KafkaConsumer
    except ImportError:
        print("[Kafka Consumer] kafka-python 未安装")
        return

    try:
        consumer = KafkaConsumer(
            *TOPICS,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="paper-behavior-group",
            consumer_timeout_ms=5000,
            api_version_auto_timeout_ms=3000,
        )
    except Exception as e:
        print(f"[Kafka Consumer] 连接失败：{e}")
        return

    print(f"[Kafka Consumer] 监听 topics：{TOPICS}")
    try:
        for record in consumer:
            topic = record.topic
            msg   = record.value
            try:
                if topic == "user-behaviors":
                    handle_behavior(msg)
                elif topic == "hot-score-delta":
                    handle_hot_delta(msg)
            except Exception as e:
                print(f"[Kafka Consumer] 处理失败 topic={topic} err={e}")
    except Exception as e:
        print(f"[Kafka Consumer] 消费中断：{e}")
    finally:
        consumer.close()
        print("[Kafka Consumer] 已关闭")


if __name__ == "__main__":
    run()
