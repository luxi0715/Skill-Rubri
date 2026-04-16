"""
metrics.py
-----------
Prometheus 指标定义（Python FastAPI 侧）。

指标清单：
  search_requests_total        Counter   搜索请求总数，按 mode/cache 标签
  search_duration_seconds      Histogram 搜索耗时，p50/p95/p99
  cache_hits_total             Counter   Redis 缓存命中次数
  cache_misses_total           Counter   Redis 缓存未命中次数
  active_ws_connections        Gauge     当前 WebSocket 连接数（Go 侧同步）
  paper_likes_total            Counter   论文点赞累计
  paper_views_total            Counter   论文浏览累计
  grpc_requests_total          Counter   gRPC 请求总数，按 method 标签
  grpc_duration_seconds        Histogram gRPC 调用耗时

使用方式：
  from metrics import (
      search_requests_total, search_duration_seconds,
      cache_hits_total, cache_misses_total,
      paper_likes_total, paper_views_total,
  )
  # 在 api_server.py 的路由里调用
"""

from prometheus_client import Counter, Histogram, Gauge, REGISTRY

# ── 搜索 ──────────────────────────────────────────────────
search_requests_total = Counter(
    "search_requests_total",
    "搜索请求总数",
    ["mode", "cache"],          # mode=bm25/vector/hybrid, cache=hit/miss
)

search_duration_seconds = Histogram(
    "search_duration_seconds",
    "搜索接口响应耗时（秒）",
    ["mode"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── 缓存 ──────────────────────────────────────────────────
cache_hits_total = Counter(
    "cache_hits_total",
    "Redis 搜索缓存命中次数",
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Redis 搜索缓存未命中次数",
)

# ── 用户行为 ───────────────────────────────────────────────
paper_likes_total = Counter(
    "paper_likes_total",
    "论文点赞累计次数",
)

paper_views_total = Counter(
    "paper_views_total",
    "论文浏览累计次数",
)

paper_comments_total = Counter(
    "paper_comments_total",
    "论文评论累计次数",
)

# ── gRPC ──────────────────────────────────────────────────
grpc_requests_total = Counter(
    "grpc_requests_total",
    "gRPC 请求总数",
    ["method", "status"],       # method=Search/Recommend, status=ok/error
)

grpc_duration_seconds = Histogram(
    "grpc_duration_seconds",
    "gRPC 调用耗时（秒）",
    ["method"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ── 系统 ──────────────────────────────────────────────────
faiss_search_duration_seconds = Histogram(
    "faiss_search_duration_seconds",
    "FAISS 向量搜索耗时（秒）",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

kafka_produce_total = Counter(
    "kafka_produce_total",
    "Kafka 消息发送次数",
    ["topic", "status"],        # status=ok/fail
)
