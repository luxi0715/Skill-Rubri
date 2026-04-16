"""
test_metrics.py
----------------
验证 Step 9：Prometheus 指标埋点（Python FastAPI 侧）

测试：
  - 指标对象类型正确（Counter / Histogram）
  - 标签定义完整
  - 指标自增逻辑正确
  - /metrics/python 端点返回标准 Prometheus 格式
  - 搜索路由触发指标更新

运行：
  python -m pytest tests/test_metrics.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestMetricTypes:
    def test_search_requests_is_counter(self):
        from prometheus_client import Counter
        from metrics import search_requests_total
        assert isinstance(search_requests_total, Counter)

    def test_search_duration_is_histogram(self):
        from prometheus_client import Histogram
        from metrics import search_duration_seconds
        assert isinstance(search_duration_seconds, Histogram)

    def test_cache_hits_is_counter(self):
        from prometheus_client import Counter
        from metrics import cache_hits_total
        assert isinstance(cache_hits_total, Counter)

    def test_faiss_duration_is_histogram(self):
        from prometheus_client import Histogram
        from metrics import faiss_search_duration_seconds
        assert isinstance(faiss_search_duration_seconds, Histogram)

    def test_kafka_produce_is_counter(self):
        from prometheus_client import Counter
        from metrics import kafka_produce_total
        assert isinstance(kafka_produce_total, Counter)


class TestMetricLabels:
    def test_search_requests_has_mode_label(self):
        from metrics import search_requests_total
        # 有 mode 标签：labelnames 应包含 mode
        assert "mode" in search_requests_total._labelnames

    def test_search_requests_has_cache_label(self):
        from metrics import search_requests_total
        assert "cache" in search_requests_total._labelnames

    def test_grpc_requests_has_method_label(self):
        from metrics import grpc_requests_total
        assert "method" in grpc_requests_total._labelnames

    def test_kafka_produce_has_topic_label(self):
        from metrics import kafka_produce_total
        assert "topic" in kafka_produce_total._labelnames


class TestMetricIncrement:
    def test_counter_increments(self):
        from metrics import paper_likes_total
        before = paper_likes_total._value.get()
        paper_likes_total.inc()
        after  = paper_likes_total._value.get()
        assert after == before + 1

    def test_histogram_observe(self):
        from metrics import faiss_search_duration_seconds
        # observe 不报错即可
        faiss_search_duration_seconds.observe(0.05)
        faiss_search_duration_seconds.observe(0.2)

    def test_labeled_counter_increments(self):
        from metrics import search_requests_total
        c = search_requests_total.labels(mode="hybrid", cache="miss")
        before = c._value.get()
        c.inc()
        assert c._value.get() == before + 1


class TestPrometheusEndpoint:
    """测试 /metrics/python 端点输出标准格式"""

    def _get_app(self):
        import api_server
        return api_server.app

    def test_metrics_endpoint_returns_200(self):
        from fastapi.testclient import TestClient
        app = self._get_app()
        client = TestClient(app)
        resp = client.get("/metrics/python")
        assert resp.status_code == 200

    def test_metrics_endpoint_content_type(self):
        from fastapi.testclient import TestClient
        app = self._get_app()
        client = TestClient(app)
        resp = client.get("/metrics/python")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_help_and_type(self):
        from fastapi.testclient import TestClient
        app = self._get_app()
        client = TestClient(app)
        resp = client.get("/metrics/python")
        body = resp.text
        assert "# HELP" in body
        assert "# TYPE" in body

    def test_metrics_contains_search_counter(self):
        from fastapi.testclient import TestClient
        app = self._get_app()
        client = TestClient(app)
        resp = client.get("/metrics/python")
        assert "search_requests_total" in resp.text

    def test_metrics_contains_faiss_histogram(self):
        from fastapi.testclient import TestClient
        app = self._get_app()
        client = TestClient(app)
        resp = client.get("/metrics/python")
        assert "faiss_search_duration_seconds" in resp.text
