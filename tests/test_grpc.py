"""
test_grpc.py
-------------
验证 Step 7：gRPC 接口定义与 Server 实现

分两组：
  A. 单元测试（mock servicer，不依赖真实 gRPC 服务器）
  B. 集成测试（启动真实 gRPC server，自动跳过若端口被占用）

运行：
  python -m pytest tests/test_grpc.py -v
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ── 公共：检查 proto 生成文件 ────────────────────────────

class TestProtoGenerated:
    def test_pb2_importable(self):
        import search_pb2
        assert hasattr(search_pb2, "SearchRequest")
        assert hasattr(search_pb2, "SearchResponse")
        assert hasattr(search_pb2, "PaperResult")
        assert hasattr(search_pb2, "RecommendRequest")
        assert hasattr(search_pb2, "RecommendResponse")

    def test_grpc_importable(self):
        import search_pb2_grpc
        assert hasattr(search_pb2_grpc, "SearchServiceServicer")
        assert hasattr(search_pb2_grpc, "add_SearchServiceServicer_to_server")

    def test_search_request_fields(self):
        import search_pb2
        req = search_pb2.SearchRequest(query="test", mode="hybrid", top_k=5, alpha=0.7)
        assert req.query == "test"
        assert req.mode  == "hybrid"
        assert req.top_k == 5
        assert abs(req.alpha - 0.7) < 1e-5

    def test_paper_result_fields(self):
        import search_pb2
        r = search_pb2.PaperResult(rank=1, paper="a.pdf", score=0.9,
                                   quality_score=0.8, rubric=0.7, comment="good")
        assert r.rank  == 1
        assert r.paper == "a.pdf"
        assert abs(r.score - 0.9) < 1e-5

    def test_search_response_has_results(self):
        import search_pb2
        r1 = search_pb2.PaperResult(rank=1, paper="a.pdf", score=0.9)
        r2 = search_pb2.PaperResult(rank=2, paper="b.pdf", score=0.7)
        resp = search_pb2.SearchResponse(query="q", mode="hybrid", results=[r1, r2])
        assert len(resp.results) == 2
        assert resp.results[0].paper == "a.pdf"


# ── A. Servicer 单元测试（mock 检索函数）────────────────

class TestSearchServicer:

    def _make_servicer(self):
        from grpc_server import SearchServiceServicer
        return SearchServiceServicer()

    def test_search_bm25_mode(self, monkeypatch):
        import grpc_server, search_pb2
        monkeypatch.setattr(grpc_server, "get_records", lambda: [])
        monkeypatch.setattr(grpc_server, "search_bm25",
                            lambda q, r, top_k: {1: 1.0, 2: 0.5})
        monkeypatch.setattr(grpc_server, "top_k_results", lambda scores, top_k, label: [
            {"rank": 1, "paper": "a.pdf", "score": 1.0,
             "quality_score": 0.8, "rubric": 0.9, "comment": "ok"}
        ])

        svc = self._make_servicer()
        req = search_pb2.SearchRequest(query="test", mode="bm25", top_k=5)
        ctx = MagicMock()
        resp = svc.Search(req, ctx)
        assert len(resp.results) == 1
        assert resp.results[0].paper == "a.pdf"

    def test_search_low_score_filtered(self, monkeypatch):
        """score < 0.3 的结果应被过滤"""
        import grpc_server, search_pb2
        monkeypatch.setattr(grpc_server, "get_records", lambda: [])
        monkeypatch.setattr(grpc_server, "search_bm25",
                            lambda q, r, top_k: {1: 0.1})
        monkeypatch.setattr(grpc_server, "top_k_results", lambda scores, top_k, label: [
            {"rank": 1, "paper": "weak.pdf", "score": 0.1,
             "quality_score": 0, "rubric": 0, "comment": ""}
        ])
        svc = self._make_servicer()
        req = search_pb2.SearchRequest(query="test", mode="bm25", top_k=5)
        ctx = MagicMock()
        resp = svc.Search(req, ctx)
        assert len(resp.results) == 0   # 被过滤

    def test_recommend_returns_papers(self, monkeypatch):
        import grpc_server, search_pb2
        monkeypatch.setattr(grpc_server, "get_records", lambda: [])
        monkeypatch.setattr(grpc_server, "recommend_similar",
                            lambda paper, records, top_k: [
                                {"paper": "sim1.pdf", "score": 0.85,
                                 "paper_quality_score": 0.8, "comment": "similar"}
                            ])
        svc = self._make_servicer()
        req = search_pb2.RecommendRequest(paper="target.pdf", top_k=3)
        ctx = MagicMock()
        resp = svc.Recommend(req, ctx)
        assert len(resp.recommendations) == 1
        assert resp.recommendations[0].paper == "sim1.pdf"

    def test_search_exception_sets_grpc_error(self, monkeypatch):
        import grpc_server, search_pb2, grpc
        monkeypatch.setattr(grpc_server, "get_records", lambda: [])
        monkeypatch.setattr(grpc_server, "search_bm25",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        svc = self._make_servicer()
        req = search_pb2.SearchRequest(query="err", mode="bm25", top_k=5)
        ctx = MagicMock()
        svc.Search(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


# ── B. 集成测试（启动真实 gRPC server）──────────────────

def _grpc_port_free():
    import socket
    with socket.socket() as s:
        return s.connect_ex(("localhost", 50051)) != 0

requires_grpc = pytest.mark.skipif(
    not _grpc_port_free(), reason="端口 50051 已占用，跳过集成测试"
)

class TestGRPCIntegration:

    @requires_grpc
    def test_server_starts_and_responds(self):
        """启动 server 线程，用 stub 调用 Search"""
        import threading, time
        import grpc, search_pb2, search_pb2_grpc
        from grpc_server import serve, SearchServiceServicer
        import concurrent.futures

        # 在线程里跑 server
        server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
        search_pb2_grpc.add_SearchServiceServicer_to_server(
            SearchServiceServicer(), server)
        server.add_insecure_port("[::]:50051")
        server.start()
        time.sleep(0.2)

        channel = grpc.insecure_channel("localhost:50051")
        stub    = search_pb2_grpc.SearchServiceStub(channel)
        req     = search_pb2.SearchRequest(query="test", mode="bm25", top_k=3)
        resp    = stub.Search(req, timeout=5)

        server.stop(grace=0)
        assert isinstance(resp, search_pb2.SearchResponse)
