"""
grpc_server.py
---------------
gRPC Search Service — 暴露 Search 和 Recommend 接口。

Go 网关通过 gRPC 调用此服务，替代直接 HTTP 调用 FastAPI。

运行：
  python grpc_server.py          # 默认监听 0.0.0.0:50051

接口：
  SearchService.Search    — BM25+向量+混合检索
  SearchService.Recommend — 相似论文推荐
"""

import os
import sys
import concurrent.futures

import grpc

sys.path.insert(0, os.path.dirname(__file__))

import search_pb2
import search_pb2_grpc
from db import get_all_papers
from hybrid_search import (
    encode_api, search_bm25, search_vector,
    hybrid_scores, top_k_results, recommend_similar,
)

GRPC_PORT = os.getenv("GRPC_PORT", "50051")

# ── 缓存 records（与 api_server 一致）────────────────────
_records = None

def get_records():
    global _records
    _records = get_all_papers()
    return _records


# ── ServiceImpl ──────────────────────────────────────────
class SearchServiceServicer(search_pb2_grpc.SearchServiceServicer):

    def Search(self, request, context):
        records  = get_records()
        q        = request.query
        mode     = request.mode or "hybrid"
        top_k    = request.top_k or 5
        alpha    = request.alpha or 0.7

        try:
            if mode == "bm25":
                scores = search_bm25(q, records, top_k=top_k)
            elif mode == "vector":
                vec    = encode_api(q)
                scores = search_vector(vec, top_k=top_k)
            else:  # hybrid
                bm25_sc = search_bm25(q, records, top_k=top_k)
                vec     = encode_api(q)
                vec_sc  = search_vector(vec, top_k=top_k)
                scores  = hybrid_scores(bm25_sc, vec_sc, alpha=alpha)

            results = top_k_results(scores, top_k=top_k, label=mode)
            results = [r for r in results if r["score"] >= 0.3]

            pb_results = [
                search_pb2.PaperResult(
                    rank         = r["rank"],
                    paper        = r["paper"],
                    score        = r["score"],
                    quality_score= r.get("quality_score", 0),
                    rubric       = r.get("rubric", 0),
                    comment      = r.get("comment", ""),
                )
                for r in results
            ]
            return search_pb2.SearchResponse(
                query=q, mode=mode, results=pb_results, cache="miss"
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.SearchResponse()

    def Recommend(self, request, context):
        records = get_records()
        paper   = request.paper
        top_k   = request.top_k or 5

        try:
            recs = recommend_similar(paper, records, top_k=top_k)
            pb_recs = [
                search_pb2.PaperResult(
                    paper        = r["paper"],
                    score        = r["score"],
                    quality_score= r.get("paper_quality_score", 0),
                    comment      = r.get("comment", ""),
                )
                for r in recs
            ]
            return search_pb2.RecommendResponse(paper=paper, recommendations=pb_recs)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.RecommendResponse()


def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    search_pb2_grpc.add_SearchServiceServicer_to_server(SearchServiceServicer(), server)
    addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(addr)
    server.start()
    print(f"[gRPC Server] 监听 {addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
