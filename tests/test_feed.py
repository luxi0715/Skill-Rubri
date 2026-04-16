"""
test_feed.py
-------------
测试首页信息流：分页、热度排序、领域筛选
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import db as db_module


@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_feed.db")
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()

    # 插入10篇测试论文
    for i in range(10):
        db_module.upsert_paper({
            "paper":               f"2024_Author{i}_Paper_Title_{i}.pdf",
            "paper_quality_score": round(0.5 + i * 0.04, 2),
            "skill_rubric_total":  0.8,
            "comment":             f"Test comment {i}",
            "comment_cn":          f"测试评语 {i}",
            "title_cn":            f"测试论文{i}",
            "domain":              "推荐系统" if i % 2 == 0 else "对比学习",
            "year":                "2024",
            "venue":               "ICML",
            "open_source":         "yes",
        })
    yield


# ── 分页 ────────────────────────────────────────────────────
def test_first_page_returns_10():
    result = db_module.get_papers_page(sort="quality", page=1, size=10)
    assert len(result["papers"]) == 10
    assert result["total"] == 10
    assert result["has_more"] is False


def test_page_size_5_returns_5():
    result = db_module.get_papers_page(sort="quality", page=1, size=5)
    assert len(result["papers"]) == 5
    assert result["has_more"] is True


def test_second_page():
    result = db_module.get_papers_page(sort="quality", page=2, size=5)
    assert len(result["papers"]) == 5
    assert result["has_more"] is False


def test_page_beyond_total_returns_empty():
    result = db_module.get_papers_page(sort="quality", page=99, size=10)
    assert len(result["papers"]) == 0
    assert result["has_more"] is False


# ── 排序 ────────────────────────────────────────────────────
def test_quality_sort_descending():
    result = db_module.get_papers_page(sort="quality", page=1, size=10)
    scores = [p["paper_quality_score"] for p in result["papers"]]
    assert scores == sorted(scores, reverse=True)


def test_hot_sort_includes_hot_score():
    result = db_module.get_papers_page(sort="hot", page=1, size=10)
    for p in result["papers"]:
        assert "hot_score" in p


def test_hot_score_increases_with_likes():
    """点赞多的论文热度分更高"""
    paper = "2024_Author0_Paper_Title_0.pdf"
    db_module.like_paper(paper)
    db_module.like_paper(paper)
    db_module.like_paper(paper)

    result = db_module.get_papers_page(sort="hot", page=1, size=1)
    assert result["papers"][0]["paper"] == paper


# ── 领域筛选 ────────────────────────────────────────────────
def test_domain_filter():
    result = db_module.get_papers_page(sort="quality", page=1, size=10, domain="推荐系统")
    assert result["total"] == 5
    for p in result["papers"]:
        assert p["domain"] == "推荐系统"


def test_domain_filter_no_match():
    result = db_module.get_papers_page(sort="quality", page=1, size=10, domain="不存在的领域")
    assert result["total"] == 0
    assert result["papers"] == []


# ── 浏览记录影响热度 ──────────────────────────────────────
def test_views_affect_hot_score():
    paper = "2024_Author0_Paper_Title_0.pdf"
    for _ in range(10):
        db_module.record_behavior(paper, "view", anon_id="test-anon")

    result = db_module.get_papers_page(sort="hot", page=1, size=1)
    assert result["papers"][0]["paper"] == paper
