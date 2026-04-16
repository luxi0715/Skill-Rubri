"""
test_likes_comments.py
-----------------------
测试点赞和评论功能（SQLite层）
"""

import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# 使用临时数据库隔离测试
import db as db_module

TEST_DB = os.path.join(os.path.dirname(__file__), "test_temp.db")


@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    # 插入一条测试论文
    db_module.upsert_paper({
        "paper": "test_paper.pdf",
        "paper_quality_score": 0.8,
        "skill_rubric_total": 0.9,
        "comment": "test comment",
    })
    yield


def test_like_paper():
    likes1 = db_module.like_paper("test_paper.pdf")
    likes2 = db_module.like_paper("test_paper.pdf")
    assert likes1 == 1
    assert likes2 == 2


def test_add_and_get_comment():
    c = db_module.add_comment("test_paper.pdf", "张三", "这篇论文很有意思")
    assert c["nickname"] == "张三"
    assert c["content"] == "这篇论文很有意思"
    assert c["likes"] == 0

    comments = db_module.get_comments("test_paper.pdf")
    assert len(comments) == 1
    assert comments[0]["nickname"] == "张三"


def test_like_comment():
    c = db_module.add_comment("test_paper.pdf", "李四", "不错")
    likes = db_module.like_comment(c["id"])
    assert likes == 1
    likes = db_module.like_comment(c["id"])
    assert likes == 2


def test_save_and_get_images():
    images = [
        {"image_path": "data/images/test/img_01.png", "caption": "架构图", "order_idx": 0},
        {"image_path": "data/images/test/img_02.png", "caption": "结果图", "order_idx": 1},
    ]
    db_module.save_paper_images("test_paper.pdf", images, "这篇论文展示了模型架构和实验结果")
    imgs, summary = db_module.get_paper_images("test_paper.pdf")
    assert len(imgs) == 2
    assert imgs[0]["caption"] == "架构图"
    assert "架构" in summary


def test_multiple_comments_order():
    db_module.add_comment("test_paper.pdf", "A", "第一条")
    db_module.add_comment("test_paper.pdf", "B", "第二条")
    comments = db_module.get_comments("test_paper.pdf")
    # 最新的排在前面
    assert comments[0]["nickname"] == "B"
