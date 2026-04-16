"""
test_image_extraction.py
-------------------------
测试 PDF 图片提取（不调用 Vision API，只测提取逻辑）
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from extract_images import extract_top_images

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")


def test_extract_images_from_first_paper(tmp_path):
    """从第一篇 PDF 提取图片，验证返回路径列表"""
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    if not pdfs:
        pytest.skip("papers/ 目录为空")

    pdf_path = os.path.join(PAPERS_DIR, pdfs[0])
    paths = extract_top_images(pdf_path, str(tmp_path))

    # 提取结果应为列表
    assert isinstance(paths, list)
    # 每个路径应存在
    for p in paths:
        assert os.path.exists(p), f"图片文件不存在：{p}"
    # 不超过 MAX_IMAGES
    assert len(paths) <= 5


def test_extract_images_nonexistent_pdf(tmp_path):
    """不存在的 PDF 应抛出异常"""
    with pytest.raises(Exception):
        extract_top_images("/nonexistent/path.pdf", str(tmp_path))
