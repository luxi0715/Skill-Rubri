"""
extract_images.py
------------------
从每篇论文 PDF 提取最多5张大图，用 GPT-4o Vision 生成图注和总结，
存入 SQLite（paper_images 表 + papers.image_summary）。

运行方式：PyCharm 右键 Run
"""

import base64
import json
import os
import sys

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, get_all_papers, save_paper_images, get_paper_images

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
IMG_DIR    = os.path.join(BASE_DIR, "data", "images")
MAX_IMAGES = 5
MIN_SIZE   = 100 * 1024   # 忽略小于100KB的图（多为图标/装饰）


def extract_top_images(pdf_path: str, out_dir: str) -> list[str]:
    """提取 PDF 中最大的几张图，返回保存路径列表"""
    os.makedirs(out_dir, exist_ok=True)
    doc    = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base = doc.extract_image(xref)
            data = base["image"]
            ext  = base["ext"]
            if len(data) < 10 * 1024:  # 跳过小于10KB的图
                continue
            images.append((len(data), xref, data, ext))

    # 按大小降序，取前 MAX_IMAGES 张
    images.sort(key=lambda x: x[0], reverse=True)
    saved = []
    for i, (size, xref, data, ext) in enumerate(images[:MAX_IMAGES]):
        fname = f"img_{i+1:02d}.{ext}"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(data)
        saved.append(fpath)

    doc.close()
    return saved


def describe_images(image_paths: list[str], paper_name: str) -> tuple[list[str], str]:
    """用 GPT-4o Vision 为每张图生成图注，再生成总结"""
    from openai import OpenAI
    client  = OpenAI()
    captions = []

    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = path.rsplit(".", 1)[-1]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{ext};base64,{b64}", "detail": "low"}},
                    {"type": "text",
                     "text": "用一句中文描述这张学术论文图片展示的内容（不超过50字）。"}
                ]
            }],
            max_tokens=100,
        )
        captions.append(resp.choices[0].message.content.strip())

    # 整体总结
    if captions:
        bullets = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"以下是论文《{paper_name}》中提取的图片描述：\n{bullets}\n\n用两到三句中文总结这篇论文的核心图示内容。"
            }],
            max_tokens=200,
        )
        summary = resp.choices[0].message.content.strip()
    else:
        summary = ""

    return captions, summary


def run():
    print("=" * 55)
    print("  extract_images.py")
    print("  PDF图片提取 + GPT-4o Vision 描述")
    print("=" * 55)

    init_db()
    records = get_all_papers()
    print(f"\n  共 {len(records)} 篇论文\n")

    for i, r in enumerate(records, 1):
        paper     = r["paper"]
        name      = paper.replace(".pdf", "")
        pdf_path  = os.path.join(PAPERS_DIR, paper)
        out_dir   = os.path.join(IMG_DIR, name)

        # 检查是否已处理
        existing, _ = get_paper_images(paper)
        if existing:
            print(f"  [{i}/{len(records)}]  {name[:45]}  已处理，跳过")
            continue

        if not os.path.exists(pdf_path):
            print(f"  [{i}/{len(records)}]  {name[:45]}  PDF不存在，跳过")
            continue

        print(f"  [{i}/{len(records)}]  {name[:45]}  提取中...")
        try:
            paths    = extract_top_images(pdf_path, out_dir)
            if not paths:
                print(f"           无可用图片，跳过")
                continue

            print(f"           提取 {len(paths)} 张，Vision描述中...")
            captions, summary = describe_images(paths, name)

            # 转为相对路径存入数据库
            images = [
                {
                    "image_path": os.path.relpath(p, BASE_DIR).replace("\\", "/"),
                    "caption":    captions[j] if j < len(captions) else "",
                    "order_idx":  j,
                }
                for j, p in enumerate(paths)
            ]
            save_paper_images(paper, images, summary)
            print(f"           完成，{len(images)} 张图已存入数据库")
        except Exception as e:
            print(f"           失败：{e}")

    print(f"\n  全部完成\n")


if __name__ == "__main__":
    run()
