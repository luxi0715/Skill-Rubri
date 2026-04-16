

import os
import re
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from paper_rubric_eval import evaluate_folder

# ================== 配置 ==================
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))   # agenteval_test01/
CONFIG_FILE = os.path.join(BASE_DIR, "config", "topics_config.json")
ARXIV_API   = "https://export.arxiv.org/api/query"
ARXIV_NS    = "http://www.w3.org/2005/Atom"
HEADERS     = {"User-Agent": "paper-eval-bot/1.0"}

# 日期范围定义
# 热门：近6个月
# 经典：2020~2022（时间够久，已被验证）
TODAY         = datetime.now()
TRENDING_FROM = (TODAY - timedelta(days=180)).strftime("%Y%m%d")
TRENDING_TO   = TODAY.strftime("%Y%m%d")
CLASSIC_FROM  = "20200101"
CLASSIC_TO    = "20221231"


# ================== 工具函数 ==================
def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(text: str, max_len: int = 40) -> str:
    """把文本转成合法文件名片段"""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def make_pdf_name(paper: dict) -> str:
    """生成 PDF 文件名：{year}_{first_author}_{title_short}.pdf"""
    year   = str(paper.get("year") or "0000")
    author = paper.get("first_author", "Unknown")
    title  = safe_filename(paper.get("title", "NoTitle"))
    return f"{year}_{author}_{title}.pdf"


# ================== arXiv 解析 ==================
def parse_arxiv_xml(xml_text: str) -> list:
    """解析 arXiv API 返回的 Atom XML"""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        # arXiv ID
        id_url   = entry.findtext(f"{{{ARXIV_NS}}}id", "")
        arxiv_id = id_url.split("/abs/")[-1].strip()

        # 标题
        title = entry.findtext(f"{{{ARXIV_NS}}}title", "").strip().replace("\n", " ")

        # 第一作者姓氏
        authors = entry.findall(f"{{{ARXIV_NS}}}author")
        first_author = ""
        if authors:
            name = authors[0].findtext(f"{{{ARXIV_NS}}}name", "")
            first_author = safe_filename(name.split()[-1]) if name else "Unknown"

        # 发布年份
        published = entry.findtext(f"{{{ARXIV_NS}}}published", "")
        year = published[:4] if published else "0000"

        papers.append({
            "arxiv_id":     arxiv_id,
            "title":        title,
            "year":         year,
            "first_author": first_author,
            "pdf_url":      f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        })
    return papers


# ================== arXiv 搜索 ==================
def search_arxiv(query: str, date_from: str, date_to: str,
                 limit: int, retries: int = 5) -> list:
    """
    搜索 arXiv，按提交时间降序排列
    date_from / date_to 格式：YYYYMMDD
    """
    search_query = (
        f"all:{query} AND "
        f"(cat:cs.IR OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL) AND "
        f"submittedDate:[{date_from}0000 TO {date_to}2359]"
    )
    params = {
        "search_query": search_query,
        "start":        0,
        "max_results":  min(limit, 100),
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    wait = 30
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(ARXIV_API, params=params,
                                headers=HEADERS, timeout=120)
            if resp.status_code in (429, 503):
                print(f"  ⏳ 触发限速，等待 {wait}s 后重试（第{attempt}次）...")
                time.sleep(wait)
                wait *= 2
                continue
            resp.raise_for_status()
            results = parse_arxiv_xml(resp.text)
            time.sleep(5)   # 成功后也等一下，避免连续请求
            return results
        except Exception as e:
            print(f"  ⚠️  arXiv 查询失败：{e}")
            if attempt < retries:
                print(f"  ⏳ 等待 {wait}s 后重试（第{attempt}次）...")
                time.sleep(wait)
                wait *= 2
    return []


# ================== PDF 下载 ==================
def download_pdf(url: str, save_path: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
        return False
    except Exception as e:
        print(f"    ⚠️  下载失败：{e}")
        return False


# ================== 主流程 ==================
def fetch_and_evaluate():
    cfg          = load_config()
    download_dir = os.path.join(BASE_DIR, cfg["download_dir"])
    os.makedirs(download_dir, exist_ok=True)

    enabled_topics = [t["name"] for t in cfg["topics"] if t.get("enabled")]
    if not enabled_topics:
        print("⚠️  没有启用任何领域，请在 topics_config.json 里把想要的领域改为 \"enabled\": true")
        return

    print(f"📚 已启用领域：{enabled_topics}")
    print(f"   热门范围：{TRENDING_FROM} ~ {TRENDING_TO}（近6个月）")
    print(f"   经典范围：{CLASSIC_FROM} ~ {CLASSIC_TO}（2020~2022）\n")

    seen_ids         = set()
    total_downloaded = 0

    for topic in enabled_topics:
        print(f"\n{'='*55}")
        print(f"🔍 领域：{topic}")

        trending = search_arxiv(topic, TRENDING_FROM, TRENDING_TO,
                                cfg["trending_count"])
        time.sleep(12)
        classic  = search_arxiv(topic, CLASSIC_FROM,  CLASSIC_TO,
                                cfg["classic_count"])
        time.sleep(12)

        all_papers = trending + classic
        print(f"  获取 {len(trending)} 篇热门 + {len(classic)} 篇经典，去重下载中...")

        downloaded = 0
        for paper in all_papers:
            pid = paper.get("arxiv_id", "")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            filename  = make_pdf_name(paper)
            save_path = os.path.join(download_dir, filename)

            if os.path.exists(save_path):
                print(f"  ⏭️  已存在，跳过：{filename}")
                continue

            print(f"  ⬇️  下载：{filename}")
            if download_pdf(paper["pdf_url"], save_path):
                downloaded += 1
                total_downloaded += 1
            time.sleep(3)

        print(f"  ✅ 本领域完成：{downloaded} 篇")

    print(f"\n{'='*55}")
    print(f"📥 全部下载完成，共 {total_downloaded} 篇新 PDF")
    print(f"📂 保存路径：{download_dir}\n")

    evaluate_folder(download_dir)


# ================== 入口 ==================
if __name__ == "__main__":
    fetch_and_evaluate()
