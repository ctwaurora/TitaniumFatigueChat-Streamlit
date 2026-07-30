"""
auto_collector.py — 自动开放文献采集模块

命令: python app.py auto

按关键词从 OpenAlex 搜索钛合金疲劳开放文献，
只下载 OA PDF，保存到 papers/，自动去重，
不绕过付费墙，不使用非法镜像。
"""

import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
PAPERS_DIR = BASE_DIR / "papers"

# ── 检索关键词 ──────────────────────────────────────────────────────────────

SEARCH_KEYWORDS = [
    "Ti-6Al-4V fatigue",
    "Ti-6Al-4V fatigue crack growth",
    "additive manufacturing Ti-6Al-4V fatigue",
    "LPBF Ti-6Al-4V fatigue",
    "SLM Ti-6Al-4V fatigue",
    "titanium alloy fatigue crack growth",
    "Ti-6Al-4V pore defect fatigue",
    "Ti-6Al-4V micro-CT fatigue",
    "titanium alloy very high cycle fatigue",
    "Ti-6Al-4V Paris model fatigue",
]

# ── API 配置 ─────────────────────────────────────────────────────────────────

OPENALEX_BASE = "https://api.openalex.org/works"
USER_AGENT = "TitaniumFatigueChat/2.0 (auto collector; mailto:research@example.com)"
REQUEST_DELAY = 1.0  # seconds between API calls
MAX_RESULTS_PER_KEYWORD = 25  # max works to fetch per keyword

# 绕过系统代理 — OpenAlex 为公开 API
_NO_PROXY = {"http": "", "https": ""}
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.headers.update({"User-Agent": USER_AGENT})
_SESSION.proxies = _NO_PROXY


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _normalize_title(title: str) -> str:
    """规范化标题用于去重比较。"""
    t = title.lower().strip()
    # 去除标点
    t = re.sub(r'[^\w\s]', '', t)
    # 合并空白
    t = re.sub(r'\s+', ' ', t)
    return t


def _sanitize_filename(title: str) -> str:
    """从标题生成安全的文件名。"""
    # 提取第一作者（如果有author信息则更好）
    clean = re.sub(r'[^\w\s-]', '', title)
    words = clean.split()
    key_words = [w for w in words if len(w) > 3][:5]
    if not key_words:
        key_words = words[:5]
    stem = "_".join(k.lower() for k in key_words)
    return f"auto_{stem}.pdf"


def _extract_doi(work: Dict) -> str:
    doi = work.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi


def _extract_first_author(work: Dict) -> str:
    authorships = work.get("authorships", []) or []
    if authorships:
        first = authorships[0].get("author", {}).get("display_name", "") or ""
        if first:
            parts = first.split()
            return parts[-1] if len(parts) > 1 else first
    return "unknown"


def _extract_year(work: Dict) -> str:
    date_str = work.get("publication_date", "") or ""
    if date_str and len(date_str) >= 4:
        return date_str[:4]
    year = work.get("publication_year", "")
    return str(year) if year else "unknown"


def _get_arxiv_id(work: Dict) -> Optional[str]:
    """从 work 中提取 arXiv ID（如果有）。"""
    ids = work.get("ids", {}) or {}
    arxiv = ids.get("arxiv", "") or ""
    if arxiv:
        # 格式通常是 https://arxiv.org/abs/xxxx.xxxxx
        arxiv = arxiv.replace("https://arxiv.org/abs/", "").strip()
        return arxiv if arxiv else None
    return None


def _get_oa_pdf_url(work: Dict) -> Tuple[Optional[str], str]:
    """提取开放获取 PDF URL。"""
    if "error" in work:
        return None, f"API error: {work['error']}"

    best_oa = work.get("best_oa_location") or {}
    if best_oa.get("pdf_url"):
        if best_oa.get("is_oa") or best_oa.get("license"):
            return best_oa["pdf_url"], "best_oa_location"

    primary = work.get("primary_location") or {}
    if primary.get("pdf_url"):
        return primary["pdf_url"], "primary_location"

    locations = work.get("locations", []) or []
    for loc in locations:
        if loc.get("is_oa") and loc.get("pdf_url"):
            return loc["pdf_url"], f"locations"

    # arXiv PDF from IDs
    ids = work.get("ids", {}) or {}
    arxiv = ids.get("arxiv", "") or ""
    if arxiv:
        arxiv_id = arxiv.replace("https://arxiv.org/abs/", "").strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return pdf_url, "arxiv"

    return None, "no_oa_location"


def _download_pdf(url: str, save_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """下载 PDF，检查 Content-Type 和文件头。"""
    try:
        resp = _SESSION.get(url, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()

        raw_start = resp.content[:4]
        if raw_start != b"%PDF":
            return False, f"Not a PDF (header: {raw_start!r})"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)

        file_size = save_path.stat().st_size
        if file_size < 1024:
            save_path.unlink(missing_ok=True)
            return False, f"File too small ({file_size} bytes)"

        return True, f"OK ({file_size // 1024} KB)"
    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected: {e}"


def _make_safe_filename(first_author: str, year: str, title: str) -> str:
    """生成安全的文件名：作者_年_关键词.pdf"""
    clean_title = re.sub(r'[^\w\s-]', '', title)
    words = clean_title.split()
    keywords = [w for w in words if len(w) > 3][:3]
    if not keywords:
        keywords = words[:3]
    stem = "_".join(k.lower() for k in keywords)
    safe_author = re.sub(r'[^\w]', '', first_author)
    return f"auto_{safe_author}_{year}_{stem}.pdf"


def _search_openalex_keyword(keyword: str, per_page: int = 25) -> List[Dict]:
    """通过 OpenAlex API 按关键词搜索文献。"""
    query = quote(keyword)
    url = f"{OPENALEX_BASE}?search={query}&per_page={per_page}&sort=relevance"
    try:
        resp = _SESSION.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.RequestException as e:
        print(f"  ⚠️  OpenAlex API error: {e}")
        return []


# ── 去重 ─────────────────────────────────────────────────────────────────────


class DedupSet:
    """跨多篇文献的去重集合。"""

    def __init__(self):
        self._seen_dois: Set[str] = set()
        self._seen_arxiv_ids: Set[str] = set()
        self._seen_titles: Set[str] = set()

    def is_duplicate(self, work: Dict) -> bool:
        doi = _extract_doi(work)
        arxiv_id = _get_arxiv_id(work)
        title_norm = _normalize_title(work.get("title", ""))

        if doi and doi in self._seen_dois:
            return True
        if arxiv_id and arxiv_id in self._seen_arxiv_ids:
            return True
        if title_norm and title_norm in self._seen_titles:
            return True
        return False

    def add(self, work: Dict):
        doi = _extract_doi(work)
        arxiv_id = _get_arxiv_id(work)
        title_norm = _normalize_title(work.get("title", ""))

        if doi:
            self._seen_dois.add(doi)
        if arxiv_id:
            self._seen_arxiv_ids.add(arxiv_id)
        if title_norm:
            self._seen_titles.add(title_norm)


# ── 主流程 ───────────────────────────────────────────────────────────────────


def run_auto_collect() -> Dict:
    """执行自动开放文献采集。

    Returns:
        stats dict with keys: searched, new, skipped_dedup, skipped_exists,
        missing_oa, failed, downloaded, downloaded_csv, missing_csv, report_path
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dedup = DedupSet()
    all_works: List[Dict] = []  # all unique works collected
    seen_titles_local: Set[str] = set()

    print(f"\n{'='*60}")
    print(f"  🔍 自动开放文献采集 — 搜索 {len(SEARCH_KEYWORDS)} 个关键词")
    print(f"{'='*60}\n")

    for kw in SEARCH_KEYWORDS:
        print(f"  📌 搜索: {kw}")
        time.sleep(REQUEST_DELAY)
        results = _search_openalex_keyword(kw)

        new_count = 0
        for work in results:
            title = work.get("title", "")
            if not title:
                continue

            # Deduplicate
            if dedup.is_duplicate(work):
                continue

            title_norm = _normalize_title(title)
            if title_norm in seen_titles_local:
                continue

            dedup.add(work)
            seen_titles_local.add(title_norm)
            all_works.append(work)
            new_count += 1

        print(f"     → 新增 {new_count} 篇（累计唯一 {len(all_works)} 篇）")

    print(f"\n  共找到 {len(all_works)} 篇唯一文献\n")

    # ── 第二阶段：尝试下载 OA PDF ──
    downloaded = []
    missing = []
    skipped_dedup_count = 0
    skipped_exists_count = 0
    failed_count = 0
    missing_oa_count = 0

    print(f"  开始下载 OA PDF...\n")

    for i, work in enumerate(all_works, 1):
        title = work.get("title", "")
        doi = _extract_doi(work)
        first_author = _extract_first_author(work)
        year = _extract_year(work)

        print(f"  [{i}/{len(all_works)}] {title[:70]}...")

        # 检查 papers/ 中是否已存在（按文件名关键词匹配）
        filename = _make_safe_filename(first_author, year, title)
        save_path = PAPERS_DIR / filename

        # 也检查是否有类似文件名的 PDF（关键词匹配）
        title_lower = title.lower()
        title_keywords = {w for w in title_lower.split() if len(w) > 4}
        already_exists = False
        for existing_path in PAPERS_DIR.glob("*.pdf"):
            existing_stem = existing_path.stem.lower()
            if any(kw in existing_stem for kw in title_keywords):
                already_exists = True
                break
            # Also check by author/year pattern
            if first_author.lower()[:4] in existing_stem and year in existing_stem:
                already_exists = True
                break

        if already_exists:
            print(f"     ⏭️  已存在")
            skipped_exists_count += 1
            continue

        # 获取 OA PDF URL
        pdf_url, source = _get_oa_pdf_url(work)

        if not pdf_url:
            print(f"     ❌ 未找到 OA PDF")
            missing.append({
                "title": title, "doi": doi, "first_author": first_author,
                "year": year, "reason": source,
            })
            missing_oa_count += 1
            continue

        # 下载
        success, message = _download_pdf(pdf_url, save_path)
        if success:
            print(f"     ✅ {message}")
            downloaded.append({
                "title": title, "doi": doi, "first_author": first_author,
                "year": year, "filename": filename, "source": source,
                "url": pdf_url,
            })
        else:
            print(f"     ❌ 下载失败: {message}")
            failed_count += 1
            missing.append({
                "title": title, "doi": doi, "first_author": first_author,
                "year": year, "reason": f"download_failed: {message}",
            })

    # ── 写入 CSV ──
    # Downloaded CSV
    dw_path = DATA_DIR / "auto_downloaded_papers.csv"
    dw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dw_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "doi", "first_author", "year", "filename", "source", "url"])
        w.writeheader()
        for r in downloaded:
            w.writerow(r)

    # Missing CSV
    ms_path = DATA_DIR / "auto_missing_papers.csv"
    with open(ms_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "doi", "first_author", "year", "reason"])
        w.writeheader()
        for r in missing:
            w.writerow(r)

    stats = {
        "searched": len(SEARCH_KEYWORDS),
        "total_found": len(all_works),
        "new": len(downloaded),
        "skipped_dedup": skipped_dedup_count,
        "skipped_exists": skipped_exists_count,
        "missing_oa": missing_oa_count,
        "failed": failed_count,
        "downloaded": len(downloaded),
        "downloaded_csv": str(dw_path.resolve()),
        "missing_csv": str(ms_path.resolve()),
        "downloaded_list": downloaded,
        "missing_list": missing,
    }

    # ── 写入 Pipeline 报告 ──
    _write_pipeline_report(stats)

    return stats


def _write_pipeline_report(stats: Dict) -> str:
    """生成 outputs/auto_pipeline_report.md"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Auto Pipeline Report（自动采集流程报告）",
        "",
        f"- **生成时间**: {now}",
        f"- **搜索关键词数**: {stats['searched']}",
        f"- **发现唯一文献**: {stats['total_found']}",
        f"- **成功下载 OA PDF**: {stats['downloaded']}",
        f"- **已存在跳过**: {stats['skipped_exists']}",
        f"- **未找到 OA PDF**: {stats['missing_oa']}",
        f"- **下载失败**: {stats['failed']}",
        "",
        "---",
        "",
        "## 下载的文献\n",
    ]

    if stats["downloaded_list"]:
        for r in stats["downloaded_list"]:
            lines.append(f"- **{r['title'][:80]}**")
            lines.append(f"  - 文件: `{r['filename']}`")
            lines.append(f"  - DOI: {r['doi'] or 'N/A'}")
            lines.append(f"  - 来源: {r['source']}")
            lines.append("")
    else:
        lines.append("（无新下载文献）\n")

    lines.extend([
        "---",
        "",
        "## 缺失 OA PDF 的文献\n",
    ])

    if stats["missing_list"]:
        for r in stats["missing_list"]:
            lines.append(f"- **{r['title'][:80]}**")
            lines.append(f"  - DOI: {r['doi'] or 'N/A'}")
            lines.append(f"  - 原因: {r.get('reason', 'N/A')}")
            lines.append("")
    else:
        lines.append("（无缺失文献）\n")

    lines.extend([
        "---",
        "",
        "## 后续 Pipeline 执行结果\n",
        "",
        "以下为自动运行的串联流程结果：",
        "",
        "1. `python app.py ingest`      — 文献卡片抽取",
        "2. `python app.py discover`    — 研究空白发现",
        "3. `python app.py validate`    — 质量门禁 + 推荐卡片",
        "4. `python app.py demo`        — 完整导出",
        "",
        "各步骤输出见对应 outputs/ 文件。",
        "",
        "---",
        "",
        "> ⚠️ 本工具仅下载开放获取（Open Access）PDF。",
        "> 不绕过付费墙，不使用非法镜像。",
        "> 缺失文献可通过 institutional access 或联系作者获取。",
    ])

    report_path = OUTPUTS_DIR / "auto_pipeline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    stats["report_path"] = str(report_path)
    return str(report_path)
