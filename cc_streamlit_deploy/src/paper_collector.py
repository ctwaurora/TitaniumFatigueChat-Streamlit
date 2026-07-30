"""
paper_collector.py — 开放获取文献自动下载模块

命令: python app.py collect-papers

只下载 Open Access PDF（OpenAlex best_oa_location / publisher OA / arXiv / PMC 等），
不绕过付费墙，不从非法镜像下载。
"""

import csv
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ── 内置目标文献清单 ──────────────────────────────────────────────────────

TARGET_PAPERS: Dict[str, List[Dict[str, str]]] = {
    "early_papers": [
        {"title": "A Review of the Fatigue Properties of Additively Manufactured Ti-6Al-4V",
         "notes": "综述：AM Ti-6Al-4V 疲劳性能"},
        {"title": "Critical assessment of the fatigue performance of additively manufactured Ti-6Al-4V and perspective for future research",
         "notes": "综述：AM Ti-6Al-4V 疲劳性能评估与展望"},
        {"title": "A Review of the As-Built SLM Ti-6Al-4V Mechanical Properties towards Achieving Fatigue Resistant Designs",
         "notes": "综述：As-built SLM Ti-6Al-4V 力学性能"},
    ],
    "papers": [
        {"title": "On the mechanical behaviour of titanium alloy TiAl6V4 manufactured by selective laser melting: Fatigue resistance and crack growth performance",
         "notes": "核心：SLM Ti-6Al-4V 疲劳抗力与裂纹扩展"},
        {"title": "Fatigue crack growth behavior and microstructural mechanisms in Ti-6Al-4V manufactured by laser engineered net shaping",
         "notes": "核心：LENS Ti-6Al-4V FCGR 与微观机制"},
        {"title": "Evaluation of fatigue crack propagation behaviour in Ti-6Al-4V manufactured by selective laser melting",
         "notes": "核心：SLM Ti-6Al-4V 裂纹扩展评估"},
        {"title": "Fatigue behaviour of additive manufactured Ti6Al4V, with as-built surfaces, exposed to variable amplitude loading",
         "notes": "核心：AM Ti6Al4V 变幅载荷疲劳"},
        {"title": "Fatigue life of additively manufactured Ti-6Al-4V in the very high cycle fatigue regime",
         "notes": "核心：AM Ti-6Al-4V VHCF 疲劳寿命"},
        {"title": "Fatigue crack growth behavior of laser powder bed fusion additive manufactured Ti-6Al-4V: Roles of post heat treatment and build orientation",
         "notes": "核心：LPBF Ti-6Al-4V 热处理与成形方向 FCGR"},
        {"title": "Impact of Surface and Pore Characteristics on Fatigue Life of Laser Powder Bed Fusion Ti-6Al-4V Alloy Described by Neural Network Models",
         "notes": "核心：表面/孔隙特征 + 神经网络预测疲劳寿命"},
        {"title": "Internal crack characteristics in very-high-cycle fatigue of a gradient structured Ti-6Al-4V alloy",
         "notes": "核心：梯度结构 Ti-6Al-4V VHCF 内部裂纹"},
        {"title": "Very High Cycle Fatigue Failure Mechanism of TC17 Alloy",
         "notes": "核心：TC17 VHCF 失效机制（次要案例）"},
    ],
    "followup_papers": [
        {"title": "Fatigue crack growth behavior of laser powder bed fusion additive manufactured Ti-6Al-4V: Roles of post heat treatment and build orientation",
         "notes": "补充：LPBF Ti-6Al-4V 热处理与成形方向 FCGR（与核心重复，下载时跳过）"},
        {"title": "Impact of Surface and Pore Characteristics on Fatigue Life of Laser Powder Bed Fusion Ti-6Al-4V Alloy Described by Neural Network Models",
         "notes": "补充：同核心，跳过"},
        {"title": "Fatigue crack segmentation and characterization of additively manufactured Ti-6Al-4V using X-ray computed tomography",
         "notes": "补充：CT 裂纹表征"},
        {"title": "Additively manufactured Ti-6Al-4V microstructure tailoring for improved fatigue life performance",
         "notes": "补充：微观组织调控提升疲劳寿命"},
    ],
}

OPENALEX_BASE = "https://api.openalex.org/works"
USER_AGENT = "TitaniumFatigueChat/2.0 (paper collector; mailto:research@example.com)"
REQUEST_DELAY = 1.0  # seconds between API calls

# 绕过系统代理检测 — OpenAlex 为公开 API，不需要代理
# 如用户需要代理，可设置环境变量 HTTP_PROXY / HTTPS_PROXY
_NO_PROXY = {"http": "", "https": ""}
_SESSION = requests.Session()
_SESSION.trust_env = False  # 不读取系统代理
_SESSION.headers.update({"User-Agent": USER_AGENT})
_SESSION.proxies = _NO_PROXY


def _sanitize_filename(title: str) -> str:
    """从标题生成文件名：第一作者_年份_关键词.pdf"""
    # 简化：从标题提取前 3-4 个关键单词
    import re
    # 去除特殊字符
    clean = re.sub(r'[^\w\s-]', '', title)
    words = clean.split()
    # 取前 4 个有意义的词
    key_words = [w for w in words if len(w) > 3][:4]
    if not key_words:
        key_words = words[:4]
    stem = "_".join(key_words).lower() if key_words else "unknown"
    return f"auto_{stem}.pdf"


def _search_openalex(title: str) -> Optional[Dict]:
    """通过 OpenAlex API 按标题搜索文献。"""
    query = quote(title[:200])
    url = f"{OPENALEX_BASE}?search={query}&per_page=3"
    try:
        resp = _SESSION.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        # 取第一个结果
        work = results[0]
        # 简单校验标题相似度
        returned_title = work.get("title", "").lower()
        query_lower = title.lower()[:60]
        # 检查是否匹配
        if returned_title and query_lower and (
            query_lower in returned_title or
            returned_title[:40] in query_lower
        ):
            return work
        # 如果第一个不匹配，检查后续结果
        for r in results[1:3]:
            rt = r.get("title", "").lower()
            if query_lower in rt or rt[:40] in query_lower:
                return r
        # 返回第一个结果但标记为低置信度
        return results[0]
    except requests.RequestException as e:
        return {"error": str(e)}


def _get_oa_pdf_url(work: Dict) -> Tuple[Optional[str], str]:
    """从 OpenAlex work 对象中提取开放获取 PDF URL。

    Returns:
        (pdf_url, source) — pdf_url 为 None 表示未找到 OA PDF
    """
    if "error" in work:
        return None, f"API error: {work['error']}"

    # 1. best_oa_location
    best_oa = work.get("best_oa_location") or {}
    if best_oa.get("pdf_url"):
        if best_oa.get("is_oa") or best_oa.get("license"):
            return best_oa["pdf_url"], "best_oa_location"

    # 2. primary_location
    primary = work.get("primary_location") or {}
    if primary.get("pdf_url"):
        return primary["pdf_url"], "primary_location"

    # 3. locations 中 is_oa=true 的 pdf_url
    locations = work.get("locations", []) or []
    for loc in locations:
        if loc.get("is_oa") and loc.get("pdf_url"):
            return loc["pdf_url"], f"locations[{locations.index(loc)}]"

    # 4. best_oa_location landing_page (无 PDF 但有 OA 版本)
    if best_oa.get("landing_page_url") and best_oa.get("is_oa"):
        return None, "oa_landing_only"

    # 5. primary_location landing_page
    if primary.get("landing_page_url"):
        return None, "landing_page_only"

    return None, "no_oa_location"


def _try_mdpi_direct_pdf(doi: str) -> Optional[str]:
    """MDPI 文章的官方 PDF URL 可直接从 DOI 构造。"""
    if not doi:
        return None
    doi_lower = doi.lower()
    if "mdpi" in doi_lower or "10.3390" in doi_lower:
        # MDPI PDF: https://www.mdpi.com/DOI/pdf
        pdf_url = f"https://www.mdpi.com/{doi.replace('doi:', '').strip()}/pdf"
        return pdf_url
    return None


def _download_pdf(url: str, save_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """下载 PDF，检查 Content-Type 和文件头。"""
    try:
        resp = _SESSION.get(url, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()

        # 检查 Content-Type
        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and "application/octet-stream" not in content_type:
            # 有些服务器返回 text/html 但仍可能是 PDF，检查前 4 字节
            pass  # 继续检查文件头

        # 检查文件头是否为 %PDF
        raw_start = resp.content[:4]
        if raw_start != b"%PDF":
            # 有些重定向后可能不是直接 PDF，尝试从 text 中提取 PDF 链接
            return False, f"Not a PDF (header: {raw_start!r}, Content-Type: {content_type})"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)

        file_size = save_path.stat().st_size
        if file_size < 1024:  # 小于 1KB 可能是错误页面
            save_path.unlink(missing_ok=True)
            return False, f"File too small ({file_size} bytes), likely error page"

        return True, f"OK ({file_size // 1024} KB)"

    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def _extract_doi(work: Dict) -> str:
    """从 work 中提取 DOI。"""
    doi = work.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi


def _extract_first_author(work: Dict) -> str:
    """从 work 中提取第一作者姓氏。"""
    authorships = work.get("authorships", []) or []
    if authorships:
        first = authorships[0].get("author", {}).get("display_name", "") or ""
        # 取姓氏（中文全名或英文 last name）
        if first:
            parts = first.split()
            return parts[-1] if len(parts) > 1 else first
    return "unknown"


def _extract_year(work: Dict) -> str:
    """从 work 中提取出版年份。"""
    date_str = work.get("publication_date", "") or ""
    if date_str and len(date_str) >= 4:
        return date_str[:4]
    year = work.get("publication_year", "")
    return str(year) if year else "unknown"


def _make_safe_filename(first_author: str, year: str, title: str) -> str:
    """生成安全的文件名。"""
    import re
    # 从标题提取关键词
    clean_title = re.sub(r'[^\w\s-]', '', title)
    words = clean_title.split()
    # 取前 3 个有意义的关键词
    keywords = [w for w in words if len(w) > 3][:3]
    if not keywords:
        keywords = words[:3]
    stem = "_".join(k.lower() for k in keywords)
    safe_author = re.sub(r'[^\w]', '', first_author)
    return f"{safe_author}_{year}_{stem}.pdf"


def _deduplicate_skip(title: str, target_folder: str, log: list) -> bool:
    """检查是否已在目标文件夹中存在（同名或内容相同）。"""
    folder_map = {
        "early_papers": BASE_DIR / "early_papers",
        "papers": BASE_DIR / "papers",
        "followup_papers": BASE_DIR / "followup_papers",
    }
    folder = folder_map.get(target_folder)
    if not folder or not folder.exists():
        return False
    # 简单的标题关键词匹配检查
    title_lower = title.lower()
    for existing_path in folder.glob("*.pdf"):
        # 检查文件名是否包含标题中的关键词
        existing_stem = existing_path.stem.lower()
        key_words = [w for w in title_lower.split() if len(w) > 5]
        if any(kw in existing_stem for kw in key_words):
            log.append(f"  ⏭️ 跳过: {title[:60]}... (已存在: {existing_path.name})")
            return True
    return False


def _write_collection_report(
    results: Dict[str, List[Dict]],
    stats: Dict,
    paper_targets: Dict[str, List[Dict]],
) -> str:
    """生成 outputs/paper_collection_report.md"""
    total_target = sum(len(papers) for papers in paper_targets.values())
    total_downloaded = stats["downloaded"]
    total_missing = stats["missing"]
    total_failed = stats["failed"]
    total_skipped = stats["skipped"]

    lines = [
        "# Paper Collection Report（文献收集报告）",
        "",
        f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **目标文献总数**: {total_target}",
        f"- **成功下载**: {total_downloaded}",
        f"- **未找到 OA PDF**: {total_missing}",
        f"- **下载失败**: {total_failed}",
        f"- **已存在跳过**: {total_skipped}",
        "",
        "---",
        "",
    ]

    for folder_name in ["early_papers", "papers", "followup_papers"]:
        folder_results = results.get(folder_name, [])
        targets = paper_targets.get(folder_name, [])
        lines.append(f"## {folder_name}/ （目标 {len(targets)} 篇）\n")
        for r in folder_results:
            status_icon = {
                "downloaded": "✅", "skipped": "⏭️", "missing": "❌", "failed": "⚠️",
            }.get(r["status"], "❓")
            lines.append(f"{status_icon} **{r['title'][:80]}**")
            lines.append(f"  - 状态: {r['status']}")
            if r["status"] == "downloaded":
                lines.append(f"  - 文件: `{r['filename']}`")
                lines.append(f"  - 来源: {r.get('source', 'N/A')}")
            elif r["status"] == "missing":
                lines.append(f"  - DOI: {r.get('doi', 'N/A')}")
                lines.append(f"  - 入口: {r.get('landing_page', 'N/A')}")
            elif r["status"] == "failed":
                lines.append(f"  - 原因: {r.get('error', 'N/A')}")
            lines.append("")

    lines.append("---")
    lines.append("> 注意：本工具仅下载开放获取（Open Access）PDF。")
    lines.append("> 未找到 OA PDF 的文献已记录在 data/missing_papers.csv，")
    lines.append("> 请通过 institutional access 或作者邮件手动获取。")

    out_path = OUTPUTS_DIR / "paper_collection_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def _write_csvs(results: Dict[str, List[Dict]]) -> None:
    """写入 downloaded_papers.csv 和 missing_papers.csv"""
    # Downloaded
    downloaded = []
    missing = []
    for folder_name, items in results.items():
        for r in items:
            if r["status"] == "downloaded":
                downloaded.append({
                    "folder": folder_name,
                    "title": r["title"],
                    "filename": r.get("filename", ""),
                    "source": r.get("source", ""),
                    "doi": r.get("doi", ""),
                    "url": r.get("url", ""),
                })
            elif r["status"] in ("missing", "failed"):
                missing.append({
                    "folder": folder_name,
                    "title": r["title"],
                    "status": r["status"],
                    "doi": r.get("doi", ""),
                    "landing_page": r.get("landing_page", ""),
                    "notes": r.get("error", r.get("source", "")),
                })

    # Write downloaded_papers.csv
    dw_path = DATA_DIR / "downloaded_papers.csv"
    dw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dw_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "title", "filename", "source", "doi", "url"])
        w.writeheader()
        for r in downloaded:
            w.writerow(r)

    # Write missing_papers.csv
    ms_path = DATA_DIR / "missing_papers.csv"
    with open(ms_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "title", "status", "doi", "landing_page", "notes"])
        w.writeheader()
        for r in missing:
            w.writerow(r)

    return dw_path, ms_path


def run_collect_papers() -> Dict:
    """执行文献收集主流程。

    Returns:
        统计信息 dict
    """
    stats = {
        "downloaded": 0,
        "skipped": 0,
        "missing": 0,
        "failed": 0,
        "total_target": sum(len(v) for v in TARGET_PAPERS.values()),
    }
    all_results = {}

    # Deduplicate check: if same title appears in multiple folders, skip later
    seen_titles = set()

    for folder_name, papers in TARGET_PAPERS.items():
        target_folder = BASE_DIR / folder_name
        target_folder.mkdir(parents=True, exist_ok=True)
        folder_results = []

        for paper in papers:
            title = paper["title"]
            notes = paper.get("notes", "")

            # Deduplicate across folders
            if title in seen_titles:
                folder_results.append({
                    "title": title,
                    "status": "skipped",
                    "filename": "",
                    "source": "duplicate_across_folders",
                    "doi": "",
                    "error": "已在之前文件夹中出现过",
                })
                stats["skipped"] += 1
                continue
            seen_titles.add(title)

            # Check if already exists in target folder (by filename keyword match)
            if _deduplicate_skip(title, folder_name, []):
                folder_results.append({
                    "title": title,
                    "status": "skipped",
                    "filename": "",
                    "source": "already_exists",
                    "doi": "",
                    "error": "文件中已存在",
                })
                stats["skipped"] += 1
                continue

            # Step 1: Search OpenAlex
            print(f"  🔍 搜索: {title[:70]}...")
            time.sleep(REQUEST_DELAY)
            work = _search_openalex(title)

            if work is None:
                folder_results.append({
                    "title": title,
                    "status": "missing",
                    "doi": "",
                    "landing_page": "",
                    "source": "openalex_no_result",
                    "error": "OpenAlex 未找到该文献",
                })
                stats["missing"] += 1
                continue

            if "error" in work:
                folder_results.append({
                    "title": title,
                    "status": "failed",
                    "doi": "",
                    "landing_page": "",
                    "source": "openalex_error",
                    "error": work["error"],
                })
                stats["failed"] += 1
                continue

            doi = _extract_doi(work)
            first_author = _extract_first_author(work)
            year = _extract_year(work)

            # Step 2: Get OA PDF URL
            pdf_url, source = _get_oa_pdf_url(work)

            # Step 3: Try MDPI direct PDF if applicable
            if pdf_url is None and doi:
                mdpi_url = _try_mdpi_direct_pdf(doi)
                if mdpi_url:
                    pdf_url = mdpi_url
                    source = "mdpi_direct"

            # Step 4: Download or record missing
            landing_page = ""
            best_oa = work.get("best_oa_location") or {}
            if best_oa.get("landing_page_url"):
                landing_page = best_oa["landing_page_url"]
            primary = work.get("primary_location") or {}
            if not landing_page and primary.get("landing_page_url"):
                landing_page = primary["landing_page_url"]

            if pdf_url:
                # Generate filename
                filename = _make_safe_filename(first_author, year, title)
                save_path = target_folder / filename

                print(f"  ⬇️  下载: {filename}")
                success, message = _download_pdf(pdf_url, save_path)

                if success:
                    folder_results.append({
                        "title": title,
                        "status": "downloaded",
                        "filename": filename,
                        "source": source,
                        "doi": doi,
                        "url": pdf_url,
                    })
                    stats["downloaded"] += 1
                    print(f"     ✅ {message}")
                else:
                    # Download failed — try alternative: check all locations
                    alt_pdf = None
                    alt_source = ""
                    locations = work.get("locations", []) or []
                    for loc in locations:
                        if loc.get("pdf_url") and loc["pdf_url"] != pdf_url:
                            alt_pdf = loc["pdf_url"]
                            alt_source = f"locations_alt_{locations.index(loc)}"
                            break

                    if alt_pdf:
                        print(f"  ⬇️  重试(备选链接): {filename}")
                        success2, message2 = _download_pdf(alt_pdf, save_path)
                        if success2:
                            folder_results.append({
                                "title": title,
                                "status": "downloaded",
                                "filename": filename,
                                "source": alt_source,
                                "doi": doi,
                                "url": alt_pdf,
                            })
                            stats["downloaded"] += 1
                            print(f"     ✅ {message2}")
                            continue

                    folder_results.append({
                        "title": title,
                        "status": "failed",
                        "doi": doi,
                        "landing_page": landing_page,
                        "source": source,
                        "error": message,
                    })
                    stats["failed"] += 1
                    print(f"     ❌ {message}")
            else:
                folder_results.append({
                    "title": title,
                    "status": "missing",
                    "doi": doi,
                    "landing_page": landing_page,
                    "source": source,
                    "error": "没有开放的 PDF 链接",
                })
                stats["missing"] += 1
                print(f"     ❌ 未找到 OA PDF (入口: {landing_page[:80] if landing_page else 'N/A'})")

        all_results[folder_name] = folder_results

    # Generate reports
    report_path = _write_collection_report(all_results, stats, TARGET_PAPERS)
    dw_path, ms_path = _write_csvs(all_results)

    stats["report_path"] = report_path
    stats["downloaded_csv"] = str(dw_path)
    stats["missing_csv"] = str(ms_path)
    stats["results"] = all_results

    return stats
