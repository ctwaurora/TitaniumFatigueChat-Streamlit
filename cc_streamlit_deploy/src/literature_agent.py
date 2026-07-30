"""
literature_agent.py — Auto literature search, ranking, ingestion & database growth
面向 L-PBF Ti-6Al-4V 疲劳文献自动补充。

依赖:
    src/literature_search_agent.py (基础检索)
    data/literature_database.csv (正式文献库)
    data/candidate_papers.csv (候选文献库)

流程:
    user_question → extract_variable_pair → assess_coverage →
    if weak: search → rank → dedup → save to candidate_papers.csv →
    if high_relevance: ingest into literature_database.csv →
    if OA: track download_status → report
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── CSV field definitions ──

from src.paper_classifier import CANDIDATE_FIELDS as PC_CANDIDATE_FIELDS

# Use the unified field definitions from paper_classifier
CANDIDATE_FIELDS = PC_CANDIDATE_FIELDS

SEARCH_RESULT_FIELDS = [
    "title", "authors", "year", "doi", "url", "source_database",
    "is_open_access", "pdf_url", "relevance_score", "matched_query",
    "download_status",
]

MANUAL_DOWNLOAD_FIELDS = [
    "title", "doi", "url", "reason", "suggested_action",
]

DOWNLOAD_STATUS_FIELDS = [
    "title", "doi", "pdf_url", "download_status", "local_path", "error_message",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load existing databases
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(path: Path, fields: List[str]) -> List[Dict[str, str]]:
    """加载 CSV 文件，返回 list of dict。如果文件不存在则返回空列表。"""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception:
        return []


def save_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]):
    """保存 rows 到 CSV 文件。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def load_literature_database() -> List[Dict[str, str]]:
    return load_csv(DATA_DIR / "literature_database.csv",
                    ["paper_id", "title", "authors", "year", "doi", "material",
                     "manufacturing_process", "heat_treatment", "surface_state",
                     "defect_type", "pore_size", "pore_location", "porosity",
                     "stress_ratio_R", "Nf", "fatigue_limit", "da_dN",
                     "Paris_C", "Paris_m", "Delta_Kth", "main_conclusion"])


def load_candidate_papers() -> List[Dict[str, str]]:
    return load_csv(DATA_DIR / "candidate_papers.csv", CANDIDATE_FIELDS)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Deduplication
# ═══════════════════════════════════════════════════════════════════════════

def normalize_title(title: str) -> str:
    """标准化标题用于去重比较。"""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9]", "", t)
    return t[:80]


def is_duplicate(
    paper: Dict[str, Any],
    existing_papers: List[Dict[str, str]],
) -> Tuple[bool, str]:
    """
    检查 paper 是否与已有文献重复。
    返回 (is_dup, reason)。

    去重规则:
    1. DOI 精确匹配
    2. 标题归一化后匹配
    """
    new_doi = (paper.get("doi") or "").strip().lower()
    new_title = normalize_title(paper.get("title", ""))

    for existing in existing_papers:
        existing_doi = (existing.get("doi") or "").strip().lower()
        existing_title = normalize_title(existing.get("title", ""))

        if new_doi and existing_doi and new_doi == existing_doi:
            return True, f"DOI 重复: {new_doi}"

        if new_title and existing_title and new_title == existing_title:
            return True, f"标题重复: {paper.get('title', '')[:60]}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# 3. Relevance scoring for literature ingestion
# ═══════════════════════════════════════════════════════════════════════════

def score_paper_for_ingestion(
    paper: Dict[str, Any],
    ind_var: Optional[str] = None,
    dep_var: Optional[str] = None,
) -> float:
    """
    对论文进行加入正式库的评分 (0.0 - 1.0)。

    规则:
    - 标题/摘要包含 L-PBF/SLM/Ti-6Al-4V/fatigue → +0.3
    - 同时包含自变量和因变量 → +0.3
    - 有 DOI → +0.1
    - 有 OA 全文 → +0.1
    - 有实验数据关键词 → +0.2
    """
    score = 0.0
    text = (
        (paper.get("title") or "") + " " +
        (paper.get("abstract") or "") + " " +
        (paper.get("keywords") or "")
    ).lower()

    # Domain match (+0.3)
    domain_kw = ["l-pbf", "slm", "laser powder bed", "ti-6al-4v", "ti64",
                  "additive manufactur", "titanium", "fatigue"]
    if any(kw in text for kw in domain_kw):
        score += 0.3

    # Variable match (+0.3)
    from src.literature_search_agent import VAR_SEARCH_KEYWORDS
    if ind_var:
        ind_kw = VAR_SEARCH_KEYWORDS.get(ind_var, [f'"{ind_var}"'])
        if any(kw.strip('"').lower() in text for kw in ind_kw):
            score += 0.15
    if dep_var:
        dep_kw = VAR_SEARCH_KEYWORDS.get(dep_var, [f'"{dep_var}"'])
        if any(kw.strip('"').lower() in text for kw in dep_kw):
            score += 0.15

    # DOI presence (+0.1)
    if paper.get("doi"):
        score += 0.1

    # OA availability (+0.1)
    if paper.get("is_open_access"):
        score += 0.1

    # Experimental data keywords (+0.2)
    exp_kw = ["experiment", "fatigue test", "fcgr", "s-n", "micro-ct",
              "hcf", "vhcf", "crack growth", "fractography", "paris",
              "murakami", "kitagawa", "el haddad"]
    if any(kw in text for kw in exp_kw):
        score += 0.2

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Search and ingest pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_auto_literature_update(
    user_query: str,
    ind_var: Optional[str] = None,
    dep_var: Optional[str] = None,
    max_results: int = 10,
    use_api: bool = True,
) -> Dict[str, Any]:
    """
    执行自动文献补充流程。

    Args:
        user_query: 用户原始输入
        ind_var: 自变量 (canonical name)
        dep_var: 因变量 (canonical name)
        max_results: 每个 API 最大返回数
        use_api: 是否尝试联网检索

    Returns:
        {
            "queries_generated": int,
            "api_success": bool,
            "candidates_new": int,      # 新增到 candidate_papers.csv
            "ingested_new": int,         # 新增到 literature_database.csv
            "oa_found": int,             # OA 文献数
            "manual_needed": int,        # 需手动下载数
            "total_literature": int,     # literature_database.csv 当前总量
            "search_plan_path": str,
            "report_path": str,
            "errors": [str],
        }
    """
    from src.literature_search_agent import (
        generate_search_queries,
        search_openalex,
        search_semantic_scholar,
        search_crossref,
        merge_search_results,
        save_search_results,
    )

    result: Dict[str, Any] = {
        "queries_generated": 0,
        "api_success": False,
        "candidates_new": 0,
        "ingested_new": 0,
        "oa_found": 0,
        "manual_needed": 0,
        "total_literature": 0,
        "search_plan_path": "",
        "report_path": "",
        "errors": [],
    }

    # Step 1: Generate search queries
    queries = generate_search_queries(ind_var, dep_var, user_query)
    result["queries_generated"] = len(queries)

    # Step 2: Search APIs
    all_results: List[Dict] = []
    if use_api and queries:
        try:
            oa = search_openalex(queries[0], max_results)
            time.sleep(0.3)
        except Exception as e:
            result["errors"].append(f"OpenAlex error: {e}")
            oa = []

        try:
            s2 = search_semantic_scholar(
                f"{ind_var or ''} {dep_var or ''} L-PBF Ti-6Al-4V fatigue",
                max_results,
            )
            time.sleep(0.3)
        except Exception as e:
            result["errors"].append(f"Semantic Scholar error: {e}")
            s2 = []

        try:
            cr = search_crossref(
                f"{ind_var or ''} {dep_var or ''} Ti-6Al-4V fatigue",
                max_results,
            )
        except Exception as e:
            result["errors"].append(f"Crossref error: {e}")
            cr = []

        all_results = merge_search_results(oa, s2, cr, ind_var, dep_var, user_query)
        if all_results:
            result["api_success"] = True

    # Step 3: Save raw search results
    if all_results:
        save_search_results(all_results)

    # Step 4: Load existing databases
    candidate_papers = load_candidate_papers()
    lit_db = load_literature_database()

    existing_dois = set()
    for p in candidate_papers:
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            existing_dois.add(doi)
    for p in lit_db:
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            existing_dois.add(doi)

    existing_titles = set()
    for p in candidate_papers:
        t = normalize_title(p.get("title", ""))
        if t:
            existing_titles.add(t)
    for p in lit_db:
        t = normalize_title(p.get("title", ""))
        if t:
            existing_titles.add(t)

    # Step 5: Process results — dedup, score, categorize
    candidates_added = 0
    ingested_added = 0
    oa_count = 0
    manual_count = 0

    now_str = datetime.now().isoformat()

    # Import paper classifier
    from src.paper_classifier import classify_paper_type, generate_filename, ensure_paper_dirs

    ensure_paper_dirs()

    for paper in all_results:
        # Dedup against existing
        dup, reason = is_duplicate(paper, candidate_papers + lit_db)
        if dup:
            continue

        # Score for ingestion
        relevance = score_paper_for_ingestion(paper, ind_var, dep_var)
        paper["relevance_score"] = relevance

        # ── Classify paper type ──
        title = (paper.get("title") or "")
        abstract = (paper.get("abstract") or "")
        classification = classify_paper_type(title, abstract)

        # Decide status
        is_oa = paper.get("is_open_access", False)
        if is_oa:
            oa_count += 1
        else:
            manual_count += 1

        # ── Create candidate entry (with classification fields) ──
        candidate_entry = {
            "candidate_id": f"CAND_{len(candidate_papers) + candidates_added + 1:04d}",
            "title": title[:300],
            "authors": (paper.get("authors") or "")[:500],
            "year": str(paper.get("year") or ""),
            "journal": (paper.get("source_name") or "")[:200],
            "doi": (paper.get("doi") or ""),
            "url": (paper.get("url") or ""),
            "source_database": (paper.get("source_database") or ""),
            "abstract": abstract[:2000],
            "keywords": "",
            "is_open_access": str(is_oa),
            "pdf_url": (paper.get("pdf_url") or ""),
            "matched_query": queries[0] if queries else "",
            "relevance_score": f"{relevance:.2f}",
            "recommended_reason": "",
            "status": "candidate",
            "full_text_status": "open_access" if is_oa else "unavailable",
            "added_time": now_str,
            "last_updated": now_str,
            # Classification fields
            "paper_type_primary": classification["paper_type_primary"],
            "paper_type_secondary": classification["paper_type_secondary"],
            "research_topic": classification["research_topic"],
            "storage_folder": classification["storage_folder"],
            "tags": classification["tags"],
            "is_review": str(classification["is_review"]),
            "is_experimental": str(classification["is_experimental"]),
            "is_fcgr": str(classification["is_fcgr"]),
            "is_micro_ct": str(classification["is_micro_ct"]),
            "is_heat_treatment": str(classification["is_heat_treatment"]),
            "is_surface_roughness": str(classification["is_surface_roughness"]),
            "is_conflict_relevant": str(classification["is_conflict_relevant"]),
            "classification_reason": classification["classification_reason"],
            "classification_confidence": classification["classification_confidence"],
        }
        candidate_papers.append(candidate_entry)
        candidates_added += 1

        # If high relevance, also ingest into literature_database
        if relevance >= 0.7:
            ingested_from = f"auto_search_{paper.get('source_database', 'unknown')}"
            lit_entry = {
                "paper_id": f"LPBF_{len(lit_db) + ingested_added + 1:04d}",
                "title": title[:300],
                "authors": (paper.get("authors") or "")[:500],
                "year": str(paper.get("year") or ""),
                "doi": (paper.get("doi") or ""),
                "material": "Ti-6Al-4V",
                "manufacturing_process": "L-PBF",
                "heat_treatment": "",
                "surface_state": "",
                "defect_type": "",
                "pore_size": "",
                "pore_location": "",
                "porosity": "",
                "stress_ratio_R": "",
                "Nf": "",
                "fatigue_limit": "",
                "da_dN": "",
                "Paris_C": "",
                "Paris_m": "",
                "Delta_Kth": "",
                "main_conclusion": "",
                # Classification fields
                "paper_type_primary": classification["paper_type_primary"],
                "paper_type_secondary": classification["paper_type_secondary"],
                "research_topic": classification["research_topic"],
                "storage_folder": classification["storage_folder"],
                "tags": classification["tags"],
                "is_review": str(classification["is_review"]),
                "is_experimental": str(classification["is_experimental"]),
                "is_fcgr": str(classification["is_fcgr"]),
                "is_micro_ct": str(classification["is_micro_ct"]),
                "is_heat_treatment": str(classification["is_heat_treatment"]),
                "is_surface_roughness": str(classification["is_surface_roughness"]),
                "is_conflict_relevant": str(classification["is_conflict_relevant"]),
                "classification_reason": classification["classification_reason"],
                "classification_confidence": classification["classification_confidence"],
                "ingested_from": ingested_from,
            }
            lit_db.append(lit_entry)
            ingested_added += 1

    # Step 6: Save updated databases
    save_csv(DATA_DIR / "candidate_papers.csv", candidate_papers, CANDIDATE_FIELDS)

    from src.paper_classifier import LIT_DB_FIELDS
    lit_fields = LIT_DB_FIELDS
    save_csv(DATA_DIR / "literature_database.csv", lit_db, lit_fields)

    # Step 7: Generate search plan markdown
    plan = _generate_auto_report(queries, all_results, candidate_papers, lit_db,
                                  ind_var, dep_var, user_query, result)
    plan_path = OUTPUTS_DIR / "22_literature_search_plan.md"
    plan_path.write_text(plan, encoding="utf-8")
    result["search_plan_path"] = str(plan_path)

    # Step 8: Generate auto update report
    report = _generate_update_report(result, queries, all_results,
                                      candidates_added, ingested_added,
                                      oa_count, manual_count, len(lit_db))
    report_path = OUTPUTS_DIR / "23_auto_literature_update_report.md"
    report_path.write_text(report, encoding="utf-8")
    result["report_path"] = str(report_path)

    # Step 9: Save download status
    download_rows = []
    for paper in all_results:
        is_oa = paper.get("is_open_access", False)
        download_rows.append({
            "title": (paper.get("title") or "")[:200],
            "doi": (paper.get("doi") or ""),
            "pdf_url": (paper.get("pdf_url") or ""),
            "download_status": "available" if is_oa else "manual_needed",
            "local_path": "",
            "error_message": "",
        })
    save_csv(DATA_DIR / "download_status.csv", download_rows, DOWNLOAD_STATUS_FIELDS)

    # Save manual download list
    manual_rows = [
        r for r in download_rows if r["download_status"] == "manual_needed"
    ]
    save_csv(DATA_DIR / "manual_download_needed.csv", manual_rows, MANUAL_DOWNLOAD_FIELDS)

    result["candidates_new"] = candidates_added
    result["ingested_new"] = ingested_added
    result["oa_found"] = oa_count
    result["manual_needed"] = manual_count
    result["total_literature"] = len(lit_db)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Report generators
# ═══════════════════════════════════════════════════════════════════════════

def _generate_auto_report(
    queries: List[str],
    results: List[Dict],
    candidates: List[Dict],
    lit_db: List[Dict],
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str],
    pipeline_result: Dict[str, Any],
) -> str:
    """生成文献搜索计划 markdown (覆盖 outputs/22_literature_search_plan.md)。"""
    lines = [
        "# Literature Search Plan（文献补充计划）",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        "",
    ]
    if user_query:
        lines.append(f"## 用户问题\n\n{user_query}\n")
    if ind_var or dep_var:
        lines.append(f"**关注变量**: {ind_var} → {dep_var}\n")
    lines.append("---\n")

    lines.append("## 推荐检索式\n")
    for i, q in enumerate(queries, 1):
        lines.append(f"### Query {i}\n```\n{q}\n```\n")

    lines.append("---\n")
    if results:
        lines.append("## 检索结果\n")
        lines.append(f"共 {len(results)} 篇：\n")
        lines.append("| # | 标题 | 年份 | OA | 相关度 | 来源 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, r in enumerate(results[:20], 1):
            title = (r.get("title", "") or "")[:55]
            year = r.get("year", "")
            oa = "✅" if r.get("is_open_access") else "❌"
            score = r.get("relevance_score", 0)
            src = r.get("source_database", "")
            lines.append(f"| {i} | {title} | {year} | {oa} | {score} | {src} |\n")

    lines.append("\n---\n")
    lines.append("## 下载后需提取的字段\n")
    fields = [
        "paper_id", "title", "material", "manufacturing_process", "heat_treatment",
        "surface_state", "defect_type", "pore_size", "sqrt_area", "pore_location",
        "distance_to_surface", "porosity", "stress_ratio_R", "stress_amplitude",
        "Nf", "fatigue_limit", "crack_initiation_site", "micro_CT_available",
        "SEM_available", "main_conclusion",
    ]
    for f in fields:
        lines.append(f"- {f}\n")

    return "".join(lines)


def _generate_update_report(
    result: Dict[str, Any],
    queries: List[str],
    search_results: List[Dict],
    candidates_added: int,
    ingested_added: int,
    oa_count: int,
    manual_count: int,
    total_lit: int,
) -> str:
    """生成自动文献更新报告 (outputs/23_auto_literature_update_report.md)。"""
    lines = [
        "# Auto Literature Update Report（自动文献更新报告）",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## 本次更新摘要",
        "",
        f"- 生成检索式: {result['queries_generated']} 条",
        f"- API 检索成功: {'是' if result['api_success'] else '否（离线模式）'}",
        f"- 新增候选文献: {candidates_added} 篇",
        f"- 新纳入正式库: {ingested_added} 篇",
        f"- 开放获取 (OA): {oa_count} 篇",
        f"- 需手动下载: {manual_count} 篇",
        f"- 正式库当前总量: {total_lit} 篇",
        "",
    ]

    if result.get("errors"):
        lines.append("### 检索异常\n")
        for e in result["errors"]:
            lines.append(f"- ⚠️ {e}\n")
        lines.append("\n")

    if search_results:
        lines.append("## 本次检索到的文献\n")
        lines.append("| # | 标题 | 来源 | 评分 | OA | 状态 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, r in enumerate(search_results[:30], 1):
            title = (r.get("title", "") or "")[:50]
            src = r.get("source_database", "")
            score = r.get("relevance_score", 0)
            oa = "✅" if r.get("is_open_access") else "❌"
            status = "已纳入" if score >= 0.7 else "仅候选"
            lines.append(f"| {i} | {title} | {src} | {score} | {oa} | {status} |\n")

    lines.append("\n---\n")
    lines.append("## 库状态\n")
    cand_path = DATA_DIR / "candidate_papers.csv"
    lit_path = DATA_DIR / "literature_database.csv"
    lines.append(f"- **候选库**: {cand_path} ({len(load_csv(cand_path, CANDIDATE_FIELDS))} 篇)\n")
    lines.append(f"- **正式库**: {lit_path} ({total_lit} 篇)\n")
    lines.append(f"- **手工下载清单**: {DATA_DIR / 'manual_download_needed.csv'}\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def cli_search_literature(args: Optional[List[str]] = None):
    """命令行入口: python app.py search-literature --query "pore size fatigue" """
    import argparse
    parser = argparse.ArgumentParser(description="TitaniumFatigueChat - 自动文献检索")
    parser.add_argument("--query", required=True, help="检索查询")
    parser.add_argument("--ind-var", default=None, help="自变量 canonical name")
    parser.add_argument("--dep-var", default=None, help="因变量 canonical name")
    parser.add_argument("--max-results", type=int, default=10, help="每 API 最大返回数")
    parser.add_argument("--no-api", action="store_true", help="离线模式")
    parsed = parser.parse_args(args)

    from src.variable_mapper import extract_variable_pair
    ind, dep, cls = extract_variable_pair(parsed.query)
    ind = parsed.ind_var or ind
    dep = parsed.dep_var or dep

    print(f"🔍 检索: {parsed.query}")
    print(f"  自变量: {ind}, 因变量: {dep}")
    print(f"  离线模式: {parsed.no_api}")
    print()

    result = run_auto_literature_update(
        user_query=parsed.query,
        ind_var=ind,
        dep_var=dep,
        max_results=parsed.max_results,
        use_api=not parsed.no_api,
    )

    print(f"✅ 完成:")
    print(f"  检索式: {result['queries_generated']} 条")
    print(f"  API 成功: {result['api_success']}")
    print(f"  新候选: {result['candidates_new']} 篇")
    print(f"  新纳入: {result['ingested_new']} 篇")
    print(f"  OA: {result['oa_found']}, 需手动: {result['manual_needed']}")
    print(f"  正式库总量: {result['total_literature']} 篇")
    print(f"  报告: {result.get('report_path', '')}")


if __name__ == "__main__":
    cli_search_literature()
