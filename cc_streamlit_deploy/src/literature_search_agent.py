"""
literature_search_agent.py — Automatic literature search & gap analysis
面向 L-PBF Ti-6Al-4V 疲劳文献自动检索。

数据源（免费开放 API，无需密钥）:
    - OpenAlex (openalex.org)
    - Semantic Scholar (semanticscholar.org)
    - Crossref (crossref.org)

离线模式:
    即使无法联网，也能生成检索式 + search_recommendations.csv。
"""

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Search query generation from variable pair
# ═══════════════════════════════════════════════════════════════════════════

# Variable → English search keyword mapping
VAR_SEARCH_KEYWORDS: Dict[str, List[str]] = {
    "pore_size": [
        '"pore size"', '"defect size"', '"sqrt area"', '"pore diameter"',
        '"pore dimension"', 'porosity', '"lack of fusion"',
    ],
    "fatigue_life": [
        '"fatigue life"', 'Nf', '"cycles to failure"', '"S-N"', '"stress-life"',
    ],
    "fatigue_limit": [
        '"fatigue limit"', '"fatigue strength"', '"endurance limit"',
        'sigma_w', '"fatigue threshold"',
    ],
    "da_dn": [
        '"crack growth rate"', 'FCGR', '"da/dN"', '"crack propagation"',
        '"fatigue crack growth"',
    ],
    "delta_k": [
        '"stress intensity factor"', 'Delta_K', '"ΔK"', '"SIF range"',
        '"crack driving force"',
    ],
    "surface_roughness": [
        '"surface roughness"', 'Ra', 'Rz', '"as-built surface"',
        '"surface finish"', '"surface quality"',
    ],
    "pore_location": [
        '"pore location"', '"distance to surface"', '"subsurface pore"',
        '"pore position"', '"near-surface defect"',
    ],
    "stress_amplitude": [
        '"stress amplitude"', '"stress range"', '"cyclic stress"',
        'sigma_a', '"applied stress"',
    ],
    "stress_ratio": [
        '"stress ratio"', '"R ratio"', '"load ratio"',
    ],
    "heat_treatment": [
        'HIP', '"hot isostatic pressing"', 'annealing', '"heat treatment"',
        'STA', '"solution treated"', '"stress relief"',
    ],
    "microstructure": [
        '"microstructure"', '"alpha lath"', '"beta phase"', '"grain size"',
        '"martensite"', '"alpha prime"', '"grain orientation"',
    ],
    "residual_stress": [
        '"residual stress"', '"residual strain"', '"stress relaxation"',
    ],
    "paris_c_m": [
        '"Paris law"', '"Paris C"', '"Paris m"', '"Paris coefficient"',
        '"Paris exponent"',
    ],
    "porosity": [
        'porosity', '"pore morphology"', '"pore aspect ratio"',
        '"volumetric defect"',
    ],
}

BASE_QUERY = '("L-PBF" OR "laser powder bed fusion" OR "SLM" OR "additive manufacturing") AND ("Ti-6Al-4V" OR "Ti64" OR "titanium alloy")'


def generate_search_queries(
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str] = None,
    n_queries: int = 3,
) -> List[str]:
    """
    根据变量对自动生成英文检索式。
    最多返回 n_queries 条。
    """
    queries = []

    ind_keywords = VAR_SEARCH_KEYWORDS.get(ind_var, [f'"{ind_var}"']) if ind_var else []
    dep_keywords = VAR_SEARCH_KEYWORDS.get(dep_var, [f'"{dep_var}"']) if dep_var else []

    # Query 1: Both variables + base
    if ind_keywords and dep_keywords:
        kw1 = " OR ".join(ind_keywords[:3])
        kw2 = " OR ".join(dep_keywords[:3])
        queries.append(
            f'{BASE_QUERY}\nAND ({kw1})\nAND ({kw2})'
        )

    # Query 2: Include characterization method
    char_options = [
        '"micro-CT" OR "X-ray computed tomography" OR "synchrotron"',
        '"SEM" OR "EBSD" OR "fractography" OR "fracture surface"',
    ]
    if ind_keywords:
        kw1 = " OR ".join(ind_keywords[:2])
        queries.append(
            f'{BASE_QUERY}\nAND ({kw1})\nAND ({char_options[0]})'
        )

    # Query 3: Model-specific
    model_keywords = {
        "paris_law": '"Paris law" OR "da/dN" OR "crack growth"',
        "murakami": '"Murakami" OR "sqrt area" OR "√area"',
        "kitagawa": '"Kitagawa" OR "Takahashi"',
        "el_haddad": '"El Haddad" OR "short crack" OR "small crack"',
    }

    for model_name, model_kw in model_keywords.items():
        if dep_keywords:
            kw2 = " OR ".join(dep_keywords[:2])
            queries.append(
                f'{BASE_QUERY}\nAND ({model_kw})\nAND ({kw2})'
            )
            break

    # Query 4: Fatigue + defect general
    if ind_var in ("pore_size", "surface_roughness", "porosity"):
        queries.append(
            f'{BASE_QUERY}\nAND ("fatigue" OR "fatigue life" OR "fatigue crack")\n'
            f'AND ("defect" OR "pore" OR "porosity" OR "surface roughness")'
        )

    # Remove duplicates and limit
    seen = set()
    unique_queries = []
    for q in queries:
        q_norm = q.replace(" ", "").lower()
        if q_norm not in seen:
            seen.add(q_norm)
            unique_queries.append(q)

    return unique_queries[:n_queries]


# ═══════════════════════════════════════════════════════════════════════════
# 2. API clients (free, no key required)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_request(url: str, timeout: int = 15) -> Optional[Dict]:
    """安全的 HTTP GET 请求，失败返回 None。"""
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "TitaniumFatigueChat/1.0 (mailto:research@example.com)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data) if data else None
    except Exception:
        return None


def search_openalex(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    通过 OpenAlex API 检索文献（免费，无需密钥）。
    https://docs.openalex.org/
    """
    encoded = quote(f'({BASE_QUERY}) AND ({query})')
    url = (
        f"https://api.openalex.org/works?"
        f"filter=title_and_abstract.search:{encoded}"
        f"&sort=relevance_score:desc"
        f"&per_page={max_results}"
        f"&select=id,title,authorships,publication_year,doi,primary_location,open_access,cited_by_count"
    )
    data = _safe_request(url)
    if not data or "results" not in data:
        return []

    results = []
    for work in data["results"]:
        title = work.get("title", "")
        year = work.get("publication_year")
        doi = work.get("doi", "")
        oa_info = work.get("open_access", {}) or {}
        is_oa = oa_info.get("is_oa", False)
        oa_url = oa_info.get("oa_url", "")

        # Authors
        authors_list = []
        for a in work.get("authorships", []):
            author_name = a.get("author", {}).get("display_name", "")
            if author_name:
                authors_list.append(author_name)

        # Primary location
        loc = work.get("primary_location", {}) or {}
        source = loc.get("source", {}) or {}
        source_name = source.get("display_name", "")

        results.append({
            "title": title,
            "authors": "; ".join(authors_list[:5]),
            "year": year,
            "doi": doi.replace("https://doi.org/", "") if doi else "",
            "url": f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else "",
            "source_database": "OpenAlex",
            "is_open_access": is_oa,
            "pdf_url": oa_url or "",
            "relevance_score": 0,
            "source_name": source_name,
        })

    return results


def search_semantic_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    通过 Semantic Scholar API 检索（免费，无需密钥）。
    https://api.semanticscholar.org/
    """
    encoded = quote(f'{BASE_QUERY} {query}')
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded}"
        f"&limit={max_results}"
        f"&fields=title,authors,year,externalIds,openAccessPdf,url,publicationVenue"
    )
    data = _safe_request(url)
    if not data or "data" not in data:
        return []

    results = []
    for paper in data["data"]:
        title = paper.get("title", "")
        year = paper.get("year")
        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        oa_pdf = paper.get("openAccessPdf", {}) or {}
        oa_url = oa_pdf.get("url", "")

        authors_list = []
        for a in paper.get("authors", []):
            name = a.get("name", "")
            if name:
                authors_list.append(name)

        venue = paper.get("publicationVenue", {}) or {}
        venue_name = venue.get("name", "")

        results.append({
            "title": title,
            "authors": "; ".join(authors_list[:5]),
            "year": year,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "source_database": "SemanticScholar",
            "is_open_access": bool(oa_url),
            "pdf_url": oa_url,
            "relevance_score": 0,
            "source_name": venue_name,
        })

    return results


def search_crossref(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    通过 Crossref API 检索（免费，无需密钥）。
    https://api.crossref.org/
    """
    encoded = quote(query)
    url = (
        f"https://api.crossref.org/works?"
        f"query={encoded}"
        f"&rows={max_results}"
        f"&filter=type:journal-article"
        f"&select=DOI,title,author,issued,URL,abstract,license,link"
    )
    data = _safe_request(url)
    if not data or "message" not in data:
        return []

    items = data["message"].get("items", [])
    results = []
    for item in items:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        doi = item.get("DOI", "") or ""
        links = item.get("link", []) or []
        oa_url = ""
        is_oa = False
        for link in links:
            if link.get("content-type") == "application/pdf":
                oa_url = link.get("URL", "")
                is_oa = True
                break

        authors_list = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors_list.append(name)

        results.append({
            "title": title,
            "authors": "; ".join(authors_list[:5]),
            "year": year,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "source_database": "Crossref",
            "is_open_access": is_oa,
            "pdf_url": oa_url,
            "relevance_score": 0,
            "source_name": "Crossref",
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Merge + score results
# ═══════════════════════════════════════════════════════════════════════════

def score_relevance(
    paper: Dict[str, Any],
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str] = None,
) -> int:
    """
    对检索结果进行相关度评分 0–100。
    评分维度：
        - 变量匹配度
        - 是否开放获取
        - 引用数
        - 年份新近度
    """
    score = 0
    title_lower = (paper.get("title", "") or "").lower()
    text = title_lower + " " + (user_query or "").lower()

    # Variable match (max 50)
    var_score = 0
    if ind_var:
        for kw in VAR_SEARCH_KEYWORDS.get(ind_var, []):
            clean_kw = kw.strip('"').lower()
            if clean_kw in text:
                var_score += 15
    if dep_var:
        for kw in VAR_SEARCH_KEYWORDS.get(dep_var, []):
            clean_kw = kw.strip('"').lower()
            if clean_kw in text:
                var_score += 15
    score += min(var_score, 50)

    # OA bonus (max 20)
    if paper.get("is_open_access"):
        score += 20

    # Year recency (max 15)
    year = paper.get("year")
    current_year = datetime.now().year
    if year and isinstance(year, int):
        age = current_year - year
        if age <= 2:
            score += 15
        elif age <= 5:
            score += 10
        elif age <= 10:
            score += 5

    # Source quality (max 15)
    source = (paper.get("source_name", "") or "").lower()
    quality_sources = [
        "international journal of fatigue", "fatigue", "acta materialia",
        "materials science and engineering", "journal of materials",
        "additive manufacturing", "metallurgical",
    ]
    for qs in quality_sources:
        if qs in source:
            score += 15
            break

    paper["relevance_score"] = score
    return score


def merge_search_results(
    openalex_results: List[Dict],
    s2_results: List[Dict],
    crossref_results: List[Dict],
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str],
) -> List[Dict]:
    """
    合并多个 API 的检索结果，去重（按 DOI），评分排序。
    """
    seen_dois = set()
    all_results = []

    for source_list in [openalex_results, s2_results, crossref_results]:
        for paper in source_list:
            doi = paper.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            score_relevance(paper, ind_var, dep_var, user_query)
            all_results.append(paper)

    # Sort by relevance score descending
    all_results.sort(key=lambda x: -x.get("relevance_score", 0))
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Save results to CSV
# ═══════════════════════════════════════════════════════════════════════════

def save_search_results(results: List[Dict], filename: str = "search_results.csv") -> str:
    """保存检索结果到 CSV。"""
    path = DATA_DIR / filename
    fields = [
        "title", "authors", "year", "doi", "url", "source_database",
        "is_open_access", "pdf_url", "relevance_score", "source_name",
        "download_status",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            row["is_open_access"] = str(r.get("is_open_access", False))
            row["relevance_score"] = r.get("relevance_score", 0)
            row["download_status"] = "open_access" if r.get("is_open_access") else "manual_download_needed"
            writer.writerow(row)

    # Also save manual_download_needed.csv
    manual = [r for r in results if not r.get("is_open_access") and r.get("doi")]
    manual_path = DATA_DIR / "manual_download_needed.csv"
    with open(manual_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in manual:
            row = {k: r.get(k, "") for k in fields}
            row["is_open_access"] = "False"
            writer.writerow(row)

    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main search pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_literature_search(
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str] = None,
    max_results: int = 10,
    use_api: bool = False,
) -> Dict[str, Any]:
    """
    执行文献检索完整流程。

    Args:
        ind_var: 自变量 canonical name
        dep_var: 因变量 canonical name
        user_query: 用户原始输入
        max_results: 每个 API 返回的最大结果数
        use_api: 是否尝试联网检索

    Returns:
        {
            "search_queries": [...],
            "results": [...],
            "results_saved_to": "",
            "search_plan_saved_to": "",
            "summary": str,
            "api_success": bool,
        }
    """
    # Step 1: Generate search queries
    queries = generate_search_queries(ind_var, dep_var, user_query)

    # Step 2: Try API search
    all_results = []
    api_success = False

    if use_api:
        # Try OpenAlex first (most generous rate limits)
        try:
            oa_results = search_openalex(queries[0] if queries else "", max_results)
            time.sleep(0.5)  # rate limit
        except Exception:
            oa_results = []

        try:
            s2_results = search_semantic_scholar(
                f"{ind_var or ''} {dep_var or ''} L-PBF Ti-6Al-4V fatigue",
                max_results,
            )
            time.sleep(0.5)
        except Exception:
            s2_results = []

        try:
            cr_results = search_crossref(
                f"{ind_var or ''} {dep_var or ''} L-PBF Ti-6Al-4V fatigue",
                max_results,
            )
        except Exception:
            cr_results = []

        all_results = merge_search_results(
            oa_results, s2_results, cr_results,
            ind_var, dep_var, user_query,
        )

        if all_results:
            api_success = True

    # Step 3: Save results
    results_path = ""
    if all_results:
        results_path = save_search_results(all_results)

    # Step 4: Generate search plan markdown
    plan = generate_search_plan_markdown(
        ind_var, dep_var, user_query,
        queries, all_results,
    )
    plan_path = OUTPUTS_DIR / "22_literature_search_plan.md"
    plan_path.write_text(plan, encoding="utf-8")

    # Step 5: Save search recommendations CSV
    save_search_recommendations(queries, ind_var, dep_var)

    # Step 6: Build summary
    if api_success:
        n_oa = sum(1 for r in all_results if r.get("is_open_access"))
        n_manual = len(all_results) - n_oa
        summary = (
            f"通过开放 API 检索到 {len(all_results)} 篇相关文献，"
            f"其中 {n_oa} 篇为开放获取（可下载），"
            f"{n_manual} 篇需手动下载。"
        )
    else:
        summary = (
            "当前为离线模式，未进行 API 检索。"
            "已生成检索式和文献补充计划。"
        )

    return {
        "search_queries": queries,
        "results": all_results,
        "results_saved_to": results_path,
        "search_plan_saved_to": str(plan_path),
        "summary": summary,
        "api_success": api_success,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Markdown plan generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_search_plan_markdown(
    ind_var: Optional[str],
    dep_var: Optional[str],
    user_query: Optional[str],
    queries: List[str],
    results: List[Dict],
) -> str:
    """生成文献搜索计划 markdown。"""
    lines = [
        "# Literature Search Plan（文献补充计划）",
        "",
    ]

    # Question
    if user_query:
        lines.append(f"## 用户问题\n\n{user_query}\n")
    if ind_var or dep_var:
        lines.append(f"**关注变量**: {ind_var} → {dep_var}\n")

    lines.append("---\n")

    # Search queries
    lines.append("## 推荐检索式\n")
    if queries:
        for i, q in enumerate(queries, 1):
            lines.append(f"### Query {i}\n```\n{q}\n```\n")
    else:
        lines.append("未生成检索式。\n")

    lines.append("---\n")

    # Results
    if results:
        lines.append("## 检索结果\n")
        lines.append(f"共找到 {len(results)} 篇文献（按相关度排序）：\n")
        lines.append("| # | 标题 | 年份 | OA | 相关度 | 来源 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, r in enumerate(results[:20], 1):
            title = (r.get("title", "") or "")[:60]
            year = r.get("year", "")
            oa = "✅" if r.get("is_open_access") else "❌"
            score = r.get("relevance_score", 0)
            source = r.get("source_database", "")
            lines.append(f"| {i} | {title} | {year} | {oa} | {score} | {source} |\n")
        lines.append("\n")

        # OA papers
        oa_papers = [r for r in results if r.get("is_open_access")]
        if oa_papers:
            lines.append("### 开放获取文献（可直接下载）\n")
            for r in oa_papers:
                title = r.get("title", "")[:60]
                url = r.get("pdf_url") or r.get("url", "")
                lines.append(f"- [{title}]({url})\n")

        # Manual download
        manual = [r for r in results if not r.get("is_open_access")]
        if manual:
            lines.append("### 需手动下载的文献\n")
            for r in manual:
                title = r.get("title", "")[:60]
                doi = r.get("doi", "")
                url = r.get("url", f"https://doi.org/{doi}") if doi else ""
                lines.append(f"- {title} ({url})\n")

    lines.append("---\n")

    # Required fields
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


def save_search_recommendations(
    queries: List[str],
    ind_var: Optional[str],
    dep_var: Optional[str],
) -> str:
    """保存检索推荐到 CSV。"""
    path = DATA_DIR / "search_recommendations.csv"
    fields = [
        "ind_var", "dep_var", "query", "database",
        "field_to_extract",
    ]
    databases = [
        "Google Scholar", "Web of Science", "Scopus",
        "OpenAlex", "Semantic Scholar", "Crossref",
    ]
    extract_fields = [
        "pore_size", "sqrt_area", "pore_location", "porosity",
        "Nf", "fatigue_limit", "crack_initiation_site",
    ]

    rows = []
    for q in queries:
        for db in databases[:3]:
            rows.append({
                "ind_var": ind_var or "",
                "dep_var": dep_var or "",
                "query": q[:200],
                "database": db,
                "field_to_extract": "; ".join(extract_fields),
            })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return str(path)
