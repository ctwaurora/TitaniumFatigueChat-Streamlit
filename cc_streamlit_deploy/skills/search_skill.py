"""自动文献发现与联网检索技能：OpenAlex / Semantic Scholar API 搜索 + DeepSeek 筛选。"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import deepseek_skill, library_skill

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_TOPIC = "钛合金疲劳断裂、寿命预测、裂纹扩展、机器学习、有限元、热处理-显微组织-疲劳性能"


# ── 工具函数 ──────────────────────────────────────────────────────────


def inverted_index_to_text(index: Optional[Dict[str, List[int]]]) -> str:
    """将 OpenAlex 的 abstract_inverted_index 转换成普通摘要字符串。"""
    if not index:
        return ""
    word_positions: Dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            word_positions[pos] = word
    if not word_positions:
        return ""
    max_pos = max(word_positions.keys())
    words = [word_positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words)


# 向后兼容
_reconstruct_abstract = inverted_index_to_text


# ── OpenAlex ──────────────────────────────────────────────────────────


def search_openalex(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """调用 OpenAlex Works API 搜索文献。

    Args:
        query: 搜索关键词
        max_results: 最多返回结果数

    Returns:
        标准化后的论文列表；失败时返回空列表。
    """
    if not query or not query.strip():
        return []

    params = {
        "search": query.strip(),
        "per_page": min(max_results, 200),
        "sort": "relevance_score:desc",
    }
    headers = {"User-Agent": "TitaniumFatigueChat/1.0 (mailto:research@example.com)"}

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.get(
            OPENALEX_URL,
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    raw_results = data.get("results", [])
    if not raw_results:
        return []

    normalized = []
    for raw in raw_results:
        paper = normalize_paper_metadata(raw)
        if paper.get("title"):
            normalized.append(paper)

    return normalized[:max_results]


def normalize_paper_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 OpenAlex 原始返回标准化为统一文献卡片格式。"""
    # --- 标题 ---
    title = (raw.get("title") or "").strip()

    # --- 作者 ---
    authors_list: List[str] = []
    for authorship in raw.get("authorships", []):
        author_obj = authorship.get("author", {})
        name = (author_obj.get("display_name") or "").strip()
        if name:
            authors_list.append(name)

    # --- 年份 ---
    year = raw.get("publication_year")
    if year is not None:
        year = str(int(year))
    else:
        year = ""

    # --- 期刊 ---
    primary_location = raw.get("primary_location") or {}
    source = primary_location.get("source") or {}
    journal = (source.get("display_name") or "").strip()

    # --- DOI ---
    doi = raw.get("doi") or ""
    if doi:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi).strip()

    # --- 摘要 ---
    abstract = inverted_index_to_text(raw.get("abstract_inverted_index"))

    # --- URL ---
    landing_page = (raw.get("primary_location") or {}).get("landing_page_url") or ""

    # --- PDF URL ---
    best_oa = (raw.get("best_oa_location") or {})
    pdf_url = (best_oa.get("pdf_url") or "")
    if not pdf_url:
        pdf_url = (primary_location.get("pdf_url") or "")

    # --- 引用数 ---
    citation_count = raw.get("cited_by_count", 0)

    return {
        "title": title,
        "authors": authors_list,
        "year": year,
        "journal": journal,
        "doi": doi,
        "abstract": abstract,
        "url": landing_page or (raw.get("id") or ""),
        "pdf_url": pdf_url,
        "source": "OpenAlex",
        "citation_count": citation_count if citation_count else 0,
    }


def add_search_result_to_library(paper: Dict[str, Any]) -> bool:
    """将搜索到的文献元数据加入文献库（无需 PDF）。

    Returns:
        True 表示成功加入，False 表示已存在未加入。
    """
    title = (paper.get("title") or "").strip()
    if not title:
        return False

    existing = library_skill.get_all_papers()
    new_doi = (paper.get("doi") or "").strip().lower()
    new_title = library_skill.normalize_text(title)
    new_year = str(paper.get("year") or "")

    for p in existing:
        existing_doi = str(p.get("doi") or "").strip().lower()
        existing_title = library_skill.normalize_text(str(p.get("title") or ""))
        existing_year = str(p.get("year") or p.get("publication_year") or "")

        if new_doi and existing_doi and new_doi == existing_doi:
            return False
        if not new_doi and new_title and existing_title and new_title == existing_title and new_year and existing_year and new_year == existing_year:
            return False

    card = {
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year", ""),
        "journal": paper.get("journal", ""),
        "doi": paper.get("doi", ""),
        "abstract": paper.get("abstract", ""),
        "url": paper.get("url", ""),
        "source_type": "在线检索元数据",
        "source_file": "",
        "file_hash": "",
        "material_system": "",
        "processing_method": "",
        "heat_treatment": "",
        "microstructure": "",
        "loading_condition": "",
        "model_or_method": "",
        "key_findings": "",
        "limitations": "",
        "possible_innovation": "",
        "evidence_text": "",
    }

    dedup_key = new_doi or (title + new_year)
    fake_hash = "online_" + hashlib.md5(dedup_key.encode("utf-8")).hexdigest()
    card["file_hash"] = fake_hash

    existing.append(card)
    with open(library_skill.CARDS_PATH, "w", encoding="utf-8") as f:
        for item in existing:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    library_skill.save_csv(existing)
    return True


# ── Semantic Scholar ──────────────────────────────────────────────────


def search_semantic_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """调用 Semantic Scholar Paper Search API 搜索文献。

    Args:
        query: 搜索关键词
        max_results: 最多返回结果数

    Returns:
        标准化后的论文列表；失败时返回空列表。
    """
    if not query or not query.strip():
        return []

    fields = "title,authors,year,journal,externalIds,abstract,url,citationCount,openAccessPdf"
    params = {
        "query": query.strip(),
        "limit": min(max_results, 100),
        "fields": fields,
    }

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.get(
            SEMANTIC_SCHOLAR_URL,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    raw_results = data.get("data", [])
    if not raw_results:
        return []

    normalized = []
    for raw in raw_results:
        paper = _normalize_semantic_scholar(raw)
        if paper.get("title"):
            normalized.append(paper)

    return normalized[:max_results]


def _normalize_semantic_scholar(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 Semantic Scholar 原始返回标准化为统一文献格式。"""
    # --- 标题 ---
    title = (raw.get("title") or "").strip()

    # --- 作者 ---
    authors_list: List[str] = []
    for author_obj in raw.get("authors", []):
        name = (author_obj.get("name") or "").strip()
        if name:
            authors_list.append(name)

    # --- 年份 ---
    year = raw.get("year")
    if year is not None:
        year = str(int(year))
    else:
        year = ""

    # --- 期刊 ---
    journal_obj = raw.get("journal") or {}
    journal = (journal_obj.get("name") or "").strip()

    # --- DOI ---
    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI") or ""
    if doi:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi).strip()

    # --- 摘要 ---
    abstract = (raw.get("abstract") or "").strip()

    # --- URL ---
    url = raw.get("url") or ""

    # --- PDF URL ---
    pdf_url = ""
    open_access = raw.get("openAccessPdf")
    if open_access and isinstance(open_access, dict):
        pdf_url = open_access.get("url") or ""

    # --- 引用数 ---
    citation_count = raw.get("citationCount", 0)

    return {
        "title": title,
        "authors": authors_list,
        "year": year,
        "journal": journal,
        "doi": doi,
        "abstract": abstract,
        "url": url,
        "pdf_url": pdf_url,
        "source": "SemanticScholar",
        "citation_count": citation_count if citation_count else 0,
    }


# ── 综合联网搜索 ──────────────────────────────────────────────────────


def online_literature_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """综合联网文献搜索：优先 OpenAlex，不足时补充 Semantic Scholar。

    对结果按标题 + DOI 去重，返回标准化列表。

    Args:
        query: 搜索关键词
        max_results: 最多返回结果数

    Returns:
        去重后的标准化论文列表；全部失败时返回空列表。
    """
    if not query or not query.strip():
        return []

    all_papers: List[Dict[str, Any]] = []

    # 1. 先搜 OpenAlex
    try:
        openalex_results = search_openalex(query, max_results)
        all_papers.extend(openalex_results)
    except Exception:
        pass

    # 2. 如果结果太少，补充 Semantic Scholar
    if len(all_papers) < max_results:
        try:
            ss_needed = max(max_results - len(all_papers), 3)
            ss_results = search_semantic_scholar(query, ss_needed)
            all_papers.extend(ss_results)
        except Exception:
            pass

    if not all_papers:
        return []

    # 3. 去重
    seen_dois: set = set()
    seen_titles: set = set()
    deduped: List[Dict[str, Any]] = []

    for p in all_papers:
        doi = (p.get("doi") or "").strip().lower()
        title = library_skill.normalize_text(p.get("title", ""))

        doi_key = doi if doi else None
        title_key = title if title else None

        if doi_key and doi_key in seen_dois:
            continue
        if title_key and title_key in seen_titles:
            continue

        if doi_key:
            seen_dois.add(doi_key)
        if title_key:
            seen_titles.add(title_key)

        deduped.append(p)

    return deduped[:max_results]


# ── DeepSeek 相关性评分 ──────────────────────────────────────────────


def score_relevance_with_deepseek(
    paper: Dict[str, Any],
    topic: str = DEFAULT_TOPIC,
) -> Tuple[int, str]:
    """调用 DeepSeek 判断论文与给定主题的相关性。

    Returns:
        (score_0_100, reason_string)
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not title and not abstract:
        return (0, "标题和摘要为空，无法判断相关性。")

    text_for_analysis = f"标题：{title}\n摘要：{abstract[:1500]}"

    prompt = f"""你是一名材料科学与工程领域的研究助手。请判断以下论文是否与「{topic}」相关。

论文信息：
{text_for_analysis}

请根据以下标准评分：
- 90-100：直接相关，核心研究主题完全匹配
- 70-89：高度相关，研究对象或方法高度匹配
- 40-69：部分相关，涉及部分相关领域
- 10-39：弱相关，仅有边缘性关联
- 0-9：不相关

请只输出以下 JSON 格式（不要添加任何其他文字或 markdown 代码块）：
{{"score": <0-100的整数>, "reason": "<评分理由，中文，20-50字>"}}"""

    try:
        response_text = deepseek_skill.call_deepseek_text(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1,
        )
    except Exception as e:
        return (0, f"DeepSeek 调用失败：{str(e)}")

    try:
        json_match = re.search(r"\{.*\}", response_text, re.S)
        if json_match:
            parsed = json.loads(json_match.group())
            score = int(parsed.get("score", 0))
            score = max(0, min(100, score))
            reason = str(parsed.get("reason", "无理由"))
            return (score, reason)
    except Exception:
        pass

    return (0, "相关性评分解析失败，请重试。")


def search_and_rank_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """综合执行：搜索 OpenAlex → 标准化 → DeepSeek 相关性评分 → 按分数排序。

    Returns:
        每篇论文增加 'relevance_score' 和 'relevance_reason' 字段。
    """
    raw_papers = search_openalex(query, max_results=max_results)
    if not raw_papers:
        return []

    scored: List[Dict[str, Any]] = []
    for paper in raw_papers:
        score, reason = score_relevance_with_deepseek(paper, DEFAULT_TOPIC)
        paper["relevance_score"] = score
        paper["relevance_reason"] = reason
        scored.append(paper)

    scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return scored
