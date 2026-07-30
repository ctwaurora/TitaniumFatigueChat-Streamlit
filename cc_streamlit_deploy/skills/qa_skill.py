"""AI Scientist 问答技能：本地文献库 + 联网检索 + DeepSeek 综合回答。"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import deepseek_skill, library_skill, rag_skill, search_skill

CHUNKS_PATH = Path("data/rag/chunks")
CARDS_PATH = Path("data/literature_cards.jsonl")


# ── 本地证据检索 ──────────────────────────────────────────────────────


def local_retrieve_for_question(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """从本地文献库检索证据：chunks + literature_cards。

    Args:
        question: 用户问题
        top_k: 每类最多返回条数

    Returns:
        统一证据列表：
        { "type": "local_chunk" / "local_card", "title": "", "source_file": "",
          "text": "", "year": "", "doi": "" }
    """
    evidence: List[Dict[str, Any]] = []

    # 1. 从 chunks 检索（RAG 片段）
    try:
        chunk_results = rag_skill.keyword_retrieve(question, top_k=top_k)
        for c in chunk_results:
            evidence.append({
                "type": "local_chunk",
                "title": c.get("source_file", ""),
                "source_file": c.get("source_file", ""),
                "text": c.get("text", ""),
                "year": "",
                "doi": "",
            })
    except Exception:
        pass

    # 2. 从 literature_cards 检索
    try:
        card_results = library_skill.search_papers(question)
        for c in card_results[:top_k]:
            # 拼接关键字段作为证据文本
            snippets = []
            for field in ["key_findings", "limitations", "possible_innovation", "evidence_text", "abstract"]:
                val = c.get(field, "")
                if val and str(val).strip() not in ("", "未说明"):
                    snippets.append(f"[{field}] {val}")
            if not snippets:
                snippets.append(c.get("abstract", "") or c.get("key_findings", "") or "")

            text_block = "\n".join(snippets) if isinstance(snippets, list) else str(snippets)

            evidence.append({
                "type": "local_card",
                "title": c.get("title", ""),
                "source_file": c.get("source_file", ""),
                "text": text_block,
                "year": str(c.get("year", "") or c.get("publication_year", "") or ""),
                "doi": c.get("doi", ""),
            })
    except Exception:
        pass

    # 按 type 排序：chunk 在前，card 在后
    evidence.sort(key=lambda x: (0 if x["type"] == "local_chunk" else 1))
    return evidence


# ── 参考文献列表 ──────────────────────────────────────────────────────


def build_reference_list(
    local_evidence: List[Dict[str, Any]],
    online_papers: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """合并本地文献和在线论文，按 DOI / title 去重。

    Returns:
        参考文献列表，每个包含 title, authors, year, journal, doi, url。
    """
    refs: List[Dict[str, str]] = []
    seen_dois: set = set()
    seen_titles: set = set()

    # 1. 从 local_evidence 提取文献信息
    #    对于 local_chunk，只有 source_file，可能没有完整文献信息
    #    对于 local_card，有 title / year / doi
    for ev in local_evidence:
        if ev["type"] == "local_card" and ev.get("title"):
            doi = (ev.get("doi") or "").strip().lower()
            title = library_skill.normalize_text(ev.get("title", ""))

            if doi and doi in seen_dois:
                continue
            if title and title in seen_titles:
                continue

            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)

            refs.append({
                "title": ev.get("title", ""),
                "authors": "",
                "year": ev.get("year", ""),
                "journal": "",
                "doi": ev.get("doi", ""),
                "url": "",
            })
        elif ev["type"] == "local_chunk" and ev.get("source_file"):
            # 尝试通过 source_file 在 literature_cards 中找完整信息
            try:
                all_cards = library_skill.get_all_papers()
                for card in all_cards:
                    if card.get("source_file") == ev.get("source_file"):
                        title = card.get("title", "")
                        if title:
                            doi = (card.get("doi") or "").strip().lower()
                            norm_title = library_skill.normalize_text(title)
                            if doi and doi in seen_dois:
                                break
                            if norm_title and norm_title in seen_titles:
                                break
                            if doi:
                                seen_dois.add(doi)
                            if norm_title:
                                seen_titles.add(norm_title)
                            refs.append({
                                "title": title,
                                "authors": ", ".join(card.get("authors", [])) if isinstance(card.get("authors"), list) else str(card.get("authors", "")),
                                "year": str(card.get("year", "") or card.get("publication_year", "") or ""),
                                "journal": card.get("journal", ""),
                                "doi": card.get("doi", ""),
                                "url": "",
                            })
                            break
            except Exception:
                pass

    # 2. 从 online_papers 添加
    for p in online_papers:
        doi = (p.get("doi") or "").strip().lower()
        title = library_skill.normalize_text(p.get("title", ""))

        if doi and doi in seen_dois:
            continue
        if title and title in seen_titles:
            continue

        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)

        authors_list = p.get("authors", [])
        if isinstance(authors_list, list):
            authors_str = ", ".join(authors_list)
        else:
            authors_str = str(authors_list)

        refs.append({
            "title": p.get("title", ""),
            "authors": authors_str,
            "year": p.get("year", ""),
            "journal": p.get("journal", ""),
            "doi": p.get("doi", ""),
            "url": p.get("url", ""),
        })

    # 3. 去重后按类型排序：在线文献在前
    return refs


# ── AI Scientist 问答 ─────────────────────────────────────────────────


def answer_question_with_online_literature(
    question: str,
    max_online: int = 8,
) -> Dict[str, Any]:
    """AI Scientist 问答：本地证据 + 在线检索 + DeepSeek 综合回答。

    Args:
        question: 用户的研究问题
        max_online: 在线检索最多返回论文数

    Returns:
        {
          "answer": "DeepSeek 生成的回答（含引用标记和参考文献小节）",
          "local_evidence": [...],
          "online_papers": [...],
          "references": [...]
        }
    """
    if not question or not question.strip():
        return {
            "answer": "请输入有效问题。",
            "local_evidence": [],
            "online_papers": [],
            "references": [],
        }

    result: Dict[str, Any] = {
        "answer": "",
        "local_evidence": [],
        "online_papers": [],
        "references": [],
    }

    # 第一步：本地证据检索
    local_evidence = local_retrieve_for_question(question, top_k=5)

    # 第二步：在线文献检索
    online_papers: List[Dict[str, Any]] = []
    try:
        online_papers = search_skill.online_literature_search(question, max_results=max_online)
    except Exception:
        pass  # 在线检索失败不影响本地问答

    # 第三步：构建参考文献列表
    references = build_reference_list(local_evidence, online_papers)

    result["local_evidence"] = local_evidence
    result["online_papers"] = online_papers
    result["references"] = references

    # 第四步：构建 DeepSeek prompt
    if not local_evidence and not online_papers:
        result["answer"] = "当前证据不足。本地文献库和在线检索均未找到相关文献。"
        return result

    # --- 构建本地证据块 ---
    local_blocks = []
    for j, ev in enumerate(local_evidence):
        label = j + 1
        source_info = f"来源：{ev.get('source_file', '') or ev.get('title', '未知')}"
        local_blocks.append(
            f"[本地证据{label}] {source_info}\n{ev['text'][:2000]}"
        )
    local_str = "\n\n".join(local_blocks) if local_blocks else "（无本地证据）"

    # --- 构建在线论文块 ---
    online_blocks = []
    for j, p in enumerate(online_papers):
        label = j + 1
        authors_str = ", ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors", ""))
        online_blocks.append(
            f"[在线文献{label}] 标题：{p.get('title', '')}\n"
            f"作者：{authors_str}\n"
            f"年份：{p.get('year', '')}\n"
            f"期刊：{p.get('journal', '')}\n"
            f"DOI：{p.get('doi', '')}\n"
            f"摘要：{p.get('abstract', '')[:1500]}"
        )
    online_str = "\n\n".join(online_blocks) if online_blocks else "（无在线文献）"

    # --- 构建参考文献块 ---
    ref_blocks = []
    for j, ref in enumerate(references):
        label = j + 1
        ref_blocks.append(
            f"[参考文献{label}] {ref.get('title', '')}, "
            f"{ref.get('authors', '')}, "
            f"{ref.get('year', '')}, "
            f"{ref.get('journal', '')}, "
            f"DOI: {ref.get('doi', '')}"
        )
    ref_str = "\n".join(ref_blocks) if ref_blocks else "（无参考文献）"

    # --- 构造 DeepSeek prompt ---
    prompt = f"""你是一名材料科学与工程领域的 AI Scientist 研究助手。请根据以下本地证据和在线文献，回答用户的研究问题。

## 用户问题
{question.strip()}

## 本地证据（来自已上传的 PDF 文献片段和文献卡片）
{local_str}

## 在线文献（来自 OpenAlex / Semantic Scholar 学术搜索引擎）
{online_str}

## 可引用的参考文献列表（严格按照此列表引用，不要编造新文献）
{ref_str}

## 回答要求
1. 直接回答用户问题，回答要自然、专业、深入，像 AI 科学家一样。
2. 如果依据来自本地证据，请在对应内容后标注 [本地证据X]（X 为证据编号）。
3. 如果依据来自在线文献，请在对应内容后标注 [在线文献X]（X 为文献编号）。
4. 回答末尾必须生成「参考文献」小节，列出你实际引用到的文献。
5. 参考文献必须来自上面提供的「可引用的参考文献列表」，不允许编造不存在的文献。
6. 如果所有可用证据都不足以回答，请明确说「当前证据不足」，并说明缺少哪些信息。
7. 不要凭空编造数据或结论。
8. 用中文回答。"""

    # 第五步：调用 DeepSeek
    try:
        answer = deepseek_skill.call_deepseek_text(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.3,
        )
        result["answer"] = answer
    except Exception as e:
        result["answer"] = f"DeepSeek 回答生成失败：{str(e)}"

    return result
