"""User-triggered smart search with evidence-first degradation behavior."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from src.api_keys import get_deepseek_settings
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError
from src.query_understanding import understand_user_query
from src.unified_rag import answer_research_question


ProgressCallback = Callable[[dict[str, Any]], None]
PORE_PATTERN = re.compile(r"孔隙|孔洞|气孔|pore|porosity", re.IGNORECASE)


def smart_search_dataset_version(base_dir: Path) -> str:
    parts = []
    for relative in ("data/rag/manifest.json", "data/system/corpus_statistics.json"):
        path = Path(base_dir) / relative
        try:
            payload = path.read_bytes()
        except OSError:
            payload = b"missing"
        parts.append(hashlib.sha256(payload).hexdigest())
    return ":".join(parts)


@functools.lru_cache(maxsize=64)
def _cached_retrieval(
    base_dir_text: str,
    dataset_version: str,
    question: str,
    top_k: int,
) -> dict[str, Any]:
    del dataset_version
    return answer_research_question(
        question, top_k=top_k, base_dir=Path(base_dir_text)
    )


@functools.lru_cache(maxsize=64)
def _cached_first_stage_retrieval(
    base_dir_text: str,
    dataset_version: str,
    question: str,
    top_k: int,
) -> dict[str, Any]:
    del dataset_version
    return answer_research_question(
        question,
        top_k=top_k,
        base_dir=Path(base_dir_text),
        include_counter=False,
    )


def clear_smart_search_cache() -> None:
    _cached_retrieval.cache_clear()
    _cached_first_stage_retrieval.cache_clear()


def _compact_conditions(value: Any) -> str:
    conditions = value if isinstance(value, dict) else {}
    reported = {key: item for key, item in conditions.items() if item not in (None, "", [], {}, "NOT_REPORTED")}
    return json.dumps(reported, ensure_ascii=False) if reported else "NOT_REPORTED"


def _evidence_markdown(
    result: dict[str, Any],
    *,
    per_group_limit: int = 10,
    original_text_limit: int | None = None,
) -> str:
    sufficiency = result["evidence_sufficiency"]
    lines = [
        "## 检索结果", "",
        f"- 召回文献数：{len(result['retrieved_papers'])}",
        f"- 证据充分度：`{sufficiency['status']}`",
        f"- 页码可追溯率：{sufficiency['page_traceability_rate']:.1%}",
    ]
    if result.get("insufficient"):
        lines.append("- **本地证据不足**：当前结果不得表述为高置信研究结论。")
    groups = (
        ("支持证据", result["supporting_evidence"]),
        ("反向证据", result["counter_evidence"]),
        ("条件依赖证据", result["condition_dependent_evidence"]),
    )
    for label, rows in groups:
        lines.extend(["", f"### {label}", ""])
        if not rows:
            lines.append("本地证据不足。")
            continue
        for row in rows[:per_group_limit]:
            original_text = str(row.get("original_text") or "")
            if original_text_limit and len(original_text) > original_text_limit:
                original_text = original_text[:original_text_limit].rstrip() + "..."
            lines.extend([
                f"- **{row.get('title') or '题名未报告'}**",
                f"  - Evidence ID：{row.get('doc_id')}；页码：{row.get('page_number')}；章节：{row.get('section') or 'NOT_REPORTED'}",
                f"  - 直接性：{row.get('directness') or 'NOT_REPORTED'}；条件：{_compact_conditions(row.get('experimental_conditions'))}",
                f"  - 原文：{original_text}",
            ])
    diagnostics = result.get("retrieval_diagnostics") or {}
    lines.extend(["", "### 检索诊断", "", f"- BM25：{diagnostics.get('bm25_executed')}", f"- 向量召回：{diagnostics.get('vector_executed')}"])
    if diagnostics.get("errors"):
        lines.append(f"- 降级原因：{'; '.join(diagnostics['errors'])}")
    return "\n".join(lines)


def _first_stage_markdown(
    result: dict[str, Any], understood: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Render five compact paper rows while preserving each evidence role."""
    selected: list[tuple[str, dict[str, Any]]] = []
    seen_papers: set[str] = set()
    groups = (
        ("支持", result.get("supporting_evidence") or [], 3),
        ("反向", result.get("counter_evidence") or [], 1),
        ("条件依赖", result.get("condition_dependent_evidence") or [], 1),
    )
    for label, rows, limit in groups:
        added = 0
        for row in rows:
            paper_id = str(row.get("paper_id") or row.get("doc_id") or "")
            if not paper_id or paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)
            selected.append((label, row))
            added += 1
            if added >= limit or len(selected) >= 5:
                break
    if len(selected) < 5:
        for label, rows, _ in groups:
            for row in rows:
                paper_id = str(row.get("paper_id") or row.get("doc_id") or "")
                if not paper_id or paper_id in seen_papers:
                    continue
                seen_papers.add(paper_id)
                selected.append((label, row))
                if len(selected) >= 5:
                    break
            if len(selected) >= 5:
                break

    lines = [
        "## 检索结果",
        "",
        f"- 识别主题：{understood.get('task_intent') or 'general_explanation'}",
        "- 检索边界：当前正式可信 RAG（153 篇）",
        f"- 召回文献数：{len(result.get('retrieved_papers') or [])}",
        "",
        "### 首批证据",
        "",
    ]
    preview_rows: list[dict[str, Any]] = []
    for label, row in selected:
        preview = str(row.get("original_text") or "").strip()
        if len(preview) > 240:
            preview = preview[:240].rstrip() + "..."
        preview_rows.append(
            {
                "role": label,
                "title": str(row.get("title") or "题名未报告"),
                "evidence_id": str(row.get("doc_id") or ""),
                "page_number": row.get("page_number"),
                "section": str(row.get("section") or "NOT_REPORTED"),
                "preview": preview,
            }
        )
        lines.extend(
            [
                f"- **[{label}] {row.get('title') or '题名未报告'}**",
                f"  - Evidence ID：{row.get('doc_id')}；页码：{row.get('page_number')}；章节：{row.get('section') or 'NOT_REPORTED'}",
                f"  - 原文：{preview}",
            ]
        )
    lines.extend(["", "完整综合正在生成；完整证据、反向证据与条件依赖证据将在下一阶段显示。"])
    return "\n".join(lines), preview_rows


def _llm_prompt(question: str, result: dict[str, Any]) -> str:
    rows = (result["supporting_evidence"][:6] + result["counter_evidence"][:4] + result["condition_dependent_evidence"][:4])
    evidence = "\n".join(
        f"[{row.get('doc_id')} | {row.get('title')} | p.{row.get('page_number')} | {row.get('section')}] {row.get('original_text')} CONDITIONS={_compact_conditions(row.get('experimental_conditions'))}"
        for row in rows
    )
    return (
        "仅依据下列本地证据回答科研问题。每个事实结论必须用[Evidence ID, p.页码]标注；"
        "分别写支持、反向、条件依赖与证据不足；不得补造文献、页码、参数或机制边界。\n"
        f"问题：{question}\n证据：\n{evidence}"
    )


def run_smart_search(
    question: str,
    *,
    base_dir: Path,
    use_llm: bool = True,
    top_k: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_epoch_ms = time.time_ns() / 1_000_000
    if progress_callback:
        progress_callback({"stage": "UNDERSTANDING", "message": "正在识别查询意图与主题。"})
    understood = understand_user_query(question)
    effective = understood.get("corrected_query") or question
    scope_guard_applied = bool(not PORE_PATTERN.search(question) and PORE_PATTERN.search(effective))
    if scope_guard_applied:
        effective = question
    normalization_complete_epoch_ms = time.time_ns() / 1_000_000
    dataset_version = smart_search_dataset_version(base_dir)
    retrieval_started = time.perf_counter()
    retrieval_started_epoch_ms = time.time_ns() / 1_000_000
    retrieval_function = _cached_retrieval if use_llm else _cached_first_stage_retrieval
    cache_before = retrieval_function.cache_info()
    result = copy.deepcopy(
        retrieval_function(
            str(Path(base_dir).resolve()), dataset_version, effective, top_k
        )
    )
    cache_after = retrieval_function.cache_info()
    retrieval_elapsed = time.perf_counter() - retrieval_started
    retrieval_complete_epoch_ms = time.time_ns() / 1_000_000
    diagnostics = result.setdefault("retrieval_diagnostics", {})
    diagnostics["intent"] = understood.get("task_intent")
    diagnostics["corrected_query"] = effective
    diagnostics["llm_executed"] = False
    diagnostics["llm_error"] = ""
    diagnostics["dataset_version"] = dataset_version
    diagnostics["retrieval_elapsed_seconds"] = round(retrieval_elapsed, 4)
    diagnostics["retrieval_cache_hit"] = cache_after.hits > cache_before.hits
    diagnostics["scope_guard_prevented_unsolicited_porosity"] = scope_guard_applied
    phase_seconds = result.get("retrieval_phase_seconds") or {}
    if diagnostics["retrieval_cache_hit"]:
        bm25_epoch_ms = retrieval_complete_epoch_ms
        vector_epoch_ms = retrieval_complete_epoch_ms
        rerank_epoch_ms = retrieval_complete_epoch_ms
    else:
        bm25_epoch_ms = retrieval_started_epoch_ms + 1000 * float(
            phase_seconds.get("bm25_complete") or 0
        )
        vector_epoch_ms = retrieval_started_epoch_ms + 1000 * float(
            phase_seconds.get("vector_complete") or 0
        )
        rerank_epoch_ms = retrieval_started_epoch_ms + 1000 * float(
            phase_seconds.get("rerank_complete") or 0
        )
    diagnostics["stage_timestamps_epoch_ms"] = {
        "t2_normalization": round(normalization_complete_epoch_ms, 3),
        "t3_bm25": round(bm25_epoch_ms, 3),
        "t4_vector": round(vector_epoch_ms, 3),
        "t5_rerank": round(rerank_epoch_ms, 3),
        "t6_first_evidence_ready": round(retrieval_complete_epoch_ms, 3),
    }
    evidence_markdown = _evidence_markdown(result)
    first_stage_markdown, preview_rows = _first_stage_markdown(result, understood)
    first_stage = {
        "identified_topic": understood.get("task_intent") or "general_explanation",
        "retrieval_scope": "CURRENT_FORMAL_RAG",
        "retrieved_paper_count": len(result.get("retrieved_papers") or []),
        "evidence_count": sum(
            len(result.get(key) or [])
            for key in (
                "supporting_evidence",
                "counter_evidence",
                "condition_dependent_evidence",
            )
        ),
        "preview_paper_count": first_stage_markdown.count("- **["),
        "preview_evidence_count": len(preview_rows),
        "preview_rows": preview_rows,
        "elapsed_seconds": round(retrieval_elapsed, 4),
        "cache_hit": diagnostics["retrieval_cache_hit"],
        "evidence_markdown": first_stage_markdown,
    }
    if progress_callback:
        progress_callback(
            {
                "stage": "EVIDENCE_READY",
                "message": "首批证据已完成召回与重排序，正在生成完整科研回答。",
                **first_stage,
            }
        )
    answer = ""
    settings = get_deepseek_settings()
    if use_llm and settings.configured:
        try:
            from src.data_cache import load_llm_client

            client = load_llm_client() or DeepSeekClient(settings)
            llm_started = time.perf_counter()
            diagnostics["stage_timestamps_epoch_ms"]["t9_deepseek_start"] = round(
                time.time_ns() / 1_000_000, 3
            )
            answer = client.chat(
                [{"role": "system", "content": "你是钛合金疲劳科研助手，必须严格服从证据引用约束。"}, {"role": "user", "content": _llm_prompt(effective, result)}],
                temperature=0.1, max_tokens=1600, timeout=12, connect_timeout=3, max_retries=0,
            )
            diagnostics["llm_executed"] = True
            diagnostics["llm_usage"] = client.usage_snapshot()
            diagnostics["llm_elapsed_seconds"] = round(
                time.perf_counter() - llm_started, 4
            )
        except (DeepSeekRequestError, ValueError, OSError) as exc:
            diagnostics["llm_error"] = f"{type(exc).__name__}:{exc}"
    if not answer:
        visible_evidence = evidence_markdown if use_llm else first_stage_markdown
        answer = "## 证据优先回答（DeepSeek不可用或未启用）\n\n" + visible_evidence
    else:
        answer += "\n\n" + evidence_markdown
    diagnostics["total_elapsed_seconds"] = round(time.perf_counter() - started, 4)
    diagnostics["stage_timestamps_epoch_ms"]["t10_final_answer"] = round(
        time.time_ns() / 1_000_000, 3
    )
    diagnostics["timeout_exceeded"] = diagnostics["total_elapsed_seconds"] > 30
    if progress_callback:
        progress_callback(
            {
                "stage": "COMPLETE",
                "message": "完整科研回答已生成。" if not diagnostics["timeout_exceeded"] else "完整回答超过30秒，已保留全部检索证据并标明超时。",
                "total_elapsed_seconds": diagnostics["total_elapsed_seconds"],
            }
        )
    return {
        "answer": answer,
        "first_stage": first_stage,
        "research_result": result,
        "query_understanding": understood,
        "diagnostics": diagnostics,
    }
