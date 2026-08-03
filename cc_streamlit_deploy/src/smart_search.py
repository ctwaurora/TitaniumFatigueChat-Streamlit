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
from src.research_skills.contracts import SkillInput
from src.research_skills.router import get_research_skill
from src.unified_rag import answer_research_question
from src.variable_mapper import extract_variable_pair


ProgressCallback = Callable[[dict[str, Any]], None]
PORE_PATTERN = re.compile(r"孔隙|孔洞|气孔|pore|porosity", re.IGNORECASE)

ENTITY_LABELS = {
    "pore_size": "孔隙尺寸（√area）",
    "fatigue_life": "疲劳寿命（Nf）",
    "fatigue_limit": "疲劳极限（σw）",
    "da_dn": "裂纹扩展速率（da/dN）",
    "delta_k": "应力强度因子范围（ΔK）",
    "surface_roughness": "表面粗糙度（Ra/Rz）",
    "pore_location": "孔隙距表面距离",
    "stress_amplitude": "应力幅（σa）",
    "stress_ratio": "应力比（R）",
    "heat_treatment": "热处理/HIP状态",
    "microstructure": "微观组织",
    "residual_stress": "残余应力",
    "porosity": "孔隙率",
    "paris_c_m": "Paris参数（C、m）",
    "build_orientation": "建造方向",
    "short_crack": "短裂纹",
    "long_crack": "长裂纹",
}


def resolve_question_entities(question: str) -> dict[str, Any]:
    """Resolve concrete scientific entities before any answer template runs."""
    independent, dependent, classification = extract_variable_pair(question)
    lowered = str(question).lower()
    aliases = (
        (("建造方向", "build orientation", "build direction"), "build_orientation"),
        (("短裂纹", "小裂纹", "short crack", "small crack"), "short_crack"),
        (("长裂纹", "long crack", "long-crack"), "long_crack"),
        (("内部孔隙", "internal pore", "near-surface pore"), "pore_size"),
    )
    mentioned = [canonical for terms, canonical in aliases if any(term in lowered for term in terms)]
    if mentioned:
        if independent in {None, "fatigue_life", "da_dn"}:
            independent = mentioned[0]
        elif dependent is None and mentioned[0] != independent:
            dependent = mentioned[0]
    # Questions phrased as “whether X changes Y” should preserve causal order.
    if independent in {"fatigue_life", "fatigue_limit", "da_dn"} and dependent in {
        "heat_treatment", "surface_roughness", "residual_stress", "microstructure",
    }:
        independent, dependent = dependent, independent
    if not independent and re.search(r"paris|c/m|c和m|c、m|公式|方程", lowered):
        independent, dependent, classification = "delta_k", "da_dn", "equation_query"
    if dependent is None and independent in {"build_orientation", "short_crack", "long_crack"}:
        if independent == "short_crack" and (
            "长裂纹" in lowered or "long crack" in lowered
        ):
            dependent = "long_crack"
        else:
            dependent = "fatigue_life"
        classification = "quantitative_relation"
    return {
        "independent": independent,
        "dependent": dependent,
        "independent_label": ENTITY_LABELS.get(independent, independent or ""),
        "dependent_label": ENTITY_LABELS.get(dependent, dependent or ""),
        "classification": classification,
        "specific": bool(independent and dependent),
    }


def validate_generated_answer(answer: str, question: str, module: str) -> dict[str, Any]:
    """Hard gate against vague placeholders leaking into user-facing output."""
    banned = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应")
    hits = [term for term in banned if term in answer]
    entities = resolve_question_entities(question)
    missing_entities = []
    if entities["specific"]:
        for label in (entities["independent_label"], entities["dependent_label"]):
            if label and label.split("（", 1)[0] not in answer and label not in answer:
                missing_entities.append(label)
    return {
        "passed": not hits and not missing_entities,
        "banned_terms": hits,
        "missing_entities": missing_entities,
        "module": module,
    }


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
    labels = {
        "alloy_grade": "材料", "material": "材料",
        "manufacturing_process": "制造工艺", "process": "制造工艺",
        "build_orientation": "建造方向", "stress_ratio_R": "应力比",
        "heat_treatment": "热处理", "hip": "HIP",
        "surface_treatment": "表面处理", "surface_state": "表面状态",
        "fatigue_regime": "疲劳区间", "temperature": "温度",
        "environment": "环境", "frequency": "频率",
    }
    reported = []
    for key, item in conditions.items():
        if key not in labels or item in (None, "", [], {}, "NOT_REPORTED"):
            continue
        if isinstance(item, (list, tuple, set)):
            text = "、".join(str(value) for value in item if value not in (None, "", "NOT_REPORTED"))
        else:
            text = str(item).strip().strip('"')
        if text and text != "NOT_REPORTED":
            reported.append(f"{labels[key]}：{text}")
    return "；".join(reported) if reported else "该证据未完整报告实验条件。"


def _evidence_markdown(
    result: dict[str, Any],
    *,
    per_group_limit: int = 10,
    original_text_limit: int | None = None,
) -> str:
    sufficiency = result["evidence_sufficiency"]
    lines = ["## 文献证据", ""]
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
                f"#### {row.get('title') or '题名未报告'}",
                f"- 作者与年份：{row.get('authors') or '未报告'}，{row.get('year') or '未报告'}",
                f"- 证据作用：{label.replace('证据', '')}",
                f"- 原文摘录：{original_text}",
                f"- 页码：{row.get('page_number') or '未报告'}；章节：{row.get('section') or '未报告'}",
                f"- Evidence ID：{row.get('doc_id') or row.get('evidence_id') or '未报告'}",
                f"- 关键实验条件：{_compact_conditions(row.get('experimental_conditions'))}",
                "",
            ])
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
        f"- 检索边界：当前正式可信 RAG（{int(result.get('formal_paper_count') or 0)} 篇）",
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


def _build_skill_input(
    question: str,
    parsed_entities: dict[str, Any],
    result: dict[str, Any],
    dataset_version: str,
) -> SkillInput:
    support = list(result.get("supporting_evidence") or [])
    counter = list(result.get("counter_evidence") or [])
    conditional = list(result.get("condition_dependent_evidence") or [])
    retrieved = support + counter + conditional
    condition_evidence = [
        row for row in retrieved if row.get("experimental_conditions")
    ]
    formula_records = [
        row for row in retrieved
        if row.get("index_type") == "formula"
        or "FORMULA_" in str(row.get("doc_id") or "")
    ]
    return SkillInput(
        user_query=question,
        parsed_entities=parsed_entities,
        retrieved_evidence=retrieved,
        condition_evidence=condition_evidence,
        formula_records=formula_records,
        support_evidence=support,
        counter_evidence=counter,
        condition_dependent_evidence=conditional,
        dataset_version=dataset_version,
    )


def run_smart_search(
    question: str,
    *,
    base_dir: Path,
    use_llm: bool = True,
    top_k: int = 10,
    progress_callback: ProgressCallback | None = None,
    module: str = "research_analysis",
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
    # All four skills require support, counter and condition-dependent evidence.
    # ``use_llm`` controls synthesis only and never weakens retrieval.
    retrieval_function = _cached_retrieval
    cache_before = retrieval_function.cache_info()
    result = copy.deepcopy(
        retrieval_function(
            str(Path(base_dir).resolve()), dataset_version, effective, top_k
        )
    )
    try:
        manifest = {
            row["paper_id"]: row
            for row in (
                json.loads(line)
                for line in (Path(base_dir) / "data" / "paper_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
    except (OSError, json.JSONDecodeError, KeyError):
        manifest = {}
    for key in ("supporting_evidence", "counter_evidence", "condition_dependent_evidence"):
        for row in result.get(key) or []:
            paper = manifest.get(str(row.get("paper_id") or ""), {})
            row["authors"] = paper.get("authors") or ""
            row["year"] = paper.get("publication_date") or paper.get("year") or ""
    result["formal_paper_count"] = len(manifest)
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
        "formal_paper_count": len(manifest),
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
    parsed_entities = resolve_question_entities(question)
    skill = get_research_skill(module)
    skill_input = _build_skill_input(
        question, parsed_entities, result, dataset_version
    )
    synthesis = ""
    settings = get_deepseek_settings()
    if use_llm and settings.configured:
        try:
            from src.data_cache import load_llm_client

            client = load_llm_client() or DeepSeekClient(settings)
            llm_started = time.perf_counter()
            diagnostics["stage_timestamps_epoch_ms"]["t9_deepseek_start"] = round(
                time.time_ns() / 1_000_000, 3
            )
            synthesis = client.chat(
                [
                    {"role": "system", "content": "你是钛合金疲劳科研助手，必须严格服从证据引用约束。"},
                    {"role": "user", "content": skill.build_prompt(skill_input)},
                ],
                temperature=0.1, max_tokens=1600, timeout=12, connect_timeout=3, max_retries=0,
            )
            diagnostics["llm_executed"] = True
            diagnostics["llm_usage"] = client.usage_snapshot()
            diagnostics["llm_elapsed_seconds"] = round(
                time.perf_counter() - llm_started, 4
            )
        except (DeepSeekRequestError, ValueError, OSError) as exc:
            diagnostics["llm_error"] = f"{type(exc).__name__}:{exc}"
    skill_output = skill.generate(skill_input, synthesis=synthesis)
    answer = skill.render_output(skill_output)
    quality = skill_output.quality_gate
    answer += "\n\n" + evidence_markdown
    diagnostics["total_elapsed_seconds"] = round(time.perf_counter() - started, 4)
    diagnostics["stage_timestamps_epoch_ms"]["t10_final_answer"] = round(
        time.time_ns() / 1_000_000, 3
    )
    diagnostics["timeout_exceeded"] = diagnostics["total_elapsed_seconds"] > 30
    diagnostics["specificity_gate"] = quality
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
        "skill_output": skill_output.as_dict(),
        "diagnostics": diagnostics,
    }
