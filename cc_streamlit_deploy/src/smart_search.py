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


def _llm_prompt(question: str, result: dict[str, Any], module: str) -> str:
    rows = (result["supporting_evidence"][:6] + result["counter_evidence"][:4] + result["condition_dependent_evidence"][:4])
    evidence = "\n".join(
        f"[{row.get('doc_id')} | {row.get('title')} | p.{row.get('page_number')} | {row.get('section')}] {row.get('original_text')} CONDITIONS={_compact_conditions(row.get('experimental_conditions'))}"
        for row in rows
    )
    return (
        f"当前模块是{module}。仅依据下列本地证据，用中文直接回答问题。"
        "本段只写结论、机制、条件边界、反向观点与不确定性，不列文献清单，"
        "不显示JSON、代码字段或NOT_REPORTED；不得补造文献、页码、参数或机制边界。\n"
        f"问题：{question}\n证据：\n{evidence}"
    )


def _module_analysis_markdown(
    module: str, question: str, result: dict[str, Any], synthesis: str
) -> str:
    sufficiency = result.get("evidence_sufficiency") or {}
    level = str(sufficiency.get("status") or "本地证据不足")
    supporting = len(result.get("supporting_evidence") or [])
    counter = len(result.get("counter_evidence") or [])
    conditional = len(result.get("condition_dependent_evidence") or [])
    direct = synthesis.strip() or (
        f"本地正式文献库检索到 {supporting} 条支持证据、{counter} 条反向证据和"
        f" {conditional} 条条件依赖证据。结论必须限定在已报告材料、工艺与载荷条件内。"
    )
    analyses: dict[str, str] = {
        "research_gap": f"""### 研究空白标题

**现有研究已经解决的问题**

现有研究已经建立了若干材料、工艺、后处理和载荷因素与疲劳行为之间的条件化联系。

**尚未解决的问题**

不同因素在匹配材料批次、微观组织、表面状态和载荷条件下的相对主导边界仍缺少独立验证。

**为什么构成研究空白**

现有证据分散在不同试验条件中，不能把跨论文差异直接解释为单一变量的因果效应。

**可能反驳该空白的证据**

本次检索得到 {counter} 条反向证据和 {conditional} 条条件依赖证据；若其条件已覆盖目标组合，该空白应缩小或取消。

**已覆盖条件**

| 维度 | 当前覆盖 |
|---|---|
| 文献来源 | {len(result.get('retrieved_papers') or [])} 篇正式文献 |
| 支持证据 | {supporting} 条 |
| 反向证据 | {counter} 条 |
| 条件依赖证据 | {conditional} 条 |

**缺失条件**

- 匹配材料批次与微观组织的对照；
- 匹配应力比、疲劳区间和环境的组合验证；
- 跨批次独立重复。

**候选研究问题**

1. 在匹配应力与组织条件下，目标因素的效应方向是否仍保持一致？
2. 哪一组工艺或载荷条件会改变主导失效机制？
3. 该边界能否在独立材料批次中复现？

**可验证性**

采用受控分组疲劳试验、断口溯源和匹配条件统计模型进行验证。

**证据充分度**

{level}
""",
        "hypothesis_generation": f"""**候选假设**

在其他条件匹配时，问题中的目标因素会以条件依赖方式改变疲劳寿命或裂纹起裂机制。

**科学依据**

当前有 {supporting} 条支持证据、{counter} 条反向证据和 {conditional} 条条件依赖证据。

**机制链**

目标因素变化 → 局部应力或微观损伤演化变化 → 起裂位置或扩展行为变化 → 疲劳响应变化。

**自变量**：问题中的目标因素。  
**因变量**：疲劳寿命、疲劳极限、裂纹起裂或扩展指标。  
**控制变量**：材料批次、组织、试样几何、表面状态、应力比、频率、温度与环境。  
**预测方向**：以支持证据中的主方向为候选预测，但遇到反向证据所对应条件时允许改变。  
**支持证据**：{supporting} 条。  
**反向证据**：{counter} 条。  
**适用边界**：仅限于证据卡片中明确报告的材料、工艺和载荷条件。  
**哪些结果会推翻假设**：匹配条件后效应消失、方向稳定反转，或由未控制变量完全解释。  
**证据充分度**：{level}。
""",
        "experiment_design": f"""**研究对象**：与问题相符的钛合金疲劳试样。  
**核心假设**：目标因素在条件匹配后仍会改变疲劳响应。  
**自变量**：目标因素及必要分层水平。  
**因变量**：疲劳寿命、疲劳极限、起裂源和裂纹扩展指标。  
**控制变量**：材料批次、组织、几何、表面状态、应力比、频率、温度和环境。  
**协变量**：残余应力、缺陷几何、粗糙度与织构。  
**分组方案**：基准组与目标因素多水平组，并保留关键条件的交互组。  
**样本建议**：先做每组不少于5件的先导试验，正式样本量由效应量与方差功效分析确定。  
**制样方法**：同批粉末、同一工艺窗口，记录加工和后处理全过程。  
**加载条件**：应力水平、应力比和疲劳区间预注册，不混合拟合 HCF 与 VHCF。  
**表征方法**：表面轮廓、micro-CT、残余应力、EBSD和SEM断口溯源按问题选用。  
**数据分析**：S-N/裂纹扩展拟合、混合效应模型、交互项检验与不确定性区间。  
**预测方向**：采用证据主方向作为预注册预测。  
**支持假设的结果**：效应在匹配条件和独立批次中方向一致。  
**推翻假设的结果**：效应消失、稳定反转或由混杂因素完全解释。  
**最低成本方案**：复用现有试样完成表面测量、断口复核和小规模对照。  
**完整方案**：跨批次全因子疲劳试验并进行盲法断口判定。  
**风险与混杂因素**：批次差异、检测分辨率、残余应力、组织差异和删失数据。
""",
        "formula_explanation": f"""**公式**

仅解释本次证据中能够追溯到原文上下文的疲劳模型；没有完整参数和单位时不做数值代入。

**中文解释**：公式用于描述载荷、寿命或裂纹扩展速率之间的经验或机制关系。  
**变量与单位**：以原文定义为准，必须统一应力、长度、循环数和扩展速率单位。  
**适用条件**：材料、应力比、温度、疲劳区间和拟合区间与原文一致。  
**前提假设**：模型形式、参数稳定性和数据区间满足原文假设。  
**不适用情况**：跨疲劳区间、跨失效机制或单位未统一。  
**多公式结果差异**：参数标定、应力比修正、缺陷表征和适用区间会造成差异。  
**哪些公式不能直接比较**：输入量定义、单位、拟合区间或物理假设不同的公式。
""",
    }
    generic = f"""**结论**

本次检索得到 {supporting} 条支持证据、{counter} 条反向证据和 {conditional} 条条件依赖证据。

**机制解释**

目标因素通过局部应力、组织或损伤演化影响疲劳响应，具体方向受工艺和载荷条件约束。

**条件边界**

只在文献证据卡片明确报告的材料、工艺、表面状态、热处理和载荷条件内成立。

**支持与反向观点**

支持证据与反向证据并存时，应优先解释条件差异，不把跨条件差异直接当作矛盾。

**不确定性与研究意义**

证据充分度为 {level}；缺失条件需要通过补充检索或受控实验验证。
"""
    analysis = analyses.get(module, generic)
    return f"""## 第一部分：直接回答

{direct}

## 第二部分：科研分析

{analysis}

## 第三部分：结论边界

- 证据充分度：{level}。
- 可直接支持：证据原文明确报告且条件一致的结论。
- 系统推断：跨文献机制串联、候选假设与实验建议。
- 本地证据不足：缺少匹配条件、独立重复或必要参数的部分。
"""


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
    retrieval_function = _cached_retrieval if use_llm else _cached_first_stage_retrieval
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
                [{"role": "system", "content": "你是钛合金疲劳科研助手，必须严格服从证据引用约束。"}, {"role": "user", "content": _llm_prompt(effective, result, module)}],
                temperature=0.1, max_tokens=1600, timeout=12, connect_timeout=3, max_retries=0,
            )
            diagnostics["llm_executed"] = True
            diagnostics["llm_usage"] = client.usage_snapshot()
            diagnostics["llm_elapsed_seconds"] = round(
                time.perf_counter() - llm_started, 4
            )
        except (DeepSeekRequestError, ValueError, OSError) as exc:
            diagnostics["llm_error"] = f"{type(exc).__name__}:{exc}"
    answer = _module_analysis_markdown(module, effective, result, answer)
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
