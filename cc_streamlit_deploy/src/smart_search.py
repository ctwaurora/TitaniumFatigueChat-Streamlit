"""User-triggered smart search with evidence-first degradation behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.api_keys import get_deepseek_settings
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError
from src.query_understanding import understand_user_query
from src.unified_rag import answer_research_question


def _compact_conditions(value: Any) -> str:
    conditions = value if isinstance(value, dict) else {}
    reported = {key: item for key, item in conditions.items() if item not in (None, "", [], {}, "NOT_REPORTED")}
    return json.dumps(reported, ensure_ascii=False) if reported else "NOT_REPORTED"


def _evidence_markdown(result: dict[str, Any]) -> str:
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
        for row in rows[:10]:
            lines.extend([
                f"- **{row.get('title') or '题名未报告'}**",
                f"  - Evidence ID：`{row.get('doc_id')}`；页码：{row.get('page_number')}；章节：{row.get('section') or 'NOT_REPORTED'}",
                f"  - 直接性：{row.get('directness') or 'NOT_REPORTED'}；条件：{_compact_conditions(row.get('experimental_conditions'))}",
                f"  - 原文：{row.get('original_text')}",
            ])
    diagnostics = result.get("retrieval_diagnostics") or {}
    lines.extend(["", "### 检索诊断", "", f"- BM25：{diagnostics.get('bm25_executed')}", f"- 向量召回：{diagnostics.get('vector_executed')}"])
    if diagnostics.get("errors"):
        lines.append(f"- 降级原因：{'; '.join(diagnostics['errors'])}")
    return "\n".join(lines)


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


def run_smart_search(question: str, *, base_dir: Path, use_llm: bool = True, top_k: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    understood = understand_user_query(question)
    effective = understood.get("corrected_query") or question
    result = answer_research_question(effective, top_k=top_k, base_dir=base_dir)
    diagnostics = result.setdefault("retrieval_diagnostics", {})
    diagnostics["intent"] = understood.get("task_intent")
    diagnostics["corrected_query"] = effective
    diagnostics["llm_executed"] = False
    diagnostics["llm_error"] = ""
    evidence_markdown = _evidence_markdown(result)
    answer = ""
    settings = get_deepseek_settings()
    if use_llm and settings.configured:
        try:
            client = DeepSeekClient(settings)
            answer = client.chat(
                [{"role": "system", "content": "你是钛合金疲劳科研助手，必须严格服从证据引用约束。"}, {"role": "user", "content": _llm_prompt(effective, result)}],
                temperature=0.1, max_tokens=1600, timeout=20, connect_timeout=5, max_retries=1,
            )
            diagnostics["llm_executed"] = True
            diagnostics["llm_usage"] = client.usage_snapshot()
        except (DeepSeekRequestError, ValueError, OSError) as exc:
            diagnostics["llm_error"] = f"{type(exc).__name__}:{exc}"
    if not answer:
        answer = "## 证据优先回答（DeepSeek不可用或未启用）\n\n" + evidence_markdown
    else:
        answer += "\n\n" + evidence_markdown
    diagnostics["total_elapsed_seconds"] = round(time.perf_counter() - started, 4)
    diagnostics["timeout_exceeded"] = diagnostics["total_elapsed_seconds"] > 30
    return {"answer": answer, "research_result": result, "query_understanding": understood, "diagnostics": diagnostics}
