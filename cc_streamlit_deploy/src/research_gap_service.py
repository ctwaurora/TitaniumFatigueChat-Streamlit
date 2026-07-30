"""Evidence-gated research-gap analysis for selected canonical papers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Sequence

from src.hypothesis_service import (
    COUNTER_RE,
    OUTCOME_PATTERNS,
    VARIABLE_PATTERNS,
    _evidence_text,
    _first_detected,
    _reference,
)
from src.literature_library import eligible_paper_ids, trusted_evidence_rows
from src.stage1_store import BASE_DIR


def analyze_research_gaps(
    paper_ids: Sequence[str],
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    gate = eligible_paper_ids(paper_ids, base_dir=base_dir)
    if not gate["eligible"]:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "gaps": [],
            "rejected": gate["rejected"],
        }
    selected = set(gate["eligible"])
    rows = [
        row
        for row in trusted_evidence_rows(base_dir)
        if str(row.get("paper_id") or "") in selected
        and str(row.get("directness") or "") in {"DIRECT", "INDIRECT"}
        and int(float(row.get("page_number") or 0)) > 0
        and "REVIEW_REQUIRED" not in str(row.get("review_status") or "")
    ]
    independent, iv_pattern = _first_detected(rows, VARIABLE_PATTERNS)
    dependent, dv_pattern = _first_detected(rows, OUTCOME_PATTERNS)
    if len(rows) < 2 or not independent or not dependent:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "gaps": [],
            "rejected": gate["rejected"],
            "reason": "可信正文证据不足以同时识别研究变量和结果变量。",
        }
    relevant = [
        row
        for row in rows
        if re.search(iv_pattern, _evidence_text(row), re.I)
        or re.search(dv_pattern, _evidence_text(row), re.I)
    ]
    support = [row for row in relevant if not COUNTER_RE.search(_evidence_text(row))]
    counter = [row for row in relevant if COUNTER_RE.search(_evidence_text(row))]
    papers_with_support = {str(row.get("paper_id") or "") for row in support}
    gap = {
        "gap_id": f"GAP_{len(selected)}_{len(relevant)}",
        "gap_statement": (
            f"现有本地可信证据涉及 {independent} 与 {dependent}，但尚缺少统一工况、"
            "配对变量和跨论文验证，不能据此声称已建立普适关系。"
        ),
        "supporting_papers": sorted(papers_with_support),
        "supporting_evidence": [_reference(row) for row in support[:12]],
        "real_page_numbers": sorted(
            {int(float(row.get("page_number") or 0)) for row in relevant}
        ),
        "counter_evidence": [_reference(row) for row in counter[:8]],
        "condition_boundary": (
            "仅限所选证据真实报告的材料、制造、热处理、表面状态和疲劳加载条件。"
        ),
        "missing_evidence": [
            "统一变量定义与单位后的跨论文配对数据。",
            "独立论文验证与留一文献敏感性分析。",
            *([] if counter else ["相同工况下的反向或零效应证据。"]),
        ],
        "local_evidence_only": True,
    }
    return {"status": "GENERATED", "gaps": [gap], "rejected": gate["rejected"]}

