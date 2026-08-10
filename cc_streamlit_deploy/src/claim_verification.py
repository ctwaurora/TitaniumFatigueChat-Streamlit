"""Deterministic claim-to-evidence audit for generated scientific answers."""

from __future__ import annotations

import re
from typing import Any

from src.feature_flags import feature_enabled
from src.claim_evidence_verifier import (
    ALTERNATIVE_MECHANISM,
    CONDITION_DEPENDENT,
    DIRECT_COUNTER,
    DIRECT_SUPPORT,
    INSUFFICIENT,
    SUPPORTING_CONTEXT,
    verify_claim_evidence,
)
from src.skill_scientific_evaluation import classify_claim


_SENTENCE = re.compile(r"(?<=[。！？!?])|\n+")
_EVIDENCE_LABEL = re.compile(r"Evidence\s*ID\s*[：:]\s*([A-Za-z0-9_.:-]+)", re.I)
_PAGE_LABEL = re.compile(r"(?:p\.|页码[：:]?)\s*(\d+)", re.I)
_SCIENTIFIC_SIGNAL = re.compile(
    r"提高|降低|增加|减少|影响|导致|相关|控制|促进|抑制|机制|疲劳|裂纹|应力|条件|公式|模型|"
    r"increase|decrease|affect|fatigue|crack|stress|mechanism|model",
    re.I,
)
_NUMBER_WITH_UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:MPa|GPa|Hz|mm|µm|μm|%|cycles?)", re.I)
_INFERENCE_MARKERS = (
    "系统推断", "可以推测", "候选模型", "待拟合", "尚未直接验证",
    "【系统候选推断】", "【条件化综合判断】", "【替代机制】", "【证据不足】",
)
_METHOD_OR_BOUNDARY_MARKERS = (
    "建议", "需要补充", "可采用", "应采用", "应补充", "当前不能确定",
    "当前证据不足", "本地证据不足", "缺少", "不能外推", "不能跨", "不得外推",
    "只支持在已报告条件", "用于验证",
)


def _claim_sentences(answer: str) -> list[str]:
    body = str(answer or "").split("## 文献证据", 1)[0]
    claims = []
    for item in _SENTENCE.split(body):
        cleaned = re.sub(r"^#{1,6}\s+|^[-*]\s+", "", item).strip()
        # A rendered Markdown table is a structured presentation block. Treating
        # each pipe-delimited row as prose causes the guard label to corrupt the
        # table and leak raw Markdown into Streamlit.
        if cleaned.startswith("|") and cleaned.endswith("|"):
            continue
        if claims and _EVIDENCE_LABEL.search(cleaned):
            claims[-1] = f"{claims[-1]} {cleaned}"
            continue
        if len(cleaned) >= 12 and _SCIENTIFIC_SIGNAL.search(cleaned):
            claims.append(cleaned)
    return claims


def verify_answer_claims(answer: str, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    body = str(answer or "").split("## 文献证据", 1)[0]
    paragraphs = [item.strip() for item in body.splitlines() if item.strip()]
    citations = evidence_bundle.get("citation_index") or {}
    formulas = {str(item.get("formula_id") or ""): item for item in evidence_bundle.get("formulas") or []}
    known_ids = set(citations) | {key for key in formulas if key}
    records = []
    critical_failures: list[str] = []
    for index, claim in enumerate(_claim_sentences(answer), start=1):
        paragraph = next((item for item in paragraphs if claim in item or item in claim), claim)
        citation_context = f"{claim} {paragraph}"
        labeled_ids = set(_EVIDENCE_LABEL.findall(citation_context))
        mentioned_ids = {evidence_id for evidence_id in known_ids if evidence_id and evidence_id in citation_context}
        cited_ids = sorted(labeled_ids | mentioned_ids)
        unknown_ids = sorted(set(cited_ids) - known_ids)
        pages = set(_PAGE_LABEL.findall(citation_context))
        valid_pages = {
            str((citations.get(evidence_id) or formulas.get(evidence_id) or {}).get("page_number"))
            for evidence_id in cited_ids
            if (citations.get(evidence_id) or formulas.get(evidence_id) or {}).get("page_number")
        }
        invalid_pages = sorted(pages - valid_pages) if cited_ids else []
        inference = any(marker in claim for marker in _INFERENCE_MARKERS)
        method_or_boundary = any(marker in claim for marker in _METHOD_OR_BOUNDARY_MARKERS)
        # Exact experimental values require traceable evidence regardless of
        # whether the surrounding sentence is labelled inference/proposal.
        # Boundary language cannot turn an invented temperature, load, rate,
        # size, or cycle count into a scientifically acceptable parameter.
        unsupported_numeric = bool(_NUMBER_WITH_UNIT.search(claim)) and not cited_ids
        cited_rows = []
        for evidence_id in cited_ids:
            row = dict(citations.get(evidence_id) or formulas.get(evidence_id) or {})
            row.setdefault("evidence_id", evidence_id)
            cited_rows.append(row)
        semantic_available = any(
            row.get("original_text") or row.get("claim") for row in cited_rows
        )
        semantic = (
            verify_claim_evidence(
                claim,
                cited_rows,
                query_frame=evidence_bundle.get("query_frame") or {},
            )
            if semantic_available else None
        )
        best_role = semantic["best_role"] if semantic else "LEGACY_TRACEABLE"
        counter_claim = bool(re.search(r"反证|反向|无显著|没有显著|效应消失|方向相反", claim))
        conditional_claim = bool(re.search(r"条件|仅在|取决于|不能外推|边界", claim))
        alternative_claim = bool(re.search(r"替代机制|替代解释|另一种解释|不自动否定", claim))
        context_claim = bool(re.search(r"背景|支持性上下文|相关事实|不能支持完整因果", claim))
        role_supports_claim = (
            best_role == DIRECT_SUPPORT
            or (best_role == DIRECT_COUNTER and counter_claim)
            or (best_role == CONDITION_DEPENDENT and conditional_claim)
            or (best_role == ALTERNATIVE_MECHANISM and alternative_claim)
            or (best_role == SUPPORTING_CONTEXT and context_claim)
            or best_role == "LEGACY_TRACEABLE"
        )
        status = (
            "SYSTEM_INFERENCE" if inference
            else "METHOD_OR_BOUNDARY" if method_or_boundary
            else "SUPPORTED" if cited_ids and not unknown_ids and not invalid_pages and role_supports_claim
            else "MISALIGNED_EVIDENCE" if cited_ids and semantic_available
            else "UNSUPPORTED"
        )
        if unknown_ids:
            critical_failures.append(f"claim_{index}:unknown_evidence_id")
        if invalid_pages:
            critical_failures.append(f"claim_{index}:invalid_page")
        if unsupported_numeric:
            critical_failures.append(f"claim_{index}:unsupported_numeric_value")
        if status == "MISALIGNED_EVIDENCE":
            critical_failures.append(f"claim_{index}:evidence_role_mismatch:{best_role}")
        records.append({
            "claim_index": index,
            "claim_text": claim,
            "status": status,
            "evidence_ids": cited_ids,
            "unknown_evidence_ids": unknown_ids,
            "page_references": sorted(pages),
            "invalid_page_references": invalid_pages,
            "system_inference": inference,
            "method_or_boundary": method_or_boundary,
            "claim_category": classify_claim(
                claim,
                system_inference=inference,
                method_or_boundary=method_or_boundary,
            ),
            "unsupported_numeric_value": unsupported_numeric,
            "verified_evidence_role": best_role,
            "semantic_evidence_audit": semantic or {},
            "alignment_status": (
                "PASS" if status == "SUPPORTED" and best_role == DIRECT_SUPPORT
                else "CONDITIONAL" if status in {"SUPPORTED", "SYSTEM_INFERENCE"}
                else "NOT_APPLICABLE" if status == "METHOD_OR_BOUNDARY"
                else "FAIL"
            ),
        })
    supported = sum(record["status"] == "SUPPORTED" for record in records)
    verifiable = [record for record in records if record["status"] != "METHOD_OR_BOUNDARY"]
    cited_verifiable = [record for record in verifiable if record["evidence_ids"]]
    traceable = [
        record for record in cited_verifiable
        if not record["unknown_evidence_ids"] and not record["invalid_page_references"]
    ]
    direct_presentations = [
        record for record in cited_verifiable
        if not any(marker in record["claim_text"] for marker in _INFERENCE_MARKERS)
        and not re.search(r"条件|仅在|取决于|反证|反向|替代机制|背景", record["claim_text"])
    ]
    direct_correct = [
        record for record in direct_presentations
        if record["status"] == "SUPPORTED" and record["verified_evidence_role"] == DIRECT_SUPPORT
    ]
    condition_scores = []
    for record in cited_verifiable:
        assessments = (record.get("semantic_evidence_audit") or {}).get("assessments") or []
        if assessments:
            condition_scores.append(max(float(row.get("condition_match_score") or 0) for row in assessments))
    aligned = sum(record["alignment_status"] in {"PASS", "CONDITIONAL"} for record in verifiable)
    unsupported = sum(record["alignment_status"] == "FAIL" for record in verifiable)
    return {
        "passed": not critical_failures,
        "claim_count": len(records),
        "supported_claim_count": supported,
        "system_inference_count": sum(record["status"] == "SYSTEM_INFERENCE" for record in records),
        "method_or_boundary_count": sum(record["status"] == "METHOD_OR_BOUNDARY" for record in records),
        "unsupported_claim_count": sum(record["status"] == "UNSUPPORTED" for record in records),
        "misaligned_evidence_count": sum(record["status"] == "MISALIGNED_EVIDENCE" for record in records),
        "claim_evidence_alignment_rate": round(aligned / len(verifiable), 4) if verifiable else 1.0,
        "direct_evidence_precision": round(len(direct_correct) / len(direct_presentations), 4) if direct_presentations else 1.0,
        "condition_match_rate": round(sum(condition_scores) / len(condition_scores), 4) if condition_scores else 1.0,
        "unsupported_claim_rate": round(unsupported / len(verifiable), 4) if verifiable else 0.0,
        "citation_traceability_rate": round(len(traceable) / len(cited_verifiable), 4) if cited_verifiable else 1.0,
        "critical_failures": critical_failures,
        "claims": records,
    }


def apply_claim_guard(answer: str, evidence_bundle: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Downgrade unsupported assertions in place without generating new content."""
    before = verify_answer_claims(answer, evidence_bundle)
    if not feature_enabled("TFC_CLAIM_GUARD", default=True):
        return answer, {"enabled": False, "before": before, "after": before, "changed_claim_count": 0}
    guarded = str(answer)
    changed = 0
    for record in before["claims"]:
        if record["status"] not in {"UNSUPPORTED", "MISALIGNED_EVIDENCE"}:
            continue
        claim = record["claim_text"]
        if claim not in guarded:
            continue
        verified_role = record.get("verified_evidence_role")
        if record["unsupported_numeric_value"]:
            prefix = "本地证据不足，不能确认以下数值性陈述："
        elif verified_role == SUPPORTING_CONTEXT:
            prefix = "【证据不足】该引文只支持背景，不能直接支持以下完整主张："
        elif verified_role == CONDITION_DEPENDENT:
            prefix = "【条件化综合判断】"
        elif verified_role == ALTERNATIVE_MECHANISM:
            prefix = "【替代机制】"
        elif verified_role == DIRECT_COUNTER:
            prefix = "【证据不足】该反向证据不能直接支持以下正向主张："
        elif verified_role == INSUFFICIENT:
            prefix = "【证据不足】"
        else:
            prefix = "系统推断（当前引文未直接验证）："
        cleaned = claim.lstrip("，,；; ")
        guarded = guarded.replace(claim, prefix + cleaned, 1)
        changed += 1
    after = verify_answer_claims(guarded, evidence_bundle)
    return guarded, {"enabled": True, "before": before, "after": after, "changed_claim_count": changed}
