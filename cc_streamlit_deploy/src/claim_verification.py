"""Deterministic claim-to-evidence audit for generated scientific answers."""

from __future__ import annotations

import re
from typing import Any

from src.feature_flags import feature_enabled


_SENTENCE = re.compile(r"(?<=[。！？!?])|\n+")
_EVIDENCE_LABEL = re.compile(r"Evidence\s*ID\s*[：:]\s*([A-Za-z0-9_.:-]+)", re.I)
_PAGE_LABEL = re.compile(r"(?:p\.|页码[：:]?)\s*(\d+)", re.I)
_SCIENTIFIC_SIGNAL = re.compile(
    r"提高|降低|增加|减少|影响|导致|相关|控制|促进|抑制|机制|疲劳|裂纹|应力|条件|公式|模型|"
    r"increase|decrease|affect|fatigue|crack|stress|mechanism|model",
    re.I,
)
_NUMBER_WITH_UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:MPa|GPa|Hz|mm|µm|μm|%|cycles?)", re.I)
_INFERENCE_MARKERS = ("系统推断", "可以推测", "候选模型", "待拟合", "尚未直接验证")
_METHOD_OR_BOUNDARY_MARKERS = (
    "建议", "需要补充", "可采用", "应采用", "应补充", "当前不能确定",
    "当前证据不足", "本地证据不足", "缺少", "不能外推", "不得外推", "用于验证",
)


def _claim_sentences(answer: str) -> list[str]:
    body = str(answer or "").split("## 文献证据", 1)[0]
    claims = []
    for item in _SENTENCE.split(body):
        cleaned = re.sub(r"^#{1,6}\s+|^[-*]\s+", "", item).strip()
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
        unsupported_numeric = bool(_NUMBER_WITH_UNIT.search(claim)) and not cited_ids and not inference and not method_or_boundary
        status = (
            "SUPPORTED" if cited_ids and not unknown_ids and not invalid_pages
            else "SYSTEM_INFERENCE" if inference
            else "METHOD_OR_BOUNDARY" if method_or_boundary
            else "UNSUPPORTED"
        )
        if unknown_ids:
            critical_failures.append(f"claim_{index}:unknown_evidence_id")
        if invalid_pages:
            critical_failures.append(f"claim_{index}:invalid_page")
        if unsupported_numeric:
            critical_failures.append(f"claim_{index}:unsupported_numeric_value")
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
            "unsupported_numeric_value": unsupported_numeric,
        })
    supported = sum(record["status"] == "SUPPORTED" for record in records)
    verifiable = [record for record in records if record["status"] != "METHOD_OR_BOUNDARY"]
    return {
        "passed": not critical_failures,
        "claim_count": len(records),
        "supported_claim_count": supported,
        "system_inference_count": sum(record["status"] == "SYSTEM_INFERENCE" for record in records),
        "method_or_boundary_count": sum(record["status"] == "METHOD_OR_BOUNDARY" for record in records),
        "unsupported_claim_count": sum(record["status"] == "UNSUPPORTED" for record in records),
        "grounded_claim_rate": round(supported / len(verifiable), 4) if verifiable else 1.0,
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
        if record["status"] != "UNSUPPORTED":
            continue
        claim = record["claim_text"]
        if claim not in guarded:
            continue
        prefix = (
            "本地证据不足，不能确认以下数值性陈述："
            if record["unsupported_numeric_value"]
            else "系统推断（当前引文未直接验证）："
        )
        cleaned = claim.lstrip("，,；; ")
        guarded = guarded.replace(claim, prefix + cleaned, 1)
        changed += 1
    after = verify_answer_claims(guarded, evidence_bundle)
    return guarded, {"enabled": True, "before": before, "after": after, "changed_claim_count": changed}
