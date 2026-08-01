"""DeepSeek semantic enrichment that preserves page-level provenance.

The model may interpret relationships and uncertainty, but it never supplies
or changes evidence source text, page numbers, or section locations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.data_contracts import EvidenceRecord, PageRecord, SectionCoverageRecord
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError


MAX_EVIDENCE_RECORDS = 72
BATCH_SIZE = 12


def _json_payload(text: str) -> Any:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start:end + 1])


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: Dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)[:80]
        if isinstance(item, (str, int, float, bool)) or item is None:
            safe[name] = item
        elif isinstance(item, list):
            safe[name] = [entry for entry in item[:20] if isinstance(entry, (str, int, float, bool))]
    return safe


def _structure_prompt(
    title: str,
    pages: Sequence[PageRecord],
    sections: Sequence[SectionCoverageRecord],
) -> List[Dict[str, str]]:
    page_samples = [
        {
            "page_number": page.page_number,
            "deterministic_section": page.section_title,
            "text_excerpt": page.cleaned_text[:700],
        }
        for page in pages
        if page.cleaned_text
    ][:24]
    section_rows = [section.to_dict() for section in sections]
    return [
        {
            "role": "system",
            "content": (
                "You analyze titanium-alloy fatigue papers. Return strict JSON only. "
                "Use only supplied excerpts. Mark uncertainty explicitly and never invent pages, "
                "measurements, formulas, or conclusions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": (
                        "Interpret paper structure, major experimental condition groups, material/"
                        "process/fatigue mechanisms, and uncertainty."
                    ),
                    "required_schema": {
                        "paper_structure": ["section name and purpose"],
                        "condition_groups": ["explicit condition group"],
                        "mechanism_relations": ["explicit or cautiously inferred relation"],
                        "formula_context_notes": ["formula context or uncertainty"],
                        "uncertainty_notes": ["items requiring human review"],
                    },
                    "title": title,
                    "deterministic_sections": section_rows,
                    "page_samples": page_samples,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _evidence_prompt(batch: Sequence[EvidenceRecord]) -> List[Dict[str, str]]:
    rows = [
        {
            "evidence_id": record.evidence_id,
            "original_text": record.original_text[:1400],
            "page_number": record.page_number,
            "section": record.section,
            "deterministic_directness": record.directness,
            "formula_reference": record.formula_reference[:1000],
        }
        for record in batch
    ]
    return [
        {
            "role": "system",
            "content": (
                "You semantically enrich traceable scientific evidence. Return a strict JSON array. "
                "Do not rewrite, quote, or relocate source evidence. Never invent a value or formula. "
                "If support is ambiguous, state uncertainty."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": (
                        "For each evidence_id interpret its concise claim, explicit variables, "
                        "conditions, result, mechanism relation, support/counter/condition-dependent "
                        "role, formula context, and uncertainty."
                    ),
                    "allowed_relation": ["SUPPORT", "COUNTER", "CONDITION_DEPENDENT"],
                    "required_fields": [
                        "evidence_id", "claim", "variables", "conditions", "result",
                        "mechanism_relation", "relation", "formula_context", "uncertain",
                        "uncertainty_reason",
                    ],
                    "evidence": rows,
                },
                ensure_ascii=False,
            ),
        },
    ]


def enrich_evidence_with_deepseek(
    *,
    client: DeepSeekClient,
    title: str,
    pages: Sequence[PageRecord],
    sections: Sequence[SectionCoverageRecord],
    evidence: Sequence[EvidenceRecord],
) -> Tuple[List[EvidenceRecord], Dict[str, Any]]:
    """Enhance semantics while keeping provenance fields immutable."""
    usage_before = client.usage_snapshot()
    result: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "structure": {},
        "enriched_record_count": 0,
        "semantic_success_count": 0,
        "semantic_failure_count": 0,
        "errors": [],
    }
    try:
        structure_text = client.chat(
            _structure_prompt(title, pages, sections), max_tokens=1400, timeout=90
        )
        parsed = _json_payload(structure_text)
        if isinstance(parsed, Mapping):
            result["structure"] = dict(parsed)
            result["semantic_success_count"] += 1
        else:
            raise ValueError("structure response is not an object")
    except (DeepSeekRequestError, ValueError, json.JSONDecodeError) as exc:
        result["semantic_failure_count"] += 1
        result["errors"].append(f"STRUCTURE:{type(exc).__name__}")

    eligible = [
        record for record in evidence
        if record.original_text and record.directness != "INVALID"
    ][:MAX_EVIDENCE_RECORDS]
    by_id = {record.evidence_id: record for record in eligible}
    for start in range(0, len(eligible), BATCH_SIZE):
        batch = eligible[start:start + BATCH_SIZE]
        try:
            response_text = client.chat(
                _evidence_prompt(batch), max_tokens=2400, timeout=90
            )
            parsed = _json_payload(response_text)
            if not isinstance(parsed, list):
                raise ValueError("evidence response is not an array")
            result["semantic_success_count"] += 1
            for item in parsed:
                if not isinstance(item, Mapping):
                    continue
                record = by_id.get(str(item.get("evidence_id") or ""))
                if record is None:
                    continue
                # original_text, page_number, section and source_method remain untouched.
                claim = str(item.get("claim") or "").strip()
                if claim:
                    record.claim = claim[:2000]
                record.variables.update(_safe_mapping(item.get("variables")))
                record.conditions.update(_safe_mapping(item.get("conditions")))
                mechanism = str(item.get("mechanism_relation") or "").strip()
                if mechanism:
                    record.variables["deepseek_mechanism_relation"] = mechanism[:1600]
                interpreted_result = str(item.get("result") or "").strip()
                if interpreted_result:
                    record.result = interpreted_result[:2000]
                relation = str(item.get("relation") or "").upper()
                if relation in {"SUPPORT", "COUNTER", "CONDITION_DEPENDENT"}:
                    record.support_or_counter = relation
                formula_context = str(item.get("formula_context") or "").strip()
                if formula_context:
                    record.conditions["deepseek_formula_context"] = formula_context[:1600]
                uncertain = bool(item.get("uncertain"))
                uncertainty_reason = str(item.get("uncertainty_reason") or "").strip()
                if uncertain:
                    record.review_status = "DEEPSEEK_UNCERTAIN_REVIEW_REQUIRED"
                    if uncertainty_reason:
                        record.conditions["deepseek_uncertainty"] = uncertainty_reason[:1000]
                record.extraction_method = (
                    f"{record.extraction_method}+DEEPSEEK_SEMANTIC"
                    if "DEEPSEEK_SEMANTIC" not in record.extraction_method
                    else record.extraction_method
                )
                result["enriched_record_count"] += 1
        except (DeepSeekRequestError, ValueError, json.JSONDecodeError) as exc:
            result["semantic_failure_count"] += 1
            result["errors"].append(f"EVIDENCE_BATCH:{start}:{type(exc).__name__}")

    result["applied"] = result["enriched_record_count"] > 0
    usage_after = client.usage_snapshot()
    result["usage"] = {
        key: int(usage_after.get(key, 0)) - int(usage_before.get(key, 0))
        for key in usage_after
    }
    return list(evidence), result
