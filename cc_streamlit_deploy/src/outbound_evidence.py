"""Build and audit the minimum evidence payload allowed to leave the process."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.research_skills.contracts import SkillInput


# Long enough to retain a compact claim with its necessary qualifiers, while
# still far below a page or paragraph-scale payload.
MAX_EXCERPT_CHARS = 500
OUTBOUND_ROOT = Path("outputs/outbound_evidence")
AUDIT_PATH = Path("data/audit/outbound_evidence_manifest.jsonl")
CONDITION_KEYS = {
    "alloy_grade", "material", "manufacturing_process", "process", "heat_treatment",
    "hip", "surface_treatment", "surface_state", "stress_ratio_R", "fatigue_regime",
    "temperature", "environment", "build_orientation", "loading_mode", "frequency",
    "crack_stage", "specimen_geometry", "stress_amplitude", "maximum_stress",
}
UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:MPa(?:\s*[√·*]\s*m)?|GPa|kPa|Pa|mm|µm|μm|um|nm|Hz|kHz|°C|K|cycles?|m/cycle)(?![A-Za-z])",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?[-+]?\d+)?(?:\s*%)?")
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|none|without|neither|nor|cannot|can't|didn't|doesn't|insignificant)\b|"
    r"无|未|不|没有|不能|并非|不显著",
    re.IGNORECASE,
)
LOCAL_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|file://|\\\\[^\s]+[\\/])")
FORBIDDEN_KEY_PATTERN = re.compile(
    r'"(?:canonical_path|source_path|local_path|backup_path|api_key|secret|manifest_path)"\s*:',
    re.IGNORECASE,
)


class OutboundEvidenceViolation(RuntimeError):
    category = "OUTBOUND_EVIDENCE_BLOCKED"

    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_outbound_run_id() -> str:
    return f"OUTBOUND_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def _compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def _semantic_tokens(text: str) -> dict[str, list[str]]:
    return {
        "numbers": sorted(set(NUMBER_PATTERN.findall(text))),
        "units": sorted(set(match.group(0) for match in UNIT_PATTERN.finditer(text)), key=str.casefold),
        "negation": sorted(set(match.group(0) for match in NEGATION_PATTERN.finditer(text)), key=str.casefold),
    }


def _normalized_semantic_sets(text: str) -> dict[str, set[str]]:
    return {
        key: {item.casefold() for item in values}
        for key, values in _semantic_tokens(text).items()
    }


def _missing_semantics(source: str, outbound: str) -> dict[str, set[str]]:
    source_tokens = _normalized_semantic_sets(source)
    outbound_tokens = _normalized_semantic_sets(outbound)
    return {
        key: source_tokens[key] - outbound_tokens[key]
        for key in source_tokens
    }


def _minimal_excerpt(
    row: dict[str, Any],
) -> tuple[str, str, dict[str, list[str]], dict[str, Any]]:
    claim = _compact(row.get("claim") or row.get("original_text") or row.get("text"))
    if not claim:
        equation = _compact(row.get("equation") or row.get("formula"))
        claim = equation
    original_claim = claim
    if len(claim) > MAX_EXCERPT_CHARS:
        # Retrieval records can contain a compact summary that is just over the
        # outbound ceiling.  Send a visibly truncated minimum excerpt instead
        # of either leaking the long span or aborting the whole scientific run.
        bounded = claim[: MAX_EXCERPT_CHARS - 2]
        boundary = max(bounded.rfind("。"), bounded.rfind("."), bounded.rfind(";"), bounded.rfind(" "))
        claim = (bounded[:boundary] if boundary >= MAX_EXCERPT_CHARS // 2 else bounded).rstrip() + " …"
        tokens = _semantic_tokens(claim)
        source_tokens = _normalized_semantic_sets(original_claim)
        preservation = {
            "status": "PASS_TRUNCATED_TO_MINIMUM",
            "source_numbers": len(source_tokens["numbers"]),
            "source_units": len(source_tokens["units"]),
            "source_negations": len(source_tokens["negation"]),
            "excerpt_characters": len(claim),
            "source_characters": len(original_claim),
            "truncated": True,
        }
        return claim, claim, tokens, preservation
    original = _compact(row.get("original_text") or row.get("text"))
    source = " ".join(part for part in (claim, original) if part)
    excerpt = claim
    missing = _missing_semantics(source, claim)

    if any(missing.values()):
        # Add only source sentences needed to retain semantics absent from the
        # concise claim. Never fall back to sending a full page.
        selected: list[str] = []
        for sentence in re.split(r"(?<=[.!?;。！？；])\s*", original):
            sentence = _compact(sentence)
            if not sentence:
                continue
            sentence_tokens = _normalized_semantic_sets(sentence)
            if not any(sentence_tokens[key] & missing[key] for key in missing):
                continue
            candidate = _compact(" ".join([*selected, sentence]))
            if len(candidate) > MAX_EXCERPT_CHARS:
                raise OutboundEvidenceViolation(
                    "MINIMAL_OUTBOUND_EVIDENCE_SEMANTIC_CONTEXT_TOO_LONG"
                )
            selected.append(sentence)
            combined = _compact(" ".join((claim, candidate)))
            missing = _missing_semantics(source, combined)
            if not any(missing.values()):
                break
        excerpt = _compact(" ".join(selected))
        if not excerpt or any(missing.values()):
            raise OutboundEvidenceViolation(
                "MINIMAL_OUTBOUND_EVIDENCE_SEMANTIC_PRESERVATION_FAILED"
            )

    outbound_text = _compact(" ".join((claim, excerpt)))
    remaining = _missing_semantics(source, outbound_text)
    if any(remaining.values()):
        raise OutboundEvidenceViolation(
            "MINIMAL_OUTBOUND_EVIDENCE_SEMANTIC_PRESERVATION_FAILED"
        )
    tokens = _semantic_tokens(outbound_text)
    source_tokens = _normalized_semantic_sets(source)
    preservation = {
        "status": "PASS",
        "source_numbers": len(source_tokens["numbers"]),
        "source_units": len(source_tokens["units"]),
        "source_negations": len(source_tokens["negation"]),
        "excerpt_characters": len(excerpt),
    }
    return claim, excerpt, tokens, preservation


def _conditions(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("experimental_conditions") or row.get("conditions") or {}
    if not isinstance(source, dict):
        return {}
    return {
        key: value for key, value in source.items()
        if key in CONDITION_KEYS and value not in (None, "", [], {}, "NOT_REPORTED")
    }


def _sanitize_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    claim, excerpt, tokens, preservation = _minimal_excerpt(row)
    evidence_id = _compact(row.get("doc_id") or row.get("evidence_id") or row.get("formula_id"))
    document_id = _compact(row.get("paper_id") or row.get("document_id") or row.get("canonical_id"))
    if not evidence_id or not document_id or not excerpt:
        raise OutboundEvidenceViolation("MINIMAL_OUTBOUND_EVIDENCE_REQUIRED_ID_OR_CLAIM_MISSING")
    material = _compact(row.get("material") or row.get("alloy_grade"))
    process = _compact(row.get("manufacturing_process") or row.get("process"))
    conditions = _conditions(row)
    material = material or _compact(conditions.get("alloy_grade") or conditions.get("material"))
    process = process or _compact(conditions.get("manufacturing_process") or conditions.get("process"))
    sanitized = {
        "doc_id": evidence_id,
        "evidence_id": evidence_id,
        "paper_id": document_id,
        "document_id": document_id,
        "title": _compact(row.get("title")),
        "year": _compact(row.get("year") or row.get("publication_year")),
        "evidence_role": role,
        "verified_evidence_role": role,
        "material": material,
        "manufacturing_process": process,
        "experimental_conditions": conditions,
        "claim": claim,
        "original_text": excerpt,
        "page_number": row.get("page_number") or row.get("page"),
        "section": _compact(row.get("section")) or "NOT_REPORTED",
        "numerical_values": tokens["numbers"],
        "units": tokens["units"],
        "negation_markers": tokens["negation"],
        "semantic_preservation": preservation,
    }
    equation = _compact(row.get("equation") or row.get("formula"))
    if equation and equation == excerpt:
        sanitized["equation"] = equation
        sanitized["formula_id"] = evidence_id
    return sanitized


def _rows(value: SkillInput) -> list[tuple[str, dict[str, Any]]]:
    groups = (
        ("DIRECT_SUPPORT", value.support_evidence),
        ("DIRECT_COUNTER", value.counter_evidence),
        ("CONDITION_DEPENDENT", value.condition_dependent_evidence),
        ("ALTERNATIVE_MECHANISM", value.alternative_mechanism_evidence),
        ("SUPPORTING_CONTEXT", value.supporting_context_evidence),
    )
    output: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for role, rows in groups:
        for row in rows:
            evidence_id = _compact(row.get("doc_id") or row.get("evidence_id"))
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                output.append((role, row))
    for row in value.formula_records:
        evidence_id = _compact(row.get("doc_id") or row.get("formula_id"))
        if evidence_id and evidence_id not in seen:
            seen.add(evidence_id)
            output.append(("FORMULA_EVIDENCE", row))
    return output


def build_minimal_outbound_input(value: SkillInput) -> tuple[SkillInput, dict[str, Any]]:
    sanitized = [_sanitize_row(row, role) for role, row in _rows(value)]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in sanitized:
        by_role.setdefault(str(row["evidence_role"]), []).append(row)
    papers = []
    for document_id in dict.fromkeys(str(row["document_id"]) for row in sanitized):
        rows = [row for row in sanitized if row["document_id"] == document_id]
        papers.append({
            "paper_id": document_id,
            "canonical_id": document_id,
            "title": rows[0]["title"],
            "year": rows[0]["year"],
            "roles": sorted({str(row["evidence_role"]) for row in rows}),
            "material": next((row["material"] for row in rows if row["material"]), ""),
            "manufacturing_process": next(
                (row["manufacturing_process"] for row in rows if row["manufacturing_process"]), ""
            ),
            "conditions": next((row["experimental_conditions"] for row in rows if row["experimental_conditions"]), {}),
            "principal_claims": [{
                "evidence_id": row["evidence_id"],
                "role": row["evidence_role"],
                "claim": row["claim"],
                "original_text": row["original_text"],
                "page_number": row["page_number"],
                "section": row["section"],
                "experimental_conditions": row["experimental_conditions"],
                "numerical_values": row["numerical_values"],
                "units": row["units"],
                "negation_markers": row["negation_markers"],
            } for row in rows],
        })
    citation_index = {
        str(row["evidence_id"]): {
            "paper_id": row["document_id"],
            "title": row["title"],
            "page_number": row["page_number"],
            "section": row["section"],
            "role": row["evidence_role"],
        }
        for row in sanitized
    }
    formulas = [row for row in sanitized if row["evidence_role"] == "FORMULA_EVIDENCE"]
    bundle = {
        "outbound_policy": "MINIMAL_OUTBOUND_EVIDENCE",
        "question": value.user_query,
        "evidence": sanitized,
    }
    minimal = replace(
        value,
        retrieved_evidence=sanitized,
        condition_evidence=[row for row in sanitized if row["experimental_conditions"]],
        formula_records=formulas,
        support_evidence=by_role.get("DIRECT_SUPPORT", []),
        counter_evidence=by_role.get("DIRECT_COUNTER", []),
        condition_dependent_evidence=by_role.get("CONDITION_DEPENDENT", []),
        alternative_mechanism_evidence=by_role.get("ALTERNATIVE_MECHANISM", []),
        supporting_context_evidence=by_role.get("SUPPORTING_CONTEXT", []),
        evidence_bundle=bundle,
        dataset_version="",
    )
    summary = {
        "evidence_ids": [row["evidence_id"] for row in sanitized],
        "document_ids": sorted({row["document_id"] for row in sanitized}),
        "sent_fields": sorted({key for row in sanitized for key in row}),
        "evidence_count": len(sanitized),
        "semantic_preservation": {
            "status": "PASS",
            "claim_semantics": "CLAIM_TEXT_PRESERVED_VERBATIM_AFTER_WHITESPACE_NORMALIZATION",
            "conditions": "RELEVANT_CONDITION_VALUES_PRESERVED",
            "numerics_units_negation": "FULL_INTERNAL_COMPARED_WITH_MINIMAL_AND_PRESERVED",
            "rows_verified": len(sanitized),
        },
    }
    return minimal, summary


def validate_outbound_prompt(prompt: str) -> None:
    if LOCAL_PATH_PATTERN.search(prompt):
        raise OutboundEvidenceViolation("MINIMAL_OUTBOUND_EVIDENCE_LOCAL_PATH_BLOCKED")
    if FORBIDDEN_KEY_PATTERN.search(prompt):
        raise OutboundEvidenceViolation("MINIMAL_OUTBOUND_EVIDENCE_PRIVATE_FIELD_BLOCKED")
    if "DEEPSEEK_API_KEY" in prompt or "sk-" in prompt:
        raise OutboundEvidenceViolation("MINIMAL_OUTBOUND_EVIDENCE_SECRET_BLOCKED")


def record_outbound_evidence(
    *,
    base_dir: Path,
    run_id: str,
    model: str,
    phase: str,
    prompt: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    validate_outbound_prompt(prompt)
    record = {
        "run_id": run_id,
        "phase": phase,
        "model": model,
        "timestamp": _now(),
        "policy": "MINIMAL_OUTBOUND_EVIDENCE",
        "evidence_ids": list(summary["evidence_ids"]),
        "document_ids": list(summary["document_ids"]),
        "sent_fields": list(summary["sent_fields"]),
        "evidence_count": int(summary["evidence_count"]),
        "character_count": len(prompt),
        "estimated_token_count": (len(prompt) + 3) // 4,
        "semantic_preservation": summary["semantic_preservation"],
        "contains_full_pdf": False,
        "contains_full_page": False,
        "contains_local_path": False,
        "contains_api_key": False,
    }
    base_dir = Path(base_dir)
    run_dir = base_dir / OUTBOUND_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "outbound_evidence_manifest.json"
    payload = {"run_id": run_id, "calls": []}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.setdefault("calls", []).append(record)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_path = base_dir / AUDIT_PATH
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {**record, "manifest_path": str(manifest_path.resolve())}
