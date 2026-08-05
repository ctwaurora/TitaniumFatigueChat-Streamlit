"""Query-specific, provenance-preserving compression for model prompt context."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from typing import Any

from src.feature_flags import feature_enabled


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+./-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_ROLE_ORDER = ("SUPPORT", "COUNTER", "CONDITION_DEPENDENT")


def _tokens(text: Any) -> set[str]:
    values = set()
    for token in _TOKEN.findall(str(text or "").casefold()):
        values.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            values.update(token[index:index + 2] for index in range(len(token) - 1))
    return values


def _claim_score(question_tokens: set[str], claim: dict[str, Any]) -> float:
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("claim", "section", "directness", "role")
    )
    claim_tokens = _tokens(text)
    overlap = len(question_tokens & claim_tokens) / max(1, len(question_tokens))
    direct_bonus = 0.15 if str(claim.get("directness") or "").upper() == "DIRECT" else 0.0
    trace_bonus = 0.10 if claim.get("evidence_id") and claim.get("page_number") else 0.0
    role_bonus = 0.05 if claim.get("role") in _ROLE_ORDER else 0.0
    return round(overlap + direct_bonus + trace_bonus + role_bonus, 6)


def _compact_bundle(bundle: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    question = str(bundle.get("question") or "")
    question_tokens = _tokens(question)
    candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    for paper in bundle.get("papers") or []:
        for claim in paper.get("principal_claims") or []:
            score = _claim_score(question_tokens, claim)
            candidates.append((score, str(claim.get("role") or ""), paper, claim))

    selected: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for role in _ROLE_ORDER:
        rows = sorted((item for item in candidates if item[1] == role), reverse=True, key=lambda item: item[0])
        if rows:
            selected.append(rows[0])
            selected_ids.add(str(rows[0][3].get("evidence_id") or ""))
    for item in sorted(candidates, reverse=True, key=lambda row: row[0]):
        evidence_id = str(item[3].get("evidence_id") or "")
        if evidence_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(evidence_id)
        if len(selected) >= 18:
            break

    papers: dict[str, dict[str, Any]] = {}
    for score, _, paper, claim in selected:
        paper_id = str(paper.get("paper_id") or paper.get("canonical_id") or "")
        target = papers.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "year": paper.get("year"),
                "conditions": paper.get("conditions") or {},
                "mechanisms": (paper.get("mechanisms") or [])[:3],
                "principal_claims": [],
            },
        )
        compact_claim = copy.deepcopy(claim)
        compact_claim["query_relevance_score"] = score
        compact_claim["query_specific_summary"] = str(claim.get("claim") or "")[:420]
        target["principal_claims"].append(compact_claim)

    citation_index = bundle.get("citation_index") or {}
    compact = {
        "question": question,
        "topics": bundle.get("topics") or [],
        "query_frame": bundle.get("query_frame") or {},
        "dataset_version": bundle.get("dataset_version") or "",
        "papers": list(papers.values()),
        "formulas": (bundle.get("formulas") or [])[:6],
        "synthesis": bundle.get("synthesis") or {},
        "citation_index": {
            evidence_id: citation_index[evidence_id]
            for evidence_id in selected_ids
            if evidence_id in citation_index
        },
        "compression": {
            "enabled": True,
            "strategy": "query_overlap_with_role_preservation",
            "source_claim_count": len(candidates),
            "selected_claim_count": len(selected),
            "roles_preserved": [role for role in _ROLE_ORDER if any(item[1] == role for item in selected)],
            "full_evidence_retained_outside_prompt": True,
        },
    }

    # Reduce complete paper groups, never bytes from serialized JSON. The result
    # therefore remains parseable and every retained claim keeps its provenance.
    while len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) > max_chars and len(compact["papers"]) > 3:
        removed = compact["papers"].pop()
        removed_ids = {
            str(claim.get("evidence_id") or "")
            for claim in removed.get("principal_claims") or []
        }
        for evidence_id in removed_ids:
            compact["citation_index"].pop(evidence_id, None)
        compact["compression"]["selected_claim_count"] -= len(removed_ids)
    if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) > max_chars:
        synthesis = compact.get("synthesis") or {}
        compact["synthesis"] = {
            key: value[:4] if isinstance(value, list) else {
                nested_key: nested_value[:4] if isinstance(nested_value, list) else nested_value
                for nested_key, nested_value in value.items()
            } if isinstance(value, dict) else value
            for key, value in synthesis.items()
            if key in {
                "consensus", "conflicts", "condition_matches", "condition_mismatches",
                "covered_conditions", "missing_conditions", "supported_conclusions",
                "unsupported_conclusions", "formula_comparability", "evidence_limitations",
            }
        }
        for paper in compact["papers"]:
            paper["mechanisms"] = paper.get("mechanisms", [])[:2]
            paper["principal_claims"] = paper.get("principal_claims", [])[:2]
            for claim in paper["principal_claims"]:
                claim["claim"] = str(claim.get("claim") or "")[:360]
                claim["query_specific_summary"] = str(claim.get("query_specific_summary") or "")[:180]
    if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) > max_chars:
        # Final structured fallback keeps one claim for every evidence role and
        # the exact citation tuple. It removes optional prose fields, never JSON
        # bytes or provenance fields.
        minimal_papers = []
        role_seen: set[str] = set()
        for paper in compact["papers"]:
            for claim in paper.get("principal_claims") or []:
                role = str(claim.get("role") or "")
                if role in role_seen:
                    continue
                role_seen.add(role)
                minimal_papers.append({
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "conditions": dict(list((paper.get("conditions") or {}).items())[:4]),
                    "principal_claims": [{
                        "role": role,
                        "claim": str(claim.get("claim") or "")[:195],
                        "query_specific_summary": str(claim.get("query_specific_summary") or "")[:90],
                        "evidence_id": claim.get("evidence_id"),
                        "page_number": claim.get("page_number"),
                        "section": claim.get("section"),
                        "directness": claim.get("directness"),
                        "query_relevance_score": claim.get("query_relevance_score"),
                    }],
                })
        compact["papers"] = minimal_papers
        compact["formulas"] = (compact.get("formulas") or [])[:2]
        compact["synthesis"] = {
            key: value[:2] if isinstance(value, list) else value
            for key, value in (compact.get("synthesis") or {}).items()
            if key in {"consensus", "conflicts", "condition_mismatches", "missing_conditions", "formula_comparability"}
        }
        compact["compression"]["selected_claim_count"] = len(minimal_papers)
        compact["citation_index"] = {
            evidence_id: {
                "page_number": value.get("page_number"),
                "section": value.get("section"),
            }
            for evidence_id, value in (compact.get("citation_index") or {}).items()
            if evidence_id in {
                str(claim.get("evidence_id") or "")
                for paper in minimal_papers
                for claim in paper.get("principal_claims") or []
            }
        }
    return compact


@lru_cache(maxsize=128)
def _cached_compact(serialized_bundle: str, max_chars: int) -> str:
    bundle = json.loads(serialized_bundle)
    compact = _compact_bundle(bundle, max_chars=max_chars)
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def evidence_prompt_json(bundle: dict[str, Any], *, max_chars: int = 18000) -> str:
    """Return model context; the full bundle remains untouched for verification/UI."""
    serialized = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not feature_enabled("TFC_QUERY_EVIDENCE_COMPRESSION", default=True):
        return serialized
    return _cached_compact(serialized, max_chars)


def clear_evidence_compression_cache() -> None:
    _cached_compact.cache_clear()
