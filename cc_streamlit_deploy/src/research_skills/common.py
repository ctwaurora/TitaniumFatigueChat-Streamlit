"""Evidence utilities shared by skills without sharing reasoning templates."""

from __future__ import annotations

from typing import Any

from src.research_skills.contracts import SkillInput


BANNED_PLACEHOLDERS = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应")


def entity_labels(value: SkillInput) -> tuple[str, str]:
    entities = value.parsed_entities
    return (
        str(entities.get("independent_label") or ""),
        str(entities.get("dependent_label") or ""),
    )


def evidence_counts(value: SkillInput) -> tuple[int, int, int]:
    return (
        len(value.support_evidence),
        len(value.counter_evidence),
        len(value.condition_dependent_evidence),
    )


def bundle_prompt(value: SkillInput) -> str:
    """Return the compact EvidenceBundle supplied to a concrete Skill."""
    import json

    return json.dumps(value.evidence_bundle, ensure_ascii=False, separators=(",", ":"))


def primary_citation(value: SkillInput, role: str = "SUPPORT") -> str:
    for paper in value.evidence_bundle.get("papers") or []:
        for claim in paper.get("principal_claims") or []:
            if claim.get("role") != role:
                continue
            return f"[Evidence ID：{claim.get('evidence_id')}，页码：{claim.get('page_number')}]"
    rows = value.support_evidence if role == "SUPPORT" else value.counter_evidence
    if rows:
        row = rows[0]
        return f"[Evidence ID：{row.get('doc_id') or row.get('evidence_id')}，页码：{row.get('page_number')}]"
    return ""


def traceable_cards(value: SkillInput, limit: int = 30) -> list[dict[str, Any]]:
    cards = []
    for role, rows in (
        ("SUPPORT", value.support_evidence),
        ("COUNTER", value.counter_evidence),
        ("CONDITION_DEPENDENT", value.condition_dependent_evidence),
    ):
        for row in rows:
            cards.append({
                "role": role,
                "title": row.get("title") or "题名未报告",
                "authors": row.get("authors") or "未报告",
                "year": row.get("year") or "未报告",
                "original_text": row.get("original_text") or "",
                "page_number": row.get("page_number") or "未报告",
                "section": row.get("section") or "未报告",
                "evidence_id": row.get("doc_id") or row.get("evidence_id") or "未报告",
                "experimental_conditions": row.get("experimental_conditions") or {},
            })
    return cards[:limit]


def missing_evidence(value: SkillInput) -> list[str]:
    missing = []
    if not value.support_evidence:
        missing.append("缺少直接支持证据")
    if not value.counter_evidence:
        missing.append("未召回明确反向证据")
    if not value.condition_dependent_evidence:
        missing.append("缺少条件依赖证据")
    if not value.condition_evidence:
        missing.append("缺少可用实验条件记录")
    if not value.formula_records:
        missing.append("未召回可追溯原文公式")
    return missing


def base_quality_gate(
    value: SkillInput,
    text: str,
    *,
    skill_name: str,
    required_terms: tuple[str, ...] = (),
) -> dict[str, Any]:
    iv, dv = entity_labels(value)
    banned = [term for term in BANNED_PLACEHOLDERS if term in text]
    missing_terms = [term for term in required_terms if term not in text]
    missing_entities = [label for label in (iv, dv) if label and label.split("（", 1)[0] not in text]
    traceable = all(
        card.get("title") and card.get("page_number") and card.get("evidence_id")
        for card in traceable_cards(value)
    )
    return {
        "passed": bool(value.parsed_entities.get("specific")) and not banned and not missing_terms and not missing_entities and traceable,
        "skill_name": skill_name,
        "specific_entities": bool(value.parsed_entities.get("specific")),
        "banned_terms": banned,
        "missing_terms": missing_terms,
        "missing_entities": missing_entities,
        "citation_traceability": traceable,
        "evidence_count": len(value.retrieved_evidence),
    }


def formula_lines(value: SkillInput, limit: int = 3) -> list[str]:
    lines = []
    seen = set()
    for row in value.formula_records:
        equation = str(row.get("equation") or row.get("formula") or "").strip()
        if not equation or equation in seen:
            continue
        seen.add(equation)
        lines.append(
            f"- `{equation}`；来源：{row.get('title') or '题名未报告'}，"
            f"p.{row.get('page_number') or '未报告'}，Evidence ID：{row.get('doc_id') or '未报告'}。"
        )
        if len(lines) >= limit:
            break
    return lines


def evidence_prompt_block(value: SkillInput, limit: int = 14) -> str:
    lines = []
    for card in traceable_cards(value, limit=limit):
        lines.append(
            f"[{card['role']} | {card['evidence_id']} | {card['title']} | "
            f"p.{card['page_number']} | {card['section']}] {card['original_text']}"
        )
    return "\n".join(lines)
