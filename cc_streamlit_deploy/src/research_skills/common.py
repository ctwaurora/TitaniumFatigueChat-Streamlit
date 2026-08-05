"""Evidence utilities shared by skills without sharing reasoning templates."""

from __future__ import annotations

from typing import Any

from src.research_skills.contracts import SkillInput


BANNED_PLACEHOLDERS = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应")
FORMULA_NOISE_TERMS = (
    "figure ", "fig. ", "table ", "international journal", "et al.",
    "representative ", "defect type", "batch ", "copyright",
)


def entity_labels(value: SkillInput) -> tuple[str, str]:
    entities = value.parsed_entities
    return (
        str(entities.get("independent_label") or ""),
        str(entities.get("dependent_label") or ""),
    )


def query_variables(value: SkillInput) -> tuple[list[str], list[str]]:
    frame = value.query_frame or value.evidence_bundle.get("query_frame") or {}
    independent = [str(item) for item in frame.get("independent_variables") or []]
    dependent = [str(item) for item in frame.get("dependent_variables") or []]
    iv, dv = entity_labels(value)
    if not independent and iv:
        independent.append(iv)
    if not dependent and dv:
        dependent.append(dv)
    return independent, dependent


def evidence_level_instruction() -> str:
    return (
        "重要陈述必须区分：文献直接结果用‘该研究报告’，作者机制解释用‘作者将其解释为’，"
        "跨文献综合用‘综合条件相容的研究可以判断’，系统推断用‘可以推测但尚未直接验证’，"
        "候选模型用‘待拟合候选模型’，证据不足用‘当前证据不足以确定’。"
    )


def evidence_counts(value: SkillInput) -> tuple[int, int, int]:
    return (
        len(value.support_evidence),
        len(value.counter_evidence),
        len(value.condition_dependent_evidence),
    )


def is_usable_formula_record(row: dict[str, Any]) -> bool:
    """Return whether an extracted record is compact enough to quote verbatim."""
    equation = str(row.get("equation") or row.get("formula") or "").strip()
    lowered = equation.casefold()
    if not equation or len(equation) > 220:
        return False
    if any(term in lowered for term in FORMULA_NOISE_TERMS):
        return False
    if not any(operator in equation for operator in ("=", "≈", "∝", "≤", "≥")):
        return False
    return len(equation.split()) <= 24


def usable_formulas(value: SkillInput) -> list[dict[str, Any]]:
    """Return equation-like records that are safe to quote as formulas."""
    return [
        row
        for row in value.formula_records or (value.evidence_bundle or {}).get("formulas") or []
        if is_usable_formula_record(row)
    ]


def is_noisy_evidence_excerpt(value: Any) -> bool:
    """Identify broken captions/page furniture that should not reach evidence cards."""
    text = str(value or "").strip()
    lowered = text.casefold()
    control_character = any(ord(char) < 32 and char not in "\n\r\t" for char in text)
    noise_hits = sum(term in lowered for term in FORMULA_NOISE_TERMS)
    return not text or control_character or noise_hits >= 2


def bundle_prompt(value: SkillInput) -> str:
    """Return the compact EvidenceBundle supplied to a concrete Skill."""
    from src.evidence_compression import evidence_prompt_json

    return evidence_prompt_json(value.evidence_bundle)


def primary_citation(value: SkillInput, role: str = "SUPPORT") -> str:
    citation_index = (value.evidence_bundle or {}).get("citation_index") or {}
    for paper in value.evidence_bundle.get("papers") or []:
        for claim in paper.get("principal_claims") or []:
            if claim.get("role") != role:
                continue
            if citation_index and str(claim.get("evidence_id")) not in citation_index:
                continue
            return f"[Evidence ID：{claim.get('evidence_id')}，页码：{claim.get('page_number')}]"
    rows = value.support_evidence if role == "SUPPORT" else value.counter_evidence
    for row in rows:
        evidence_id = str(row.get("doc_id") or row.get("evidence_id") or "")
        if citation_index and evidence_id not in citation_index:
            continue
        return f"[Evidence ID：{evidence_id}，页码：{row.get('page_number')}]"
    return ""


def traceable_cards(value: SkillInput, limit: int = 30) -> list[dict[str, Any]]:
    cards = []
    for role, rows in (
        ("SUPPORT", value.support_evidence),
        ("COUNTER", value.counter_evidence),
        ("CONDITION_DEPENDENT", value.condition_dependent_evidence),
    ):
        for row in rows:
            if is_noisy_evidence_excerpt(row.get("original_text")):
                continue
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
