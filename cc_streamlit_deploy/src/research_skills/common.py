"""Evidence utilities shared by skills without sharing reasoning templates."""

from __future__ import annotations

from typing import Any

from src.research_skills.contracts import SkillInput


BANNED_PLACEHOLDERS = ("目标因素", "问题中的目标因素", "某个因素", "疲劳响应")
FORMULA_NOISE_TERMS = (
    "figure ", "fig. ", "table ", "international journal", "et al.",
    "representative ", "defect type", "batch ", "copyright",
)
ENTITY_LABEL_ALIASES = {
    "短裂纹扩展速率": ("短裂纹扩展速率", "短裂纹da/dN", "短裂纹", "da/dN"),
    "裂纹扩展速率": ("裂纹扩展速率", "da/dN"),
    "疲劳寿命": ("疲劳寿命", "Nf"),
    "残余应力": ("残余应力", "σres"),
    "微观组织": ("微观组织", "显微组织", "α片层", "织构"),
}


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
    status = str(row.get("evidence_status") or row.get("validation_status") or "").upper()
    if status and status not in {"已确认", "CONFIRMED", "VERIFIED", "HUMAN_CONFIRMED"}:
        return False
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
    role_aliases = {
        "SUPPORT": {"SUPPORT", "DIRECT_SUPPORT"},
        "COUNTER": {"COUNTER", "DIRECT_COUNTER"},
        "ALTERNATIVE_MECHANISM": {"ALTERNATIVE_MECHANISM"},
        "CONDITION_DEPENDENT": {"CONDITION_DEPENDENT"},
    }
    accepted_roles = role_aliases.get(role, {role})
    citation_index = (value.evidence_bundle or {}).get("citation_index") or {}
    for paper in value.evidence_bundle.get("papers") or []:
        for claim in paper.get("principal_claims") or []:
            if claim.get("role") not in accepted_roles:
                continue
            if citation_index and str(claim.get("evidence_id")) not in citation_index:
                continue
            return f"[Evidence ID：{claim.get('evidence_id')}，页码：{claim.get('page_number')}]"
    rows_by_role = {
        "SUPPORT": value.support_evidence,
        "COUNTER": value.counter_evidence,
        "ALTERNATIVE_MECHANISM": value.alternative_mechanism_evidence,
        "CONDITION_DEPENDENT": value.condition_dependent_evidence,
        "SUPPORTING_CONTEXT": value.supporting_context_evidence,
    }
    rows = rows_by_role.get(role, [])
    for row in rows:
        evidence_id = str(row.get("doc_id") or row.get("evidence_id") or "")
        if citation_index and evidence_id not in citation_index:
            continue
        return f"[Evidence ID：{evidence_id}，页码：{row.get('page_number')}]"
    return ""


def facet_evidence(value: SkillInput, subtask: str) -> list[dict[str, Any]]:
    rows = [
        *value.support_evidence,
        *value.counter_evidence,
        *value.condition_dependent_evidence,
        *value.alternative_mechanism_evidence,
        *value.supporting_context_evidence,
    ]
    return [row for row in rows if row.get("retrieval_subtask") == subtask]


def facet_citation(value: SkillInput, subtask: str, *, direct_only: bool = True) -> str:
    rows = facet_evidence(value, subtask)
    if direct_only:
        rows = [row for row in rows if row.get("verified_evidence_role") == "DIRECT_SUPPORT"]
    for row in rows:
        evidence_id = str(row.get("doc_id") or row.get("evidence_id") or "")
        page = row.get("page_number")
        if evidence_id and page:
            return f"[Evidence ID：{evidence_id}，页码：{page}]"
    return ""


def has_direct_interaction_evidence(value: SkillInput) -> bool:
    return any(
        row.get("retrieval_subtask") == "residual_stress_microstructure_interaction_query"
        and row.get("verified_evidence_role") == "DIRECT_SUPPORT"
        for row in value.support_evidence
    )


def joint_short_crack_candidate_models() -> str:
    return """以下均为系统提出、需要预注册和拟合的候选模型，并非文献原公式。

M0（基础驱动力基线）：`log(da/dN)=β0+β1log(ΔKeff)+u_specimen`

M1（残余应力模型）：`log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3log(a)+β4R+u_specimen`

M2（微观组织模型）：`log(da/dN)=β0+β1log(ΔKeff)+β2lα+β3T+β4log(a)+u_specimen`

M3（加性联合模型）：`log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3lα+β4T+β5log(a)+β6R+u_specimen`

M4（交互模型）：`log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3lα+β4T+β5σres·lα+β6σres·T+β7log(a)+β8R+u_specimen`

变量定义：da/dN为裂纹扩展速率（m/cycle）；ΔKeff为有效应力强度因子范围（MPa√m）；σres为裂纹路径附近残余应力（MPa，拉应力为正）；lα为α片层宽度（µm）；T为预注册的织构或裂纹相对取向无量纲指标；a为裂纹长度（m或mm，全程统一）；R为应力比；u_specimen为试样随机效应。"""


def traceable_cards(value: SkillInput, limit: int = 30) -> list[dict[str, Any]]:
    cards = []
    for role, rows in (
        ("DIRECT_SUPPORT", value.support_evidence),
        ("DIRECT_COUNTER", value.counter_evidence),
        ("CONDITION_DEPENDENT", value.condition_dependent_evidence),
        ("ALTERNATIVE_MECHANISM", value.alternative_mechanism_evidence),
        ("SUPPORTING_CONTEXT", value.supporting_context_evidence),
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
    if not value.alternative_mechanism_evidence:
        missing.append("未召回可核验的替代机制证据")
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
    missing_entities = []
    for label in (iv, dv):
        base_label = label.split("（", 1)[0]
        aliases = ENTITY_LABEL_ALIASES.get(base_label, (base_label,))
        if label and not any(alias in text for alias in aliases):
            missing_entities.append(label)
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
