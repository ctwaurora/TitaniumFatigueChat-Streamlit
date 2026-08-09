"""Claim-aware evidence-role verification for scientific answers.

The verifier is deterministic and deliberately conservative.  It separates
topic relevance from support for a complete scientific claim, so a paper
about short-crack importance cannot silently support a residual-stress to
Delta-K-effective causal chain.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DIRECT_SUPPORT = "DIRECT_SUPPORT"
SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"
CONDITION_DEPENDENT = "CONDITION_DEPENDENT"
ALTERNATIVE_MECHANISM = "ALTERNATIVE_MECHANISM"
DIRECT_COUNTER = "DIRECT_COUNTER"
LIMITATION_EVIDENCE = "LIMITATION_EVIDENCE"
REVIEW_BACKGROUND = "REVIEW_BACKGROUND"
INSUFFICIENT = "INSUFFICIENT"

EVIDENCE_ROLES = (
    DIRECT_SUPPORT,
    SUPPORTING_CONTEXT,
    CONDITION_DEPENDENT,
    ALTERNATIVE_MECHANISM,
    DIRECT_COUNTER,
    LIMITATION_EVIDENCE,
    REVIEW_BACKGROUND,
    INSUFFICIENT,
)

_CONCEPTS: dict[str, tuple[str, ...]] = {
    "residual_stress": ("residual stress", "residual-stress", "残余应力", "sigmares", "σres"),
    "crack_closure": ("crack closure", "opening load", "closure load", "裂纹闭合", "开闭载荷", "kop"),
    "delta_keff": ("deltakeff", "delta keff", "effective stress intensity", "有效应力强度", "Δkeff"),
    "stress_relaxation": ("stress relaxation", "residual stress relaxation", "应力松弛", "循环松弛"),
    "microstructure": ("microstructure", "microstructural", "显微组织", "微观组织"),
    "alpha_lath": ("alpha lath", "alpha-lath", "α片层", "片层宽度", "lα"),
    "alpha_prime": ("martensite", "martensitic", "α′", "alpha prime"),
    "prior_beta": ("prior beta", "prior-β", "先前β", "原始β"),
    "texture": ("texture", "crystallographic", "basal plane", "织构", "晶体取向"),
    "barrier": ("barrier", "grain boundary", "phase boundary", "crack deflection", "组织屏障", "晶界", "相界", "裂纹偏转"),
    "ebsd": ("ebsd", "electron backscatter", "电子背散射"),
    "short_crack": ("short crack", "small crack", "microstructurally short", "短裂纹", "小裂纹"),
    "long_crack": ("long crack", "long-crack", "长裂纹"),
    "transition": ("transition", "transfer", "转变", "过渡区", "transition region"),
    "crack_length": ("crack length", "裂纹长度", "initial crack", "a="),
    "da_dn": ("da/dn", "da d n", "crack growth rate", "crack propagation rate", "裂纹扩展速率"),
    "delta_k": ("delta k", "stress intensity factor range", "应力强度因子范围", "Δk"),
    "stress_ratio": ("stress ratio", "应力比", "r="),
    "surface": ("surface condition", "surface roughness", "machined", "polished", "表面状态", "表面粗糙度", "机加工", "抛光"),
    "heat_treatment": ("heat treatment", "stress relieved", "annealed", "热处理", "去应力退火"),
    "l_pbf": ("l-pbf", "lpbf", "slm", "laser powder bed fusion", "选区激光熔化"),
    "ti64": ("ti-6al-4v", "ti6al4v", "tc4"),
    "fatigue_life": ("fatigue life", "fatigue limit", "疲劳寿命", "疲劳极限", "hcf", "vhcf"),
    "hip": ("hot isostatic pressing", "hot-isostatic pressing", "hipping", "hip treatment", "热等静压"),
    "pore": ("pore", "porosity", "void", "孔隙", "孔洞"),
    "defect_size": ("defect size", "pore size", "sqrt area", "√area", "缺陷尺寸", "孔隙尺寸"),
    "defect_location": ("defect location", "distance from surface", "near-surface", "near surface", "internal defect", "surface defect", "缺陷位置", "距表面", "近表面", "内部缺陷", "表面缺陷"),
    "crack_initiation": ("crack initiation", "crack origin", "裂纹起裂", "裂纹萌生", "起裂位置"),
}

_GENERIC_CONCEPTS = {"ti64", "l_pbf", "surface", "heat_treatment", "stress_ratio", "delta_k"}
_COUNTER = re.compile(
    r"no significant|not significant|did not (?:affect|change|improve)|no (?:measurable )?effect|"
    r"effect (?:disappeared|vanished)|opposite direction|fully explained by|"
    r"无显著|没有显著|未改变|效应消失|方向相反|完全解释",
    re.I,
)
_MECHANISM = re.compile(
    r"mechanis|attribut|because|due to|dominat|slip|texture|microstruct|closure|roughness|pore|"
    r"机制|归因|由于|主导|滑移|织构|组织|闭合|粗糙度|孔隙",
    re.I,
)
_LIMITATION = re.compile(r"\blimit(?:ation|ed|s)?\b|uncertain|insufficient|cannot be generalized|not reported|局限|不足|未报告", re.I)
_REVIEW = re.compile(r"\breview\b|systematic review|meta-analysis|综述", re.I)


def scientific_concepts(text: Any) -> set[str]:
    normalized = " ".join(str(text or "").casefold().replace("−", "-").split())
    return {
        name
        for name, aliases in _CONCEPTS.items()
        if any(alias.casefold() in normalized for alias in aliases)
    }


def _reported(value: Any) -> bool:
    return value not in (None, "", [], {}, "NOT_REPORTED", "未报告")


def condition_match_score(
    query_frame: dict[str, Any] | None,
    evidence_conditions: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    frame = query_frame or {}
    conditions = evidence_conditions or {}
    pairs = {
        "alloy_grade": frame.get("alloy_grade") or frame.get("material"),
        "manufacturing_process": frame.get("manufacturing_process"),
        "stress_ratio_R": frame.get("stress_ratio"),
        "temperature": frame.get("temperature"),
        "environment": frame.get("environment"),
        "loading_mode": frame.get("loading_mode"),
        "heat_treatment": " ".join(frame.get("post_processing") or []),
        "surface_treatment": " ".join(frame.get("surface_condition") or []),
        "fatigue_regime": frame.get("fatigue_stage"),
        "crack_stage": frame.get("crack_stage"),
    }
    checked = matched = 0
    conflicts: list[str] = []
    for key, expected in pairs.items():
        if not _reported(expected):
            continue
        actual = conditions.get(key)
        if key == "manufacturing_process" and not _reported(actual):
            actual = conditions.get("process")
        if not _reported(actual):
            continue
        checked += 1
        expected_text = str(expected).casefold()
        actual_text = " ".join(map(str, actual)).casefold() if isinstance(actual, (list, tuple, set)) else str(actual).casefold()
        if expected_text in actual_text or actual_text in expected_text or ({"l-pbf", "slm"} & {expected_text, actual_text}):
            matched += 1
        else:
            conflicts.append(key)
    if not checked:
        return 0.5, conflicts
    return matched / checked, conflicts


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_id: str
    document_id: str
    page_number: Any
    supporting_span: str
    condition_provenance: dict[str, Any]
    role: str
    directness_score: float
    condition_match_score: float
    concept_coverage: float
    matched_concepts: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    condition_conflicts: tuple[str, ...]
    allow_direct_conclusion: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_evidence_for_claim(
    claim_text: str,
    evidence: dict[str, Any],
    *,
    query_frame: dict[str, Any] | None = None,
) -> EvidenceAssessment:
    claim_terms = scientific_concepts(claim_text)
    evidence_text = " ".join(
        str(evidence.get(key) or "")
        for key in ("claim", "original_text", "text")
    )
    evidence_terms = scientific_concepts(evidence_text)
    required = claim_terms - _GENERIC_CONCEPTS
    if not required:
        required = claim_terms
    matched = required & evidence_terms
    coverage = len(matched) / max(1, len(required))
    directness = str(evidence.get("directness") or "").upper()
    directness_score = {"DIRECT": 1.0, "INDIRECT": 0.55, "MENTION_ONLY": 0.2}.get(directness, 0.35)
    condition_score, conflicts = condition_match_score(
        query_frame, evidence.get("experimental_conditions") or evidence.get("conditions")
    )
    explicit_counter = bool(_COUNTER.search(evidence_text))
    mechanism_only = bool(_MECHANISM.search(evidence_text)) and coverage < 0.67
    is_review = bool(_REVIEW.search(" ".join((str(evidence.get("title") or ""), str(evidence.get("section") or "")))))

    role = INSUFFICIENT
    reason = "证据与主张的核心科学概念覆盖不足。"
    if is_review and coverage >= 0.20:
        role = REVIEW_BACKGROUND
        reason = "Review evidence is retained as background and cannot displace matched primary evidence."
    elif _LIMITATION.search(evidence_text) and coverage >= 0.35:
        role = LIMITATION_EVIDENCE
        reason = "The source directly reports an applicability limit or unresolved condition."
    elif conflicts and coverage >= 0.45:
        role = CONDITION_DEPENDENT
        reason = "核心概念相关，但材料、工艺或载荷条件不完全匹配。"
    elif explicit_counter and directness_score >= 0.95 and coverage >= 0.67 and condition_score >= 0.65:
        role = DIRECT_COUNTER
        reason = "可比条件下原文明确报告无效应、效应消失或相反方向。"
    elif explicit_counter and coverage >= 0.45:
        role = CONDITION_DEPENDENT
        reason = "原文含否定信号，但直接性或条件可比性不足，不能列为真正反证。"
    elif directness_score >= 0.95 and coverage >= 0.75 and condition_score >= 0.5:
        role = DIRECT_SUPPORT
        reason = "原文直接覆盖主张的核心变量和结果。"
    elif mechanism_only:
        role = ALTERNATIVE_MECHANISM
        reason = "原文提供相关或竞争机制，但没有直接验证完整主张。"
    elif coverage >= 0.35:
        role = SUPPORTING_CONTEXT
        reason = "原文只支持背景或部分关系，不能支持完整因果链。"

    allow = role == DIRECT_SUPPORT
    return EvidenceAssessment(
        evidence_id=str(evidence.get("doc_id") or evidence.get("evidence_id") or ""),
        document_id=str(evidence.get("paper_id") or evidence.get("document_id") or ""),
        page_number=evidence.get("page_number") or evidence.get("page"),
        supporting_span=str(evidence.get("original_text") or evidence.get("claim") or evidence.get("text") or ""),
        condition_provenance={
            key: (evidence.get("experimental_conditions") or evidence.get("conditions") or {}).get(key)
            for key in (
                "alloy_grade", "manufacturing_process", "surface_treatment",
                "heat_treatment", "fatigue_regime", "crack_stage", "loading_mode",
                "stress_ratio_R", "defect_size", "defect_location",
            )
        },
        role=role,
        directness_score=directness_score,
        condition_match_score=round(condition_score, 4),
        concept_coverage=round(coverage, 4),
        matched_concepts=tuple(sorted(matched)),
        missing_concepts=tuple(sorted(required - evidence_terms)),
        condition_conflicts=tuple(conflicts),
        allow_direct_conclusion=allow,
        reason=reason,
    )


def verify_claim_evidence(
    claim_text: str,
    evidence_rows: Iterable[dict[str, Any]],
    *,
    query_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessments = [
        classify_evidence_for_claim(claim_text, row, query_frame=query_frame)
        for row in evidence_rows
    ]
    role_rank = {
        DIRECT_SUPPORT: 5,
        DIRECT_COUNTER: 4,
        LIMITATION_EVIDENCE: 3,
        CONDITION_DEPENDENT: 3,
        ALTERNATIVE_MECHANISM: 2,
        SUPPORTING_CONTEXT: 1,
        REVIEW_BACKGROUND: 1,
        INSUFFICIENT: 0,
    }
    best = max(assessments, key=lambda item: (role_rank[item.role], item.concept_coverage), default=None)
    return {
        "claim_text": claim_text,
        "best_role": best.role if best else INSUFFICIENT,
        "allow_direct_conclusion": bool(best and best.allow_direct_conclusion),
        "assessments": [item.as_dict() for item in assessments],
    }


def batch_verify_claims(
    claims: Iterable[dict[str, Any]],
    evidence_rows: Iterable[dict[str, Any]],
    *,
    query_frame: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = list(evidence_rows)
    output = []
    for claim in claims:
        audit = verify_claim_evidence(
            str(claim.get("claim_text") or claim.get("text") or ""),
            rows,
            query_frame=query_frame,
        )
        output.append({**claim, **audit})
    return output
