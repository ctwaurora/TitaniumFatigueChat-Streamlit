"""Skill-aware scientific reliability metrics.

The evaluator deliberately avoids a single global grounded-claim score.  It
separates factual assertions, evidence-informed inference, and novel proposals,
then applies the metric family appropriate to each research skill.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


EVIDENCE_GROUNDED_FACTUAL_CLAIM = "EVIDENCE_GROUNDED_FACTUAL_CLAIM"
EVIDENCE_INFORMED_INFERENCE = "EVIDENCE_INFORMED_INFERENCE"
NOVEL_SCIENTIFIC_PROPOSAL = "NOVEL_SCIENTIFIC_PROPOSAL"

CLAIM_CATEGORIES = (
    EVIDENCE_GROUNDED_FACTUAL_CLAIM,
    EVIDENCE_INFORMED_INFERENCE,
    NOVEL_SCIENTIFIC_PROPOSAL,
)

_PROPOSAL = re.compile(
    r"假设|候选机制|建议|实验设计|验证方案|预注册|判废|证伪|falsif|hypothes|"
    r"we propose|provisional|testable proposal",
    re.I,
)
_INFERENCE = re.compile(
    r"推断|推测|可能|条件化|综合判断|替代机制|尚未直接验证|inference|suggests|may|could",
    re.I,
)
_FALSIFIABLE = re.compile(
    r"证伪|否证|判废|失败标准|若.*则|如果.*则|支持标准|拒绝.*假设|falsif|reject|"
    r"failure criterion|acceptance criterion|if .* then",
    re.I | re.S,
)
_FEASIBILITY = re.compile(
    r"试样|样本|重复|测量|表征|加载|疲劳试验|统计|功效|specimen|sample size|replicat|"
    r"measure|characteri[sz]|fatigue test|statistical",
    re.I,
)
_CONFOUNDER = re.compile(
    r"混杂|对照组|随机|盲法|分层|协变量|保持.*不变|控制变量|confound|control group|"
    r"randomi[sz]|covariate|stratif",
    re.I,
)
_NUMBER_WITH_UNIT = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:MPa|GPa|kPa|Hz|mm|μm|um|nm|°C|K|%|cycles?)",
    re.I,
)


def classify_claim(
    text: str,
    *,
    system_inference: bool = False,
    method_or_boundary: bool = False,
) -> str:
    """Classify what kind of scientific statement the system is making."""
    value = str(text or "")
    if _PROPOSAL.search(value):
        return NOVEL_SCIENTIFIC_PROPOSAL
    if system_inference or method_or_boundary or _INFERENCE.search(value):
        return EVIDENCE_INFORMED_INFERENCE
    return EVIDENCE_GROUNDED_FACTUAL_CLAIM


def _rate(numerator: float, denominator: float, *, empty: float = 1.0) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else empty


def _records(claim_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in claim_audit.get("claims") or []]
    for row in rows:
        row.setdefault(
            "claim_category",
            classify_claim(
                str(row.get("claim_text") or ""),
                system_inference=bool(row.get("system_inference")),
                method_or_boundary=bool(row.get("method_or_boundary")),
            ),
        )
    return rows


def _factual_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    factual = [row for row in rows if row["claim_category"] == EVIDENCE_GROUNDED_FACTUAL_CLAIM]
    supported = [row for row in factual if row.get("status") == "SUPPORTED"]
    aligned = [row for row in factual if row.get("alignment_status") in {"PASS", "CONDITIONAL"}]
    unsupported = [row for row in factual if row.get("status") in {"UNSUPPORTED", "MISALIGNED_EVIDENCE"}]
    condition_scores: list[float] = []
    for row in factual:
        assessments = (row.get("semantic_evidence_audit") or {}).get("assessments") or []
        if assessments:
            condition_scores.append(max(float(item.get("condition_match_score") or 0) for item in assessments))
    return {
        "factual_claim_count": len(factual),
        "factual_grounding": _rate(len(supported), len(factual)),
        "claim_evidence_alignment": _rate(len(aligned), len(factual)),
        "condition_match": round(sum(condition_scores) / len(condition_scores), 4) if condition_scores else 1.0,
        "unsupported_factual_claim_rate": _rate(len(unsupported), len(factual), empty=0.0),
    }


def _category_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    output = {category: 0 for category in CLAIM_CATEGORIES}
    for row in rows:
        output[str(row.get("claim_category"))] = output.get(str(row.get("claim_category")), 0) + 1
    return output


def evaluate_skill_output(
    module: str,
    *,
    answer: str,
    claim_audit: dict[str, Any],
    evidence_bundle: dict[str, Any] | None = None,
    skill_output: dict[str, Any] | None = None,
    generation_state: str = "",
) -> dict[str, Any]:
    """Return one skill-specific metric family, never a global aggregate."""
    rows = _records(claim_audit)
    factual = _factual_metrics(rows)
    category_counts = _category_counts(rows)
    bundle = evidence_bundle or {}
    output = skill_output or {}
    text = "\n".join(
        str(value or "")
        for value in (
            answer,
            output.get("direct_answer"),
            output.get("structured_reasoning"),
            output.get("uncertainty"),
            " ".join(str(item) for item in output.get("missing_evidence") or []),
        )
    )

    common = {
        "schema_version": "skill-scientific-evaluation-1.0",
        "module": module,
        "claim_category_counts": category_counts,
        "global_grounded_claim_rate": "NOT_APPLICABLE",
    }
    if module == "research_analysis":
        common["metrics"] = factual
        return common

    if module == "research_gap":
        cited = {evidence_id for row in rows for evidence_id in row.get("evidence_ids") or []}
        for card in output.get("evidence_cards") or []:
            evidence_id = str(card.get("evidence_id") or card.get("doc_id") or "")
            if evidence_id:
                cited.add(evidence_id)
        available = len(bundle.get("citation_index") or {})
        coverage_denominator = min(max(available, 1), 12)
        evidence_coverage = min(
            1.0,
            _rate(len(cited), coverage_denominator, empty=0.0),
        )
        direct = sum(
            str(row.get("role") or row.get("verified_evidence_role") or "") == "DIRECT_SUPPORT"
            for row in (bundle.get("citation_index") or {}).values()
        )
        gap_state = "GAP" in str(generation_state).upper()
        explicit_gap = bool(re.search(r"缺口|空白|不足|未见|缺少|gap|insufficient|not found", text, re.I))
        false_gap_risk = 0.0
        if gap_state or explicit_gap:
            # A few direct records may support adjacent concepts while the
            # interaction remains missing (Q04). A broad pool across many
            # primary papers raises the risk of overstating a gap.
            false_gap_risk = min(1.0, round(max(0, direct - 3) / 7.0, 4))
            if factual["condition_match"] < 0.75:
                false_gap_risk = round(false_gap_risk * 0.5, 4)
        gap_validity = round(max(0.0, 1.0 - false_gap_risk), 4) if (gap_state or explicit_gap) else 0.0
        common["metrics"] = {
            "evidence_coverage": evidence_coverage,
            "gap_validity": gap_validity,
            "false_gap_risk": false_gap_risk,
        }
        return common

    if module == "hypothesis_generation":
        proposal_count = category_counts[NOVEL_SCIENTIFIC_PROPOSAL]
        common["metrics"] = {
            "premise_grounding": factual["factual_grounding"],
            "evidence_consistency": factual["claim_evidence_alignment"],
            "falsifiability": 1.0 if _FALSIFIABLE.search(text) else 0.0,
            "novelty": 1.0 if proposal_count > 0 else 0.0,
        }
        return common

    if module == "experiment_design":
        parameter_rows = [row for row in rows if _NUMBER_WITH_UNIT.search(str(row.get("claim_text") or ""))]
        unsupported_parameters = [row for row in parameter_rows if row.get("unsupported_numeric_value")]
        unsupported_rate = _rate(len(unsupported_parameters), len(parameter_rows), empty=0.0)
        common["metrics"] = {
            "evidence_grounded_methods_parameters": round(1.0 - unsupported_rate, 4),
            "feasibility": 1.0 if _FEASIBILITY.search(text) else 0.0,
            "confounder_control": 1.0 if _CONFOUNDER.search(text) else 0.0,
            "falsification_criteria": 1.0 if _FALSIFIABLE.search(text) else 0.0,
            "unsupported_parameter_rate": unsupported_rate,
        }
        return common

    raise ValueError(f"Unsupported scientific evaluation module: {module}")
