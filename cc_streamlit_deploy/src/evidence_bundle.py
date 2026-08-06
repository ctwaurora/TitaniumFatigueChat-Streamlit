"""Build compact, traceable per-paper and cross-paper evidence bundles."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from src.research_topics import document_topics, identify_topics


EMPTY_VALUES = (None, "", "NOT_REPORTED", [], {})
COMPARISON_FIELDS = (
    "alloy_grade", "material", "manufacturing_process", "process", "heat_treatment",
    "hip", "surface_treatment", "surface_state", "stress_ratio_R", "fatigue_regime",
    "temperature", "environment", "build_orientation", "loading_mode",
)


def _clean_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item not in EMPTY_VALUES}


def _text(row: dict[str, Any]) -> str:
    return str(row.get("claim") or row.get("original_text") or row.get("text") or "").strip()


def _normal_key(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def is_plausible_formula(row: dict[str, Any]) -> bool:
    equation = str(row.get("equation") or row.get("formula") or "").strip()
    if not equation or len(equation) > 280 or not row.get("paper_id"):
        return False
    if not row.get("page_number") or not (row.get("doc_id") or row.get("formula_id")):
        return False
    relation = bool(re.search(r"(?:=|≈|≃|∝|≤|≥|<|>)", equation))
    operators = bool(re.search(r"[+\-*/^√∑Δσɛε()\[\]]", equation))
    symbol = bool(re.search(r"\b(?:da/dN|Nf|Delta\s*K|Paris|C|m|R|K|max|min)\b|[Δσɛεαβγ]", equation, re.I))
    words = re.findall(r"[A-Za-z]{3,}", equation)
    sentence_like = len(words) > 28 and equation.count(" ") > 35
    return relation and operators and symbol and not sentence_like


def _result_direction(text: str) -> str:
    lower = text.casefold()
    if any(term in lower for term in ("no significant", "not significant", "independent of", "无显著", "没有显著")):
        return "NO_SIGNIFICANT_EFFECT"
    if any(term in lower for term in ("decrease", "reduced", "lower", "下降", "降低", "减小")):
        return "DECREASE"
    if any(term in lower for term in ("increase", "improved", "higher", "上升", "提高", "增加")):
        return "INCREASE"
    return "CONDITION_DEPENDENT_OR_UNRESOLVED"


@dataclass
class PaperEvidenceSummary:
    paper_id: str
    title: str
    authors: str
    year: str
    roles: list[str]
    topics: list[str]
    study_object: str
    conditions: dict[str, Any]
    principal_claims: list[dict[str, Any]]
    mechanisms: list[str]
    formulas: list[dict[str, Any]]
    result_directions: list[str]
    applicability: list[str]
    limitations: list[str]
    canonical_id: str = ""
    doi: str = ""
    material: str = ""
    manufacturing_process: str = ""
    post_processing: str = ""
    surface_condition: str = ""
    microstructure: str = ""
    residual_stress: str = ""
    specimen_geometry: str = ""
    loading_conditions: dict[str, Any] = field(default_factory=dict)
    stress_ratio: str = ""
    frequency: str = ""
    temperature: str = ""
    environment: str = ""
    fatigue_regime: str = ""
    crack_stage: str = ""
    independent_variables: list[str] = field(default_factory=list)
    dependent_variables: list[str] = field(default_factory=list)
    measurement_methods: list[str] = field(default_factory=list)
    quantitative_results: list[str] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    mechanism: list[str] = field(default_factory=list)
    authors_conclusion: list[str] = field(default_factory=list)
    evidence_role: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    source_sections: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    extraction_confidence: float = 0.0


@dataclass
class ScientificClaim:
    claim_id: str
    claim_text: str
    material: str
    manufacturing_process: str
    independent_variable: str
    dependent_variable: str
    fatigue_stage: str
    crack_stage: str
    experimental_conditions: dict[str, Any]
    mechanism_chain: list[str]
    quantitative_relationship: str
    applicable_formula_ids: list[str]
    supporting_evidence_ids: list[str]
    counter_evidence_ids: list[str]
    condition_dependent_evidence_ids: list[str]
    applicability_boundary: list[str]
    unresolved_part: list[str]
    evidence_level: str
    confidence: float
    source_papers: list[str]


@dataclass
class CrossPaperSynthesis:
    consensus: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    condition_matches: list[str] = field(default_factory=list)
    condition_mismatches: list[str] = field(default_factory=list)
    covered_conditions: dict[str, list[str]] = field(default_factory=dict)
    missing_conditions: list[str] = field(default_factory=list)
    supported_conclusions: list[str] = field(default_factory=list)
    unsupported_conclusions: list[str] = field(default_factory=list)
    formula_comparability: list[str] = field(default_factory=list)
    mechanism_map: dict[str, Any] = field(default_factory=dict)
    consistent_findings: list[dict[str, Any]] = field(default_factory=list)
    conflicting_findings: list[dict[str, Any]] = field(default_factory=list)
    condition_explanations: list[str] = field(default_factory=list)
    comparable_studies: list[list[str]] = field(default_factory=list)
    non_comparable_studies: list[dict[str, Any]] = field(default_factory=list)
    missing_condition_combinations: list[str] = field(default_factory=list)
    evidence_limitations: list[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    question: str
    topics: list[str]
    papers: list[PaperEvidenceSummary]
    formulas: list[dict[str, Any]]
    synthesis: CrossPaperSynthesis
    citation_index: dict[str, dict[str, Any]]
    query_frame: dict[str, Any] = field(default_factory=dict)
    scientific_claims: list[ScientificClaim] = field(default_factory=list)
    dataset_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_json(self, max_chars: int = 18000) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))
        return payload if len(payload) <= max_chars else payload[:max_chars] + "\n[EvidenceBundle truncated]"


def _merge_conditions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for key, item in _clean_conditions(row.get("experimental_conditions")).items():
            items = item if isinstance(item, (list, tuple, set)) else [item]
            for entry in items:
                text = str(entry).strip()
                if text and text not in values[key]:
                    values[key].append(text)
    return {key: items[0] if len(items) == 1 else items for key, items in values.items()}


def _condition_comparison(papers: list[PaperEvidenceSummary]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    covered: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        for key in COMPARISON_FIELDS:
            item = paper.conditions.get(key)
            if item in EMPTY_VALUES:
                continue
            for value in item if isinstance(item, list) else [item]:
                if str(value) not in covered[key]:
                    covered[key].append(str(value))
    matches = [f"{key}={values[0]}" for key, values in covered.items() if len(values) == 1]
    mismatches = [f"{key}存在不可直接合并的条件：{' / '.join(values)}" for key, values in covered.items() if len(values) > 1]
    return matches, mismatches, dict(covered)


def build_evidence_bundle(
    question: str,
    supporting: list[dict[str, Any]],
    counter: list[dict[str, Any]],
    conditional: list[dict[str, Any]],
    alternative: list[dict[str, Any]] | None = None,
    supporting_context: list[dict[str, Any]] | None = None,
    retrieved_pool: list[dict[str, Any]] | None = None,
    query_frame: dict[str, Any] | None = None,
    dataset_version: str = "",
) -> EvidenceBundle:
    role_rows = (
        ("DIRECT_SUPPORT", supporting),
        ("DIRECT_COUNTER", counter),
        ("CONDITION_DEPENDENT", conditional),
        ("ALTERNATIVE_MECHANISM", alternative or []),
        ("SUPPORTING_CONTEXT", supporting_context or []),
    )
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for role, rows in role_rows:
        for row in rows:
            paper_id = str(row.get("paper_id") or "")
            claim = _text(row)
            key = (paper_id, _normal_key(claim))
            if not paper_id or not claim or key in seen:
                continue
            seen.add(key)
            grouped[paper_id].append((role, row))

    formula_source = list(retrieved_pool or []) + supporting + counter + conditional
    formula_rows: list[dict[str, Any]] = []
    formula_seen: set[str] = set()
    for row in formula_source:
        formula_id = str(row.get("doc_id") or row.get("formula_id") or "")
        if formula_id in formula_seen or not is_plausible_formula(row):
            continue
        from src.formula_validation import CONFIRMED, validate_formula_candidate

        validated_formula = validate_formula_candidate({
            **row,
            "formula_id": formula_id,
            "original_formula": row.get("equation") or row.get("formula"),
            "paper_title": row.get("title"),
            "context_before_after": row.get("original_text") or row.get("claim"),
            "symbol_definitions": row.get("parameters") or [],
            "symbol_units": row.get("units") or [],
            "equation_number": row.get("equation_number") or "",
            "raw_review_status": row.get("review_status") or "",
        })
        if validated_formula["validation_status"] != CONFIRMED:
            continue
        formula_seen.add(formula_id)
        formula_rows.append({
            "formula_id": formula_id,
            "equation": str(row.get("equation") or row.get("formula")),
            "equation_latex": str(row.get("equation_latex") or row.get("equation") or row.get("formula")),
            "equation_text": str(row.get("original_text") or row.get("equation") or row.get("formula")),
            "paper_id": str(row.get("paper_id")),
            "canonical_id": str(row.get("paper_id")),
            "title": str(row.get("title") or ""),
            "page_number": row.get("page_number"),
            "section": str(row.get("section") or ""),
            "parameters": row.get("parameters") or [],
            "units": row.get("units") or [],
            "parameter_values": row.get("parameter_values") or {},
            "parameter_units": row.get("parameter_units") or row.get("units") or {},
            "variable_units": row.get("variable_units") or row.get("units") or {},
            "dependent_variable": str(row.get("dependent_variable") or ""),
            "independent_variables": row.get("independent_variables") or [],
            "applicable_conditions": _clean_conditions(row.get("applicable_conditions") or row.get("experimental_conditions")),
            "material_scope": str(row.get("material_scope") or ""),
            "process_scope": str(row.get("process_scope") or ""),
            "fatigue_stage": str(row.get("fatigue_stage") or ""),
            "crack_stage": str(row.get("crack_stage") or ""),
            "stress_ratio_scope": str(row.get("stress_ratio_scope") or ""),
            "temperature_scope": str(row.get("temperature_scope") or ""),
            "environment_scope": str(row.get("environment_scope") or ""),
            "assumptions": row.get("assumptions") or [],
            "calibration_dataset": str(row.get("calibration_dataset") or ""),
            "validation_dataset": str(row.get("validation_dataset") or ""),
            "extraction_confidence": float(row.get("confidence") or row.get("extraction_confidence") or 0.0),
            "evidence_status": "已确认",
            "validation_status": CONFIRMED,
            "validation_basis": validated_formula.get("validation_basis"),
        })

    papers: list[PaperEvidenceSummary] = []
    citation_index: dict[str, dict[str, Any]] = {}
    for paper_id, entries in grouped.items():
        rows = [row for _, row in entries]
        claims = []
        for role, row in entries[:6]:
            evidence_id = str(row.get("doc_id") or row.get("evidence_id") or "")
            claim = {
                "role": role,
                "claim": _text(row)[:650],
                "evidence_id": evidence_id,
                "page_number": row.get("page_number"),
                "section": str(row.get("section") or ""),
                "directness": str(row.get("directness") or ""),
                "conditions": _clean_conditions(row.get("experimental_conditions")),
                "evidence_level": (
                    "DIRECT_LITERATURE_FINDING"
                    if str(row.get("directness") or "").upper() == "DIRECT"
                    else "AUTHOR_INTERPRETATION"
                    if re.search(r"mechanis|because|due to|attribut|机制|由于|归因", _text(row), re.I)
                    else "CROSS_PAPER_SYNTHESIS"
                ),
            }
            claims.append(claim)
            if evidence_id:
                citation_index[evidence_id] = {
                    "paper_id": paper_id,
                    "title": str(row.get("title") or ""),
                    "page_number": row.get("page_number"),
                    "section": str(row.get("section") or ""),
                    "claim": str(row.get("claim") or ""),
                    "original_text": str(row.get("original_text") or row.get("text") or ""),
                    "directness": str(row.get("directness") or ""),
                    "evidence_role": role,
                    "experimental_conditions": _clean_conditions(row.get("experimental_conditions")),
                }
        conditions = _merge_conditions(rows)
        all_text = " ".join(_text(row) for row in rows)
        paper_formulas = [formula for formula in formula_rows if formula["paper_id"] == paper_id]
        mechanisms = [
            _text(row)[:350] for row in rows
            if re.search(r"mechanis|because|due to|attribut|机制|由于|归因", _text(row), re.I)
        ][:3]
        papers.append(PaperEvidenceSummary(
            paper_id=paper_id,
            title=str(rows[0].get("title") or "题名未报告"),
            authors=str(rows[0].get("authors") or "未报告"),
            year=str(rows[0].get("year") or "未报告"),
            roles=list(dict.fromkeys(role for role, _ in entries)),
            topics=document_topics(all_text, conditions),
            study_object=str(conditions.get("alloy_grade") or conditions.get("material") or "钛合金疲劳（具体牌号以证据条件为准）"),
            conditions=conditions,
            principal_claims=claims,
            mechanisms=mechanisms,
            formulas=paper_formulas,
            result_directions=list(dict.fromkeys(_result_direction(_text(row)) for row in rows)),
            applicability=[f"{key}={value}" for key, value in conditions.items() if key in COMPARISON_FIELDS],
            limitations=["未报告完整关键实验条件"] if len(conditions) < 3 else [],
            canonical_id=paper_id,
            doi=str(rows[0].get("doi") or ""),
            material=str(conditions.get("alloy_grade") or conditions.get("material") or ""),
            manufacturing_process=str(conditions.get("manufacturing_process") or conditions.get("process") or ""),
            post_processing=str(conditions.get("post_processing") or conditions.get("heat_treatment") or ""),
            surface_condition=str(conditions.get("surface_treatment") or conditions.get("surface_state") or ""),
            microstructure=str(conditions.get("microstructure") or ""),
            residual_stress=str(conditions.get("residual_stress") or ""),
            specimen_geometry=str(conditions.get("specimen_geometry") or conditions.get("sample_geometry") or ""),
            loading_conditions={key: conditions.get(key) for key in ("loading_mode", "stress_ratio_R", "frequency", "temperature", "environment") if conditions.get(key) not in EMPTY_VALUES},
            stress_ratio=str(conditions.get("stress_ratio_R") or ""),
            frequency=str(conditions.get("frequency") or ""),
            temperature=str(conditions.get("temperature") or ""),
            environment=str(conditions.get("environment") or ""),
            fatigue_regime=str(conditions.get("fatigue_regime") or ""),
            crack_stage=str(conditions.get("crack_stage") or ""),
            independent_variables=list((query_frame or {}).get("independent_variables") or []),
            dependent_variables=list((query_frame or {}).get("dependent_variables") or []),
            measurement_methods=list(dict.fromkeys(str(conditions.get(key)) for key in ("testing_method", "characterization_method") if conditions.get(key) not in EMPTY_VALUES)),
            quantitative_results=[_text(row)[:350] for row in rows if re.search(r"\d+(?:\.\d+)?\s*(?:MPa|GPa|Hz|cycles?|mm|µm|μm|%)", _text(row), re.I)][:5],
            equations=paper_formulas,
            mechanism=mechanisms,
            authors_conclusion=[claim["claim"] for claim in claims if claim["evidence_level"] in {"DIRECT_LITERATURE_FINDING", "AUTHOR_INTERPRETATION"}][:4],
            evidence_role=list(dict.fromkeys(role for role, _ in entries)),
            source_pages=sorted({int(row.get("page_number")) for row in rows if str(row.get("page_number") or "").isdigit()}),
            source_sections=list(dict.fromkeys(str(row.get("section") or "") for row in rows if row.get("section"))),
            evidence_ids=[claim["evidence_id"] for claim in claims if claim["evidence_id"]],
            extraction_confidence=round(sum(float(row.get("confidence") or 0.0) for row in rows) / max(1, len(rows)), 4),
        ))

    papers.sort(key=lambda paper: ("DIRECT_SUPPORT" not in paper.roles, paper.title.casefold()))
    matches, mismatches, covered = _condition_comparison(papers)
    directions = {direction for paper in papers for direction in paper.result_directions}
    consensus = [
        paper.principal_claims[0]["claim"]
        for paper in papers if "DIRECT_SUPPORT" in paper.roles and paper.principal_claims
    ][:4]
    conflicts = [
        paper.principal_claims[0]["claim"]
        for paper in papers if "DIRECT_COUNTER" in paper.roles and paper.principal_claims
    ][:4]
    if len(directions - {"CONDITION_DEPENDENT_OR_UNRESOLVED"}) > 1:
        conflicts.insert(0, "不同文献报告的结果方向不一致，必须按材料、处理、表面、载荷和疲劳阶段分层解释。")
    missing = [key for key in COMPARISON_FIELDS if key not in covered]
    formula_comparability = []
    if len(formula_rows) > 1:
        condition_sets = [formula["applicable_conditions"] for formula in formula_rows]
        formula_comparability.append(
            "公式条件一致后才可数值比较。" if all(item == condition_sets[0] for item in condition_sets[1:])
            else "这些公式的适用条件不同，不能直接进行数值比较。"
        )
    topics = identify_topics(question)
    comparable = []
    non_comparable = []
    for index, left in enumerate(papers):
        for right in papers[index + 1:]:
            differences = [key for key in COMPARISON_FIELDS if left.conditions.get(key) not in EMPTY_VALUES and right.conditions.get(key) not in EMPTY_VALUES and left.conditions.get(key) != right.conditions.get(key)]
            if differences:
                non_comparable.append({"paper_ids": [left.paper_id, right.paper_id], "different_conditions": differences, "reason": "条件不相容，只能用于解释冲突，不能定量合并。"})
            else:
                comparable.append([left.paper_id, right.paper_id])
    structured_conflicts = [
        {"paper_id": paper.paper_id, "finding": paper.principal_claims[0]["claim"], "conditions": paper.conditions, "classification": "CONDITION_DIFFERENCE" if mismatches else "POTENTIAL_TRUE_CONFLICT"}
        for paper in papers if "DIRECT_COUNTER" in paper.roles and paper.principal_claims
    ]
    synthesis = CrossPaperSynthesis(
        consensus=consensus,
        conflicts=conflicts,
        condition_matches=matches,
        condition_mismatches=mismatches,
        covered_conditions=covered,
        missing_conditions=missing,
        supported_conclusions=consensus,
        unsupported_conclusions=[f"缺少{key}条件，不能外推到该维度。" for key in missing[:5]],
        formula_comparability=formula_comparability,
        mechanism_map={
            "type": "QUALITATIVE_MECHANISM_MAP",
            "message": "当前证据只能形成定性机制主导区，尚不足以确定精确转换边界。",
            "candidate_axes": topics[:2],
        } if len(topics) >= 2 else {},
        consistent_findings=[{"finding": item, "evidence_level": "CROSS_PAPER_SYNTHESIS"} for item in consensus],
        conflicting_findings=structured_conflicts,
        condition_explanations=mismatches,
        comparable_studies=comparable,
        non_comparable_studies=non_comparable,
        missing_condition_combinations=[f"缺少同时报告{key}且与问题变量匹配的研究。" for key in missing[:8]],
        evidence_limitations=[f"{key}覆盖不足" for key in missing[:8]],
    )
    claims: list[ScientificClaim] = []
    for index, paper in enumerate(papers):
        if not paper.principal_claims:
            continue
        first = paper.principal_claims[0]
        claims.append(ScientificClaim(
            claim_id=f"SCI_CLAIM_{index + 1:03d}",
            claim_text=first["claim"], material=paper.material,
            manufacturing_process=paper.manufacturing_process,
            independent_variable=", ".join((query_frame or {}).get("independent_variables") or []),
            dependent_variable=", ".join((query_frame or {}).get("dependent_variables") or []),
            fatigue_stage=paper.fatigue_regime, crack_stage=paper.crack_stage,
            experimental_conditions=paper.conditions, mechanism_chain=paper.mechanisms,
            quantitative_relationship=(paper.quantitative_results[0] if paper.quantitative_results else ""),
            applicable_formula_ids=[item["formula_id"] for item in paper.formulas],
            supporting_evidence_ids=[item["evidence_id"] for item in paper.principal_claims if item["role"] == "DIRECT_SUPPORT"],
            counter_evidence_ids=[item["evidence_id"] for item in paper.principal_claims if item["role"] == "DIRECT_COUNTER"],
            condition_dependent_evidence_ids=[item["evidence_id"] for item in paper.principal_claims if item["role"] == "CONDITION_DEPENDENT"],
            applicability_boundary=paper.applicability,
            unresolved_part=paper.limitations,
            evidence_level=first["evidence_level"],
            confidence=paper.extraction_confidence,
            source_papers=[paper.paper_id],
        ))
    return EvidenceBundle(question, topics, papers, formula_rows, synthesis, citation_index, query_frame or {}, claims, dataset_version)
