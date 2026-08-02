"""Evidence-bound multi-paper research reasoning workflow.

All extracted facts retain paper/title/page/section/Evidence-ID provenance.
Derived proposals are explicitly labelled and never presented as source text.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.library_management import TARGET_AUDIT
from src.metadata_gate import read_jsonl
from src.unified_rag import generate_counter_targets, retrieve_research_evidence


NOT_REPORTED = "NOT_REPORTED"
CONDITION_LABELS = {
    "alloy_grade": "材料",
    "material": "材料",
    "manufacturing_process": "工艺",
    "process": "工艺",
    "heat_treatment": "热处理",
    "surface_treatment": "表面状态",
    "surface_state": "表面状态",
    "loading_mode": "载荷",
    "stress_ratio_R": "应力比",
    "fatigue_regime": "疲劳区间",
    "environment": "环境",
    "crack_detection_method": "检测方法",
    "characterization_method": "检测方法",
}
RELEVANCE_PATTERNS = (
    r"surface roughness|\bRa\b|\bRz\b|as-built|machined|polished",
    r"near[- ]surface|subsurface|distance (?:to|from) (?:the )?surface|depth",
    r"pore|porosity|defect|sqrt.?area|√area",
    r"stress ratio|\bR\s*=|stress amplitude|maximum stress",
    r"\bHCF\b|\bVHCF\b|high[- ]cycle|very[- ]high[- ]cycle",
    r"\bHIP\b|hot isostatic|heat treatment|surface treatment",
    r"crack initiation|crack origin|initiation site",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_evidence(base_dir: Path) -> list[dict[str, Any]]:
    with (base_dir / "data/evidence/trusted_evidence.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _conditions(base_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("evidence_id") or ""): row for row in read_jsonl(base_dir / "data/evidence/condition_evidence_records.jsonl")}


def _role(row: dict[str, Any]) -> str:
    explicit = str(row.get("evidence_role") or row.get("support_or_counter") or "").upper()
    if explicit in {"SUPPORT", "COUNTER", "CONDITION_DEPENDENT", "INSUFFICIENT"}:
        return explicit
    text = f"{row.get('claim','')} {row.get('original_text','')}".casefold()
    if re.search(r"no significant|not significant|did not|contrary|opposite|无显著|相反", text):
        return "COUNTER"
    if re.search(r"depend(?:s|ed)? on|under .*condition|whereas|however|only when|条件|取决于", text):
        return "CONDITION_DEPENDENT"
    return "SUPPORT"


def normalize_ui_value(value: Any) -> Any:
    """Remove nested JSON/string quoting before values enter reports or UI."""
    if isinstance(value, dict):
        return {str(key): normalize_ui_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_ui_value(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    for _ in range(3):
        if len(text) < 2 or text[0] not in {'"', "'"} or text[-1] != text[0]:
            break
        try:
            decoded = json.loads(text) if text[0] == '"' else text[1:-1]
        except json.JSONDecodeError:
            decoded = text[1:-1]
        if not isinstance(decoded, str):
            return normalize_ui_value(decoded)
        text = decoded.strip()
    return text


def _reference(row: dict[str, Any], title: str, conditions: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_paper_id": row.get("canonical_paper_id") or row.get("paper_id"),
        "title": normalize_ui_value(title),
        "evidence_id": row.get("evidence_id") or row.get("doc_id"),
        "original_text": normalize_ui_value(row.get("original_text")),
        "claim": normalize_ui_value(row.get("claim")),
        "page_number": int(float(row.get("page_number") or 0)),
        "section": normalize_ui_value(row.get("section") or NOT_REPORTED),
        "directness": row.get("directness") or NOT_REPORTED,
        "evidence_role": _role(row),
        "experimental_conditions": normalize_ui_value(conditions),
        "source_hash": row.get("source_hash") or NOT_REPORTED,
    }


def build_formula_comparison(paper_ids: Sequence[str], manifest: dict[str, dict[str, Any]], base_dir: Path) -> dict[str, Any]:
    formulas = []
    for paper_id in paper_ids:
        for index, row in enumerate(read_jsonl(base_dir / "data/deep_read" / paper_id / "equations.jsonl"), 1):
            context = str(row.get("original_text") or "")
            original = str(row.get("latex_candidate") or "").strip() or next((line.strip() for line in context.splitlines() if "=" in line), NOT_REPORTED)
            variables = re.findall(r"\b(?:Nf|sigma|epsilon|da|dN|DeltaK|R|C|m|b|c|sqrt_area|HV)\b", context, re.I)
            units = re.findall(r"\b(?:MPa|GPa|Pa|Hz|kHz|mm|µm|μm|um|cycles?|m/cycle)\b", context, re.I)
            conditions = [sentence.strip() for sentence in re.split(r"(?<=[.;])\s+", " ".join(context.split())) if re.search(r"\b(?:R\s*=|temperature|frequency|condition|range|valid)\b", sentence, re.I)][:4]
            formulas.append({
                "formula_id": f"{paper_id}:p{row.get('page_number')}:e{index}",
                "canonical_paper_id": paper_id, "title": manifest.get(paper_id, {}).get("title", NOT_REPORTED),
                "page_number": int(row.get("page_number") or 0), "section": row.get("section") or NOT_REPORTED,
                "original_formula": original, "normalized_formula": row.get("latex_candidate") or NOT_REPORTED,
                "variable_definitions": sorted(set(variables)) or NOT_REPORTED,
                "parameter_units": sorted(set(units)) or NOT_REPORTED,
                "applicable_range": conditions or NOT_REPORTED,
                "assumptions": NOT_REPORTED,
                "same_input_result": "NOT_COMPUTED_MISSING_PARAMETERS_OR_INCOMPARABLE",
                "uncertainty": "Formula context may require human symbol/unit verification.",
            })
    return {
        "formula_count": len(formulas), "formulas": formulas,
        "comparison_status": "COMPARABLE_ONLY_AFTER_UNIT_RANGE_AND_ASSUMPTION_ALIGNMENT" if len(formulas) > 1 else "INSUFFICIENT_MULTIPLE_FORMULAS",
        "result_difference": "NOT_COMPUTED" if formulas else NOT_REPORTED,
        "difference_source": "Applicability ranges, units, fitted parameters and assumptions must be aligned first.",
        "missing_parameter_policy": "Do not impute; mark NOT_REPORTED and skip numerical comparison.",
    }


def build_dominance_map(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    quantitative = [row for row in evidence if any(row["experimental_conditions"].get(key) not in (None, "", [], {}, NOT_REPORTED) for key in ("defect_size", "defect_distance_to_surface", "surface_roughness"))]
    papers = sorted({row["canonical_paper_id"] for row in quantitative})
    paired = [
        row for row in quantitative
        if all(
            row["experimental_conditions"].get(key) not in (None, "", [], {}, NOT_REPORTED)
            for key in ("defect_size", "defect_distance_to_surface", "surface_roughness")
        )
    ]
    paired_papers = {row["canonical_paper_id"] for row in paired}
    has_numeric_boundary = len(paired) >= 10 and len(paired_papers) >= 3
    return {
        "map_type": "QUANTITATIVE_CANDIDATE_REQUIRES_FIT_REVIEW" if has_numeric_boundary else "QUALITATIVE_ONLY",
        "x_axis": "defect_distance_to_surface / defect_size",
        "y_axis": "surface_roughness or equivalent_defect_size",
        "zones": ["surface_roughness_dominated", "near_surface_pore_dominated", "internal_pore_dominated", "microstructure_dominated", "insufficient_evidence", "literature_conflict"],
        "numeric_boundaries": [] if not has_numeric_boundary else "NOT_FITTED_IN_RULE_BASED_STAGE",
        "boundary_basis": "No numeric boundary is emitted unless quantitative coverage is sufficient and a separately reviewed fit exists.",
        "source_evidence_ids": [row["evidence_id"] for row in quantitative],
        "paper_count": len(papers), "experimental_condition_range": [row["experimental_conditions"] for row in quantitative[:30]],
        "confidence": "LOW" if not has_numeric_boundary else "PENDING_FIT_REVIEW",
        "paired_quantitative_record_count": len(paired),
        "uncertainty": "Sparse paired roughness/defect-distance/defect-size observations.",
    }


def _counter_search(question: str, base_dir: Path) -> dict[str, Any]:
    targets = generate_counter_targets(question)
    rows = []
    for query in targets:
        result = retrieve_research_evidence(query, task_type="reverse_evidence", top_k=5, base_dir=base_dir)
        for row in result["results"]:
            item = _reference(row, str(row.get("title") or NOT_REPORTED), row.get("experimental_conditions") or {})
            text = f"{item['claim']} {item['original_text']}".casefold()
            item["classification"] = (
                "COUNTER" if re.search(r"no significant|not significant|did not|opposite|contrary|相反|无显著", text)
                else "CONDITION_DEPENDENT" if re.search(r"depend|whereas|however|condition|only|条件", text)
                else "SUPPORT"
            )
            item["reverse_query"] = query; rows.append(item)
    unique = {}
    for row in rows: unique.setdefault(str(row["evidence_id"]), row)
    classified = list(unique.values())
    return {
        "completed": True, "queries": targets, "search_scope": "CURRENT_FORMAL_RAG",
        "results": classified,
        "counts": dict(Counter(row["classification"] for row in classified)),
    }


def _experiment_design(hypothesis: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "research_object": "L-PBF Ti-6Al-4V fatigue specimens; exact grade and powder route require human confirmation",
        "core_hypothesis": hypothesis["statement"],
        "independent_variables": ["surface_roughness", "defect_size", "defect_distance_to_surface", "stress_ratio_R", "post_processing"],
        "dependent_variables": ["fatigue_life", "crack_initiation_location", "fatigue_limit"],
        "control_variables": ["alloy_heat", "build_orientation", "specimen_geometry", "machine/process window", "test_environment", "frequency"],
        "covariates": ["porosity", "defect_morphology", "residual_stress", "microstructure"],
        "grouping": "Factorial roughness × defect-location/size × R ratio × post-processing groups; retain an as-built baseline.",
        "sample_size_recommendation_and_basis": "Pilot ≥5 specimens/cell; final n must follow variance/effect-size power analysis. This is a system-derived planning minimum, not a literature-reported universal value.",
        "specimen_preparation": "Same build and heat; controlled machining/polishing; pre-test micro-CT and roughness metrology.",
        "loading_mode": "Axial HCF/VHCF as applicable; do not mix regimes in one fitted boundary.",
        "stress_ratio": "At least two registered R levels selected from evidence; exact levels require protocol approval.",
        "stress_levels": "Multiple S-N levels bracketing finite-life and run-out regions; set after pilot.",
        "cycle_range": "HCF and VHCF analysed separately; run-out definition fixed before testing.",
        "characterization": ["micro-CT", "3D surface profilometry", "SEM fractography", "EBSD where microstructure competition is tested", "residual-stress measurement"],
        "data_analysis": ["competing-risk initiation model", "factorial/mixed-effects model", "S-N or Basquin fit by regime", "leave-one-build validation", "uncertainty intervals"],
        "predicted_direction": "Surface-origin fraction should fall as roughness is removed; pore-origin fraction should rise when a sufficiently severe near-surface/internal defect remains. Exact transition boundary is not asserted.",
        "success_criterion": "Pre-registered interaction or competing-risk boundary is reproducible across held-out builds with traceable initiation origins.",
        "supports_hypothesis_if": ["Initiation mechanism changes systematically with controlled roughness/defect geometry after covariate adjustment.", "Held-out build predicts the same direction."],
        "falsifies_hypothesis_if": ["No mechanism-transition association after controlling stress and microstructure.", "Observed transition direction reverses reproducibly.", "A single uncontrolled covariate explains the apparent transition."],
        "confounders": ["build-to-build variation", "unresolved pores below CT resolution", "residual stress", "texture/microstructure", "surface measurement bandwidth", "run-out censoring"],
        "minimum_cost_validation": "Existing specimens: profilometry + SEM origin classification + available CT, two R levels, limited factorial pilot.",
        "complete_validation": "Purpose-built factorial campaign with CT-tracked defects, roughness control, residual-stress/microstructure characterization, HCF and VHCF cohorts, blinded fractography.",
        "evidence_sources": sources[:20],
        "provenance_labels": {
            "directly_extracted": ["evidence_sources and their reported conditions"],
            "system_derived": ["grouping, analysis plan, predicted direction, success/falsification criteria"],
            "human_confirmation_required": ["exact R levels, stress levels, sample size, run-out threshold, machine parameters, safety and budget"],
        },
    }


def run_selected_research_workflow(paper_ids: Sequence[str], *, question: str = "", base_dir: Path) -> dict[str, Any]:
    selected = list(dict.fromkeys(str(value) for value in paper_ids if value))
    manifest = {row["paper_id"]: row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl")}
    allowed, rejected = [], {}
    for paper_id in selected:
        row = manifest.get(paper_id) or {}
        if row.get("library_status") == "FORMAL" and row.get("rag_status") == "INDEXED_STAGE3_UNIFIED": allowed.append(paper_id)
        else: rejected[paper_id] = "NOT_FORMAL_INDEXED"
    condition_by_evidence = _conditions(base_dir)
    raw = [row for row in _read_evidence(base_dir) if str(row.get("canonical_paper_id") or row.get("paper_id") or "") in set(allowed)]
    if question:
        relevant = []
        for row in raw:
            text = f"{row.get('claim','')} {row.get('original_text','')} {row.get('experimental_conditions','')}"
            if sum(bool(re.search(pattern, text, re.I)) for pattern in RELEVANCE_PATTERNS) >= 2:
                relevant.append(row)
        # For a small hand-picked selection, retain all evidence if the strict
        # transition-variable filter would erase the evidence context.
        if relevant or len(allowed) > 10:
            raw = relevant
    evidence = []
    for row in raw:
        paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "")
        cond = condition_by_evidence.get(str(row.get("evidence_id") or ""), {})
        clean_conditions = {key: value for key, value in cond.items() if key not in {"original_text", "claim"}}
        reference = _reference(
            row,
            str(manifest.get(paper_id, {}).get("title") or NOT_REPORTED),
            clean_conditions,
        )
        reference["authors"] = normalize_ui_value(
            manifest.get(paper_id, {}).get("authors") or NOT_REPORTED
        )
        reference["year"] = normalize_ui_value(
            manifest.get(paper_id, {}).get("publication_date")
            or manifest.get(paper_id, {}).get("year")
            or NOT_REPORTED
        )
        evidence.append(reference)
    support = [row for row in evidence if row["evidence_role"] == "SUPPORT"]
    counter_selected = [row for row in evidence if row["evidence_role"] == "COUNTER"]
    conditional = [row for row in evidence if row["evidence_role"] == "CONDITION_DEPENDENT"]
    mechanisms = defaultdict(list)
    for row in evidence:
        mechanism = str(row["experimental_conditions"].get("mechanism") or row["experimental_conditions"].get("fracture_mechanism") or NOT_REPORTED)
        if mechanism != NOT_REPORTED: mechanisms[mechanism].append(row["evidence_id"])
    condition_fields = sorted({key for row in evidence for key, value in row["experimental_conditions"].items() if value not in (None, "", [], {}, NOT_REPORTED)})
    effective_question = question or "What relationship remains untested under matched fatigue conditions in the selected papers?"
    reverse = _counter_search(effective_question, base_dir) if allowed else {"completed": False, "queries": [], "results": [], "counts": {}}
    for row in reverse.get("results", []):
        paper_id = str(row.get("canonical_paper_id") or "")
        row["authors"] = normalize_ui_value(
            manifest.get(paper_id, {}).get("authors") or NOT_REPORTED
        )
        row["year"] = normalize_ui_value(
            manifest.get(paper_id, {}).get("publication_date")
            or manifest.get(paper_id, {}).get("year")
            or NOT_REPORTED
        )
    reverse_counter = [
        row for row in reverse.get("results", [])
        if row.get("classification") == "COUNTER"
    ]
    support_ids = {str(row.get("evidence_id") or "") for row in support}
    counter_all = []
    conditional_ids = {str(row.get("evidence_id") or "") for row in conditional}
    for row in [*counter_selected, *reverse_counter]:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id in support_ids:
            if evidence_id not in conditional_ids:
                converted = dict(row)
                converted["evidence_role"] = "CONDITION_DEPENDENT"
                converted["classification_explanation"] = (
                    "同一 Evidence ID 不能同时作为支持与反驳，已降级为条件依赖证据。"
                )
                conditional.append(converted)
                conditional_ids.add(evidence_id)
            continue
        if evidence_id and evidence_id not in {
            str(item.get("evidence_id") or "") for item in counter_all
        }:
            counter_all.append(row)
    covered = {
        key: sorted(
            {
                json.dumps(
                    normalize_ui_value(row["experimental_conditions"].get(key)),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for row in evidence
                if row["experimental_conditions"].get(key)
                not in (None, "", [], {}, NOT_REPORTED)
            }
        )
        for key in condition_fields
    }
    mandatory = ["defect_size", "defect_distance_to_surface", "surface_roughness", "stress_ratio_R", "fatigue_regime", "heat_treatment", "hip"]
    uncovered = [key for key in mandatory if key not in covered]
    support_papers = {str(row["canonical_paper_id"]) for row in support}
    counter_papers = {str(row["canonical_paper_id"]) for row in counter_all}
    threshold_met = (
        len(allowed) >= 3
        and len(support_papers) >= 2
        and bool(reverse.get("completed"))
    )
    evidence_sufficiency = (
        "HIGH"
        if threshold_met and len(support_papers) >= 5 and bool(counter_all)
        else "MEDIUM"
        if threshold_met and evidence
        else "LOW"
    )
    dual_role_explanations = []
    for paper_id in sorted(support_papers & counter_papers):
        support_rows = [row for row in support if row["canonical_paper_id"] == paper_id]
        counter_rows = [row for row in counter_all if row["canonical_paper_id"] == paper_id]
        if {row["evidence_id"] for row in support_rows}.isdisjoint(
            {row["evidence_id"] for row in counter_rows}
        ):
            dual_role_explanations.append(
                {
                    "paper_id": paper_id,
                    "explanation": "同一论文中的不同原文证据在不同实验条件下方向不同，因此分别列为支持与可能反驳。",
                    "supporting_evidence_ids": [row["evidence_id"] for row in support_rows],
                    "refuting_evidence_ids": [row["evidence_id"] for row in counter_rows],
                }
            )
    gap = {
        "gap_id": "GAP_" + hashlib.sha256(("|".join(allowed) + effective_question).encode()).hexdigest()[:16].upper(),
        "title_cn": "匹配疲劳条件下的机制边界仍缺乏独立验证",
        "description": "当前正式文献已覆盖表面状态、缺陷几何、应力比及部分后处理条件，但尚未建立在应力、组织和试验环境一致时可独立复现的机制转换边界。",
        "description_paragraphs_cn": [
            "当前选中文献已经研究了表面粗糙度、缺陷尺寸与位置、应力比以及疲劳区间对裂纹起裂和寿命的影响。现有证据说明这些因素可能相互竞争，而不是由单一变量在所有条件下主导。",
            "已覆盖条件分散在不同论文和不同试验方案中，缺少同一材料批次、相同组织与载荷条件下对表面严重度和临界缺陷几何的成组比较，因此无法给出可迁移的转换边界。",
            "这构成的是条件组合与独立验证层面的研究空白。由于本地检索范围有限，仍可能存在尚未纳入的论文已经部分解决该问题，结论必须与补充检索共同使用。",
        ],
        "supporting_papers": sorted(support_papers),
        "possibly_refuting_papers": sorted(counter_papers),
        "supporting_evidence": support[:20],
        "refuting_evidence": counter_all[:20],
        "condition_dependent_evidence": conditional[:20],
        "dual_role_explanations": dual_role_explanations,
        "studied_conditions": covered, "uncovered_conditions": uncovered,
        "possible_retrieval_omission": True,
        "missing_evidence": ["paired roughness–defect distance–defect size data", "matched R/HCF-VHCF/post-processing cohorts", "independent validation boundary"],
        "testability": "HIGH", "importance": "HIGH", "evidence_sufficiency": evidence_sufficiency,
        "continued_search_keywords": reverse.get("queries", []),
        "counter_search_completed": reverse.get("completed", False),
        "confidence": "HIGH" if evidence_sufficiency == "HIGH" else "NOT_HIGH_CONFIDENCE",
        "reliable_gap_threshold_met": threshold_met,
        "threshold_message": "" if threshold_met else "当前选中文献不足以可靠判定研究空白。",
        "importance_cn": "该空白影响疲劳寿命模型对起裂机制转换的表达，并关系到表面处理、HIP 和缺陷验收标准的工程选择，值得通过匹配条件实验验证。",
        "testability_cn": "可使用 micro-CT、三维轮廓测量、受控疲劳试验和 SEM 断口定位验证。最低成本方案是复用现有试样完成表面测量与起裂源复核；主要困难是缺陷空间分辨率和跨批次组织差异。",
        "search_keywords_cn": ["钛合金 疲劳 表面粗糙度 缺陷位置 起裂机制", "匹配应力比 热处理 条件边界"],
        "search_keywords_en": reverse.get("queries", []),
    }
    hypothesis = {
        "hypothesis_id": "HYP_" + gap["gap_id"].split("_")[-1],
        "statement": "At matched stress and microstructure, the crack-initiation mechanism changes from surface-controlled to pore-controlled as surface severity falls relative to the effective severity of the nearest critical pore.",
        "status": "SYSTEM_DERIVED_FALSIFIABLE_CANDIDATE",
        "supporting_evidence_ids": [row["evidence_id"] for row in support[:20]],
        "counter_evidence_ids": [row["evidence_id"] for row in counter_all[:20]],
        "falsification_standard": "No reproducible mechanism switch after matched-condition adjustment, or a reproducible opposite switch.",
    }
    formulas = build_formula_comparison(allowed, manifest, base_dir)
    dominance = build_dominance_map(evidence)
    experiment = _experiment_design(hypothesis, evidence)
    conflicts = [{"topic": "same mechanism under differing conditions", "support": support[:5], "counter": counter_all[:5]}] if counter_all else []
    return {
        "generated_at": _now(), "status": "GENERATED" if evidence and threshold_met else "INSUFFICIENT_EVIDENCE",
        "selected_paper_ids": selected, "eligible_paper_ids": allowed, "rejected": rejected,
        "evidence_matrix": evidence, "condition_matrix": [{"evidence_id": row["evidence_id"], "paper_id": row["canonical_paper_id"], **row["experimental_conditions"]} for row in evidence],
        "mechanism_comparison": dict(mechanisms), "formula_comparison": formulas,
        "supporting_evidence": support, "counter_evidence": counter_all,
        "condition_dependent_evidence": conditional, "literature_conflicts": conflicts,
        "research_gaps": [gap], "hypotheses": [hypothesis], "experiment_designs": [experiment],
        "reverse_evidence_retrieval": reverse, "condition_mechanism_dominance_map": dominance,
    }
