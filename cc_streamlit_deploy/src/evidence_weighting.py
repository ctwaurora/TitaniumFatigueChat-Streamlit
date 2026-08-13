"""Configurable evidence scoring, role budgeting, diversity, and gap priority."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.query_frame import parse_query_frame

DEFAULT_CONFIG = {
    "dimension_weights": {"relevance": .25, "condition_match": .20, "evidence_directness": .15,
                           "study_quality": .12, "method_quality": .10, "material_process_match": .10,
                           "temporal_version_quality": .08},
    "role_coefficients": {"DIRECT_SUPPORT": 1.0, "DIRECT_COUNTER": 1.0, "CONDITION_DEPENDENT": .82,
                           "LIMITATION_EVIDENCE": .80, "ALTERNATIVE_MECHANISM": .72,
                           "SUPPORTING_CONTEXT": .55, "REVIEW_BACKGROUND": .40},
    "tier_coefficients": {"TIER1_CORE_DIRECT": 1.0, "TIER2_NEAR_DOMAIN": .82,
                          "TIER3_FOUNDATIONAL_MECHANICS": .68, "OUT_OF_SCOPE": .30},
    "material_process_match": {"lpbf_ti6al4v": 1.0, "pbf_lb_m_ti6al4v": .95, "ebm_ti6al4v": .78,
                               "other_am_ti6al4v": .72, "wrought_ti6al4v": .62,
                               "other_alpha_beta_titanium": .45, "other_metal": .20},
    "budget": {"total": 12, "role_caps": {"DIRECT_SUPPORT": 5, "CONDITION_DEPENDENT": 2,
            "DIRECT_COUNTER": 2, "LIMITATION_EVIDENCE": 2, "ALTERNATIVE_MECHANISM": 1},
            "max_single_paper_fraction": .25},
    "diversity": {"same_paper_penalty": .08, "same_author_penalty": .04,
                   "same_condition_penalty": .03, "same_method_penalty": .02},
}


def load_weight_config(base_dir: Path | str | None = None) -> dict[str, Any]:
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    path = root / "config" / "evidence_weight_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, item in value.items():
        if isinstance(item, dict) and isinstance(merged.get(key), dict):
            merged[key].update(item)
        else:
            merged[key] = item
    return merged


ROLE_COEFFICIENTS = DEFAULT_CONFIG["role_coefficients"]


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("original_text") or row.get("claim") or row.get("title") or "")


def _values(row: dict[str, Any], key: str) -> str:
    conditions = row.get("experimental_conditions") or {}
    value = row.get(key) if key in row else conditions.get(key)
    text = " ".join(map(str, value)) if isinstance(value, (list, tuple, set)) else str(value or "")
    return "" if text.strip().upper() in {"NOT_REPORTED", "NOT APPLICABLE", "N/A", "NONE"} else text


def material_process_match(row: dict[str, Any], frame: dict[str, Any], config: dict[str, Any]) -> float:
    title = str(row.get("title") or "")
    reported_alloy = _values(row, "alloy_grade")
    reported_process = _values(row, "manufacturing_process") or _values(row, "process")
    # Structured experimental conditions are authoritative. Full text is a
    # fallback only; otherwise a comparison sentence can mislabel the study.
    title_has_alloy = bool(re.search(r"ti[- ]?6al[- ]?4v|tc4|titanium|steel|aluminium|aluminum|nickel", title, re.I))
    title_has_process = bool(re.search(r"l[- ]?pbf|lpbf|pbf[- ]?lb/?m|slm|laser powder bed|electron beam|\bebm\b|directed energy deposition|\bded\b|wrought|forged|rolled", title, re.I))
    if title_has_alloy or title_has_process:
        text = f"{title if title_has_alloy else reported_alloy} {title if title_has_process else reported_process}".casefold()
    elif reported_alloy or reported_process:
        text = f"{reported_alloy} {reported_process}".casefold()
    else:
        text = (str(row.get("title") or "") + " " + _text(row)).casefold()
    alloy = str(frame.get("alloy_grade") or "").casefold()
    process = str(frame.get("manufacturing_process") or "").casefold()
    ti = bool(re.search(r"ti[- ]?6al[- ]?4v|tc4", text))
    lpbf = bool(re.search(r"l[- ]?pbf|lpbf|pbf[- ]?lb/?m|slm|laser powder bed", text))
    ebm = "ebm" in text or "electron beam" in text
    wrought = bool(re.search(r"wrought|forged|rolled", text))
    requested_ti = bool(re.search(r"ti[- ]?6al[- ]?4v|tc4", alloy))
    requested_process = bool(process.strip())
    # Do not punish an otherwise exact alloy match for an unspecified process.
    # A missing query constraint is not an evidence mismatch.
    if requested_ti and ti and not requested_process:
        return 1.0
    if ti and lpbf:
        return config["material_process_match"]["lpbf_ti6al4v"]
    if ti and ("pbf" in text or "pbf" in process):
        return config["material_process_match"]["pbf_lb_m_ti6al4v"]
    if ti and ebm:
        return config["material_process_match"]["ebm_ti6al4v"]
    if ti and re.search(r"am|additive|printed|selective", text):
        return config["material_process_match"]["other_am_ti6al4v"]
    if ti and wrought:
        return config["material_process_match"]["wrought_ti6al4v"]
    if re.search(r"titanium|ti[- ]?\d|alpha.*beta|α\+β", text):
        return config["material_process_match"]["other_alpha_beta_titanium"]
    return config["material_process_match"]["other_metal"]


def condition_match_score(row: dict[str, Any], frame: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    conditions = row.get("experimental_conditions") or {}
    expected = {
        "alloy_grade": frame.get("alloy_grade"), "manufacturing_process": frame.get("manufacturing_process"),
        "heat_treatment": " ".join(frame.get("post_processing") or []),
        "surface_treatment": " ".join(frame.get("surface_condition") or []),
        "stress_ratio_R": frame.get("stress_ratio"), "environment": frame.get("environment"),
        "temperature": frame.get("temperature"), "loading_mode": frame.get("loading_mode"),
        "fatigue_regime": frame.get("fatigue_stage"), "crack_stage": frame.get("crack_stage"),
    }
    query = str(frame.get("original_query") or "")
    dynamic_patterns = {
        "defect_type": r"lack[- ]of[- ]fusion|\bLOF\b|pore|porosity|void|孔隙|未熔合",
        "defect_size": r"\d+(?:\.\d+)?\s*(?:um|μm|mm)|defect size|pore size|缺陷尺寸|孔隙尺寸",
        "defect_location": r"near[- ]surface|subsurface|internal|surface|distance from surface|距表面|近表面|内部",
        "stress_amplitude": r"\d+(?:\.\d+)?\s*MPa|stress level|stress amplitude|应力水平|应力幅",
        "frequency": r"\d+(?:\.\d+)?\s*(?:Hz|kHz)|frequency|频率",
        "crack_detection_method": r"micro[- ]?CT|\bCT\b|\bSEM\b|\bEBSD\b|DIC|fractograph|测量方法|表征方法",
    }
    for key, pattern in dynamic_patterns.items():
        match = re.search(pattern, query, re.I)
        if match:
            expected[key] = match.group(0)
    expected = {key: str(value) for key, value in expected.items() if value}
    if not expected:
        return .5, []
    matches = 0
    conflicts = []
    for key, wanted in expected.items():
        actual = _values(row, key)
        if not actual:
            continue
        wanted_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", wanted.casefold()))
        actual_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", actual.casefold()))
        if wanted.casefold() in actual.casefold() or actual.casefold() in wanted.casefold() or bool(wanted_tokens & actual_tokens):
            matches += 1
        else:
            conflicts.append({"field": key, "query": wanted, "evidence": actual})
    missing = len(expected) - matches - len(conflicts)
    score = (matches + .5 * missing) / len(expected)
    # Explicit conflict on a supplied key is more harmful than an unreported key.
    if conflicts:
        score = max(0.0, score - 0.15 * len(conflicts) / len(expected))
    return max(0.0, min(1.0, score)), conflicts


def score_evidence(row: dict[str, Any], question: str, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_weight_config()
    frame = parse_query_frame(question).as_dict()
    semantic = float(row.get("semantic_score") or row.get("vector_score") or 0.0)
    lexical = float(row.get("lexical_score") or 0.0)
    relevance = max(0.0, min(1.0, .55 * semantic + .25 * lexical + .20 * float(row.get("entity_match_score") or 0.0)))
    condition, conflicts = condition_match_score(row, frame)
    directness = {"DIRECT": 1.0, "INDIRECT": .55, "INFERRED": .35, "MENTION_ONLY": .20}.get(str(row.get("directness") or ""), .30)
    study_quality = max(0.0, min(1.0, .45 * float(row.get("confidence") or 0.0) + .35 * float(int(row.get("page_number") or 0) > 0) + .20 * float(row.get("index_type") == "evidence")))
    method_quality = 1.0 if row.get("index_type") in {"evidence", "condition", "formula"} else .55
    material = material_process_match(row, frame, cfg)
    year_match = re.search(r"(?:19|20)\d{2}", str(row.get("year") or row.get("publication_year") or ""))
    year = int(year_match.group(0)) if year_match else 0
    recency = 1.0 if year >= 2015 else .9 if year >= 2000 else .75 if not year else .7
    version_quality = 1.0 if str(row.get("data_version") or "").startswith("stage") else .75
    temporal = .6 * version_quality + .4 * recency
    dims = {"relevance": relevance, "condition_match": condition, "evidence_directness": directness,
            "study_quality": study_quality, "method_quality": method_quality,
            "material_process_match": material, "temporal_version_quality": temporal}
    original_role = str(row.get("verified_evidence_role") or row.get("evidence_role") or "SUPPORTING_CONTEXT").upper()
    role = original_role
    tier = str(row.get("evidence_tier") or "TIER1_CORE_DIRECT").upper()
    study_type = str(row.get("study_type") or row.get("document_type") or "").casefold()
    title = str(row.get("title") or "").casefold()
    if "review" in study_type or re.search(r"\b(?:systematic |literature |brief )?review\b|state[- ]of[- ]the[- ]art", title):
        role = "REVIEW_BACKGROUND"
    elif role in {"DIRECT_SUPPORT", "DIRECT_COUNTER"} and material < .70:
        # Dynamic query mismatch changes only the role used for this ranking;
        # it never mutates the stored Evidence fact or its source annotation.
        role = "CONDITION_DEPENDENT"
    if tier == "TIER2_NEAR_DOMAIN" and role in {"DIRECT_SUPPORT", "DIRECT_COUNTER"}:
        role = "CONDITION_DEPENDENT"
    elif tier == "TIER3_FOUNDATIONAL_MECHANICS" and role in {"DIRECT_SUPPORT", "DIRECT_COUNTER", "CONDITION_DEPENDENT"}:
        role = "ALTERNATIVE_MECHANISM"
    role_coeff = float(cfg["role_coefficients"].get(role, cfg["role_coefficients"].get("REVIEW_BACKGROUND", .4)))
    tier_coeff = float(cfg["tier_coefficients"].get(tier, cfg["tier_coefficients"].get("OUT_OF_SCOPE", .3)))
    base = sum(float(cfg["dimension_weights"].get(key, 0.0)) * value for key, value in dims.items())
    scored = dict(row)
    scored.update({"relevance_score": round(relevance, 6), "condition_match_score": round(condition, 6),
                   "condition_conflicts": conflicts, "evidence_directness_score": round(directness, 6),
                   "study_quality_score": round(study_quality, 6), "method_quality_score": round(method_quality, 6),
                   "material_process_match_score": round(material, 6), "temporal_version_quality_score": round(temporal, 6),
                   "original_evidence_role": original_role, "verified_evidence_role": role,
                   "ranking_role_reason": (
                       "REVIEW_CANNOT_DISPLACE_PRIMARY_DIRECT_EVIDENCE"
                       if role == "REVIEW_BACKGROUND" and original_role != role
                       else "MATERIAL_PROCESS_MISMATCH_DYNAMIC_DOWNGRADE"
                       if role == "CONDITION_DEPENDENT" and original_role != role
                       else "ORIGINAL_ROLE_RETAINED"
                   ),
                   "evidence_role_coefficient": role_coeff, "evidence_weighted_score": round(base, 6),
                   "evidence_tier": tier, "evidence_tier_coefficient": tier_coeff,
                   "final_evidence_score": round(max(0.0, min(1.0, base * role_coeff * tier_coeff)), 6)})
    return scored


def _diversity_penalty(row: dict[str, Any], selected: list[dict[str, Any]], config: dict[str, Any]) -> float:
    penalties = config["diversity"]
    paper = row.get("paper_id")
    authors = str(row.get("authors") or row.get("author") or "").casefold()
    method = str(row.get("source_method") or row.get("measurement_method") or row.get("index_type") or "").casefold()
    condition = json.dumps(row.get("experimental_conditions") or {}, sort_keys=True, ensure_ascii=False)
    penalty = 0.0
    for other in selected:
        if paper and paper == other.get("paper_id"): penalty += penalties["same_paper_penalty"]
        if authors and authors == str(other.get("authors") or other.get("author") or "").casefold(): penalty += penalties["same_author_penalty"]
        if condition and condition == json.dumps(other.get("experimental_conditions") or {}, sort_keys=True, ensure_ascii=False): penalty += penalties["same_condition_penalty"]
        if method and method == str(other.get("source_method") or other.get("measurement_method") or other.get("index_type") or "").casefold(): penalty += penalties["same_method_penalty"]
    return penalty


def select_evidence_budget(pool: Iterable[dict[str, Any]], *, question: str, limit: int | None = None, config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    cfg = config or load_weight_config()
    total = int(limit or cfg["budget"]["total"])
    scored = [score_evidence(row, question, config=cfg) for row in pool]
    scored = [row for row in scored if str(row.get("verified_evidence_role") or "").upper() != "INSUFFICIENT"]
    scored.sort(key=lambda row: (
        -float(row.get("final_evidence_score") or 0),
        str(row.get("paper_id") or row.get("document_id") or ""),
        str(row.get("doc_id") or row.get("evidence_id") or ""),
    ))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    max_per_paper = max(1, int(total * float(cfg["budget"].get("max_single_paper_fraction", .25))))
    caps = cfg["budget"].get("role_caps", {})
    def add(row: dict[str, Any], *, enforce_role_cap: bool = True) -> bool:
        role = str(row.get("verified_evidence_role") or "SUPPORTING_CONTEXT").upper()
        if enforce_role_cap and counts[role] >= int(caps.get(role, total)): return False
        if role in {"DIRECT_COUNTER", "LIMITATION_EVIDENCE"} and counts["DIRECT_COUNTER"] + counts["LIMITATION_EVIDENCE"] >= 2:
            return False
        if role == "ALTERNATIVE_MECHANISM" and counts[role] >= int(caps.get(role, 1)):
            return False
        if sum(1 for item in selected if item.get("paper_id") == row.get("paper_id")) >= max_per_paper: return False
        adjusted = float(row.get("final_evidence_score") or 0) - _diversity_penalty(row, selected, cfg)
        row["diversity_adjusted_score"] = round(adjusted, 6)
        selected.append(row); counts[role] += 1
        return True
    # Fill explicit role quotas first, then use remaining capacity by quality.
    for role in ("DIRECT_SUPPORT", "CONDITION_DEPENDENT", "DIRECT_COUNTER", "LIMITATION_EVIDENCE", "ALTERNATIVE_MECHANISM"):
        for row in sorted(
            (r for r in scored if str(r.get("verified_evidence_role") or "").upper() == role),
            key=lambda r: (
                -float(r.get("final_evidence_score") or 0),
                str(r.get("paper_id") or r.get("document_id") or ""),
                str(r.get("doc_id") or r.get("evidence_id") or ""),
            ),
        ):
            if len(selected) >= total or counts[role] >= int(caps.get(role, total)): break
            add(row)
    while len(selected) < total:
        remaining = [row for row in scored if row not in selected]
        if not remaining:
            break
        remaining.sort(key=lambda row: (
            -(
                float(row.get("final_evidence_score") or 0)
                - _diversity_penalty(row, selected, cfg)
            ),
            str(row.get("paper_id") or row.get("document_id") or ""),
            str(row.get("doc_id") or row.get("evidence_id") or ""),
        ))
        if not any(add(row, enforce_role_cap=False) for row in remaining):
            break
    groups: dict[str, list[dict[str, Any]]] = {role: [] for role in set(caps) | {"SUPPORTING_CONTEXT", "REVIEW_BACKGROUND"}}
    for row in selected:
        groups.setdefault(str(row.get("verified_evidence_role") or "SUPPORTING_CONTEXT").upper(), []).append(row)
    groups["_selected"] = selected
    groups["_all_scored"] = scored
    return groups


def supplement_priority(gaps: Iterable[dict[str, Any]], *, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for gap in gaps:
        importance = float(gap.get("gap_importance", 0) or 0)
        expected = float(gap.get("expected_evidence_value", 0) or 0)
        scarcity = float(gap.get("current_scarcity", 0) or 0)
        coverage = float(gap.get("question_coverage", 0) or 0)
        score = importance * expected * scarcity * coverage
        tier = "P0" if gap.get("direct_match") else "P1" if gap.get("conditional_or_counter") else "P2" if gap.get("adjacent_or_review") else "P3"
        rows.append({**gap, "supplement_priority": round(score, 6), "priority_level": tier})
    return sorted(rows, key=lambda row: (row["supplement_priority"], row["priority_level"]), reverse=True)
