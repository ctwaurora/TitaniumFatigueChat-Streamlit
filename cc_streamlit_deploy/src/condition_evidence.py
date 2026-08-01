"""Deterministic condition extraction without an additional LLM pass."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


NOT_REPORTED = "NOT_REPORTED"
CONDITION_FIELDS = (
    "alloy_grade", "manufacturing_process", "build_orientation", "layer_thickness",
    "laser_power", "scan_speed", "energy_density", "relative_density", "heat_treatment",
    "hip", "surface_treatment", "defect_type", "defect_size", "defect_morphology",
    "defect_distance_to_surface", "porosity", "defect_location", "defect_distribution",
    "stress_amplitude", "maximum_stress", "stress_ratio_R", "frequency", "cycle_count",
    "fatigue_regime", "loading_mode", "environment", "temperature", "specimen_geometry",
    "surface_roughness", "ct_resolution", "sem", "ebsd", "crack_detection_method",
    "fatigue_life", "fatigue_limit", "crack_initiation_location", "da_dN", "delta_K",
    "fracture_mechanism", "mechanism_dominance_direction",
)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value or ""))
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return {}


def _first(text: str, pattern: str, flags: int = re.I) -> str:
    match = re.search(pattern, text, flags)
    return " ".join(match.group(0).split()) if match else NOT_REPORTED


def _terms(text: str, terms: tuple[tuple[str, str], ...]) -> list[str] | str:
    values = [name for name, pattern in terms if re.search(pattern, text, re.I)]
    return values or NOT_REPORTED


def extract_conditions(text: str, supplied: Any = None) -> dict[str, Any]:
    source = _mapping(supplied)
    conditions: dict[str, Any] = {key: NOT_REPORTED for key in CONDITION_FIELDS}
    aliases = {
        "material": "alloy_grade", "process": "manufacturing_process",
        "heat_treatment": "heat_treatment", "stress_ratio_R": "stress_ratio_R",
        "fatigue_regime": "fatigue_regime", "temperature": "temperature",
        "sample_geometry": "specimen_geometry", "surface_state": "surface_treatment",
    }
    for old, new in aliases.items():
        if source.get(old) not in (None, "", [], {}):
            conditions[new] = source[old]
    for key in CONDITION_FIELDS:
        if source.get(key) not in (None, "", [], {}):
            conditions[key] = source[key]

    detected = {
        "alloy_grade": _terms(text, (("Ti-6Al-4V", r"\bTi[-– ]?6Al[-– ]?4V\b|\bTi64\b|\bTC4\b"), ("Ti-6Al-4V ELI", r"Ti[-– ]?6Al[-– ]?4V\s*ELI"), ("TC21", r"\bTC21\b"), ("TC17", r"\bTC17\b"))),
        "manufacturing_process": _terms(text, (("L-PBF", r"\bL[- ]?PBF\b|laser powder bed fusion"), ("EBM", r"electron beam melt|\bEBM\b"), ("DED", r"directed energy deposition|\bDED\b|laser engineered net shaping|\bLENS\b"), ("forged", r"\bforg(?:ed|ing)\b"), ("welded", r"\bweld(?:ed|ing)?\b"))),
        "build_orientation": _first(text, r"(?:build|building|deposition)\s+(?:orientation|direction)\s*(?:of|was|=|:)?\s*(?:horizontal|vertical|diagonal|\d{1,3}\s*°|[XYZ])"),
        "layer_thickness": _first(text, r"(?:layer thickness|layer height)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:µm|μm|um|mm)"),
        "laser_power": _first(text, r"(?:laser power|power)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:W|kW)\b"),
        "scan_speed": _first(text, r"(?:scan(?:ning)? speed)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:mm/s|m/s)"),
        "energy_density": _first(text, r"(?:energy density)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:J/mm(?:3|³)|J\s*mm[-−]3)"),
        "relative_density": _first(text, r"(?:relative density|density)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*%"),
        "heat_treatment": _terms(text, (("annealed", r"\banneal(?:ed|ing)?\b"), ("stress-relieved", r"stress[- ]reliev"), ("solution-treated", r"solution treat"), ("aged", r"\bage(?:d|ing)\b"))),
        "hip": "HIP" if re.search(r"\bHIP\b|hot isostatic press", text, re.I) else NOT_REPORTED,
        "surface_treatment": _terms(text, (("as-built", r"\bas[- ]built\b"), ("machined", r"\bmachin(?:ed|ing)\b"), ("polished", r"\bpolish(?:ed|ing)\b"), ("shot peened", r"shot peen"), ("laser shock peened", r"laser shock peen"), ("grit blasted", r"grit[- ]blast"))),
        "defect_type": _terms(text, (("pore", r"\bpor(?:e|es|osity)\b"), ("lack of fusion", r"lack[- ]of[- ]fusion|\bLOF\b"), ("inclusion", r"\binclusion"), ("crack", r"\bcrack"))),
        "defect_size": _first(text, r"(?:defect|pore|inclusion)(?:\s+(?:diameter|size|length|area))?\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:µm|μm|um|mm|mm2|mm²)"),
        "defect_morphology": _terms(text, (("spherical", r"\bspherical\b"), ("irregular", r"\birregular\b"), ("lack-of-fusion shaped", r"lack[- ]of[- ]fusion"), ("elongated", r"\belongated\b"))),
        "defect_distance_to_surface": _first(text, r"(?:distance|depth)\s+(?:from|below)\s+(?:the\s+)?surface\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:µm|μm|um|mm)"),
        "porosity": _first(text, r"(?:porosity|pore volume fraction)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*%"),
        "defect_location": _terms(text, (("surface", r"surface defect|surface pore"), ("near-surface", r"near[- ]surface|subsurface"), ("internal", r"internal (?:defect|pore)|interior (?:defect|pore)"))),
        "defect_distribution": _terms(text, (("random", r"random(?:ly)? distribut"), ("clustered", r"cluster(?:ed|ing)?"), ("uniform", r"uniform(?:ly)? distribut"))),
        "stress_amplitude": _first(text, r"(?:stress amplitude|σa|sigma_a)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*MPa"),
        "maximum_stress": _first(text, r"(?:maximum|max\.?|peak)\s+stress\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*MPa"),
        "stress_ratio_R": _first(text, r"(?:stress|load)\s+ratio\s*(?:R\s*)?(?:of|was|=|:)?\s*-?\d+(?:\.\d+)?|\bR\s*=\s*-?\d+(?:\.\d+)?"),
        "frequency": _first(text, r"(?:frequency|tested at)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:Hz|kHz)"),
        "cycle_count": _first(text, r"\d+(?:\.\d+)?\s*(?:×\s*10\s*\^?\s*\d+|million|billion)?\s*cycles"),
        "fatigue_regime": _terms(text, (("LCF", r"\bLCF\b|low[- ]cycle fatigue"), ("HCF", r"\bHCF\b|high[- ]cycle fatigue"), ("VHCF", r"\bVHCF\b|very[- ]high[- ]cycle fatigue"), ("FCG", r"fatigue crack (?:growth|propagation)|\bFCG\b"))),
        "loading_mode": _terms(text, (("axial", r"\baxial\b"), ("bending", r"\bbending\b"), ("torsion", r"\btorsion(?:al)?\b"), ("multiaxial", r"\bmultiaxial\b"))),
        "environment": _terms(text, (("air", r"\bin air\b"), ("vacuum", r"\bvacuum\b"), ("saline", r"\bsaline\b|NaCl"), ("corrosive", r"corrosive environment|corrosion fatigue"), ("hydrogen", r"\bhydrogen\b"))),
        "temperature": _first(text, r"[-+]?\d+(?:\.\d+)?\s*(?:°\s*C|℃|K)\b"),
        "specimen_geometry": _terms(text, (("compact tension", r"compact tension|\bC\(T\)\b"), ("dog-bone", r"dog[- ]bone"), ("cylindrical", r"\bcylindrical\b"), ("notched", r"\bnotched\b"), ("three-point bending", r"three[- ]point bend"))),
        "surface_roughness": _first(text, r"(?:Ra|Rz|surface roughness)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:µm|μm|um)"),
        "ct_resolution": _first(text, r"(?:CT|computed tomography|voxel)(?:\s+resolution|\s+voxel size)?\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*(?:µm|μm|um)"),
        "sem": "SEM" if re.search(r"\bSEM\b|scanning electron microscop", text, re.I) else NOT_REPORTED,
        "ebsd": "EBSD" if re.search(r"\bEBSD\b|electron backscatter diffraction", text, re.I) else NOT_REPORTED,
        "crack_detection_method": _terms(text, (("replica", r"replica method"), ("DIC", r"\bDIC\b|digital image correlation"), ("in-situ CT", r"in[- ]situ (?:micro[- ]?)?CT"), ("SEM", r"\bSEM\b"))),
        "fatigue_life": _first(text, r"(?:fatigue life|Nf)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?(?:\s*×\s*10\s*\^?\s*\d+)?\s*(?:cycles)?"),
        "fatigue_limit": _first(text, r"(?:fatigue (?:limit|strength)|endurance limit)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*MPa"),
        "crack_initiation_location": _terms(text, (("surface", r"crack (?:initiated|initiation|originated) (?:at|from) (?:the )?surface"), ("near-surface pore", r"crack (?:initiated|originated).{0,80}(?:near[- ]surface|subsurface) pore"), ("internal pore", r"crack (?:initiated|originated).{0,80}internal pore"))),
        "da_dN": _first(text, r"da\s*/\s*dN\s*(?:of|was|=|:)?\s*[-+]?\d+(?:\.\d+)?(?:\s*[Ee][-+]?\d+)?(?:\s*mm/cycle)?"),
        "delta_K": _first(text, r"(?:ΔK|Delta\s*K|delta\s*K)\s*(?:of|was|=|:)?\s*\d+(?:\.\d+)?\s*MPa\s*(?:√m|m\^?0\.5)"),
        "fracture_mechanism": _terms(text, (("cleavage", r"\bcleavage\b"), ("striations", r"fatigue striation"), ("facet", r"\bfacet(?:s|ed)?\b"), ("fish-eye", r"fish[- ]eye"), ("microvoid coalescence", r"microvoid coalescence"))),
        "mechanism_dominance_direction": _terms(text, (("surface-dominated", r"surface[- ]dominat"), ("defect-dominated", r"defect[- ]dominat|pore[- ]dominat"), ("microstructure-dominated", r"microstructure[- ]dominat"))),
    }
    for key, value in detected.items():
        if conditions[key] == NOT_REPORTED and value != NOT_REPORTED:
            conditions[key] = value
    return conditions


def build_condition_records(base_dir: Path) -> dict[str, Any]:
    source = base_dir / "data/evidence/trusted_evidence.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fields = list(reader.fieldnames or []); rows = list(reader)
    records = []
    for row in rows:
        conditions = extract_conditions(str(row.get("original_text") or ""), row.get("experimental_conditions") or row.get("conditions"))
        row["experimental_conditions"] = json.dumps(conditions, ensure_ascii=False, sort_keys=True)
        record = {
            "condition_evidence_id": f"CE_{hashlib.sha256(str(row.get('evidence_id')).encode()).hexdigest()[:20].upper()}",
            "evidence_id": row.get("evidence_id"),
            "canonical_paper_id": row.get("canonical_paper_id") or row.get("paper_id"),
            "claim": row.get("claim"), "original_text": row.get("original_text"),
            "page_number": row.get("page_number"), "section": row.get("section"),
            "evidence_role": row.get("evidence_role") or row.get("support_or_counter"),
            "directness": row.get("directness"), "source_hash": row.get("source_hash"),
            **conditions,
            "independent_variables": row.get("independent_variables") or NOT_REPORTED,
            "dependent_variables": row.get("dependent_variables") or NOT_REPORTED,
            "control_variables": row.get("control_variables") or NOT_REPORTED,
            "result_direction": row.get("result_direction") or NOT_REPORTED,
            "mechanism": row.get("mechanism") or NOT_REPORTED,
            "uncertainty": row.get("uncertainty") or NOT_REPORTED,
            "extraction_basis": "EXPLICIT_TEXT_RULE_OR_EXISTING_STRUCTURED_FIELD",
        }
        records.append(record)
    temporary = source.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    temporary.replace(source)
    target = base_dir / "data/evidence/condition_evidence_records.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records: stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    coverage = {field: sum(record[field] != NOT_REPORTED for record in records) for field in CONDITION_FIELDS}
    return {"record_count": len(records), "path": str(target), "field_coverage": coverage, "missing_value_marker": NOT_REPORTED}
