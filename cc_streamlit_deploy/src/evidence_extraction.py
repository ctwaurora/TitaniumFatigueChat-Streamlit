"""
evidence_extraction.py — 条件化证据抽取模块 (论文级 v2)

将 literature_database.csv 中的每篇文献，
通过规则或 DeepSeek 抽取结构化字段，
生成 evidence_snippets.csv / variable_relation_dataset.csv / equation_parameter_dataset.csv。

v2 升级: 每条证据必须提取实验条件，并将机制结论绑定到具体条件。
如果条件缺失，标记 missing_condition_fields。

字段结构:
  [基础字段]
  evidence_id, paper_id, title, year, paper_type,
  original_sentence, extracted_claim,
  independent_variable, dependent_variable,
  moderating_variables, controlled_variables,
  mechanism, phenomenon, fatigue_indicator,

  [实验条件字段]
  material, process, build_orientation, heat_treatment,
  surface_state, surface_roughness_Ra, surface_roughness_Rz,
  stress_ratio_R, stress_amplitude, fatigue_type,
  frequency, temperature, environment, sample_geometry,
  pore_size, sqrt_area, distance_to_surface, pore_location,
  pore_aspect_ratio, porosity, defect_type,
  characterization_method, testing_method,
  condition_boundary, missing_condition_fields,

  [元数据字段]
  equation_or_model, parameter_values,
  evidence_type, evidence_strength, direct_or_indirect,
  is_conflicting, notes, creation_time
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"

# ── 证据类型枚举 ──────────────────────────────────────────────────────────

EVIDENCE_TYPES = [
    "direct_experimental_evidence",
    "indirect_mechanistic_evidence",
    "review_statement",
    "equation_parameter_evidence",
    "conflict_evidence",
    "hypothesis_candidate",
    "insufficient_evidence",
]

# ── 实验条件字段定义 ──────────────────────────────────────────────────────

EXPERIMENTAL_CONDITIONS = [
    "material",
    "process",
    "build_orientation",
    "heat_treatment",
    "surface_state",
    "surface_roughness_Ra",
    "surface_roughness_Rz",
    "stress_ratio_R",
    "stress_amplitude",
    "fatigue_type",
    "frequency",
    "temperature",
    "environment",
    "sample_geometry",
    "pore_size",
    "sqrt_area",
    "distance_to_surface",
    "pore_location",
    "pore_aspect_ratio",
    "porosity",
    "defect_type",
    "characterization_method",
    "testing_method",
    "condition_boundary",
    "missing_condition_fields",
]

# 条件字段中文名映射（用于展示）
CONDITION_FIELD_CN = {
    "material": "材料",
    "process": "工艺",
    "build_orientation": "成形方向",
    "heat_treatment": "热处理",
    "surface_state": "表面状态",
    "surface_roughness_Ra": "表面粗糙度 Ra",
    "surface_roughness_Rz": "表面粗糙度 Rz",
    "stress_ratio_R": "应力比 R",
    "stress_amplitude": "应力幅",
    "fatigue_type": "疲劳类型",
    "frequency": "频率",
    "temperature": "温度",
    "environment": "环境",
    "sample_geometry": "试样几何",
    "pore_size": "孔隙尺寸",
    "sqrt_area": "√area",
    "distance_to_surface": "距表面距离",
    "pore_location": "孔隙位置",
    "pore_aspect_ratio": "孔隙长宽比",
    "porosity": "孔隙率",
    "defect_type": "缺陷类型",
    "characterization_method": "表征方法",
    "testing_method": "测试方法",
    "condition_boundary": "条件边界",
    "missing_condition_fields": "缺失条件字段",
}

# ── 疲劳指标枚举 ──────────────────────────────────────────────────────────

FATIGUE_INDICATORS = [
    "Nf_life",
    "fatigue_limit",
    "Delta_K_threshold",
    "da_dN_crack_growth_rate",
    "Paris_C",
    "Paris_m",
    "crack_initiation_site",
    "crack_initiation_cycle_ratio",
    "S_N_curve",
    "strain_life_curve",
]

# ── 标准化变量映射 ────────────────────────────────────────────────────────

VARIABLE_CANONICAL = {
    "pore_size": ["pore size", "pore diameter", "defect size", "void size", "pore dimension"],
    "distance_to_surface": ["distance to surface", "pore depth", "defect depth", "surface distance", "subsurface depth"],
    "surface_roughness_Ra": ["surface roughness", "ra", "surface finish", "as-built surface", "roughness"],
    "porosity": ["porosity", "pore density", "void fraction", "relative density"],
    "stress_ratio_R": ["stress ratio", "R ratio", "load ratio", "R"],
    "stress_amplitude": ["stress amplitude", "stress level", "maximum stress", "applied stress"],
    "strain_amplitude": ["strain amplitude", "strain range", "total strain"],
    "build_orientation": ["build orientation", "build direction", "orientation angle"],
    "heat_treatment_temperature": ["heat treatment temperature", "annealing temperature", "aging temperature"],
    "HIP_pressure": ["HIP pressure", "hot isostatic pressing pressure"],
}

VARIABLE_CANONICAL_REVERSE = {}
for canonical, aliases in VARIABLE_CANONICAL.items():
    for alias in aliases:
        VARIABLE_CANONICAL_REVERSE[alias.lower()] = canonical


def normalize_variable(var_name: str) -> str:
    """将变量名标准化为规范形式。"""
    var_lower = var_name.strip().lower()
    if var_lower in VARIABLE_CANONICAL_REVERSE:
        return VARIABLE_CANONICAL_REVERSE[var_lower]
    return var_name.strip()


def generate_evidence_id(paper_id: str, idx: int) -> str:
    return f"EV_{paper_id}_{idx:04d}"


def extract_experimental_conditions(paper: Dict[str, str]) -> Dict[str, str]:
    """从文献数据库字段中提取实验条件。

    读取 paper dict 中的规范化字段，返回条件字典。
    未找到的字段标记为空字符串。
    """
    field_map = {
        "material": "material",
        "process": "process",
        "build_orientation": "build_orientation",
        "heat_treatment": "heat_treatment",
        "surface_state": "surface_state",
        "surface_roughness_Ra": "surface_roughness_Ra",
        "surface_roughness_Rz": "surface_roughness_Rz",
        "stress_ratio_R": "stress_ratio_R",
        "stress_amplitude": "stress_amplitude",
        "fatigue_type": "fatigue_type",
        "frequency": "frequency",
        "temperature": "temperature",
        "environment": "environment",
        "sample_geometry": "sample_geometry",
        "pore_size": "pore_size",
        "sqrt_area": "sqrt_area",
        "distance_to_surface": "distance_to_surface",
        "pore_location": "pore_location",
        "pore_aspect_ratio": "pore_aspect_ratio",
        "porosity": "porosity",
        "defect_type": "defect_type",
        "characterization_method": "characterization_method",
        "testing_method": "testing_method",
    }
    conditions = {}
    for field_key, paper_key in field_map.items():
        val = paper.get(paper_key, "").strip()
        conditions[field_key] = val

    # 从 notes / main_conclusion 尝试提取更多条件
    conclusion = (paper.get("main_conclusion", "") or "") + " " + (paper.get("notes", "") or "")
    conclusion_lower = conclusion.lower()

    # 如果为空，尝试从文本推断
    if not conditions["material"]:
        if "ti-6al-4v" in conclusion_lower or "ti64" in conclusion_lower or "tc4" in conclusion_lower:
            conditions["material"] = "Ti-6Al-4V"
    if not conditions["process"]:
        if "l-pbf" in conclusion_lower or "slm" in conclusion_lower or "dmls" in conclusion_lower:
            conditions["process"] = "L-PBF"
        elif "ebm" in conclusion_lower:
            conditions["process"] = "EBM"
    if not conditions["surface_state"]:
        if "as-built" in conclusion_lower:
            conditions["surface_state"] = "as-built"
        elif "polish" in conclusion_lower or "machin" in conclusion_lower:
            conditions["surface_state"] = "polished/machined"
    if not conditions["heat_treatment"]:
        if "hip" in conclusion_lower:
            conditions["heat_treatment"] = "HIP"
        elif "anneal" in conclusion_lower or "stress-relief" in conclusion_lower:
            conditions["heat_treatment"] = "stress-relieved"
    if not conditions["fatigue_type"]:
        if "vhcf" in conclusion_lower:
            conditions["fatigue_type"] = "VHCF"
        elif "hcf" in conclusion_lower:
            conditions["fatigue_type"] = "HCF"
        elif "fcgr" in conclusion_lower or "crack growth" in conclusion_lower:
            conditions["fatigue_type"] = "FCGR"

    return conditions


def identify_missing_conditions(paper: Dict[str, str]) -> str:
    """检查文献中缺失的关键实验条件，返回缺失字段的列表字符串。"""
    key_fields = [
        "material", "process", "surface_state", "heat_treatment",
        "stress_ratio_R", "fatigue_type", "surface_roughness_Ra",
    ]
    missing = []
    for field in key_fields:
        val = paper.get(field, "").strip()
        if not val:
            missing.append(field)
    return "; ".join(missing) if missing else ""


def build_condition_boundary(mechanism: str, conditions: Dict[str, str]) -> str:
    """根据机制和条件构建条件边界描述。"""
    if not mechanism:
        return ""

    # 条件-机制边界描述模板
    boundary_map = {
        "crack_initiation_from_defects": (
            f"实验条件: {conditions.get('material', 'Ti-6Al-4V')}, "
            f"{conditions.get('process', 'L-PBF')}, "
            f"表面状态: {conditions.get('surface_state', '未指定')}, "
            f"热处理: {conditions.get('heat_treatment', '未指定')}, "
            f"应力比 R: {conditions.get('stress_ratio_R', '未指定')}, "
            f"疲劳类型: {conditions.get('fatigue_type', '未指定')}"
        ),
        "surface_roughness_induced_cracking": (
            f"该机制适用于表面粗糙度较高的条件，如 as-built 表面。"
            f"在 polished/machined 表面中，该机制可能被内部孔隙起裂机制取代。"
            f"材料: {conditions.get('material', 'Ti-6Al-4V')}, "
            f"表面粗糙度: Ra={conditions.get('surface_roughness_Ra', '未指定')}, "
            f"Rz={conditions.get('surface_roughness_Rz', '未指定')}"
        ),
    }

    for mech_key, boundary_desc in boundary_map.items():
        if mech_key in mechanism:
            return boundary_desc
    return f"条件: {conditions.get('surface_state', '未指定表面状态')}, {conditions.get('heat_treatment', '未指定热处理')}"


def format_evidence_with_conditions(claim: str, conditions: Dict[str, str], mechanism: str) -> str:
    """将机制结论与条件绑定，生成条件化描述。

    示例:
        "surface roughness reduces fatigue life."
        → "在 as-built L-PBF Ti-6Al-4V、较高 Ra/Rz、HCF、R=0.1 条件下，
           surface roughness 可能通过表面缺口效应主导 crack initiation，从而降低 Nf。"
    """
    # 收集非空条件描述
    cond_parts = []
    if conditions.get("material"):
        cond_parts.append(conditions["material"])
    if conditions.get("process"):
        cond_parts.append(conditions["process"])
    if conditions.get("surface_state"):
        cond_parts.append(conditions["surface_state"])
    if conditions.get("heat_treatment"):
        cond_parts.append(conditions["heat_treatment"])
    if conditions.get("surface_roughness_Ra") or conditions.get("surface_roughness_Rz"):
        ra = conditions.get("surface_roughness_Ra", "")
        rz = conditions.get("surface_roughness_Rz", "")
        if ra and rz:
            cond_parts.append(f"Ra={ra}/Rz={rz}")
        elif ra:
            cond_parts.append(f"Ra={ra}")
        elif rz:
            cond_parts.append(f"Rz={rz}")
    if conditions.get("fatigue_type"):
        cond_parts.append(conditions["fatigue_type"])
    if conditions.get("stress_ratio_R"):
        cond_parts.append(f"R={conditions['stress_ratio_R']}")
    if conditions.get("temperature"):
        cond_parts.append(f"{conditions['temperature']}")

    cond_str = "、".join(cond_parts) if cond_parts else "常规实验条件"

    mech_descriptions = {
        "stress_concentration_at_pores": "通过孔隙边缘的局部应力集中效应",
        "crack_initiation_from_defects": "通过缺陷处应力集中主导裂纹起裂",
        "crack_propagation": "通过裂纹扩展机制",
        "surface_roughness_induced_cracking": "通过表面缺口效应主导裂纹起裂",
        "HIP_defect_closure": "通过 HIP 闭合孔隙缺陷",
        "microstructural_barrier": "通过微观组织屏障效应",
        "short_crack_growth": "通过短裂纹扩展机制",
        "": ""
    }
    mech_desc = mech_descriptions.get(mechanism, f"通过{mechanism}机制")

    if mech_desc:
        return f"在 {cond_str} 条件下，{claim}，{mech_desc}。"
    return f"在 {cond_str} 条件下，{claim}。"


def make_condition_record(conditions: Dict[str, str], evidence_id: str, paper_id: str) -> Dict[str, str]:
    """构建条件字段字典，用于写入 CSV。"""
    record = {}
    for field in EXPERIMENTAL_CONDITIONS:
        if field in ("missing_condition_fields", "condition_boundary"):
            continue  # 由外部设置
        record[field] = conditions.get(field, "")
    return record


def load_literature_database() -> List[Dict[str, str]]:
    """加载文献数据库。"""
    path = DATA_DIR / "literature_database.csv"
    if not path.exists():
        print(f"[WARNING] literature_database.csv not found at {path}")
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[INFO] Loaded {len(rows)} papers from literature_database.csv")
    return rows


def load_existing_evidence() -> Dict[str, Dict]:
    """加载已有证据片段，避免重复生成。"""
    path = DATA_DIR / "evidence_snippets.csv"
    if not path.exists():
        return {}
    existing = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row.get("evidence_id", "")
                if eid:
                    existing[eid] = row
    except Exception:
        pass
    print(f"[INFO] Loaded {len(existing)} existing evidence snippets")
    return existing


def extract_evidence_from_paper(
    paper: Dict[str, str],
    llm_client=None
) -> List[Dict[str, Any]]:
    """
    从单篇文献抽取证据片段。
    使用 DeepSeek（如有），否则使用基于规则的抽取。

    返回 evidence record 列表。
    """
    # Stage 1 hard safety gate.  The legacy implementation below inferred
    # claims from titles/folder names and labeled them as direct evidence.
    # It is intentionally unreachable until Stage 2 replaces it with
    # page-bound extraction.
    return []

    title = paper.get("title", "")
    paper_id = paper.get("paper_id", "")
    year = paper.get("year", "")
    evidence_type = paper.get("evidence_type", "").strip().lower()
    main_conclusion = paper.get("main_conclusion", "")
    notes = paper.get("notes", "")

    # ── 判断证据类型 ──
    paper_type = paper.get("paper_type_primary", "")
    if evidence_type == "review" or paper_type == "review":
        primary_ev_type = "review_statement"
    else:
        primary_ev_type = "direct_experimental_evidence"

    # ── 提取实验条件 ──
    experimental_conditions = extract_experimental_conditions(paper)

    # ── 判断缺失条件 ──
    missing_condition_fields = identify_missing_conditions(paper)

    # ── 从 main_conclusion 和 notes 结构化提取 ──
    evidence_list = []
    storage_folder = paper.get("storage_folder", "")
    evidence_idx = [0]  # mutable counter

    def next_ev_id():
        eid = evidence_idx[0]
        evidence_idx[0] += 1
        return generate_evidence_id(paper_id, eid)

    # 1. 从标题自动生成结构化证据（即使 main_conclusion 为空）
    title_lower = title.lower()
    folder_lower = storage_folder.lower()
    combined_title_folder = f"{title_lower} {folder_lower}"

    # ── 按分类从标题生成有针对性的证据 ──
    evidence_rules = []

    if "pore" in folder_lower or "pore" in title_lower:
        evidence_rules.append({
            "iv": "pore_size",
            "dv": "Nf_life",
            "mechanism": "stress_concentration_at_pores",
            "ev_type": "direct_experimental_evidence",
            "strength": "moderate",
            "claim": f"Paper investigates relationship between pore/defect characteristics and fatigue life of L-PBF Ti-6Al-4V",
            "orig": title,
        })
        evidence_rules.append({
            "iv": "pore_location",
            "dv": "crack_initiation_site",
            "mechanism": "crack_initiation_from_defects",
            "ev_type": "direct_experimental_evidence",
            "strength": "moderate",
            "claim": f"Paper examines effect of pore location/distribution on crack initiation",
            "orig": title,
        })

    if "surface" in folder_lower or "roughness" in folder_lower:
        evidence_rules.append({
            "iv": "surface_roughness_Ra",
            "dv": "Nf_life",
            "mechanism": "surface_roughness_induced_cracking",
            "ev_type": "direct_experimental_evidence",
            "strength": "moderate",
            "claim": f"Paper investigates effect of surface roughness on fatigue performance",
            "orig": title,
        })

    if "hip" in folder_lower or "heat" in folder_lower or "treatment" in folder_lower:
        evidence_rules.append({
            "iv": "HIP_pressure",
            "dv": "porosity",
            "mechanism": "HIP_defect_closure",
            "ev_type": "direct_experimental_evidence",
            "strength": "moderate",
            "claim": f"Paper studies effect of HIP/heat treatment on defects and fatigue behavior",
            "orig": title,
        })

    if "fcgr" in folder_lower or "paris" in folder_lower or "crack" in title_lower:
        evidence_rules.append({
            "iv": "Delta_K",
            "dv": "da_dN_crack_growth_rate",
            "mechanism": "crack_propagation",
            "ev_type": "equation_parameter_evidence",
            "strength": "moderate",
            "claim": f"Paper reports fatigue crack growth rate data and Paris law parameters",
            "orig": title,
        })
        evidence_rules.append({
            "iv": "stress_ratio_R",
            "dv": "da_dN_crack_growth_rate",
            "mechanism": "crack_propagation",
            "ev_type": "equation_parameter_evidence",
            "strength": "moderate",
            "claim": f"Paper examines effect of stress ratio on crack growth behavior",
            "orig": title,
        })

    if "review" in folder_lower:
        evidence_rules.append({
            "iv": "",
            "dv": "",
            "mechanism": "",
            "ev_type": "review_statement",
            "strength": "low",
            "claim": f"Review paper summarizing fatigue behavior of L-PBF Ti-6Al-4V",
            "orig": title,
        })

    for rule in evidence_rules:
        # 构建条件化 claim
        cond_claim = format_evidence_with_conditions(
            rule["claim"], experimental_conditions, rule["mechanism"]
        )
        # 条件边界
        cond_boundary = build_condition_boundary(rule["mechanism"], experimental_conditions)
        # 缺失条件
        missing_conds = missing_condition_fields

        evidence_list.append({
            "evidence_id": next_ev_id(),
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "paper_type": paper.get("paper_type_primary", ""),
            "original_sentence": rule["orig"][:500],
            "extracted_claim": cond_claim,
            "independent_variable": rule["iv"],
            "dependent_variable": rule["dv"],
            "moderating_variables": "",
            "controlled_variables": f"material={paper.get('material', '')}; process={paper.get('process', '')}",
            "mechanism": rule["mechanism"],
            "phenomenon": "",
            "fatigue_indicator": rule["dv"],
            # 实验条件
            "material": experimental_conditions.get("material", ""),
            "process": experimental_conditions.get("process", ""),
            "build_orientation": experimental_conditions.get("build_orientation", ""),
            "heat_treatment": experimental_conditions.get("heat_treatment", ""),
            "surface_state": experimental_conditions.get("surface_state", ""),
            "surface_roughness_Ra": experimental_conditions.get("surface_roughness_Ra", ""),
            "surface_roughness_Rz": experimental_conditions.get("surface_roughness_Rz", ""),
            "stress_ratio_R": experimental_conditions.get("stress_ratio_R", ""),
            "stress_amplitude": experimental_conditions.get("stress_amplitude", ""),
            "fatigue_type": experimental_conditions.get("fatigue_type", ""),
            "frequency": experimental_conditions.get("frequency", ""),
            "temperature": experimental_conditions.get("temperature", ""),
            "environment": experimental_conditions.get("environment", ""),
            "sample_geometry": experimental_conditions.get("sample_geometry", ""),
            "pore_size": experimental_conditions.get("pore_size", ""),
            "sqrt_area": experimental_conditions.get("sqrt_area", ""),
            "distance_to_surface": experimental_conditions.get("distance_to_surface", ""),
            "pore_location": experimental_conditions.get("pore_location", ""),
            "pore_aspect_ratio": experimental_conditions.get("pore_aspect_ratio", ""),
            "porosity": experimental_conditions.get("porosity", ""),
            "defect_type": experimental_conditions.get("defect_type", ""),
            "characterization_method": experimental_conditions.get("characterization_method", ""),
            "testing_method": experimental_conditions.get("testing_method", ""),
            "condition_boundary": cond_boundary,
            "missing_condition_fields": missing_conds,
            # 元数据
            "equation_or_model": "",
            "parameter_values": "",
            "evidence_type": rule["ev_type"],
            "evidence_strength": rule["strength"],
            "direct_or_indirect": "direct" if rule["ev_type"] == "direct_experimental_evidence" else "indirect",
            "is_conflicting": "",
            "notes": f"Auto-extracted from title/folder classification ({storage_folder})",
            "creation_time": datetime.now().isoformat(),
            "source_field": "title_classification",
            "usable_for_validation": "True" if rule["strength"] in ("high", "moderate") and rule["iv"] else "False",
        })

    # ⚠️ NOTE: The code below (Paris law, ΔKth, variable relation extraction)
    # is structurally inside _condition_fields() due to indentation mismatch.
    # It is dead code and never executes. To re-enable, it must be moved
    # inside this function with proper indentation. For now, this function
    # returns the rules-based evidence only.
    return evidence_list


# ── _condition_fields helper (module-level, used by save functions) ──

def _condition_fields(conditions: Dict[str, str], boundary: str = "", missing: str = "") -> Dict[str, str]:
    """生成实验条件字段字典，避免重复写 25 个字段。"""
    d = {f: conditions.get(f, "") for f in EXPERIMENTAL_CONDITIONS if f not in ("missing_condition_fields", "condition_boundary")}
    d["condition_boundary"] = boundary
    d["missing_condition_fields"] = missing
    return d


    # 2. 如果有方程参数，生成参数证据
    paris_c = paper.get("Paris_C", "").strip()
    paris_m = paper.get("Paris_m", "").strip()
    if paris_c or paris_m:
        param_text = f"Paris law parameters: C={paris_c}, m={paris_m}"
        cond_claim_p = format_evidence_with_conditions(
            f"Paper reports Paris law C={paris_c or 'N/A'}, m={paris_m or 'N/A'}",
            experimental_conditions, "crack_propagation"
        )
        evidence_list.append({
            "evidence_id": generate_evidence_id(paper_id, 2),
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "paper_type": paper.get("paper_type_primary", ""),
            "original_sentence": param_text,
            "extracted_claim": cond_claim_p,
            "independent_variable": "Delta_K",
            "dependent_variable": "da_dN",
            "moderating_variables": "",
            "controlled_variables": f"R={experimental_conditions.get('stress_ratio_R', '')}",
            "mechanism": "crack growth governed by Paris law",
            "phenomenon": "Paris law crack propagation",
            "fatigue_indicator": "da_dN_crack_growth_rate",
            # 实验条件
            **{f: experimental_conditions.get(f, "") for f in EXPERIMENTAL_CONDITIONS if f not in ("missing_condition_fields", "condition_boundary")},
            "condition_boundary": f"R={experimental_conditions.get('stress_ratio_R', '未指定')}, FCGR test",
            "missing_condition_fields": missing_condition_fields,
            "equation_or_model": "Paris law: da/dN = C(ΔK)^m",
            "parameter_values": json.dumps({"C": paris_c, "m": paris_m}),
            "experimental_method": "FCGR test",
            "characterization_method": "",
            "evidence_type": "equation_parameter_evidence",
            "evidence_strength": "high" if (paris_c and paris_m) else "moderate",
            "direct_or_indirect": "direct",
            "is_conflicting": "",
            "notes": f"Extracted from literature database Paris fields",
            "creation_time": datetime.now().isoformat(),
            "source_field": "Paris_C/m",
            "usable_for_validation": "True",
        })

    # 4. 如果有 ΔKth
    dkth = paper.get("Delta_Kth", "").strip()
    if dkth:
        cond_claim_dk = format_evidence_with_conditions(
            f"Fatigue crack growth threshold ΔKth = {dkth}",
            experimental_conditions, "fatigue threshold behavior"
        )
        evidence_list.append({
            "evidence_id": generate_evidence_id(paper_id, 3),
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "paper_type": paper.get("paper_type_primary", ""),
            "original_sentence": f"Fatigue threshold ΔKth = {dkth}",
            "extracted_claim": cond_claim_dk,
            "independent_variable": "stress_intensity_factor_range",
            "dependent_variable": "Delta_K_threshold",
            "moderating_variables": "stress_ratio_R",
            "controlled_variables": f"R={experimental_conditions.get('stress_ratio_R', '')}",
            "mechanism": "fatigue threshold behavior",
            "phenomenon": "fatigue threshold",
            "fatigue_indicator": "Delta_K_threshold",
            **_condition_fields(experimental_conditions,
                boundary=f"R={experimental_conditions.get('stress_ratio_R', '未指定')}",
                missing=missing_condition_fields),
            "equation_or_model": "Murakami √area model or Kitagawa-Takahashi",
            "parameter_values": json.dumps({"Delta_Kth": dkth}),
            "experimental_method": "",
            "characterization_method": "",
            "evidence_type": "equation_parameter_evidence",
            "evidence_strength": "moderate",
            "direct_or_indirect": "direct",
            "is_conflicting": "",
            "notes": f"Threshold extracted from database",
            "creation_time": datetime.now().isoformat(),
            "source_field": "Delta_Kth",
            "usable_for_validation": "True",
        })

    # 5. 如果有关键变量填写，生成变量关系证据
    has_variable_data = any([
        paper.get("pore_size", "").strip(),
        paper.get("distance_to_surface", "").strip(),
        paper.get("surface_roughness_Ra", "").strip(),
        paper.get("porosity", "").strip(),
        paper.get("fatigue_limit", "").strip(),
    ])
    if has_variable_data:
        var_relations = []
        if paper.get("pore_size", "").strip() and paper.get("Nf", "").strip():
            var_relations.append(("pore_size", "Nf_life"))
        if paper.get("distance_to_surface", "").strip() and paper.get("crack_initiation_site", "").strip():
            var_relations.append(("distance_to_surface", "crack_initiation_site"))
        if paper.get("surface_roughness_Ra", "").strip() and paper.get("Nf", "").strip():
            var_relations.append(("surface_roughness_Ra", "Nf_life"))
        if paper.get("porosity", "").strip() and paper.get("Nf", "").strip():
            var_relations.append(("porosity", "Nf_life"))

        for iv, dv in var_relations:
            idx = 4 + var_relations.index((iv, dv))
            cond_claim_vr = format_evidence_with_conditions(
                f"{iv} influences {dv}",
                experimental_conditions, "relationship exists between these variables"
            )
            evidence_list.append({
                "evidence_id": generate_evidence_id(paper_id, idx),
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "paper_type": paper.get("paper_type_primary", ""),
                "original_sentence": f"Paper reports {iv} and {dv} data",
                "extracted_claim": cond_claim_vr,
                "independent_variable": iv,
                "dependent_variable": dv,
                "moderating_variables": "",
                "controlled_variables": "",
                "mechanism": "relationship exists between these variables",
                "phenomenon": "",
                "fatigue_indicator": dv,
                **_condition_fields(experimental_conditions,
                    boundary=f"{iv}→{dv} relation under specified conditions",
                    missing=missing_condition_fields),
                "equation_or_model": "",
                "parameter_values": "",
                "experimental_method": paper.get("evidence_type", ""),
                "characterization_method": "SEM" if paper.get("SEM_available", "").lower() == "true" else "",
                "evidence_type": "direct_experimental_evidence",
                "evidence_strength": "moderate",
                "direct_or_indirect": "direct",
                "is_conflicting": "",
                "notes": f"Extracted from database variable fields",
                "creation_time": datetime.now().isoformat(),
                "source_field": f"{iv}_{dv}",
                "usable_for_validation": "True",
            })

    return evidence_list


def _extract_from_text(
    evidence_id: str,
    paper_id: str,
    title: str,
    year: str,
    paper_type: str,
    source_text: str,
    default_ev_type: str,
    paper: Dict[str, str],
) -> Dict[str, Any]:
    """从文本段落中提取证据字段。"""
    text_lower = source_text.lower()

    # ── 识别变量 ──
    iv = _extract_variable(text_lower, paper)
    dv = _extract_indicator(text_lower, paper)

    # ── 识别机制关键词 ──
    mechanism = _extract_mechanism(text_lower)

    # ── 识别方程 ──
    equation = _extract_equation(text_lower)

    # ── 判断证据强度 ──
    strength, ev_type = _classify_evidence(source_text, default_ev_type, paper)

    return {
        "evidence_id": evidence_id,
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "paper_type": paper_type,
        "original_sentence": source_text[:500],
        "extracted_claim": source_text[:300],
        "independent_variable": iv,
        "dependent_variable": dv,
        "moderating_variables": "",
        "controlled_variables": f"material={paper.get('material', '')}; process={paper.get('process', '')}",
        "mechanism": mechanism,
        "phenomenon": "",
        "fatigue_indicator": dv if dv else "",
        **_condition_fields(extract_experimental_conditions(paper),
            boundary=f"under {paper.get('surface_state', 'general')} conditions",
            missing=identify_missing_conditions(paper)),
        "equation_or_model": equation,
        "parameter_values": "",
        "experimental_method": paper.get("evidence_type", ""),
        "characterization_method": "SEM" if paper.get("SEM_available", "").lower() == "true" else "",
        "evidence_type": ev_type,
        "evidence_strength": strength,
        "direct_or_indirect": "direct" if ev_type == "direct_experimental_evidence" else "indirect",
        "is_conflicting": "",
        "notes": "Auto-extracted from literature database",
        "creation_time": datetime.now().isoformat(),
        "source_field": "main_conclusion",
        "usable_for_validation": "True" if strength in ("high", "moderate") else "False",
    }


def _extract_variable(text_lower: str, paper: Dict[str, str]) -> str:
    """从文本识别自变量。"""
    # 检查文献库中已有的结构化变量
    variable_map = [
        ("pore_size", ["pore size", "pore", "defect size", "void"]),
        ("distance_to_surface", ["distance to surface", "pore depth", "subsurface"]),
        ("surface_roughness_Ra", ["surface roughness", "ra", "roughness", "surface finish"]),
        ("porosity", ["porosity", "relative density", "pore density"]),
        ("stress_ratio_R", ["stress ratio", "r ratio", "load ratio"]),
        ("stress_amplitude", ["stress amplitude", "stress level", "maximum stress"]),
        ("build_orientation", ["build orientation", "build direction", "orientation"]),
        ("heat_treatment", ["heat treatment", "annealing", "aging", "hip"]),
    ]
    for var, keywords in variable_map:
        if any(kw in text_lower for kw in keywords):
            return var
    # 回退：检查paper字段
    for field in ["pore_size", "surface_roughness_Ra", "distance_to_surface"]:
        if paper.get(field, "").strip():
            return field
    return ""


def _extract_indicator(text_lower: str, paper: Dict[str, str]) -> str:
    """从文本识别疲劳指标。"""
    indicator_map = [
        ("Nf_life", ["fatigue life", "nf", "cycles to failure", "life"]),
        ("fatigue_limit", ["fatigue limit", "endurance limit", "fatigue strength"]),
        ("da_dN_crack_growth_rate", ["crack growth", "da/dn", "dadn", "crack propagation"]),
        ("crack_initiation_site", ["crack initiation", "initiation site", "crack origin"]),
        ("Delta_K_threshold", ["threshold", "delta kth", "Δkth"]),
    ]
    for ind, keywords in indicator_map:
        if any(kw in text_lower for kw in keywords):
            return ind
    # 回退
    for field in ["Nf", "fatigue_limit", "Delta_Kth", "crack_initiation_site"]:
        if paper.get(field, "").strip():
            field_map = {
                "Nf": "Nf_life",
                "fatigue_limit": "fatigue_limit",
                "Delta_Kth": "Delta_K_threshold",
                "crack_initiation_site": "crack_initiation_site",
            }
            return field_map.get(field, field)
    return ""


def _extract_mechanism(text_lower: str) -> str:
    """从文本识别损伤机制。"""
    mechanism_map = [
        ("crack_initiation_from_defects", ["crack initiation", "crack nucleat", "initiate"]),
        ("crack_propagation", ["crack growth", "crack propagation", "fatigue crack"]),
        ("stress_concentration_at_pores", ["stress concentration", "stress field around pore"]),
        ("surface_roughness_induced_cracking", ["roughness-induced", "surface notch", "as-built surface"]),
        ("HIP_defect_closure", ["hip", "hot isostatic pressing", "pore closure"]),
        ("microstructural_barrier", ["microstructural barrier", "grain boundary", "alpha lath"]),
        ("short_crack_growth", ["short crack", "small crack", "microstructurally short"]),
    ]
    for mech, keywords in mechanism_map:
        if any(kw in text_lower for kw in keywords):
            return mech
    return ""


def _extract_equation(text_lower: str) -> str:
    """从文本识别疲劳方程。"""
    equation_map = [
        ("Paris law: da/dN = C(ΔK)^m", ["paris law", "da/dn", "c(δk)", "paris"]),
        ("Coffin-Manson: Δε_p/2 = ε'_f(2N_f)^c", ["coffin", "manson", "strain-life"]),
        ("Basquin: Δσ/2 = σ'_f(2N_f)^b", ["basquin", "stress-life", "s-n curve"]),
        ("Murakami √area model", ["murakami", "√area", "sqrt area", "area"]),
        ("Kitagawa-Takahashi diagram", ["kitagawa", "takahashi", "kitagawa-takahashi"]),
        ("Walker equation", ["walker", "walker equation"]),
    ]
    for eq, keywords in equation_map:
        if any(kw in text_lower for kw in keywords):
            return eq
    return ""


def _classify_evidence(text: str, default_type: str, paper: Dict[str, str]) -> Tuple[str, str]:
    """判断证据强度和类型。"""
    text_lower = text.lower()
    paper_type = paper.get("paper_type_primary", "").lower()

    # 类型判断
    if paper_type == "review":
        ev_type = "review_statement"
    elif any(w in text_lower for w in ["conflict", "contradict", "disagree", "inconsistent"]):
        ev_type = "conflict_evidence"
    elif any(w in text_lower for w in ["paris", "da/dn", "coffin", "basquin", "murakami"]):
        ev_type = "equation_parameter_evidence"
    elif any(w in text_lower for w in ["hypothesis", "propose", "suggest", "may be"]):
        ev_type = "hypothesis_candidate"
    else:
        ev_type = default_type

    # 强度判断
    if text_lower.startswith("review") or paper_type == "review":
        strength = "low"
    elif any(w in text_lower for w in ["experiment", "test", "measure", "observation"]):
        strength = "high"
    elif any(w in text_lower for w in ["result", "show", "find", "indicate", "demonstrate"]):
        strength = "moderate"
    else:
        strength = "low"

    return strength, ev_type


def compute_evidence_strength(evidence_list: List[Dict]) -> Dict[str, int]:
    """统计证据库强度分布。"""
    counts: Dict[str, int] = {
        "total": len(evidence_list),
        "direct_experimental_evidence": 0,
        "indirect_mechanistic_evidence": 0,
        "review_statement": 0,
        "equation_parameter_evidence": 0,
        "conflict_evidence": 0,
        "hypothesis_candidate": 0,
        "insufficient_evidence": 0,
        "high_strength": 0,
        "moderate_strength": 0,
        "low_strength": 0,
    }
    for ev in evidence_list:
        ev_type = ev.get("evidence_type", "")
        if ev_type in counts:
            counts[ev_type] += 1
        strength = ev.get("evidence_strength", "")
        key = f"{strength}_strength"
        if key in counts:
            counts[key] += 1
    return counts


def save_evidence_snippets(evidence_list: List[Dict], append: bool = False):
    """保存证据片段到 CSV。"""
    path = DATA_DIR / "evidence_snippets.csv"
    fieldnames = [
        "evidence_id", "paper_id", "title", "year", "paper_type",
        "original_sentence", "extracted_claim",
        "independent_variable", "dependent_variable",
        "moderating_variables", "controlled_variables",
        "mechanism", "fatigue_indicator",
        "equation_or_model", "parameter_values",
        "experimental_method", "characterization_method",
        "evidence_type", "evidence_strength",
        "direct_or_indirect", "is_conflicting", "notes",
        "phenomenon",
        # 实验条件字段
    "material", "process", "build_orientation", "heat_treatment",
    "surface_state", "surface_roughness_Ra", "surface_roughness_Rz",
    "stress_ratio_R", "stress_amplitude", "fatigue_type",
    "frequency", "temperature", "environment", "sample_geometry",
    "pore_size", "sqrt_area", "distance_to_surface", "pore_location",
    "pore_aspect_ratio", "porosity", "defect_type",
    "testing_method",
    "condition_boundary", "missing_condition_fields",
    # 元数据字段
    "creation_time", "source_field", "usable_for_validation",
    ]

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not append:
            writer.writeheader()
        for ev in evidence_list:
            writer.writerow(ev)

    print(f"[INFO] Saved {len(evidence_list)} evidence snippets to {path}")


def save_variable_relation_dataset(evidence_list: List[Dict]):
    """从证据片段提取变量关系并保存（v2: 含结构化条件字段）。"""
    path = DATA_DIR / "variable_relation_dataset.csv"
    relations = []

    for ev in evidence_list:
        iv = ev.get("independent_variable", "").strip()
        dv = ev.get("dependent_variable", "").strip()
        if not iv or not dv:
            continue

        relations.append({
            "independent_variable": iv,
            "dependent_variable": dv,
            "moderating_variables": ev.get("moderating_variables", ""),
            "controlled_variables": ev.get("controlled_variables", ""),
            "phenomenon": ev.get("phenomenon", ""),
            "mechanism": ev.get("mechanism", ""),
            "fatigue_indicator": dv,
            "equation_or_model": ev.get("equation_or_model", ""),
            "supporting_evidence_ids": ev.get("evidence_id", ""),
            "direction": "positive" if "increase" in ev.get("extracted_claim", "").lower() else "unknown",
            "evidence_type": ev.get("evidence_type", ""),
            "evidence_strength": ev.get("evidence_strength", ""),
            "direct_or_indirect": ev.get("direct_or_indirect", ""),
            # 结构化条件字段
            "material": ev.get("material", ""),
            "process": ev.get("process", ""),
            "build_orientation": ev.get("build_orientation", ""),
            "heat_treatment": ev.get("heat_treatment", ""),
            "surface_state": ev.get("surface_state", ""),
            "surface_roughness_Ra": ev.get("surface_roughness_Ra", ""),
            "surface_roughness_Rz": ev.get("surface_roughness_Rz", ""),
            "stress_ratio_R": ev.get("stress_ratio_R", ""),
            "stress_amplitude": ev.get("stress_amplitude", ""),
            "fatigue_type": ev.get("fatigue_type", ""),
            "frequency": ev.get("frequency", ""),
            "temperature": ev.get("temperature", ""),
            "environment": ev.get("environment", ""),
            "sample_geometry": ev.get("sample_geometry", ""),
            "pore_size": ev.get("pore_size", ""),
            "sqrt_area": ev.get("sqrt_area", ""),
            "distance_to_surface": ev.get("distance_to_surface", ""),
            "pore_location": ev.get("pore_location", ""),
            "pore_aspect_ratio": ev.get("pore_aspect_ratio", ""),
            "porosity": ev.get("porosity", ""),
            "defect_type": ev.get("defect_type", ""),
            "testing_method": ev.get("testing_method", ""),
            "characterization_method": ev.get("characterization_method", ""),
            "condition_boundary": ev.get("condition_boundary", ""),
            "missing_condition_fields": ev.get("missing_condition_fields", ""),
        })

    # 去重合并：同一对变量合并支持证据
    merged = {}
    for r in relations:
        key = f"{r['independent_variable']}→{r['dependent_variable']}"
        if key in merged:
            merged[key]["evidence_count"] += 1
            merged[key]["supporting_evidence_ids"] += "; " + r["supporting_evidence_ids"]
        else:
            r["evidence_count"] = 1
            r["variable_relation_id"] = f"VR_{len(merged)+1:04d}"
            r["notes"] = "Merged from evidence_snippets"
            merged[key] = r

    fieldnames = [
        "variable_relation_id", "independent_variable", "dependent_variable",
        "moderating_variables", "controlled_variables",
        "phenomenon", "mechanism", "fatigue_indicator",
        "equation_or_model", "supporting_evidence_ids",
        "evidence_count", "direction",
        "evidence_type", "evidence_strength", "direct_or_indirect",
        # 结构化条件字段 (取代旧的 condition_summary)
        "material", "process", "build_orientation", "heat_treatment",
        "surface_state", "surface_roughness_Ra", "surface_roughness_Rz",
        "stress_ratio_R", "stress_amplitude", "fatigue_type",
        "frequency", "temperature", "environment", "sample_geometry",
        "pore_size", "sqrt_area", "distance_to_surface", "pore_location",
        "pore_aspect_ratio", "porosity", "defect_type",
        "testing_method", "characterization_method",
        "condition_boundary", "missing_condition_fields", "notes",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in merged.values():
            writer.writerow(r)

    print(f"[INFO] Saved {len(merged)} variable relations to {path}")


def save_equation_parameter_dataset(evidence_list: List[Dict]):
    """从证据片段提取方程参数并保存。"""
    path = DATA_DIR / "equation_parameter_dataset.csv"
    params = []
    seen_eq = set()

    for ev in evidence_list:
        eq = ev.get("equation_or_model", "").strip()
        if not eq:
            continue
        if eq in seen_eq:
            continue
        seen_eq.add(eq)

        param_values_str = ev.get("parameter_values", "")
        param_values = {}
        try:
            if param_values_str:
                param_values = json.loads(param_values_str)
        except json.JSONDecodeError:
            param_values = {"raw": param_values_str}

        params.append({
            "equation_parameter_id": f"EP_{len(params)+1:04d}",
            "equation_or_model": eq,
            "paper_id": ev.get("paper_id", ""),
            "fatigue_indicator": ev.get("fatigue_indicator", ""),
            "parameter_values": param_values_str,
            "condition": f"material={ev.get('controlled_variables', '')}",
            "evidence_id": ev.get("evidence_id", ""),
            "notes": "Extracted from evidence_snippets",
        })

    fieldnames = [
        "equation_parameter_id", "equation_or_model", "paper_id",
        "fatigue_indicator", "parameter_values", "condition",
        "evidence_id", "notes",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in params:
            writer.writerow(r)

    print(f"[INFO] Saved {len(params)} equation parameter records to {path}")


def run_evidence_extraction(use_deepseek: bool = False, force_rebuild: bool = False):
    """Disabled during Stage 1 to prevent new title-derived evidence."""
    print(
        "[EVIDENCE] Disabled in Stage 1: title/folder-derived evidence is quarantined."
    )
    return []

    # Legacy implementation is retained below for rollback/reference only.
    """
    主入口：运行结构化证据抽取。

    Args:
        use_deepseek: 是否使用 DeepSeek 进行抽取（否则用基于规则的方法）
        force_rebuild: 是否强制重新生成所有证据
    """
    print("=" * 60)
    print("  结构化证据抽取模块")
    print("=" * 60)

    papers = load_literature_database()
    if not papers:
        print("[ERROR] No papers loaded. Run ingest first.")
        return

    # 加载已有证据
    if force_rebuild:
        existing = {}
        print("[INFO] Force rebuild: clearing existing evidence")
    else:
        existing = load_existing_evidence()

    # DeepSeek 初始化（如需）
    llm_client = None
    if use_deepseek:
        try:
            from src.api_keys import get_deepseek_settings
            from src.deepseek_client import DeepSeekClient

            settings = get_deepseek_settings()
            if settings.configured:
                llm_client = DeepSeekClient(settings)
                print("[INFO] DeepSeek client initialized")
            else:
                print("[WARNING] DEEPSEEK_API_KEY is not configured; using rule-based extraction")
        except ImportError:
            print("[WARNING] DeepSeek client is unavailable; using rule-based extraction")

    # 对每篇文献抽取证据
    all_evidence = list(existing.values())
    existing_ids = set(existing.keys())
    new_count = 0

    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if not paper_id:
            continue

        # 检查是否已抽取
        has_existing = any(
            ev.get("evidence_id", "").startswith(f"EV_{paper_id}")
            for ev in all_evidence
        )
        if has_existing and not force_rebuild:
            continue

        # 抽取证据
        evidence_batch = extract_evidence_from_paper(paper, llm_client)
        if evidence_batch is None:
            print(f"[WARNING] extract_evidence_from_paper returned None for paper_id={paper_id}")
            continue
        for ev in evidence_batch:
            if ev["evidence_id"] not in existing_ids:
                all_evidence.append(ev)
                existing_ids.add(ev["evidence_id"])
                new_count += 1

        if new_count % 10 == 0 and new_count > 0:
            print(f"[INFO] Extracted {new_count} new evidence snippets...")

    print(f"\n[INFO] Total evidence snippets: {len(all_evidence)}")
    print(f"[INFO] Newly extracted: {new_count}")
    print(f"[INFO] Previously existing: {len(all_evidence) - new_count}")

    # 保存
    save_evidence_snippets(all_evidence, append=False)
    save_variable_relation_dataset(all_evidence)
    save_equation_parameter_dataset(all_evidence)

    # 统计
    stats = compute_evidence_strength(all_evidence)
    print("\n[EVIDENCE STATS]")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    return all_evidence


if __name__ == "__main__":
    run_evidence_extraction()
