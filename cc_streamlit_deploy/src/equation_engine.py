"""
equation_engine.py — Fatigue Equation Knowledge Base Engine
面向 L-PBF Ti-6Al-4V 疲劳方程推荐、参数检查与方程型假设生成。

依赖:
    data/fatigue_equation_library.csv

主要函数:
    load_equation_library()
    extract_variables_from_query()
    recommend_equations()
    generate_equation_hypothesis()
    check_required_parameters()
    build_validation_plan()
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════
# Variable synonym map
# ═══════════════════════════════════════════════════════════════════════════

VARIABLE_SYNONYMS = {
    "pore_size": [
        "pore size", "pore diameter", "pore radius", "pore dimension",
        "pore_size", "孔隙尺寸", "孔隙直径", "孔隙大小", "孔径",
        "defect size", "defect_size", "缺陷尺寸",
    ],
    "sqrt_area": [
        "sqrt_area", "√area", "sqrtarea", "sqrt area",
        "square root of area", "defect area",
        "√area", "缺陷面积平方根",
    ],
    "pore_location": [
        "pore location", "pore position", "distance to surface",
        "pore_location", "孔隙位置", "距表面距离", "距表面深度",
        "defect location", "defect_location",
    ],
    "surface_roughness": [
        "surface roughness", "surface_roughness", "ra", "rz", "rms",
        "as-built surface", "surface finish", "surface quality",
        "表面粗糙度", "粗糙度", "表面质量",
    ],
    "stress_amplitude": [
        "stress amplitude", "stress_amplitude", "sigma_a", "σa",
        "σ_a", "applied stress", "cyclic stress",
        "应力幅", "应力振幅",
    ],
    "fatigue_life": [
        "fatigue life", "fatigue_life", "nf", "cycles to failure",
        "lifetime", "疲劳寿命", "寿命", "循环寿命", "失效周次",
    ],
    "fatigue_limit": [
        "fatigue limit", "fatigue_limit", "fatigue strength",
        "sigma_w", "σw", "σ_w", "endurance limit",
        "疲劳极限", "疲劳强度",
    ],
    "delta_k": [
        "delta_k", "Δk", "ΔK", "delta K", "stress intensity range",
        "sif range", "Δk", "ΔKth",
        "应力强度因子范围", "应力强度因子幅",
    ],
    "da_dn": [
        "da/dn", "da_dn", "crack growth rate", "fcgr",
        "crack propagation rate", "da dN",
        "裂纹扩展速率", "裂纹扩展速度",
    ],
    "paris_c": [
        "paris c", "paris_c", "c", "paris coefficient",
        "paris 常数 c",
    ],
    "paris_m": [
        "paris m", "paris_m", "m", "paris exponent",
        "paris 指数 m",
    ],
    "heat_treatment": [
        "heat treatment", "heat_treatment", "hip", "hot isostatic pressing",
        "annealing", "anneal", "solution treatment", "aging",
        "热处理", "hip 处理", "退火", "固溶", "时效",
    ],
    "microstructure": [
        "microstructure", "microstructure", "alpha lath", "alpha_lath",
        "martensite", "alpha_prime", "beta phase", "grain size",
        "grain orientation", "prior beta", "微观组织", "显微组织",
        "片层", "晶粒", "相",
    ],
    "residual_stress": [
        "residual stress", "residual_stress", "residual",
        "残余应力",
    ],
    "stress_ratio": [
        "stress ratio", "stress_ratio", "r ratio", "r-ratio",
        "load ratio", "应力比", "r",
    ],
    "porosity": [
        "porosity", "porosity", "void fraction", "density",
        "孔隙率", "致密度",
    ],
    "defect_aspect_ratio": [
        "aspect ratio", "defect aspect ratio", "pore aspect ratio",
        "pore_shape", "pore morphology", "缺陷形态", "孔隙形态",
        "长宽比",
    ],
}

# Reverse map: keyword → canonical variable name
_KEYWORD_TO_VAR: List[Tuple[str, str]] = []
for canonical, synonyms in VARIABLE_SYNONYMS.items():
    for syn in synonyms:
        _KEYWORD_TO_VAR.append((syn.lower(), canonical))
# Sort by length descending so longer matches take priority
_KEYWORD_TO_VAR.sort(key=lambda x: -len(x[0]))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load equation library
# ═══════════════════════════════════════════════════════════════════════════

def load_equation_library(
    path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    读取疲劳方程知识库 CSV。

    返回 list of dict，每行一个方程。
    """
    if path is None:
        path = str(DATA_DIR / "fatigue_equation_library.csv")

    p = Path(path)
    if not p.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(p, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all fields
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)
    return rows


def get_equation_by_id(
    eq_id: str,
    library: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """按 equation_id 查找方程。"""
    if library is None:
        library = load_equation_library()
    for row in library:
        if row.get("equation_id", "").strip() == eq_id:
            return row
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extract variables from user query
# ═══════════════════════════════════════════════════════════════════════════

def extract_variables_from_query(user_query: str) -> List[str]:
    """
    从用户输入中识别变量。

    返回去重后的 canonical variable name 列表。
    """
    if not user_query:
        return []

    text = user_query.lower()
    found: List[str] = []

    for keyword, canonical in _KEYWORD_TO_VAR:
        if keyword in text:
            # Avoid duplicate entries for same canonical variable
            if canonical not in found:
                found.append(canonical)

    return found


# ═══════════════════════════════════════════════════════════════════════════
# 3. Recommend equations based on detected variables
# ═══════════════════════════════════════════════════════════════════════════

def recommend_equations(
    user_query: str,
    detected_variables: Optional[List[str]] = None,
    library: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    根据用户输入或检测到的变量推荐候选方程。

    规则:
    - 如果 Delta_K + da/dN 同时出现 → Paris law
    - 如果 stress_amplitude + fatigue_life → Basquin / S-N fitting
    - 如果 sqrt_area / pore_size + fatigue_limit → Murakami sqrt-area
    - 如果 defect_size + Delta_Kth/fatigue_limit → Kitagawa-Takahashi
    - 如果 small crack / short crack → El Haddad

    返回匹配的 equation rows（可能为空）。
    """
    if library is None:
        library = load_equation_library()
    if not library:
        return []

    if detected_variables is None:
        detected_variables = extract_variables_from_query(user_query)

    text = user_query.lower() if user_query else ""
    var_set = set(detected_variables)

    recommended: List[Dict[str, Any]] = []

    # Check each equation's applicable_question against query
    # and input_variables against detected variables
    for eq_row in library:
        eq_id = eq_row.get("equation_id", "")
        applicable_q = eq_row.get("applicable_question", "").lower()
        input_vars = eq_row.get("input_variables", "").lower()

        # Score the match
        score = 0

        # Match 1: applicable_question keyword overlap
        q_keywords = [kw.strip() for kw in applicable_q.split(";")]
        for kw in q_keywords:
            if kw and kw in text:
                score += 2

        # Match 2: input variable overlap
        input_var_list = [v.strip() for v in input_vars.split(";") if v.strip()]
        for iv in input_var_list:
            iv_canonical = iv.replace(" ", "_").lower()
            if iv_canonical in var_set:
                score += 1

        # Special rules for strong matches
        if eq_id == "paris_law":
            if "delta_k" in var_set and "da_dn" in var_set:
                score += 5
            if any(kw in text for kw in ["paris", "da/dn", "crack growth", "fcgr"]):
                score += 3

        elif eq_id == "basquin_equation":
            if "stress_amplitude" in var_set and "fatigue_life" in var_set:
                score += 5
            if any(kw in text for kw in ["basquin", "s-n", "sn curve", "stress-life"]):
                score += 3

        elif eq_id == "s_n_fitting":
            if "stress_amplitude" in var_set and "fatigue_life" in var_set:
                score += 4
            if any(kw in text for kw in ["s-n", "sn", "stress-life", "life prediction"]):
                score += 2

        elif eq_id == "murakami_sqrt_area":
            if "sqrt_area" in var_set or "pore_size" in var_set:
                if "fatigue_limit" in var_set:
                    score += 5
            if any(kw in text for kw in ["murakami", "sqrt", "√area"]):
                score += 3

        elif eq_id == "kitagawa_takahashi":
            if "pore_size" in var_set or "defect_size" in var_set or "sqrt_area" in var_set:
                if "delta_k" in var_set or "fatigue_limit" in var_set:
                    score += 5
            if any(kw in text for kw in ["kitagawa", "takahashi", "threshold", "缺陷容忍"]):
                score += 3

        elif eq_id == "el_haddad":
            if any(kw in text for kw in ["short crack", "small crack", "small defect",
                                          "小裂纹", "小缺陷", "short defect"]):
                score += 5
            if "pore_size" in var_set and ("delta_k" in var_set or "fatigue_limit" in var_set):
                score += 3

        if score >= 3:
            recommended.append((score, eq_row))

    # Sort by score descending, return highest-scoring matches
    recommended.sort(key=lambda x: -x[0])

    # Remove duplicates (same equation_id)
    seen_ids = set()
    unique: List[Dict[str, Any]] = []
    for score, eq_row in recommended:
        eq_id = eq_row.get("equation_id", "")
        if eq_id not in seen_ids:
            seen_ids.add(eq_id)
            unique.append(eq_row)

    return unique[:4]  # max 4


# ═══════════════════════════════════════════════════════════════════════════
# 4. Generate equation-based hypothesis
# ═══════════════════════════════════════════════════════════════════════════

def generate_equation_hypothesis(
    user_query: str,
    equation_row: Dict[str, Any],
    detected_variables: List[str],
) -> Dict[str, Any]:
    """
    基于方程库中的 hypothesis_template 生成具体的方程型假设。

    返回 dict:
        hypothesis_id, equation_name, formula,
        statement, controlled_vars, independent_vars, dependent_vars,
        mechanism, validation_data, falsification,
        missing_parameters, score
    """
    template = equation_row.get("hypothesis_template", "")
    eq_name = equation_row.get("equation_name", "")
    formula = equation_row.get("formula", "")
    eq_id = equation_row.get("equation_id", "")
    required_params_raw = equation_row.get("required_parameters", "")

    # Identify which detected variables are NOT standard params
    input_vars_raw = equation_row.get("input_variables", "")
    input_var_list = [v.strip() for v in input_vars_raw.split(";") if v.strip()]

    # Determine the "main variable" — the one the user is asking about
    # that is NOT a standard control variable
    standard_controls = {
        "stress_ratio", "stress_ratio_r", "r_ratio",
        "material_condition", "heat_treatment", "surface_state",
        "defect_state", "microstructure",
    }

    user_focus_vars = [
        v for v in detected_variables
        if v not in standard_controls
    ]

    # If no specific variable found, use the first detected variable
    main_variable = user_focus_vars[0] if user_focus_vars else "target variable"

    # Fill the template
    statement = template.replace("[变量]", main_variable)

    # Controlled variables: standard controls
    controlled = [v for v in detected_variables if v in standard_controls]
    if not controlled:
        controlled = ["stress_ratio_R", "heat_treatment", "surface_state"]

    # Independent (main input) and dependent (output) variables
    output_vars_raw = equation_row.get("output_variables", "")
    output_var_list = [v.strip() for v in output_vars_raw.split(";") if v.strip()]

    independent_vars = [main_variable] if main_variable != "target variable" else detected_variables[:2]
    dependent_vars = output_var_list[:3]

    # Missing parameters check
    missing, available = check_required_parameters(equation_row, detected_variables)

    # Validation plan
    validation_plan = build_validation_plan(equation_row, detected_variables)

    # Build mechanism description
    mechanism = _build_mechanism(equation_row, main_variable, detected_variables)

    # Score components
    score_info = _score_equation_hypothesis(equation_row, detected_variables, len(missing))

    return {
        "hypothesis_id": f"EQH_{eq_id}",
        "equation_name": eq_name,
        "equation_id": eq_id,
        "formula": formula,
        "statement": statement,
        "controlled_variables": "; ".join(controlled),
        "independent_variables": "; ".join(independent_vars),
        "dependent_variables": "; ".join(dependent_vars),
        "mechanism": mechanism,
        "validation_data": validation_plan.get("needed_data", []),
        "validation_method": validation_plan.get("validation_method", ""),
        "falsification": _build_falsification(equation_row, main_variable),
        "missing_parameters": missing,
        "available_parameters": available,
        "score": score_info,
    }


def _build_mechanism(
    equation_row: Dict[str, Any],
    main_variable: str,
    detected_variables: List[str],
) -> str:
    """生成机制解释。"""
    eq_id = equation_row.get("equation_id", "")

    mechanism_map = {
        "paris_law": (
            f"{main_variable} 可能通过改变材料局部裂纹扩展阻力影响 Paris 参数 C 或 m："
            "若 C 增大（或 m 减小），表示相同 ΔK 下 da/dN 增大，即扩展抗力降低。"
            "该效应可能源于缺陷处应力集中加速早期扩展，或微观组织变化改变扩展路径。"
        ),
        "basquin_equation": (
            f"{main_variable} 可能通过改变疲劳强度系数 σ_f' 或斜率 b 影响高周疲劳寿命："
            "σ_f' 降低或 |b| 增大均会导致相同应力幅下 Nf 降低。"
        ),
        "s_n_fitting": (
            f"{main_variable} 预期改变 S-N 曲线的斜率 B 或截距 A，"
            "表现为相同应力幅下 Nf 的系统变化。"
        ),
        "murakami_sqrt_area": (
            f"{main_variable}（作为缺陷尺寸参数）可能通过局部应力集中效应降低疲劳极限 σw。"
            "该效应符合 Murakami √area 模型：缺陷越大，疲劳极限越低；近表面缺陷影响更显著。"
        ),
        "kitagawa_takahashi": (
            f"{main_variable} 可能影响缺陷容忍边界：小尺寸缺陷时疲劳极限控制失效，"
            "大尺寸缺陷时裂纹扩展门槛控制失效。两者之间存在一个临界缺陷尺寸。"
        ),
        "el_haddad": (
            f"对于 L-PBF Ti-6Al-4V 中的小尺寸 {main_variable}，"
            "El Haddad 修正通过引入 intrinsic crack length a0 使 K-T 模型适用于小裂纹/小缺陷区域。"
        ),
    }

    return mechanism_map.get(eq_id, f"{main_variable} 可能通过影响方程参数改变疲劳行为。需要实验数据验证。")


def _build_falsification(
    equation_row: Dict[str, Any],
    main_variable: str,
) -> str:
    """生成推翻条件。"""
    eq_id = equation_row.get("equation_id", "")

    falsification_map = {
        "paris_law": (
            f"若在控制应力比 R 和表面状态后，不同 {main_variable} 条件下拟合得到的 "
            "Paris C/m 值无统计显著差异（p > 0.05），或差异完全由实验误差解释，则该假设应被降级。"
        ),
        "basquin_equation": (
            f"若不同 {main_variable} 条件下 S-N 拟合斜率 b 和系数 σ_f' 无系统差异，"
            "则该假设应被降级。"
        ),
        "s_n_fitting": (
            f"若引入 {main_variable} 修正后 S-N 预测精度无明显提升（R² 提升 < 0.05），"
            "则该假设不成立。"
        ),
        "murakami_sqrt_area": (
            f"若疲劳失效并非从最大或近表面 √area 缺陷起裂，"
            "或实测疲劳极限与 √area 模型预测偏差超过 ±20%，则该假设应被推翻或降级。"
        ),
        "kitagawa_takahashi": (
            "若小缺陷样品仍发生早期失效，且 Kitagawa-Takahashi 边界无法区分失效/安全区域，"
            "则该模型不适用。需要引入 El Haddad 修正或考虑表面粗糙度。"
        ),
        "el_haddad": (
            "若引入修正后仍无法解释失效边界，则说明表面粗糙度或残余应力是更主要的控制因素。"
        ),
    }

    return falsification_map.get(
        eq_id,
        f"若在控制相关条件后，{main_variable} 与疲劳指标无统计显著关系，则该假设应被降级。"
    )


def _score_equation_hypothesis(
    equation_row: Dict[str, Any],
    detected_variables: List[str],
    n_missing: int,
) -> Dict[str, Any]:
    """
    对方程型假设评分。

    维度:
    - variable_match (0-5): 检测到的变量覆盖程度
    - param_completeness (0-5): 参数完整性
    - data_feasibility (0-5): 数据可获取性
    - mechanism_clarity (0-5): 机制清晰度
    - total: /20
    """
    input_vars_raw = equation_row.get("input_variables", "")
    input_var_list = [v.strip() for v in input_vars_raw.split(";") if v.strip()]

    # Variable match
    matched = sum(1 for v in detected_variables if v in input_var_list)
    variable_match = min(matched * 2, 5)

    # Parameter completeness
    if n_missing == 0:
        param_completeness = 5
    elif n_missing <= 2:
        param_completeness = 3
    else:
        param_completeness = 1

    # Data feasibility (assume always feasible for these standard equations)
    data_feasibility = 4

    # Mechanism clarity
    mechanism_clarity = 4

    total = variable_match + param_completeness + data_feasibility + mechanism_clarity

    if total >= 18:
        grade = "good"
    elif total >= 14:
        grade = "medium"
    elif total >= 10:
        grade = "weak"
    else:
        grade = "reject"

    return {
        "variable_match": variable_match,
        "param_completeness": param_completeness,
        "data_feasibility": data_feasibility,
        "mechanism_clarity": mechanism_clarity,
        "total": total,
        "max": 20,
        "grade": grade,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Check required parameters
# ═══════════════════════════════════════════════════════════════════════════

def check_required_parameters(
    equation_row: Dict[str, Any],
    available_variables: List[str],
) -> Tuple[List[str], List[str]]:
    """
    检查方程所需参数是否齐全。

    返回 (missing_parameters, available_parameters)。
    """
    required_raw = equation_row.get("required_parameters", "")
    required_list = [p.strip().lower().replace(" ", "_") for p in required_raw.split(";") if p.strip()]

    var_set = set(available_variables)

    missing = []
    available = []
    for req in required_list:
        if req in var_set:
            available.append(req)
        else:
            missing.append(req)

    return missing, available


# ═══════════════════════════════════════════════════════════════════════════
# 6. Build validation plan
# ═══════════════════════════════════════════════════════════════════════════

def build_validation_plan(
    equation_row: Dict[str, Any],
    detected_variables: List[str],
) -> Dict[str, Any]:
    """
    根据方程和检测到的变量生成验证方案。

    返回 dict:
        needed_data: list of required data
        experiments: suggested experiments
        characterization: suggested characterization methods
        fitting_method: how to fit
        support_criteria: criteria to support hypothesis
        rejection_criteria: criteria to reject
    """
    eq_id = equation_row.get("equation_id", "")
    validation_method = equation_row.get("validation_method", "")

    plans = {
        "paris_law": {
            "needed_data": [
                "da/dN-ΔK 曲线数据（至少 3 个试样）",
                "Paris C 和 m 拟合值",
                "应力比 R",
                "孔隙缺陷三维特征（micro-CT）",
                "微观组织状态（SEM/EBSD）",
            ],
            "experiments": "FCGR 试验：compact tension (CT) 或 single edge notch (SEN) 试样，恒定 R 比，记录 da/dN-ΔK",
            "characterization": "micro-CT 预先表征孔隙特征；SEM 观察断口起裂源和扩展路径；EBSD 表征裂纹路径处微观组织",
            "fitting_method": "log(da/dN) = m * log(ΔK) + log(C)，线性拟合获取 C 和 m，比较不同条件下的参数差异",
            "support_criteria": "不同缺陷状态的 Paris C 或 m 存在统计显著差异（t-test p < 0.05）",
            "rejection_criteria": "C 和 m 均无系统变化，或变化方向与假设相反",
        },
        "basquin_equation": {
            "needed_data": [
                "S-N 曲线数据（多应力水平）",
                "应力幅 σa 和对应 Nf",
                "疲劳强度系数 σ_f' 和指数 b",
                "表面粗糙度 Ra/Rz",
                "缺陷状态（micro-CT）",
            ],
            "experiments": "HCF 试验：多应力水平（至少 4 级），应力比 R=0.1 或 -1，记录 Nf",
            "characterization": "SEM 确认起裂源（表面 vs 内部缺陷）；表面粗糙度测量",
            "fitting_method": "log(σa) = b * log(2Nf) + log(σ_f')，线性拟合",
            "support_criteria": "不同表面/缺陷状态下 b 或 σ_f' 存在系统差异",
            "rejection_criteria": "S-N 曲线在组间无显著差异（ANOVA p > 0.05）",
        },
        "s_n_fitting": {
            "needed_data": [
                "S-N 实验数据",
                "应力幅和对应 Nf",
                "R 比",
                "表面状态",
            ],
            "experiments": "HCF 试验获取 S-N 数据",
            "characterization": "断口分析确认失效模式",
            "fitting_method": "log(Nf) = A - B * log(σa)，线性拟合",
            "support_criteria": "引入缺陷/粗糙度修正后 R² 提升 ≥ 0.05",
            "rejection_criteria": "修正项无显著改善预测精度",
        },
        "murakami_sqrt_area": {
            "needed_data": [
                "缺陷 √area（micro-CT 数据）",
                "硬度 HV",
                "疲劳极限 σw（升降法获得）",
                "缺陷位置（表面/近表面/内部）",
            ],
            "experiments": "升降法测定疲劳极限；统计最大缺陷 √area",
            "characterization": "micro-CT：缺陷三维尺寸和位置；断口确认起裂源与 √area 对应",
            "fitting_method": "σw = A*(HV+120)/(√area)^{1/6}，拟合 A 值",
            "support_criteria": "预测疲劳极限与实测值的偏差在 ±15% 以内",
            "rejection_criteria": "实测 σw 与 √area 模型无关，或失效并非从最大缺陷起裂",
        },
        "kitagawa_takahashi": {
            "needed_data": [
                "缺陷尺寸 a 或 √area",
                "疲劳极限 σw",
                "ΔKth",
                "失效/安全数据点",
            ],
            "experiments": "不同缺陷尺寸试样的 HCF 试验，标记失效 vs 安全数据点",
            "characterization": "micro-CT 测缺陷尺寸；SEM 确认起裂源",
            "fitting_method": "Δσth = ΔKth / √(πa)，绘制 Kitagawa 图",
            "support_criteria": "Kitagawa 边界能清晰区分失效/安全区域",
            "rejection_criteria": "小缺陷样品在边界以下仍发生早期失效",
        },
        "el_haddad": {
            "needed_data": [
                "小缺陷/小裂纹尺寸 a",
                "ΔKth",
                "疲劳极限 σw",
                "intrinsic crack length a0",
            ],
            "experiments": "小缺陷试样 HCF 试验；短裂纹扩展试验",
            "characterization": "高分辨率 SEM/EBSD 观察短裂纹扩展行为",
            "fitting_method": "Δσth = ΔKth / √(π(a + a0))，拟合 a0",
            "support_criteria": "El Haddad 修正后预测边界与实验数据吻合",
            "rejection_criteria": "修正后仍无法解释小裂纹/小缺陷失效",
        },
    }

    default_plan = {
        "needed_data": equation_row.get("validation_data_needed", "").split(";") if equation_row.get("validation_data_needed") else [],
        "experiments": "需根据具体方程设计实验",
        "characterization": "需根据实验条件选择表征方法",
        "fitting_method": validation_method,
        "support_criteria": "实验数据与模型预测一致",
        "rejection_criteria": "实验数据与模型预测偏差超过可接受范围",
    }

    return plans.get(eq_id, default_plan)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_equation_pipeline(user_query: str) -> Dict[str, Any]:
    """
    执行完整方程匹配流水线。

    返回 dict:
        detected_variables: [...]
        recommended_equations: [...]
        equation_hypotheses: [...]  (每个推荐方程生成一个假设)
        summary: str
    """
    if not user_query:
        return {
            "detected_variables": [],
            "recommended_equations": [],
            "equation_hypotheses": [],
            "summary": "请输入科研问题。",
        }

    detected = extract_variables_from_query(user_query)
    library = load_equation_library()

    if not library:
        return {
            "detected_variables": detected,
            "recommended_equations": [],
            "equation_hypotheses": [],
            "summary": "疲劳方程知识库为空（data/fatigue_equation_library.csv 不存在或格式错误）。",
        }

    recommended = recommend_equations(user_query, detected, library)
    hypotheses = []

    for eq_row in recommended:
        h = generate_equation_hypothesis(user_query, eq_row, detected)
        hypotheses.append(h)

    # Build summary
    if not recommended:
        summary = (
            f"当前变量组合暂未匹配到可靠疲劳方程（检测到变量：{', '.join(detected) if detected else '无'}）。"
            "如需生成方程型假设，建议输入更明确的变量组合，例如：\n"
            "- ΔK 与 da/dN → 推荐 Paris law\n"
            "- 应力幅与疲劳寿命 → 推荐 Basquin equation\n"
            "- 缺陷尺寸与疲劳极限 → 推荐 Murakami √area model 或 Kitagawa-Takahashi"
        )
    else:
        eq_names = [r.get("equation_name", "") for r in recommended]
        summary = f"已匹配 {len(recommended)} 个候选方程：{' / '.join(eq_names)}。"

    return {
        "detected_variables": detected,
        "recommended_equations": recommended,
        "equation_hypotheses": hypotheses,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. Format equation results as markdown
# ═══════════════════════════════════════════════════════════════════════════

def format_equation_results_markdown(pipeline_result: Dict[str, Any]) -> str:
    """
    将方程流水线结果格式化为 markdown 字符串，
    供 streamlit_app.py 直接嵌入综合回答。
    """
    lines = []
    detected = pipeline_result["detected_variables"]
    recommended = pipeline_result["recommended_equations"]
    hypotheses = pipeline_result["equation_hypotheses"]
    summary = pipeline_result["summary"]

    if not recommended:
        lines.append("## 候选方程\n")
        lines.append(f"{summary}\n")
        return "\n".join(lines)

    lines.append("## 候选方程\n")
    lines.append(f"> 检测到变量：{'、'.join(detected) if detected else '无'}\n")

    for i, eq_row in enumerate(recommended, 1):
        eq_name = eq_row.get("equation_name", "未知方程")
        formula = eq_row.get("formula", "")
        applicable = eq_row.get("applicable_question", "")
        required = eq_row.get("required_parameters", "")
        conditions = eq_row.get("applicable_conditions", "")
        limitations = eq_row.get("limitations", "")

        lines.append(f"### {i}. {eq_name}\n")
        lines.append(f"**公式**：`{formula}`\n")
        lines.append(f"**适用问题**：{applicable}\n")
        lines.append(f"**需要参数**：{required}\n")
        lines.append(f"**适用条件**：{conditions}\n")
        lines.append(f"**局限性**：{limitations}\n")

    lines.append("---\n")

    # Equation hypotheses
    if hypotheses:
        lines.append("## 方程型候选假设\n")

        for h in hypotheses:
            eq_name = h.get("equation_name", "")
            formula = h.get("formula", "")
            statement = h.get("statement", "")
            controlled = h.get("controlled_variables", "")
            independent = h.get("independent_variables", "")
            dependent = h.get("dependent_variables", "")
            mechanism = h.get("mechanism", "")
            validation_method = h.get("validation_method", "")
            falsification = h.get("falsification", "")
            missing = h.get("missing_parameters", [])
            score = h.get("score", {})

            lines.append(f"### {eq_name} 型假设\n")
            lines.append(f"**可能方程**：`{formula}`\n")
            lines.append(f"**假设**：{statement}\n")
            lines.append(f"**控制变量**：{controlled}\n")
            lines.append(f"**自变量**：{independent}\n")
            lines.append(f"**因变量**：{dependent}\n")
            lines.append(f"**机制解释**：{mechanism}\n")
            lines.append(f"**验证方法**：{validation_method}\n")
            lines.append(f"**推翻条件**：{falsification}\n")

            # Missing parameters
            if missing:
                lines.append(f"**缺失参数**：{'、'.join(missing)}\n")
            else:
                lines.append("**缺失参数**：无（参数完整）\n")

            # Score
            score_total = score.get("total", 0)
            score_max = score.get("max", 20)
            score_grade = score.get("grade", "reject")
            grade_icon = {"good": "🟢", "medium": "🟡", "weak": "🟠", "reject": "🔴"}.get(score_grade, "⚪")
            lines.append(
                f"**评分**：{score_total}/{score_max} {grade_icon} {score_grade} "
                f"（变量匹配={score.get('variable_match')}/5, "
                f"参数完整={score.get('param_completeness')}/5, "
                f"数据可行={score.get('data_feasibility')}/5, "
                f"机制清晰={score.get('mechanism_clarity')}/5）\n"
            )

            lines.append("\n")

        lines.append("---\n")

        # Validation plans
        lines.append("## 下一步需要的数据\n")
        for h in hypotheses:
            eq_name = h.get("equation_name", "")
            validation_data = h.get("validation_data", [])
            lines.append(f"### {eq_name}\n")
            if validation_data:
                for item in validation_data:
                    lines.append(f"- {item}\n")
            else:
                lines.append("- 需查看方程库中 validation_data_needed 字段\n")
            lines.append("\n")

    return "\n".join(lines)
