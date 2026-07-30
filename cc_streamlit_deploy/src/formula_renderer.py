"""
formula_renderer.py — LaTeX formula cards for fatigue equation display
面向 L-PBF Ti-6Al-4V 疲劳方程的专业 LaTeX 渲染。

为每个公式生成以下内容的卡片:
    1. 公式名称 (中英文)
    2. LaTeX 渲染公式 (st.latex)
    3. 公式物理意义
    4. 输入/输出变量解释
    5. 适用条件
    6. 不适用情况
    7. 在当前问题中的作用
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX definitions for core fatigue models
# ═══════════════════════════════════════════════════════════════════════════

FORMULA_LATEX = {
    "paris_law": {
        "name_cn": "Paris 裂纹扩展定律",
        "name_en": "Paris Law",
        "latex": r"\frac{da}{dN} = C(\Delta K)^m",
        "meaning": "描述稳定长裂纹扩展阶段中裂纹扩展速率 da/dN 与应力强度因子范围 ΔK 的关系。C 和 m 是材料参数。",
        "input_vars": [
            ("ΔK", "应力强度因子范围，疲劳裂纹扩展的驱动力"),
            ("C", "Paris 参数 C，反映材料裂纹扩展抗力的水平"),
            ("m", "Paris 指数 m，反映 da/dN 对 ΔK 的敏感程度"),
        ],
        "output_vars": [
            ("da/dN", "疲劳裂纹扩展速率，每个循环裂纹扩展的长度"),
        ],
        "applicable": [
            "稳定长裂纹扩展阶段（Region II）",
            "同一应力比 R 和试验条件下比较",
            "线弹性断裂力学 (LEFM) 适用时",
        ],
        "not_applicable": [
            "不适合描述裂纹起裂阶段",
            "不适合近门槛值小裂纹阶段，除非引入 El Haddad 修正",
            "不适合短裂纹或小裂纹行为",
            "塑性区尺寸相对于裂纹尺寸较大时不适用",
        ],
    },
    "basquin_equation": {
        "name_cn": "Basquin 方程 / S-N 高周疲劳拟合",
        "name_en": "Basquin Equation",
        "latex": r"\sigma_a = \sigma_f' (2N_f)^b",
        "latex_alt": r"\log N_f = A - B \log \sigma_a",
        "meaning": "描述应力幅 σa 与疲劳寿命 Nf 的经验关系，常用于高周疲劳拟合。",
        "input_vars": [
            ("σa", "应力幅，循环应力的半幅值"),
            ("σf'", "疲劳强度系数（Basquin 参数）"),
            ("b", "疲劳强度指数（Basquin 指数，通常为负值）"),
            ("Nf", "疲劳寿命，到达失效的循环次数"),
        ],
        "output_vars": [
            ("Nf", "给定应力幅下的预测疲劳寿命"),
            ("σf' 和 b", "拟合得到的 S-N 曲线参数"),
        ],
        "applicable": [
            "高周疲劳 (HCF) 寿命拟合",
            "应力控制疲劳实验",
            "同一材料状态和表面状态下对比",
        ],
        "not_applicable": [
            "不直接解释孔隙位置等微观机制",
            "不适合没有应力幅和寿命数据时强行拟合",
            "不适合低周疲劳（需 Coffin-Manson 修正）",
        ],
    },
    "murakami_sqrt_area": {
        "name_cn": "Murakami √area 模型",
        "name_en": "Murakami sqrt-area Model",
        "latex": r"\sigma_w = C \cdot \frac{HV + 120}{(\sqrt{area})^{1/6}}",
        "meaning": "估算含小缺陷材料的疲劳极限。缺陷尺寸 √area 越大、硬度 HV 越低，疲劳极限 σw 越低。",
        "input_vars": [
            ("HV", "维氏硬度，反映材料强度水平"),
            ("√area", "缺陷投影面积平方根，缺陷尺寸的 Murakami 表征量"),
            ("C", "经验系数，取决于缺陷位置（表面/内部）和加载方式"),
        ],
        "output_vars": [
            ("σw", "疲劳极限或疲劳强度估值"),
        ],
        "applicable": [
            "小缺陷/夹杂对疲劳极限的影响",
            "含 pore、lack of fusion、inclusion 的材料",
            "缺陷 √area 可通过 micro-CT 获得时",
        ],
        "not_applicable": [
            "不直接预测完整疲劳寿命 Nf",
            "不描述裂纹扩展速率 da/dN",
            "对非缺陷主导的疲劳失效需谨慎使用",
        ],
    },
    "kitagawa_takahashi": {
        "name_cn": "Kitagawa-Takahashi 缺陷容限模型",
        "name_en": "Kitagawa-Takahashi Diagram",
        "latex": r"\Delta \sigma_{th} = \frac{\Delta K_{th}}{Y \sqrt{\pi a}}",
        "meaning": "判断缺陷尺寸 a 与疲劳门槛应力范围 Δσth 的关系。大缺陷时裂纹扩展门槛控制，小缺陷时疲劳极限控制。",
        "input_vars": [
            ("a", "缺陷尺寸或等效裂纹半长"),
            ("ΔKth", "疲劳裂纹扩展门槛值"),
            ("Y", "几何修正因子，与缺陷形状和位置有关"),
        ],
        "output_vars": [
            ("Δσth", "门槛应力范围，低于该值裂纹不扩展"),
            ("acritical", "临界缺陷尺寸"),
        ],
        "applicable": [
            "缺陷容限分析",
            "判断缺陷是否超过临界尺寸",
            "区分疲劳极限控制区和裂纹扩展控制区",
        ],
        "not_applicable": [
            "对极小缺陷或短裂纹区域可能偏保守",
            "需要 El Haddad 修正来处理小裂纹",
            "Y 因子未知时只能使用简化版本",
        ],
    },
    "el_haddad": {
        "name_cn": "El Haddad 小裂纹修正模型",
        "name_en": "El Haddad Small-Crack Correction",
        "latex": r"\Delta \sigma_{th} = \frac{\Delta K_{th}}{Y \sqrt{\pi (a + a_0)}}",
        "meaning": "修正 Kitagawa-Takahashi 模型在小裂纹区域的不足。引入固有裂纹长度 a₀，使模型在 a→0 时平滑过渡到疲劳极限。",
        "input_vars": [
            ("a", "缺陷或裂纹尺寸"),
            ("a₀", "固有裂纹长度，El Haddad 修正参数"),
            ("ΔKth", "裂纹扩展门槛值"),
            ("Y", "几何修正因子"),
        ],
        "output_vars": [
            ("Δσth", "修正后的门槛应力范围"),
        ],
        "applicable": [
            "小缺陷（尺寸接近 a₀）的疲劳门槛预测",
            "短裂纹扩展行为分析",
            "L-PBF Ti-6Al-4V 中尺寸孔隙的容限分析",
        ],
        "not_applicable": [
            "a₀ 值未知时无法直接应用",
            "a₀ 对材料和微观组织敏感，需要实验测定",
        ],
    },
    "s_n_fitting": {
        "name_cn": "S-N 曲线经验拟合",
        "name_en": "S-N Curve Fitting",
        "latex": r"\log N_f = A - B \log \sigma_a",
        "meaning": "最常用的疲劳寿命-应力幅经验拟合关系。A 和 B 是拟合参数。",
        "input_vars": [
            ("σa", "应力幅"),
            ("Nf", "疲劳寿命"),
            ("A", "S-N 曲线截距参数"),
            ("B", "S-N 曲线斜率参数"),
        ],
        "output_vars": [
            ("Nf", "预测疲劳寿命"),
            ("A, B", "表征材料疲劳行为的经验参数"),
        ],
        "applicable": [
            "同一加载条件和应力比下的寿命拟合",
            "工程寿命预测",
        ],
        "not_applicable": [
            "不能直接解释微观机制",
            "外推到实验应力范围外可能不可靠",
        ],
    },
    "walker_model": {
        "name_cn": "Walker 裂纹扩展模型",
        "name_en": "Walker Crack Growth Model",
        "latex": r"\frac{da}{dN} = C \bigl(\Delta K (1-R)^{p-1}\bigr)^m",
        "meaning": "Paris 定律的扩展，引入应力比 R 的修正。p 是 Walker 材料参数。",
        "input_vars": [
            ("ΔK", "应力强度因子范围"),
            ("R", "应力比"),
            ("C, m, p", "Walker 模型材料参数"),
        ],
        "output_vars": [
            ("da/dN", "裂纹扩展速率"),
        ],
        "applicable": [
            "不同应力比 R 下的 FCGR 数据统一描述",
            "多 R 比实验数据可用时",
        ],
        "not_applicable": [
            "不适合单 R 比数据（此时退化为 Paris 定律）",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Generate formula card as markdown string
# ═══════════════════════════════════════════════════════════════════════════

def get_formula_card(formula_id: str, question_context: str = "") -> str:
    """
    Generate a formatted formula card as markdown string.

    Args:
        formula_id: one of "paris_law", "basquin_equation", etc.
        question_context: user's question to contextualize recommendation

    Returns:
        Markdown string ready for st.markdown() or st.latex()
    """
    if formula_id not in FORMULA_LATEX:
        return f"未知公式: {formula_id}"

    f = FORMULA_LATEX[formula_id]
    lines = []

    # Title
    lines.append(f"### {f['name_cn']}")
    lines.append(f"*{f['name_en']}*\n")

    # Formula
    lines.append("**公式**")
    latex_code = f["latex"]
    lines.append(f"$${latex_code}$$")
    if f.get("latex_alt"):
        lines.append(f"或 $${f['latex_alt']}$$\n")

    # Physical meaning
    lines.append("**物理意义**")
    lines.append(f"{f['meaning']}\n")

    # Input variables
    lines.append("**输入变量**")
    for var_name, var_desc in f["input_vars"]:
        lines.append(f"- **{var_name}**：{var_desc}")
    lines.append("")

    # Output variables
    lines.append("**输出变量**")
    for var_name, var_desc in f["output_vars"]:
        lines.append(f"- **{var_name}**：{var_desc}")
    lines.append("")

    # Applicable conditions
    lines.append("**适用条件**")
    for cond in f["applicable"]:
        lines.append(f"- ✅ {cond}")
    lines.append("")

    # Not applicable
    lines.append("**不适用情况**")
    for cond in f["not_applicable"]:
        lines.append(f"- ❌ {cond}")
    lines.append("")

    # Context recommendation
    if question_context:
        context_note = _contextualize_recommendation(formula_id, question_context)
        lines.append("**在当前问题中的作用**")
        lines.append(context_note)
        lines.append("")

    return "\n".join(lines)


def _contextualize_recommendation(formula_id: str, question: str) -> str:
    """Generate context-aware recommendation note."""
    q = question.lower()

    context_map = {
        "paris_law": (
            "用户问题涉及 ΔK 和 da/dN 的关系，推荐 Paris 定律。"
            "该模型直接描述稳定裂纹扩展速率与驱动力之间的定量关系。"
            "若已知应力比 R，还可进一步使用 Walker 模型。"
        ),
        "basquin_equation": (
            "用户问题涉及应力幅和疲劳寿命，推荐 Basquin 方程。"
            "该模型直接拟合 S-N 曲线的高周疲劳段，可比较不同条件下的斜率 b 和系数 σf'。"
        ),
        "murakami_sqrt_area": (
            "用户问题涉及缺陷尺寸和疲劳极限，推荐 Murakami √area 模型。"
            "该模型将缺陷尺寸 √area 和硬度 HV 与疲劳极限定量关联，"
            "适合分析 L-PBF Ti-6Al-4V 中孔隙缺陷对疲劳强度的影响。"
        ),
        "kitagawa_takahashi": (
            "用户问题涉及缺陷尺寸与门槛应力的关系，推荐 Kitagawa-Takahashi 模型。"
            "该模型判断缺陷是否超过临界尺寸，区分疲劳极限控制区和裂纹扩展控制区。"
        ),
        "el_haddad": (
            "用户问题涉及小缺陷或小裂纹，推荐 El Haddad 修正模型。"
            "该模型通过引入固有裂纹长度 a₀ 使 K-T 模型适用于小缺陷区域，"
            "对 L-PBF Ti-6Al-4V 中常见的小尺寸孔隙分析有重要意义。"
        ),
        "s_n_fitting": (
            "用户问题涉及疲劳寿命预测，推荐 S-N 曲线拟合。"
            "该模型是工程中最常用的寿命预测方法，可引入缺陷尺寸等修正项。"
        ),
        "walker_model": (
            "用户问题涉及不同应力比 R 下的裂纹扩展，推荐 Walker 模型。"
            "该模型扩展了 Paris 定律，引入 p 参数考虑 R 比效应。"
        ),
    }

    return context_map.get(formula_id, "根据用户问题的变量匹配，推荐此模型。")


# ═══════════════════════════════════════════════════════════════════════════
# Batch formula display for variable pair
# ═══════════════════════════════════════════════════════════════════════════

def recommend_formulas_for_pair(
    ind_var: Optional[str],
    dep_var: Optional[str],
    question: str = "",
) -> List[str]:
    """
    Recommend appropriate formula cards based on variable pair.

    Returns list of markdown strings, one per recommended formula.
    """
    if not ind_var or not dep_var:
        return []

    pair = (ind_var, dep_var)
    reverse_pair = (dep_var, ind_var)

    formula_map = {
        ("delta_k", "da_dn"): ["paris_law", "walker_model"],
        ("da_dn", "delta_k"): ["paris_law", "walker_model"],
        ("stress_amplitude", "fatigue_life"): ["basquin_equation", "s_n_fitting"],
        ("fatigue_life", "stress_amplitude"): ["basquin_equation", "s_n_fitting"],
        ("pore_size", "fatigue_limit"): ["murakami_sqrt_area", "kitagawa_takahashi"],
        ("sqrt_area", "fatigue_limit"): ["murakami_sqrt_area", "kitagawa_takahashi"],
        ("fatigue_limit", "pore_size"): ["murakami_sqrt_area", "kitagawa_takahashi"],
        ("pore_size", "da_dn"): ["paris_law", "murakami_sqrt_area"],
        ("pore_size", "da_dn"): ["paris_law", "murakami_sqrt_area"],
        ("defect_size", "fatigue_limit"): ["murakami_sqrt_area", "kitagawa_takahashi", "el_haddad"],
        ("pore_size", "fatigue_life"): ["murakami_sqrt_area", "s_n_fitting", "kitagawa_takahashi"],
        ("fatigue_life", "pore_size"): ["murakami_sqrt_area", "s_n_fitting", "kitagawa_takahashi"],
        ("surface_roughness", "fatigue_life"): ["s_n_fitting", "kitagawa_takahashi"],
        ("surface_roughness", "fatigue_limit"): ["murakami_sqrt_area", "kitagawa_takahashi"],
        ("heat_treatment", "fatigue_life"): ["s_n_fitting", "paris_law"],
        ("paris_c_m", "da_dn"): ["paris_law", "walker_model"],
        ("microstructure", "da_dn"): ["paris_law", "walker_model"],
        ("delta_k", "fatigue_limit"): ["kitagawa_takahashi", "el_haddad"],
    }

    # Check exact match
    formula_ids = formula_map.get(pair, formula_map.get(reverse_pair, []))

    if not formula_ids:
        # Try partial match
        for key, ids in formula_map.items():
            if ind_var in key or dep_var in key:
                formula_ids = ids
                break

    cards = []
    for fid in formula_ids[:3]:  # max 3 formulas
        card = get_formula_card(fid, question)
        cards.append(card)

    return cards
