"""
formula_comparison.py — 多模型对比分析模块

功能:
  对于同一科研问题，比较多个可选模型/方程。
  输出各模型的输入、输出、适用条件、不适用情况、
  数据需求、当前数据充分性和推荐等级。

核心函数:
  compare_candidate_models(question, detected_variables, available_data)
  format_model_comparison_markdown(comparison_result)
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 可选模型知识库
# ═══════════════════════════════════════════════════════════════════════════

CANDIDATE_MODELS = [
    {
        "model_id": "s_n_basquin",
        "model_name": "S-N / Basquin fitting",
        "model_name_cn": "S-N/Basquin 曲线拟合",
        "formula": "σ_a = σ_f'(2N_f)^b 或 log(N_f) = A - B·log(σ_a)",
        "applicable_variables": "stress_amplitude → fatigue_life (Nf)",
        "input_parameters": ["σ_a (应力幅)", "N_f (疲劳寿命)", "至少 4 级应力水平"],
        "output_results": ["σ_f' (疲劳强度系数)", "b (疲劳强度指数)", "A, B (S-N 拟合参数)"],
        "applicable_conditions": [
            "相同材料状态和表面状态下的多应力级数据",
            "HCF 区 (10^4-10^7 cycles)",
            "恒定应力比 R",
            "试样几何一致",
        ],
        "inapplicable_cases": [
            "缺陷尺寸可作为额外变量时（需要修正 S-N）",
            "VHCF 区（>10^7 cycles）需双线性 S-N",
            "数据点少于 4 个应力级时拟合不可靠",
            "不同表面状态下数据直接混用",
        ],
        "data_sufficiency_check": [
            "需要至少 4 级应力水平 × 每级 ≥ 3 个试样",
            "需要记录 R 比、表面状态、热处理状态",
            "需要相同材料批次",
        ],
        "strength": "可比较不同条件下的 S-N 曲线差异，直接反映疲劳性能变化",
        "weakness": "不能区分起裂和扩展阶段，不能考虑缺陷尺寸分布",
        "recommendation_level": "recommended",
        "data_ready": False,
    },
    {
        "model_id": "murakami_sqrt_area",
        "model_name": "Murakami √area model",
        "model_name_cn": "Murakami √area 模型",
        "formula": "σ_w = C·(H_V + 120) / (√area)^{1/6}",
        "applicable_variables": "pore_size / √area → fatigue_limit (σw)",
        "input_parameters": [
            "H_V (维氏硬度)",
            "√area (缺陷面积平方根)",
            "位置系数 C（表面=1.43, 近表面=1.41, 内部=1.56）",
            "疲劳极限 σ_w 实测值（用于验证）",
        ],
        "output_results": ["σ_w_pred (预测疲劳极限)", "容许缺陷尺寸", "缺陷严重度排序"],
        "applicable_conditions": [
            "含缺陷材料（铸件/AM 件）的疲劳极限预测",
            "H_V 在 200-500 范围",
            "缺陷 √area 在 10-500μm 范围",
            "R = -1 为基准，其他 R 需修正",
        ],
        "inapplicable_cases": [
            "表面粗糙度主导时（需结合表面等效缺陷）",
            "无缺陷材料（接近锻件）预测偏差大",
            "多种缺陷类型共存时需分别处理",
            "VHCF 区可能需要考虑位置效应",
        ],
        "data_sufficiency_check": [
            "需要 micro-CT 或断口测量的最大 √area 数据",
            "需要 HV 硬度数据",
            "需要升降法获得的疲劳极限",
            "需要确认起裂源对应的缺陷",
        ],
        "strength": "工程实用，输入参数少，预测精度可接受（通常 ±15%）",
        "weakness": "不能预测完整 S-N 曲线，需预设缺陷是关键缺陷",
        "recommendation_level": "recommended",
        "data_ready": False,
    },
    {
        "model_id": "kitagawa_takahashi",
        "model_name": "Kitagawa-Takahashi diagram",
        "model_name_cn": "北川-高桥图",
        "formula": "Δσ_th = ΔK_th / (Y·√(π·a))",
        "applicable_variables": "defect_size a → fatigue_limit OR ΔKth → allowable defect_size",
        "input_parameters": [
            "ΔK_th (疲劳裂纹扩展门槛值)",
            "缺陷尺寸 a 或 √area",
            "几何因子 Y（通常取 0.65 表面缺陷）",
            "疲劳极限 σ_w",
        ],
        "output_results": ["失效/安全边界图", "临界缺陷尺寸 a_c", "容许应力水平"],
        "applicable_conditions": [
            "适用于区分失效和安全区域的边界分析",
            "长裂纹假设适用时",
            "已知 ΔKth 或可通过实验确定",
        ],
        "inapplicable_cases": [
            "小裂纹/小缺陷区需 El Haddad 修正",
            "多种缺陷类型共存时需分别计算",
            "材料各向异性显著时 Y 值需修正",
            "残余应力显著时需修正 ΔKth",
        ],
        "data_sufficiency_check": [
            "需要至少一组 ΔKth 数据",
            "需要缺陷尺寸分布数据",
            "需要疲劳极限数据验证边界",
            "需要 Y 因子估算",
        ],
        "strength": "直观给出安全/失效边界，工程实用",
        "weakness": "小裂纹区不准确，需要修正；只给出边界，不预测寿命",
        "recommendation_level": "recommended",
        "data_ready": False,
    },
    {
        "model_id": "el_haddad",
        "model_name": "El Haddad small-crack correction",
        "model_name_cn": "El Haddad 小裂纹修正",
        "formula": "Δσ_th = ΔK_th / (Y·√(π·(a + a_0)))，其中 a_0 = (1/π)·(ΔK_th/σ_w)^2",
        "applicable_variables": "small_defect_size → fatigue_limit (smooth transition)",
        "input_parameters": [
            "ΔK_th (疲劳裂纹扩展门槛值)",
            "σ_w (疲劳极限)",
            "缺陷尺寸 a",
            "a_0 = intrinsic crack length",
        ],
        "output_results": ["修正后的疲劳极限预测", "小裂纹区平滑过渡边界", "a_0 值"],
        "applicable_conditions": [
            "缺陷尺寸在短裂纹/小缺陷区（a < 1mm）",
            "需要同时已知 ΔKth 和 σw",
            "适用于从缺陷容限到疲劳极限的平滑过渡",
        ],
        "inapplicable_cases": [
            "大裂纹区（K-T 图大裂纹分支足够）",
            "表面粗糙度主导时不适用",
            "a_0 值需从已知 ΔKth 和 σw 推导",
            "残余应力效应需单独考虑",
        ],
        "data_sufficiency_check": [
            "需要 ΔKth 和 σw 数据",
            "需要验证 a_0 的合理性",
            "需要小缺陷试样的疲劳数据验证",
        ],
        "strength": "弥补 K-T 图在小裂纹区的不足，给出平滑过渡",
        "weakness": "a_0 为拟合参数，物理意义有限；需要较多的输入数据",
        "recommendation_level": "conditional",
        "data_ready": False,
    },
    {
        "model_id": "paris_law",
        "model_name": "Paris law (FCGR)",
        "model_name_cn": "Paris 裂纹扩展定律",
        "formula": "da/dN = C·(ΔK)^m",
        "applicable_variables": "Delta_K → da_dN (crack growth rate)",
        "input_parameters": [
            "ΔK (应力强度因子范围)",
            "da/dN (裂纹扩展速率)",
            "Paris C 和 m 参数",
        ],
        "output_results": ["da/dN-ΔK 曲线", "Paris C 和 m 拟合值", "裂纹扩展寿命预测"],
        "applicable_conditions": [
            "稳定裂纹扩展区（Region II）",
            "长裂纹假设成立",
            "CT 或 SEN 试样",
            "恒定或准恒定应力比 R",
        ],
        "inapplicable_cases": [
            "短裂纹区（需修正）",
            "近门槛值区（需考虑 ΔKth）",
            "裂纹起裂阶段（Paris 不适用）",
            "高 R 比下裂纹闭合效应未计入",
        ],
        "data_sufficiency_check": [
            "需要至少 3 个试样的 FCGR 数据",
            "需要 ΔK 和 da/dN 对应的数据点",
            "需要 R 比记录",
            "需要试样几何信息用于 ΔK 计算",
        ],
        "strength": "标准模型、参数含义明确、大量历史数据可对比",
        "weakness": "不能预测起裂寿命，只适用于稳定扩展阶段",
        "recommendation_level": "recommended",
        "data_ready": False,
    },
    {
        "model_id": "walker_model",
        "model_name": "Walker model (R-ratio correction)",
        "model_name_cn": "Walker 模型（R 比修正）",
        "formula": "da/dN = C·(ΔK·(1-R)^{p-1})^m",
        "applicable_variables": "Delta_K + stress_ratio_R → da_dN",
        "input_parameters": [
            "ΔK (应力强度因子范围)",
            "R (应力比)",
            "Paris C 和 m 参数",
            "Walker p 参数",
        ],
        "output_results": ["统一不同 R 比的 da/dN-ΔK 曲线", "Walker p 值", "有效驱动力 ΔK_eff"],
        "applicable_conditions": [
            "需要多个 R 比下的 FCGR 数据",
            "R 比范围 -1 到 0.7",
            "可用于归一化不同 R 比的扩展数据",
        ],
        "inapplicable_cases": [
            "裂纹闭合效应非常显著时需独立的闭合模型",
            "高温或腐蚀环境下 p 值可能不恒定",
        ],
        "data_sufficiency_check": [
            "需要至少 2 个 R 比下的 FCGR 数据",
            "Paris 参数已知",
        ],
        "strength": "可统一不同 R 比的数据，工程实用",
        "weakness": "增加一个拟合参数 p，需要多 R 比数据支持",
        "recommendation_level": "conditional",
        "data_ready": False,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def load_available_data_info() -> Dict[str, bool]:
    """检查当前系统中有哪些数据可供模型使用。"""
    info = {}

    # 检查方程参数库
    eq_path = DATA_DIR / "equation_parameter_dataset.csv"
    if eq_path.exists():
        try:
            df = pd.read_csv(eq_path, encoding="utf-8-sig", on_bad_lines="skip")
            info["has_paris_params"] = any("Paris" in str(v) for v in df.get("equation_or_model", []))
            info["has_basquin_params"] = any("Basquin" in str(v) for v in df.get("equation_or_model", []))
        except Exception:
            pass

    # 检查文献库中的字段
    lit_path = DATA_DIR / "literature_database.csv"
    if lit_path.exists():
        try:
            df = pd.read_csv(lit_path, encoding="utf-8-sig", on_bad_lines="skip")
            info["has_HV"] = any(df.get("HV", pd.Series([""])).dropna().astype(bool))
            info["has_fatigue_limit"] = any(df.get("fatigue_limit", pd.Series([""])).dropna().astype(bool))
            info["has_Nf_data"] = any(df.get("Nf", pd.Series([""])).dropna().astype(bool))
            info["has_Delta_Kth"] = any(df.get("Delta_Kth", pd.Series([""])).dropna().astype(bool))
            info["has_pore_size"] = any(df.get("pore_size", pd.Series([""])).dropna().astype(bool))
        except Exception:
            pass

    # 检查证据库 (v2: 同时检查条件字段)
    ev_path = TRUSTED_EVIDENCE_PATH
    if ev_path.exists():
        try:
            df = pd.read_csv(ev_path, encoding="utf-8-sig", on_bad_lines="skip")
            info["has_surface_roughness_data"] = any(
                "roughness" in str(v).lower() for v in df.get("extracted_claim", [])
            )
            # v2: 检查条件字段中是否包含实际数据
            if "surface_roughness_Ra" in df.columns:
                info["has_surface_roughness_data"] = info["has_surface_roughness_data"] or any(
                    df["surface_roughness_Ra"].dropna().astype(str).str.strip().str.len().gt(0)
                )
            if "pore_size" in df.columns:
                info["has_pore_size"] = info.get("has_pore_size", False) or any(
                    df["pore_size"].dropna().astype(str).str.strip().str.len().gt(0)
                )
        except Exception:
            pass

    return info


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def detect_relevant_models(detected_variables: List[str], question: str = "") -> List[Dict[str, Any]]:
    """根据检测到的变量和问题文本筛选相关模型。"""
    var_set = set(detected_variables)
    q = question.lower() if question else ""
    relevant = []

    model_variable_map = {
        "s_n_basquin": ["stress_amplitude", "fatigue_life", "stress_amplitude", "fatigue_life"],
        "murakami_sqrt_area": ["sqrt_area", "pore_size", "fatigue_limit", "HV"],
        "kitagawa_takahashi": ["pore_size", "defect_size", "fatigue_limit", "delta_k", "Delta_K_threshold"],
        "el_haddad": ["pore_size", "defect_size", "delta_k", "fatigue_limit"],
        "paris_law": ["delta_k", "da_dn", "crack growth rate"],
        "walker_model": ["delta_k", "da_dn", "stress_ratio"],
    }

    for model in CANDIDATE_MODELS:
        mid = model["model_id"]
        req_vars = model_variable_map.get(mid, [])

        # 变量匹配得分
        var_score = sum(1 for v in req_vars if v in var_set)

        # 关键词匹配
        keyword_score = 0
        keywords = {
            "s_n_basquin": ["s-n", "basquin", "stress-life", "应力-寿命", "sn"],
            "murakami_sqrt_area": ["murakami", "sqrt", "√area", "area模型"],
            "kitagawa_takahashi": ["kitagawa", "takahashi", "缺陷容限", "threshold"],
            "el_haddad": ["el haddad", "小裂纹", "short crack", "small crack"],
            "paris_law": ["paris", "da/dn", "crack growth", "裂纹扩展", "fcgr"],
            "walker_model": ["walker", "r比修正"],
        }
        for kw in keywords.get(mid, []):
            if kw in q:
                keyword_score += 1

        relevance = var_score + keyword_score * 2
        if relevance >= 2 or (q and "all" in q):
            relevant.append(model)

    return relevant


def check_data_readiness(
    model: Dict[str, Any], available_data: Dict[str, bool]
) -> Tuple[bool, List[str]]:
    """检查特定模型的数据准备情况。"""
    mid = model["model_id"]
    missing = []

    readiness_checks = {
        "s_n_basquin": ["has_Nf_data"],
        "murakami_sqrt_area": ["has_HV", "has_pore_size", "has_fatigue_limit"],
        "kitagawa_takahashi": ["has_Delta_Kth", "has_pore_size", "has_fatigue_limit"],
        "el_haddad": ["has_Delta_Kth", "has_fatigue_limit"],
        "paris_law": ["has_paris_params"],
        "walker_model": ["has_paris_params"],
    }

    checks = readiness_checks.get(mid, [])
    data_missing = []
    data_ok = True
    for check in checks:
        if not available_data.get(check, False):
            data_missing.append(check.replace("has_", ""))
            data_ok = False

    # 补充具体缺失数据描述
    data_descriptions = {
        "Nf_data": "Nf（疲劳寿命）配对数据",
        "HV": "HV（维氏硬度）",
        "pore_size": "pore_size/√area（缺陷尺寸）",
        "fatigue_limit": "fatigue_limit（疲劳极限）",
        "Delta_Kth": "ΔKth（门槛值）",
        "paris_params": "Paris C/m 参数",
        "surface_roughness_data": "表面粗糙度数据",
    }

    for d in data_missing:
        desc = data_descriptions.get(d, d)
        missing.append(desc)

    return data_ok, missing


def compare_candidate_models(
    question: str = "",
    detected_variables: Optional[List[str]] = None,
    available_data: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    主函数：比较多个候选模型。

    Args:
        question: 用户科研问题
        detected_variables: 从问题中检测到的变量列表
        available_data: 当前系统中可用数据的指示字典

    Returns:
        包含模型比较结果的字典
    """
    if detected_variables is None:
        detected_variables = []
    if available_data is None:
        available_data = load_available_data_info()

    # 检测相关模型
    relevant_models = detect_relevant_models(detected_variables, question)

    # 对每个模型评估数据准备情况
    comparisons = []
    for model in relevant_models:
        data_ready, missing_params = check_data_readiness(model, available_data)
        model_copy = dict(model)
        model_copy["data_ready"] = data_ready
        model_copy["missing_parameters_local"] = missing_params
        comparisons.append(model_copy)

    # 按推荐等级排序
    level_order = {"recommended": 0, "conditional": 1, "not_recommended": 2}
    comparisons.sort(key=lambda m: level_order.get(m.get("recommendation_level", "conditional"), 1))

    # 如果没有匹配，返回全部模型供参考
    if not comparisons:
        for model in CANDIDATE_MODELS:
            data_ready, missing_params = check_data_readiness(model, available_data)
            model_copy = dict(model)
            model_copy["data_ready"] = data_ready
            model_copy["missing_parameters_local"] = missing_params
            comparisons.append(model_copy)

    return {
        "question": question,
        "detected_variables": detected_variables,
        "total_models": len(comparisons),
        "comparisons": comparisons,
        "available_data_summary": {
            k: "✅" if v else "❌" for k, v in available_data.items()
        },
    }


def format_model_comparison_markdown(result: Dict[str, Any]) -> str:
    """
    将模型对比结果格式化为 Markdown 表格。
    """
    lines = []
    comparisons = result.get("comparisons", [])
    data_summary = result.get("available_data_summary", {})

    if not comparisons:
        lines.append("## 模型对比\n")
        lines.append("未检测到相关的候选模型。\n")
        return "".join(lines)

    lines.append("## 📐 多模型对比分析\n")
    lines.append(f"> 检测到变量：{'、'.join(result.get('detected_variables', ['无']))}\n")
    lines.append(f"> 比较 {result['total_models']} 个候选模型\n\n")

    # ── 数据可用性概览 ──
    if data_summary:
        lines.append("### 当前系统数据可用性\n\n")
        lines.append("| 数据类型 | 状态 |\n")
        lines.append("|---|---|\n")
        data_labels = {
            "has_Nf_data": "Nf（疲劳寿命）数据", "has_HV": "HV（硬度）",
            "has_pore_size": "pore_size/√area 数据", "has_fatigue_limit": "疲劳极限数据",
            "has_Delta_Kth": "ΔKth（门槛值）", "has_paris_params": "Paris C/m 参数",
            "has_surface_roughness_data": "表面粗糙度数据", "has_basquin_params": "Basquin 参数",
        }
        for k, v in data_summary.items():
            label = data_labels.get(k, k)
            lines.append(f"| {label} | {v} |\n")
        lines.append("\n")

    # ── 模型对比主表 ──
    lines.append("### 模型对比总表\n\n")
    lines.append("| 模型 | 适用关系 | 输入参数 | 输出结果 | 适用条件 | 不适用情况 | 当前数据足够？ | 推荐等级 |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")

    for model in comparisons:
        mid = model.get("model_id", "")
        name = model.get("model_name_cn", model.get("model_name", ""))
        applicable = model.get("applicable_variables", "")
        inputs = "; ".join(model.get("input_parameters", [])[:3])
        outputs = "; ".join(model.get("output_results", [])[:2])
        conditions = "; ".join(model.get("applicable_conditions", [])[:2])
        inapp = "; ".join(model.get("inapplicable_cases", [])[:2])
        data_ready = model.get("data_ready", False)
        data_tag = "✅ 足够" if data_ready else "❌ 需补充"
        rec = model.get("recommendation_level", "")
        rec_icon = {"recommended": "🟢 推荐", "conditional": "🟡 条件推荐", "not_recommended": "🔴 不推荐"}.get(rec, rec)

        lines.append(f"| {name} | {applicable} | {inputs} | {outputs} | {conditions} | {inapp} | {data_tag} | {rec_icon} |\n")

    lines.append("\n")

    # ── 各模型详细信息 ──
    lines.append("### 各模型详细说明\n\n")
    for model in comparisons:
        name = model.get("model_name_cn", model.get("model_name", ""))
        formula = model.get("formula", "")
        en_name = model.get("model_name", "")

        lines.append(f"#### {name}\n")
        lines.append(f"**{en_name}**\n\n")
        lines.append(f"`{formula}`\n\n")

        lines.append("**输入参数**:\n")
        for p in model.get("input_parameters", []):
            lines.append(f"- {p}\n")
        lines.append("\n")

        lines.append("**输出结果**:\n")
        for o in model.get("output_results", []):
            lines.append(f"- {o}\n")
        lines.append("\n")

        lines.append("**适用条件**:\n")
        for c in model.get("applicable_conditions", []):
            lines.append(f"- {c}\n")
        lines.append("\n")

        lines.append("**不适用情况**:\n")
        for ic in model.get("inapplicable_cases", []):
            lines.append(f"- {ic}\n")
        lines.append("\n")

        # 缺失参数
        missing = model.get("missing_parameters_local", [])
        if missing:
            lines.append("**当前数据缺失**:\n")
            for m in missing:
                lines.append(f"- {m}\n")
            lines.append("\n")

        lines.append(f"**推荐等级**: {rec_icon}\n\n")
        lines.append("---\n")

    # ── 总体建议 ──
    lines.append("### 模型选择建议\n\n")
    recommended = [m for m in comparisons if m.get("recommendation_level") == "recommended"]
    conditional = [m for m in comparisons if m.get("recommendation_level") == "conditional"]

    if recommended:
        lines.append("**优先推荐**:\n")
        for m in recommended:
            lines.append(f"- {m.get('model_name_cn', m.get('model_name', ''))}")
            if not m.get("data_ready"):
                lines.append("（需补充数据）")
            lines.append("\n")
        lines.append("\n")

    if conditional:
        lines.append("**条件推荐**（需满足特定条件后使用）:\n")
        for m in conditional:
            lines.append(f"- {m.get('model_name_cn', m.get('model_name', ''))}\n")
        lines.append("\n")

    lines.append("**建议**：多个模型组合使用往往比单一模型更有效。"
                 "例如，Murakami √area 预测疲劳极限 + Kitagawa-Takahashi 判断缺陷容限 + "
                 "S-N/Basquin 分析完整疲劳寿命。\n")

    return "".join(lines)
