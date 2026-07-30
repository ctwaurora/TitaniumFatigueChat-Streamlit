"""
condition_mechanism_map.py — 条件—机制主导图生成模块

目标：
当系统比较多个机制时，生成"条件—机制主导图"或"机制边界图"，
用于表达不同实验条件下主导疲劳机制的变化。

核心函数:
  generate_condition_mechanism_map(question, variable_pairs, literature_db, evidence_df)
  format_condition_map_markdown(map_result)

输出字段:
  condition_id, surface_state, roughness_level, defect_state, heat_treatment,
  fatigue_regime, stress_ratio_R,
  dominant_mechanism, secondary_mechanism, target_indicator,
  supporting_evidence, counter_evidence, confidence, missing_evidence
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 已知机制模板（基于 L-PBF Ti-6Al-4V 疲劳领域知识）
# ═══════════════════════════════════════════════════════════════════════════

# 条件-机制主导矩阵（预定义知识）
# 列: 条件变量组合，行: 可能的主导机制
CONDITION_MECHANISM_TEMPLATES = [
    {
        "condition_id": "C1",
        "surface_state": "as-built",
        "roughness_level": "high (Ra > 10μm)",
        "defect_state": "natural (with pores)",
        "heat_treatment": "stress-relieved",
        "fatigue_regime": "HCF / VHCF",
        "stress_ratio_R": "0.1",
        "dominant_mechanism": "surface_notch_effect",
        "dominant_mechanism_cn": "表面缺口效应主导裂纹起裂",
        "secondary_mechanism": "near_surface_pore_initiation",
        "secondary_mechanism_cn": "近表面孔隙辅助起裂",
        "target_indicator": "Nf (reduced), crack_initiation_site (surface)",
        "supporting_evidence_template": "as-built L-PBF Ti-6Al-4V 表面粗糙度 Ra 通常在 10-20μm 范围，表面缺口应力集中系数 Kt 可达 2-3，足以在 HCF 中主导起裂。",
        "counter_evidence_template": "部分研究在 as-built 试样中也观察到近表面孔隙起裂，说明孔隙尺寸较大时仍可竞争。",
        "confidence": "high",
        "missing_evidence": "不同 Ra 阈值下主导权切换的定量数据",
    },
    {
        "condition_id": "C2",
        "surface_state": "polished / machined",
        "roughness_level": "low (Ra < 1μm)",
        "defect_state": "natural (with pores)",
        "heat_treatment": "stress-relieved",
        "fatigue_regime": "HCF",
        "stress_ratio_R": "0.1",
        "dominant_mechanism": "near_surface_pore_initiation",
        "dominant_mechanism_cn": "近表面孔隙诱导裂纹起裂",
        "secondary_mechanism": "internal_pore_initiation",
        "secondary_mechanism_cn": "内部孔隙起裂",
        "target_indicator": "Nf (moderately reduced), crack_initiation_site (pore)",
        "supporting_evidence_template": "抛光消除表面缺口后，裂纹起裂源转向近表面孔隙（距表面 < 100μm 的大孔隙）。",
        "counter_evidence_template": "当孔隙整体尺寸较小（√area < 20μm）时，即使 polished 也可能从表面起裂。",
        "confidence": "high",
        "missing_evidence": "近表面孔隙临界 √area 与距表面距离的耦合数据",
    },
    {
        "condition_id": "C3",
        "surface_state": "polished / machined",
        "roughness_level": "low (Ra < 1μm)",
        "defect_state": "HIP (low porosity)",
        "heat_treatment": "HIP",
        "fatigue_regime": "HCF",
        "stress_ratio_R": "0.1",
        "dominant_mechanism": "residual_defect_or_roughness_initiation",
        "dominant_mechanism_cn": "残留缺陷或表面特征起裂",
        "secondary_mechanism": "microstructural_barrier",
        "secondary_mechanism_cn": "微观组织屏障效应",
        "target_indicator": "Nf (improved), fatigue_limit (improved)",
        "supporting_evidence_template": "HIP 可闭合大部分内部孔隙，但表面连通孔隙和超大未熔合缺陷可能残留，成为新起裂源。",
        "counter_evidence_template": "经过 HIP 后，部分研究报道 fatigue limit 接近锻件水平，说明残留缺陷影响有限。",
        "confidence": "medium",
        "missing_evidence": "HIP 后残留缺陷的尺寸/位置分布与疲劳寿命的系统数据",
    },
    {
        "condition_id": "C4",
        "surface_state": "polished / machined",
        "roughness_level": "low (Ra < 1μm)",
        "defect_state": "HIP (low porosity)",
        "heat_treatment": "HIP + annealing",
        "fatigue_regime": "FCGR (Region II)",
        "stress_ratio_R": "0.1",
        "dominant_mechanism": "delta_K_controlled_crack_growth",
        "dominant_mechanism_cn": "ΔK 控制的裂纹扩展（Paris law）",
        "secondary_mechanism": "crack_closure_effect",
        "secondary_mechanism_cn": "裂纹闭合效应",
        "target_indicator": "da/dN, Paris C/m",
        "supporting_evidence_template": "在长裂纹 FCGR 阶段，da/dN-ΔK 关系主要服从 Paris 定律，微观组织和残余应力的调节作用减弱。",
        "counter_evidence_template": "在近门槛区，缺陷和微观组织对 ΔKth 仍有显著影响，Paris 定律不再适用。",
        "confidence": "high",
        "missing_evidence": "L-PBF Ti-6Al-4V 在不同 R 比下 Paris 参数的完整数据集",
    },
    {
        "condition_id": "C5",
        "surface_state": "as-built",
        "roughness_level": "high (Ra > 10μm)",
        "defect_state": "HIP (low porosity)",
        "heat_treatment": "HIP",
        "fatigue_regime": "HCF",
        "stress_ratio_R": "0.1",
        "dominant_mechanism": "surface_roughness_dominated",
        "dominant_mechanism_cn": "表面粗糙度主导（HIP 不能消除表面缺陷）",
        "secondary_mechanism": "subsurface_residual_defect",
        "secondary_mechanism_cn": "亚表面残留缺陷",
        "target_indicator": "Nf (moderate improvement vs as-built SR), crack_initiation_site (surface)",
        "supporting_evidence_template": "HIP 消除内部孔隙，但 as-built 表面粗糙度仍然存在，表面缺口效应仍是主导。",
        "counter_evidence_template": "HIP 后残余应力大幅降低，部分抵消表面缺口效应，因此 Nf 仍有一定提升。",
        "confidence": "medium",
        "missing_evidence": "as-built+HIP 与 polished+HIP 组的直接对比数据",
    },
    {
        "condition_id": "C6",
        "surface_state": "polished / machined",
        "roughness_level": "low (Ra < 1μm)",
        "defect_state": "natural (with large internal pores)",
        "heat_treatment": "stress-relieved",
        "fatigue_regime": "VHCF (10^7-10^8 cycles)",
        "stress_ratio_R": "-1",
        "dominant_mechanism": "internal_pore_fish_eye",
        "dominant_mechanism_cn": "内部孔隙鱼眼型起裂（内部起裂 + 细粒区）",
        "secondary_mechanism": "surface_initiation",
        "secondary_mechanism_cn": "表面起裂",
        "target_indicator": "Nf (very high cycles), crack_initiation_site (internal)",
        "supporting_evidence_template": "在 VHCF 中，表面起裂驱动力降低，内部大孔隙周围形成鱼眼(fish-eye)和细粒区(FCG)，成为主导起裂模式。",
        "counter_evidence_template": "若表面存在划痕或微小缺陷，即使在 VHCF 中表面起裂仍可能发生。",
        "confidence": "medium",
        "missing_evidence": "L-PBF Ti-6Al-4V 在 VHCF 下的系统数据",
    },
]

# 机制中文名映射
MECHANISM_CN = {
    "surface_notch_effect": "表面缺口效应",
    "near_surface_pore_initiation": "近表面孔隙起裂",
    "internal_pore_initiation": "内部孔隙起裂",
    "residual_defect_or_roughness_initiation": "残留缺陷/粗糙度起裂",
    "microstructural_barrier": "微观组织屏障",
    "delta_K_controlled_crack_growth": "ΔK 控制裂纹扩展",
    "crack_closure_effect": "裂纹闭合效应",
    "surface_roughness_dominated": "表面粗糙度主导",
    "subsurface_residual_defect": "亚表面残留缺陷",
    "internal_pore_fish_eye": "内部孔隙鱼眼型起裂",
    "surface_initiation": "表面起裂",
    "surface_roughness_induced_cracking": "表面粗糙度诱导裂纹",
    "crack_initiation_from_defects": "缺陷诱导裂纹起裂",
    "crack_propagation": "裂纹扩展",
    "stress_concentration_at_pores": "孔隙处应力集中",
    "HIP_defect_closure": "HIP 缺陷闭合",
}


def load_evidence_data() -> pd.DataFrame:
    """加载证据数据。"""
    path = TRUSTED_EVIDENCE_PATH
    if path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_literature_data() -> pd.DataFrame:
    """加载文献数据。"""
    path = DATA_DIR / "literature_database.csv"
    if path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def classify_roughness_level(ra_val: str) -> str:
    """根据 Ra 值分类粗糙度等级。"""
    if not ra_val:
        return "unknown"
    try:
        ra = float(re.sub(r"[^\d.]", "", ra_val))
        if ra > 10:
            return "high (Ra > 10μm)"
        elif ra > 3:
            return "medium (Ra 3-10μm)"
        elif ra > 1:
            return "low (Ra 1-3μm)"
        else:
            return "very low (Ra < 1μm)"
    except ValueError:
        return "unknown"


def match_conditions_to_template(
    surface_state: str,
    roughness: str,
    defect: str,
    heat_treatment: str,
    fatigue_regime: str,
    r_ratio: str,
) -> List[Dict[str, Any]]:
    """将用户查询的条件匹配到最相关的机制模板。"""
    ss = surface_state.lower() if surface_state else ""
    ht = heat_treatment.lower() if heat_treatment else ""
    fr = fatigue_regime.lower() if fatigue_regime else ""
    rr = r_ratio if r_ratio else ""

    matched = []
    for tmpl in CONDITION_MECHANISM_TEMPLATES:
        score = 0
        # Surface state match
        if "as-built" in ss and "as-built" in tmpl["surface_state"]:
            score += 3
        elif ("polish" in ss or "machin" in ss) and "polished" in tmpl["surface_state"]:
            score += 3

        # Roughness level match
        rl = classify_roughness_level(roughness)
        if rl != "unknown":
            rl_high = "high" in rl or "Ra > 10" in rl
            rl_low = "low" in rl or "very low" in rl
            tmpl_high = "high" in tmpl["roughness_level"]
            tmpl_low = "low" in tmpl["roughness_level"]
            if (rl_high and tmpl_high) or (rl_low and tmpl_low):
                score += 2

        # Heat treatment match
        if "hip" in ht and "HIP" in tmpl["heat_treatment"]:
            score += 2
        elif ("anneal" in ht or "stress" in ht) and "stress-relieved" in tmpl["heat_treatment"]:
            score += 1

        # Fatigue regime match
        if "hcf" in fr and "HCF" in tmpl["fatigue_regime"]:
            score += 2
        elif "vhcf" in fr and "VHCF" in tmpl["fatigue_regime"]:
            score += 2
        elif "fcgr" in fr and "FCGR" in tmpl["fatigue_regime"]:
            score += 2

        # Stress ratio match
        if rr and rr in tmpl["stress_ratio_R"]:
            score += 1

        if score >= 3:
            matched.append((score, tmpl))

    matched.sort(key=lambda x: -x[0])
    return [m[1] for m in matched]


def find_relevant_evidence(
    mechanism: str, condition: Dict[str, str], evidence_df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """从证据库中查找支持该机制和条件的证据。"""
    supporting = []
    conflicting = []

    if evidence_df.empty:
        return supporting, conflicting

    # 根据机制关键词查找
    mech_keywords = mechanism.replace("_", " ").lower().split("_")
    mech_keywords = [w for w in mech_keywords if len(w) > 2]

    for _, row in evidence_df.iterrows():
        claim = str(row.get("extracted_claim", "") or "")
        mech = str(row.get("mechanism", "") or "")
        ev_type = str(row.get("evidence_type", "") or "")

        claim_lower = claim.lower()
        mech_lower = mech.lower()

        # 检查是否匹配机制
        is_match = any(kw in mech_lower or kw in claim_lower for kw in mech_keywords)
        if not is_match:
            continue

        # 检查是否冲突
        if "conflict" in ev_type or "conflict" in claim_lower:
            conflicting.append(claim[:150])
        else:
            supporting.append(claim[:150])

    return supporting[:5], conflicting[:3]


def generate_condition_mechanism_map(
    question: str = "",
    surface_state: str = "",
    roughness: str = "",
    defect_state: str = "",
    heat_treatment: str = "",
    fatigue_regime: str = "",
    stress_ratio_R: str = "",
) -> Dict[str, Any]:
    """
    主函数：生成条件—机制主导图。

    根据给定条件或问题描述，匹配预定义的机制模板，
    并结合实际证据生成条件-机制映射。

    Args:
        question: 用户原始问题（用于自动推断条件）
        surface_state: as-built / polished / machined
        roughness: Ra 值或描述
        defect_state: 缺陷状态描述
        heat_treatment: 热处理状态
        fatigue_regime: HCF / VHCF / FCGR
        stress_ratio_R: 应力比

    Returns:
        包含条件-机制映射结果的字典
    """
    evidence_df = load_evidence_data()
    literature_df = load_literature_data()

    # 尝试从问题中推断条件
    if question:
        q = question.lower()
        if not surface_state:
            if "as-built" in q:
                surface_state = "as-built"
            elif "polish" in q or "machin" in q:
                surface_state = "polished/machined"
        if not fatigue_regime:
            if "vhcf" in q:
                fatigue_regime = "VHCF"
            elif "hcf" in q or "high cycle" in q:
                fatigue_regime = "HCF"
            elif "fcgr" in q or "crack growth" in q or "paris" in q:
                fatigue_regime = "FCGR"
        if not heat_treatment:
            if "hip" in q:
                heat_treatment = "HIP"

    # 匹配模板
    matched = match_conditions_to_template(
        surface_state, roughness, defect_state,
        heat_treatment, fatigue_regime, stress_ratio_R
    )

    # 生成完整 map 条目
    map_entries = []
    if matched:
        for tmpl in matched:
            # 从证据库查找
            mech = tmpl["dominant_mechanism"]
            cond_dict = {
                "surface_state": tmpl["surface_state"],
                "heat_treatment": tmpl["heat_treatment"],
                "defect_state": tmpl["defect_state"],
                "roughness_level": tmpl["roughness_level"],
                "fatigue_regime": tmpl["fatigue_regime"],
                "stress_ratio_R": tmpl["stress_ratio_R"],
            }
            supporting_ev, conflicting_ev = find_relevant_evidence(mech, cond_dict, evidence_df)

            entry = {
                "condition_id": tmpl["condition_id"],
                "surface_state": tmpl["surface_state"],
                "roughness_level": tmpl["roughness_level"],
                "defect_state": tmpl["defect_state"],
                "heat_treatment": tmpl["heat_treatment"],
                "fatigue_regime": tmpl["fatigue_regime"],
                "stress_ratio_R": tmpl["stress_ratio_R"],
                "dominant_mechanism": tmpl["dominant_mechanism"],
                "dominant_mechanism_cn": tmpl["dominant_mechanism_cn"],
                "secondary_mechanism": tmpl["secondary_mechanism"],
                "secondary_mechanism_cn": tmpl["secondary_mechanism_cn"],
                "target_indicator": tmpl["target_indicator"],
                "supporting_evidence": supporting_ev if supporting_ev else [tmpl["supporting_evidence_template"]],
                "counter_evidence": conflicting_ev if conflicting_ev else (
                    [tmpl["counter_evidence_template"]] if tmpl["counter_evidence_template"] else []
                ),
                "confidence": tmpl["confidence"],
                "missing_evidence": tmpl["missing_evidence"],
            }
            map_entries.append(entry)
    else:
        # 无精确匹配，返回通用条件-机制对照
        for tmpl in CONDITION_MECHANISM_TEMPLATES:
            map_entries.append({
                "condition_id": tmpl["condition_id"],
                "surface_state": tmpl["surface_state"],
                "roughness_level": tmpl["roughness_level"],
                "defect_state": tmpl["defect_state"],
                "heat_treatment": tmpl["heat_treatment"],
                "fatigue_regime": tmpl["fatigue_regime"],
                "stress_ratio_R": tmpl["stress_ratio_R"],
                "dominant_mechanism": tmpl["dominant_mechanism"],
                "dominant_mechanism_cn": tmpl["dominant_mechanism_cn"],
                "secondary_mechanism": tmpl["secondary_mechanism"],
                "secondary_mechanism_cn": tmpl["secondary_mechanism_cn"],
                "target_indicator": tmpl["target_indicator"],
                "supporting_evidence": [tmpl["supporting_evidence_template"]],
                "counter_evidence": [tmpl["counter_evidence_template"]] if tmpl["counter_evidence_template"] else [],
                "confidence": tmpl["confidence"],
                "missing_evidence": tmpl["missing_evidence"],
            })

    return {
        "query_conditions": {
            "surface_state": surface_state,
            "roughness": roughness,
            "defect_state": defect_state,
            "heat_treatment": heat_treatment,
            "fatigue_regime": fatigue_regime,
            "stress_ratio_R": stress_ratio_R,
        },
        "matched_count": len(matched),
        "entries": map_entries,
        "total_conditions": len(CONDITION_MECHANISM_TEMPLATES),
    }


def generate_competition_map(
    mechanisms: List[str],
    evidence_df: pd.DataFrame,
) -> str:
    """
    生成多机制竞争的文本描述。

    当系统比较多个机制时，输出机制间的竞争与边界关系。
    """
    lines = []
    if not mechanisms:
        return ""

    lines.append("### 多机制竞争分析\n")
    lines.append("不同条件下可能存在以下竞争机制：\n")

    # 已知机制竞争关系
    competition_pairs = [
        ("surface_notch_effect", "near_surface_pore_initiation",
         "as-built 表面粗糙度较高时表面缺口效应占主导；表面改善后近表面孔隙起裂机制转为主导。"
         "存在临界 Ra 值（约 3-10μm），低于该值时主导权转移。"),
        ("near_surface_pore_initiation", "internal_pore_fish_eye",
         "HCF 中近表面孔隙起裂占主导；VHCF 中内部孔隙鱼眼型起裂可能转为主导。"
         "临界循环数约 10^6-10^7 cycles。"),
        ("surface_notch_effect", "residual_defect_or_roughness_initiation",
         "两者均由表面特征主导，但 HIP 可部分消除内部缺陷的贡献。"
         "as-built+HIP 后仍由表面粗糙度主导。"),
    ]

    for mech_a, mech_b, description in competition_pairs:
        if (mech_a in mechanisms or any(m in mech_a for m in mechanisms)) or \
           (mech_b in mechanisms or any(m in mech_b for m in mechanisms)):
            a_cn = MECHANISM_CN.get(mech_a, mech_a)
            b_cn = MECHANISM_CN.get(mech_b, mech_b)
            lines.append(f"- **{a_cn}** ↔ **{b_cn}**\n")
            lines.append(f"  {description}\n")

    return "".join(lines)


def format_condition_map_markdown(map_result: Dict[str, Any]) -> str:
    """
    将条件-机制主导图格式化为 Markdown 表格。
    供 streamlit_app.py 直接嵌入综合回答。
    """
    lines = []
    entries = map_result.get("entries", [])
    query = map_result.get("query_conditions", {})

    if not entries:
        lines.append("## 条件—机制主导图\n")
        lines.append("当前条件未匹配到已知机制模板。\n")
        return "".join(lines)

    lines.append("## 条件—机制主导图\n")
    lines.append(f"> 基于 {len(entries)} 种已知条件-机制模板的匹配结果\n")

    # 查询条件显示
    active_conds = {k: v for k, v in query.items() if v}
    if active_conds:
        lines.append("**查询条件**: ")
        cond_strs = []
        for k, v in active_conds.items():
            cn_names = {
                "surface_state": "表面状态", "roughness": "粗糙度", "defect_state": "缺陷状态",
                "heat_treatment": "热处理", "fatigue_regime": "疲劳类型", "stress_ratio_R": "应力比 R",
            }
            cond_strs.append(f"{cn_names.get(k, k)}={v}")
        lines.append("、".join(cond_strs) + "\n\n")

    # ── 简版总览表 (论文级格式) ──
    lines.append("### 总览：条件—机制主导图\n\n")
    lines.append("| 条件 | 可能主导机制 | 次要机制 | 目标指标 | 支持证据数 | 反向证据数 | 证据等级 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for entry in entries:
        dom_mech = entry.get("dominant_mechanism_cn", entry.get("dominant_mechanism", ""))
        sec_mech = entry.get("secondary_mechanism_cn", entry.get("secondary_mechanism", ""))
        n_support = len(entry.get("supporting_evidence", []))
        n_counter = len(entry.get("counter_evidence", []))
        confidence_icon = {"high": "🟢 high", "medium": "🟡 medium", "low": "🟠 low"}.get(entry.get("confidence", ""), "⚪")
        cond_str = f"C{entry.get('condition_id', '?')}"
        lines.append(
            f"| {cond_str} | {dom_mech} | {sec_mech} "
            f"| {entry.get('target_indicator', '')} "
            f"| {n_support} | {n_counter} "
            f"| {confidence_icon} |\n"
        )
    lines.append("\n")

    # ── 详细对比表 ──
    lines.append("### 详细条件对比表\n\n")
    lines.append("| 条件 | 表面状态 | 粗糙度 | 缺陷 | 热处理 | 疲劳类型 | R | 主导机制 | 次级机制 | 目标指标 | 证据等级 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")

    for entry in entries:
        dom_mech = entry.get("dominant_mechanism_cn", entry.get("dominant_mechanism", ""))
        sec_mech = entry.get("secondary_mechanism_cn", entry.get("secondary_mechanism", ""))
        confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🟠"}.get(entry.get("confidence", ""), "⚪")

        # 条件列（简写）
        cond_str = f"C{entry.get('condition_id', '?')}"

        lines.append(
            f"| {cond_str} "
            f"| {entry.get('surface_state', '')} "
            f"| {entry.get('roughness_level', '')} "
            f"| {entry.get('defect_state', '')} "
            f"| {entry.get('heat_treatment', '')} "
            f"| {entry.get('fatigue_regime', '')} "
            f"| {entry.get('stress_ratio_R', '')} "
            f"| {dom_mech} "
            f"| {sec_mech} "
            f"| {entry.get('target_indicator', '')} "
            f"| {confidence_icon} {entry.get('confidence', '')} |\n"
        )

    lines.append("\n")

    # ── 详细信息 ──
    lines.append("### 各条件详细分析\n\n")
    for entry in entries:
        cid = entry.get("condition_id", "")
        dom_cn = entry.get("dominant_mechanism_cn", "")
        sec_cn = entry.get("secondary_mechanism_cn", "")
        supporting = entry.get("supporting_evidence", [])
        conflicting = entry.get("counter_evidence", [])
        missing = entry.get("missing_evidence", "")

        lines.append(f"#### 条件 {cid}: {entry.get('surface_state', '')} / {entry.get('roughness_level', '')}\n\n")
        lines.append(f"- **主导机制**: {dom_cn}\n")
        lines.append(f"- **次级机制**: {sec_cn}\n")
        lines.append(f"- **目标指标**: {entry.get('target_indicator', '')}\n\n")

        lines.append("**支持证据**:\n")
        if supporting:
            for ev in supporting[:3]:
                lines.append(f"- {ev}\n")
        else:
            lines.append("- 无直接匹配证据\n")
        lines.append("\n")

        if conflicting:
            lines.append("**反向证据 (Counter-evidence)**:\n")
            for ev in conflicting[:2]:
                lines.append(f"- {ev}\n")
            lines.append("\n")

        if missing:
            lines.append(f"**缺失证据**: {missing}\n\n")
        else:
            lines.append(f"**缺失证据**: 当前模板暂无记录\n\n")

        lines.append("---\n")

    return "".join(lines)
