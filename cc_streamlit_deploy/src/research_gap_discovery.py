"""
research_gap_discovery.py — Research Gap Discovery Engine
面向 L-PBF Ti-6Al-4V 疲劳领域自动识别研究空白。

输入:
    - literature_review text (from docs/ or uploaded)
    - literature_database.csv
    - candidate_papers.csv
    - evidence_snippets.csv
    - variable_mechanism.csv
    - equation_parameters.csv
    - conflict_claims.csv

输出:
    - data/research_gaps.csv
    - data/gap_hypotheses.csv
    - outputs/24_research_gap_report.md
    - outputs/25_prioritized_gap_hypotheses.md

空白类型:
    evidence_gap, parameter_gap, mechanism_gap, conflict_gap,
    validation_gap, boundary_condition_gap, data_gap, translation_gap
"""

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

GAP_FIELDS = [
    "gap_id", "gap_type", "gap_title", "gap_statement",
    "related_variables", "target_indicators",
    "existing_evidence_summary", "supporting_papers", "conflicting_papers",
    "missing_evidence", "why_it_matters",
    "potential_equation_or_model", "candidate_hypothesis",
    "experimental_design", "required_data_fields", "falsification_condition",
    "novelty_score", "scientific_value_score", "testability_score",
    "feasibility_score", "parameter_potential_score",
    "conflict_usefulness_score", "paper_potential_score",
    "total_priority_score", "priority_level", "status",
]

HYPOTHESIS_FIELDS = [
    "gap_id", "hypothesis_id", "hypothesis_statement",
    "independent_variable", "dependent_variable", "controlled_variables",
    "mechanism_chain", "possible_equation_or_model",
    "experimental_design", "support_criteria", "falsification_condition",
    "hypothesis_score", "priority_rank",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data loaders (fail-safe)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        pass
    return pd.DataFrame()


def load_all_inputs() -> Dict[str, Any]:
    return {
        "lit_db": _safe_read_csv(DATA_DIR / "literature_database.csv"),
        "candidates": _safe_read_csv(DATA_DIR / "candidate_papers.csv"),
        "evidence": _safe_read_csv(TRUSTED_EVIDENCE_PATH),
        "var_mech": _safe_read_csv(DATA_DIR / "variable_mechanism.csv"),
        "equation_params": _safe_read_csv(DATA_DIR / "equation_parameters.csv"),
        "conflicts": _safe_read_csv(DATA_DIR / "conflict_claims.csv"),
    }


def load_review_text(path: Optional[Path] = None) -> str:
    """Load literature review from file or default docs location."""
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    default_paths = [
        BASE_DIR / "docs" / "literature_review.md",
        BASE_DIR / "docs" / "literature_review.txt",
    ]
    for p in default_paths:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. Gap templates — known common gaps in L-PBF Ti-6Al-4V fatigue
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_GAP_TEMPLATES: List[Dict[str, Any]] = [
    {
        "gap_type": "mechanism_gap",
        "gap_title": "近表面孔隙尺寸与距表面距离对疲劳寿命 Nf 的耦合影响缺少直接验证",
        "related_variables": "pore_size; distance_to_surface; pore_location",
        "target_indicators": "Nf; crack_initiation_site",
        "existing_evidence_summary": "文献普遍认为孔隙缺陷降低疲劳寿命，但直接同时区分 pore_size 和 distance_to_surface 对 Nf 和起裂源的耦合影响的实验数据不足。",
        "missing_evidence": "同时包含 pore_size、distance_to_surface 和 Nf 的结构化数据；micro-CT 三维定位 + HCF 对应数据",
        "why_it_matters": "确定孔隙尺寸和距表面距离的耦合效应有助于建立更准确的缺陷容限判据，指导工艺优化和寿命预测。",
        "potential_equation_or_model": "Murakami sqrt-area + surface correction f(distance); Kitagawa-Takahashi with position-dependent ΔKth",
        "novelty_score": 12,
        "scientific_value_score": 17,
        "testability_score": 18,
        "feasibility_score": 15,
        "parameter_potential_score": 8,
        "conflict_usefulness_score": 5,
        "paper_potential_score": 8,
    },
    {
        "gap_type": "conflict_gap",
        "gap_title": "HIP 后残余近表面缺陷是否仍控制疲劳起裂尚缺少系统证据",
        "related_variables": "heat_treatment; hip; surface_state; pore_size",
        "target_indicators": "Nf; crack_initiation_site; fatigue_limit",
        "existing_evidence_summary": "一些文献指出 HIP 可显著改善疲劳性能，另一些发现 HIP 后仍由残留缺陷起裂。两类结论可能受表面状态和初始孔隙特征调节。",
        "missing_evidence": "HIP + 不同表面状态 (as-built vs polished) 的系统对比；HIP 前后 micro-CT + HCF + SEM 数据",
        "why_it_matters": "明确 HIP 的局限性和适用条件，避免过度依赖 HIP 而忽略表面处理。",
        "potential_equation_or_model": "Murakami sqrt-area 对比 HIP 前后; Kitagawa-Takahashi 评估残留缺陷",
        "novelty_score": 10,
        "scientific_value_score": 16,
        "testability_score": 17,
        "feasibility_score": 14,
        "parameter_potential_score": 7,
        "conflict_usefulness_score": 9,
        "paper_potential_score": 7,
    },
    {
        "gap_type": "parameter_gap",
        "gap_title": "缺陷特征与 Paris 参数 C/m 的定量关系缺少系统提取",
        "related_variables": "pore_size; porosity; defect_state; Paris_C; Paris_m",
        "target_indicators": "da_dN; Paris_C; Paris_m",
        "existing_evidence_summary": "文献提到缺陷影响 FCGR，但缺乏系统比较不同缺陷状态下 Paris C 和 m 的定量差异。",
        "missing_evidence": "不同孔隙特征下的 FCGR 测试数据；Paris C/m 拟合值；R 比和表面状态记录",
        "why_it_matters": "Paris 参数是寿命预测模型的关键输入，明确缺陷对 C/m 的影响规律可改进预测精度。",
        "potential_equation_or_model": "Paris law: da/dN = C(ΔK)^m; Walker model",
        "novelty_score": 11,
        "scientific_value_score": 16,
        "testability_score": 16,
        "feasibility_score": 13,
        "parameter_potential_score": 10,
        "conflict_usefulness_score": 6,
        "paper_potential_score": 8,
    },
    {
        "gap_type": "boundary_condition_gap",
        "gap_title": "表面粗糙度与内部孔隙主导疲劳失效的条件边界尚未量化",
        "related_variables": "surface_roughness; pore_size; pore_location",
        "target_indicators": "Nf; crack_initiation_site",
        "existing_evidence_summary": "独立研究分别支持表面粗糙度主导或内部孔隙主导的结论，但缺少两者竞争失效的定量边界条件 (如临界 Ra 值、临界孔隙尺寸)。",
        "missing_evidence": "同时改变 Ra 和孔隙尺寸的系统实验数据；竞争主导的统计模型",
        "why_it_matters": "确定主导权转移的临界条件有助于针对性选择后处理工艺。",
        "potential_equation_or_model": "Kitagawa-type competition map; 孔隙 vs 粗糙度主导权边界方程",
        "novelty_score": 14,
        "scientific_value_score": 18,
        "testability_score": 16,
        "feasibility_score": 13,
        "parameter_potential_score": 8,
        "conflict_usefulness_score": 9,
        "paper_potential_score": 9,
    },
    {
        "gap_type": "evidence_gap",
        "gap_title": "残余应力对 ΔKth 和短裂纹扩展的影响缺少定量数据",
        "related_variables": "residual_stress; Delta_Kth; crack_closure",
        "target_indicators": "Delta_Kth; da_dN; crack_initiation",
        "existing_evidence_summary": "残余应力被提及作为调节因素，但缺少直接测量残余应力与 ΔKth 或短裂纹扩展关系的系统数据。",
        "missing_evidence": "残余应力定量测量 (XRD) + FCGR 测试对应数据；不同应力消除工艺对比",
        "why_it_matters": "残余应力是 L-PBF 工艺的固有特征，其对门槛值和短裂纹行为的影响直接决定寿命预测模型精度。",
        "potential_equation_or_model": "ΔKth = f(σ_res); crack closure model; Walker model with residual stress correction",
        "novelty_score": 13,
        "scientific_value_score": 15,
        "testability_score": 13,
        "feasibility_score": 11,
        "parameter_potential_score": 9,
        "conflict_usefulness_score": 7,
        "paper_potential_score": 8,
    },
    {
        "gap_type": "data_gap",
        "gap_title": "L-PBF Ti-6Al-4V 高周疲劳数据缺乏统一的孔隙特征-寿命对应结构",
        "related_variables": "pore_size; sqrt_area; porosity; pore_aspect_ratio; Nf",
        "target_indicators": "Nf; fatigue_limit",
        "existing_evidence_summary": "许多独立实验包含部分孔隙特征和疲劳数据，但缺少可合并用于建模的标准化结构化数据。",
        "missing_evidence": "标准化字段 (pore_size; sqrt_area; distance_to_surface; porosity; Nf; stress_ratio_R; surface_state; heat_treatment)",
        "why_it_matters": "结构化数据是建立机器学习预测模型或统一经验公式的基础。",
        "potential_equation_or_model": "多变量回归; S-N with defect correction; Murakami √area",
        "novelty_score": 9,
        "scientific_value_score": 13,
        "testability_score": 15,
        "feasibility_score": 14,
        "parameter_potential_score": 8,
        "conflict_usefulness_score": 4,
        "paper_potential_score": 6,
    },
    {
        "gap_type": "translation_gap",
        "gap_title": "基于缺陷容限的 L-PBF Ti-6Al-4V 疲劳寿命预测方法尚未建立工程可用模型",
        "related_variables": "defect_state; stress_amplitude; Nf",
        "target_indicators": "Nf; fatigue_limit",
        "existing_evidence_summary": "已有 Murakami、Kitagawa-Takahashi、Paris law 等模型，但缺少集成这些模型、考虑 L-PBF 特有缺陷特征的工程寿命预测方法。",
        "missing_evidence": "综合模型验证数据；多变量预测模型在独立数据集上的验证",
        "why_it_matters": "从机理到工程应用的关键跳板，直接影响 L-PBF Ti-6Al-4V 承力件的设计准则。",
        "potential_equation_or_model": "Murakami √area + modified S-N; 缺陷-S-N 曲面; 概率寿命预测模型",
        "novelty_score": 11,
        "scientific_value_score": 15,
        "testability_score": 12,
        "feasibility_score": 10,
        "parameter_potential_score": 8,
        "conflict_usefulness_score": 5,
        "paper_potential_score": 9,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Review-based gap discovery
# ═══════════════════════════════════════════════════════════════════════════

def extract_claims_from_review(review_text: str) -> List[str]:
    """从文献综述中提取已知结论/观点。"""
    if not review_text:
        return []
    claims = []
    lines = review_text.split("\n")
    for line in lines:
        # Look for statements containing known keywords
        keywords = [
            "影响", "关系", "相关", "降低", "提高", "改善",
            "导致", "促进", "抑制", "控制", "决定",
            "affects", "influences", "reduces", "improves",
            "controls", "determines", "depends",
        ]
        if any(kw in line.lower() for kw in keywords):
            clean = line.strip().strip("-*#")
            if len(clean) > 20 and len(clean) < 300:
                claims.append(clean)
    return claims[:30]


def match_claims_to_evidence(
    claims: List[str],
    data: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """比对综述中的 claim 与现有证据的覆盖情况。"""
    gaps = []
    # Simple approach: check if each claim's variables have matching evidence
    var_db = set()
    for df_name, df in data.items():
        if df.empty:
            continue
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("variable_name", "variable", "independent_variable",
                       "related_variables", "linked_variable"):
                vals = df[col].dropna().astype(str).str.lower().unique()
                var_db.update(vals)

    for claim in claims:
        text_lower = claim.lower()
        # Check if any key variable in this claim is NOT in evidence
        var_keywords = [
            "pore", "porosity", "roughness", "defect", "fatigue",
            "crack", "microstructure", "heat treatment", "hip",
            "stress", "residual", "orientation", "paris",
        ]
        has_evidence = any(kw in text_lower and any(kw in v for v in var_db)
                          for kw in var_keywords)
        if not has_evidence:
            gaps.append({
                "gap_type": "evidence_gap",
                "gap_title": f"综述提及但缺乏直接证据: {claim[:60]}",
                "related_variables": "",
                "target_indicators": "",
                "missing_evidence": claim[:200],
                "why_it_matters": "综述中提到的关系需要在本地数据库中有直接证据支持。",
            })
    return gaps


# ═══════════════════════════════════════════════════════════════════════════
# 4. Evidence-based gap discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_gaps_from_evidence(data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """从现有证据覆盖情况中发现空白。"""
    discovered = []

    # Check variable_mechanism for missing relations
    vm = data.get("var_mech", pd.DataFrame())
    if not vm.empty and "missing_evidence" in vm.columns:
        missing = vm["missing_evidence"].dropna().unique()
        for m in missing[:10]:
            m_str = str(m).strip()
            if len(m_str) > 10:
                discovered.append({
                    "gap_type": "evidence_gap",
                    "gap_title": f"机制表标识证据缺失: {m_str[:60]}",
                    "related_variables": "",
                    "missing_evidence": m_str,
                    "why_it_matters": "变量机制表中明确标识了当前证据不足以建立关系。",
                })

    # Check for conflicts that haven't been resolved
    conflicts = data.get("conflicts", pd.DataFrame())
    if not conflicts.empty and "conflict_type" in conflicts.columns:
        direct_conflicts = conflicts[conflicts["conflict_type"] == "direct_conflict"]
        if len(direct_conflicts) > 0:
            for _, row in direct_conflicts.iterrows():
                topic = str(row.get("topic", ""))
                if topic:
                    discovered.append({
                        "gap_type": "conflict_gap",
                        "gap_title": f"文献冲突待解释: {topic[:60]}",
                        "conflicting_papers": f"A: {row.get('paper_A', '')}; B: {row.get('paper_B', '')}",
                        "missing_evidence": "需要系统对比实验解释冲突",
                        "why_it_matters": "直接冲突表明存在未控制的中间变量。",
                    })

    return discovered


# ═══════════════════════════════════════════════════════════════════════════
# 5. Gap scoring
# ═══════════════════════════════════════════════════════════════════════════

def compute_gap_scores(gap_base: Dict[str, Any]) -> Dict[str, Any]:
    """计算研究空白的完整评分。"""
    novelty = gap_base.get("novelty_score", 8)
    scientific = gap_base.get("scientific_value_score", 12)
    testability = gap_base.get("testability_score", 12)
    feasibility = gap_base.get("feasibility_score", 10)
    param_potential = gap_base.get("parameter_potential_score", 5)
    conflict_use = gap_base.get("conflict_usefulness_score", 5)
    paper_potential = gap_base.get("paper_potential_score", 6)

    total = (novelty + scientific + testability + feasibility +
             param_potential + conflict_use + paper_potential)

    if total >= 80:
        priority = "high_priority"
    elif total >= 60:
        priority = "medium_priority"
    elif total >= 40:
        priority = "low_priority"
    else:
        priority = "reject"

    gap_base.update({
        "novelty_score": novelty,
        "scientific_value_score": scientific,
        "testability_score": testability,
        "feasibility_score": feasibility,
        "parameter_potential_score": param_potential,
        "conflict_usefulness_score": conflict_use,
        "paper_potential_score": paper_potential,
        "total_priority_score": total,
        "priority_level": priority,
    })
    return gap_base


def generate_candidate_hypothesis(gap: Dict[str, Any]) -> str:
    """为研究空白的候选项生成默认假设。"""
    vars_list = gap.get("related_variables", "").split(";")
    ind_var = vars_list[0].strip() if vars_list else "变量"
    dep_var = gap.get("target_indicators", "疲劳指标").split(";")[0].strip()

    gap_type = gap.get("gap_type", "evidence_gap")

    templates = {
        "evidence_gap": (
            f"在控制实验条件后，{ind_var} 可能通过尚未充分证明的机制影响 {dep_var}。"
            f"该假设的验证需要补充 {gap.get('missing_evidence', '直接证据')}。"
        ),
        "parameter_gap": (
            f"在控制应力比 R 和表面状态后，{ind_var} 可能改变 {dep_var} 相关的模型参数，"
            f"具体表现为方程参数的系统偏移。需要系统实验提取参数。"
        ),
        "mechanism_gap": (
            f"{ind_var} 可能通过某一中间微观机制影响 {dep_var}，"
            f"但该机制链尚未被直接表征证据证实。需要 micro-CT/SEM/EBSD 联合表征。"
        ),
        "conflict_gap": (
            f"关于 {ind_var} 对 {dep_var} 的影响，现有文献存在条件依赖型差异。"
            f"该差异可能由表面状态、应力比或其他未控制变量调节。"
        ),
        "boundary_condition_gap": (
            f"{ind_var} 对 {dep_var} 的影响存在条件边界，"
            f"但边界的具体参数范围（临界值）尚未被系统量化。"
        ),
    }
    return templates.get(gap_type, f"{ind_var} 与 {dep_var} 的关系需要进一步验证。")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main discovery pipeline
# ═══════════════════════════════════════════════════════════════════════════

def discover_research_gaps(
    review_text: str = "",
    review_path: Optional[Path] = None,
    scope: str = "all",
) -> List[Dict[str, Any]]:
    """
    执行研究空白发现完整流程。

    Args:
        review_text: 直接传入的文献综述文本
        review_path: 文献综述文件路径
        scope: "all" | "lit_db_only" | "lit_db_and_candidates" | "recent_only"

    Returns:
        研究空白列表，按 total_priority_score 排序
    """
    # Load review
    if not review_text:
        review_text = load_review_text(review_path)

    # Load data
    data = load_all_inputs()

    # Step 1: Use known gap templates
    gaps = []
    for template in KNOWN_GAP_TEMPLATES:
        gap = dict(template)  # copy
        gap["candidate_hypothesis"] = generate_candidate_hypothesis(gap)
        gap["gap_statement"] = gap.get(
            "gap_statement",
            gap.get("existing_evidence_summary", "")[:300]
        )
        gap = compute_gap_scores(gap)
        gap["gap_id"] = f"GP{len(gaps)+1:03d}"
        gap["status"] = "candidate"
        # Add default fields
        for f in GAP_FIELDS:
            if f not in gap:
                gap[f] = ""
        gaps.append(gap)

    # Step 2: Review-based gaps
    claims = extract_claims_from_review(review_text)
    if claims:
        review_gaps = match_claims_to_evidence(claims, data)
        for rg in review_gaps:
            rg = compute_gap_scores(rg)
            rg["gap_id"] = f"GP{len(gaps)+1:03d}"
            rg["status"] = "candidate"
            for f in GAP_FIELDS:
                if f not in rg:
                    rg[f] = ""
            gaps.append(rg)

    # Step 3: Evidence-based gaps
    ev_gaps = discover_gaps_from_evidence(data)
    for eg in ev_gaps:
        eg = compute_gap_scores(eg)
        eg["gap_id"] = f"GP{len(gaps)+1:03d}"
        eg["status"] = "candidate"
        for f in GAP_FIELDS:
            if f not in eg:
                eg[f] = ""
        gaps.append(eg)

    # Filter by scope
    # (scope filtering is mostly relevant for display; here we keep all)

    # Sort by priority score descending
    gaps.sort(key=lambda g: -g.get("total_priority_score", 0))

    # Update priority ranks
    for i, gap in enumerate(gaps, 1):
        gap["priority_rank"] = i

    return gaps


# ═══════════════════════════════════════════════════════════════════════════
# 7. Save results
# ═══════════════════════════════════════════════════════════════════════════

def save_gaps(gaps: List[Dict[str, Any]]):
    """保存研究空白到 CSV 和 markdown 报告。"""
    # Save CSV
    gap_path = DATA_DIR / "research_gaps.csv"
    with open(gap_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for gap in gaps:
            writer.writerow({k: gap.get(k, "") for k in GAP_FIELDS})

    # Save gap hypotheses CSV
    hyp_path = DATA_DIR / "gap_hypotheses.csv"
    hypotheses = []
    for gap in gaps:
        hypotheses.append({
            "gap_id": gap.get("gap_id", ""),
            "hypothesis_id": f"H_{gap.get('gap_id', '')}",
            "hypothesis_statement": gap.get("candidate_hypothesis", ""),
            "independent_variable": gap.get("related_variables", "").split(";")[0].strip() if gap.get("related_variables") else "",
            "dependent_variable": gap.get("target_indicators", "").split(";")[0].strip() if gap.get("target_indicators") else "",
            "controlled_variables": "surface_state; stress_ratio_R; heat_treatment",
            "mechanism_chain": "待补充",
            "possible_equation_or_model": gap.get("potential_equation_or_model", ""),
            "experimental_design": gap.get("experimental_design", "待设计"),
            "support_criteria": "待确定",
            "falsification_condition": gap.get("falsification_condition", "待确定"),
            "hypothesis_score": gap.get("total_priority_score", 0),
            "priority_rank": gap.get("priority_rank", 99),
        })
    with open(hyp_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HYPOTHESIS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for h in hypotheses:
            writer.writerow(h)

    # Generate markdown reports
    _save_gap_report_md(gaps)
    _save_hypothesis_report_md(hypotheses)


def _save_gap_report_md(gaps: List[Dict[str, Any]]):
    """生成研究空白报告 markdown。"""
    lines = [
        "# Research Gap Report（研究空白报告）",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        f"总空白数: {len(gaps)}",
        "",
        "---",
        "",
    ]

    for gap in gaps:
        pri = gap.get("priority_level", "low_priority")
        pri_icon = {"high_priority": "🔴", "medium_priority": "🟡",
                     "low_priority": "🟢", "reject": "⚪"}.get(pri, "⚪")
        total = gap.get("total_priority_score", 0)

        lines.append(f"## {gap.get('gap_id', '')}: {gap.get('gap_title', '')}\n")
        lines.append(f"- **类型**: {gap.get('gap_type', '')}\n")
        lines.append(f"- **优先级**: {pri_icon} {pri} (总分 {total}/100)\n")
        lines.append(f"- **相关变量**: {gap.get('related_variables', '')}\n")
        lines.append(f"- **目标指标**: {gap.get('target_indicators', '')}\n\n")
        lines.append("**研究空白**:\n")
        lines.append(f"{gap.get('gap_statement', gap.get('existing_evidence_summary', ''))}\n\n")
        lines.append("**为什么重要**:\n")
        lines.append(f"{gap.get('why_it_matters', '')}\n\n")
        lines.append("**缺失证据**:\n")
        lines.append(f"{gap.get('missing_evidence', '')}\n\n")

        if gap.get("candidate_hypothesis"):
            lines.append("**候选假设**:\n")
            lines.append(f"{gap['candidate_hypothesis']}\n\n")

        if gap.get("potential_equation_or_model"):
            lines.append("**可能方程/模型**:\n")
            lines.append(f"{gap['potential_equation_or_model']}\n\n")

        lines.append("**评分**:\n")
        lines.append(f"- 科学价值: {gap.get('scientific_value_score', 0)}/20\n")
        lines.append(f"- 新颖性: {gap.get('novelty_score', 0)}/15\n")
        lines.append(f"- 可验证性: {gap.get('testability_score', 0)}/20\n")
        lines.append(f"- 可行性: {gap.get('feasibility_score', 0)}/15\n")
        lines.append(f"- 参数潜力: {gap.get('parameter_potential_score', 0)}/10\n")
        lines.append(f"- 冲突利用: {gap.get('conflict_usefulness_score', 0)}/10\n")
        lines.append(f"- 论文潜力: {gap.get('paper_potential_score', 0)}/10\n")
        lines.append(f"- **总分**: {gap.get('total_priority_score', 0)}/100\n")

        lines.append("---\n")

    report_path = OUTPUTS_DIR / "24_research_gap_report.md"
    report_path.write_text("".join(lines), encoding="utf-8")


def _save_hypothesis_report_md(hypotheses: List[Dict[str, Any]]):
    """生成假设报告 markdown。"""
    lines = [
        "# Prioritized Gap Hypotheses（优先级排序假设列表）",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        f"总假设数: {len(hypotheses)}",
        "",
        "---",
        "",
    ]

    for h in sorted(hypotheses, key=lambda x: -x.get("hypothesis_score", 0)):
        lines.append(f"## {h['hypothesis_id']}\n")
        lines.append(f"**关联空白**: {h.get('gap_id', '')}\n")
        lines.append(f"**假设**: {h.get('hypothesis_statement', '')}\n")
        lines.append(f"**自变量**: {h.get('independent_variable', '')}\n")
        lines.append(f"**因变量**: {h.get('dependent_variable', '')}\n")
        lines.append(f"**控制变量**: {h.get('controlled_variables', '')}\n")
        lines.append(f"**可能方程**: {h.get('possible_equation_or_model', '')}\n")
        lines.append(f"**实验方案**: {h.get('experimental_design', '')}\n")
        lines.append(f"**支持判据**: {h.get('support_criteria', '')}\n")
        lines.append(f"**推翻条件**: {h.get('falsification_condition', '')}\n")
        lines.append(f"**评分**: {h.get('hypothesis_score', 0)}/100 | "
                     f"**优先级排名**: {h.get('priority_rank', 99)}\n")
        lines.append("---\n")

    hyp_path = OUTPUTS_DIR / "25_prioritized_gap_hypotheses.md"
    hyp_path.write_text("".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Auto link with literature search
# ═══════════════════════════════════════════════════════════════════════════

def auto_search_for_gap(gap: Dict[str, Any]) -> Optional[str]:
    """
    如果研究空白缺少关键证据，自动生成补文献计划。
    Returns path to search plan if generated.
    """
    missing = gap.get("missing_evidence", "")
    if not missing:
        return None

    from src.literature_search_agent import generate_search_queries, save_search_recommendations

    var_names = gap.get("related_variables", "").split(";")
    ind_var = var_names[0].strip() if var_names else None
    dep_var = gap.get("target_indicators", "").split(";")[0].strip() if gap.get("target_indicators") else None

    queries = generate_search_queries(ind_var, dep_var, missing)
    save_search_recommendations(queries, ind_var, dep_var)

    plan_path = OUTPUTS_DIR / "22_literature_search_plan.md"
    with open(plan_path, "a", encoding="utf-8") as f:
        f.write(f"\n## 研究空白相关检索: {gap.get('gap_id', '')}\n")
        f.write(f"**缺失证据**: {missing}\n\n")
        for i, q in enumerate(queries, 1):
            f.write(f"### Query {i}\n```\n{q}\n```\n")
        f.write("---\n")


# ═══════════════════════════════════════════════════════════════════════════
# 9. 反向证据检索 / Counter-evidence Search
# ═══════════════════════════════════════════════════════════════════════════

# 已知假设-反证知识库（领域知识）
COUNTER_EVIDENCE_KNOWLEDGE = {
    "near_surface_pore_dominates": {
        "supporting": [
            "近表面孔隙边缘的应力集中与自由表面应力场叠加，起裂驱动力增强",
            "抛光试样中裂纹起裂源多位于近表面大孔隙处",
            "距表面距离 < 100μm 的孔隙更易成为起裂源",
        ],
        "counter": [
            "as-built 表面粗糙度较高时，表面缺口效应掩盖近表面孔隙的影响",
            "深部大孔隙在 VHCF 中通过鱼眼(fish-eye)机制主导起裂",
            "HIP 处理后孔隙闭合，起裂源转向残留表面缺陷",
        ],
        "condition_dependent": [
            "polished 条件下内部孔隙作用增强",
            "as-built 条件下表面粗糙度掩盖内部孔隙",
            "高应力幅下多个孔隙可能同时起裂",
        ],
    },
    "surface_roughness_dominates": {
        "supporting": [
            "as-built 表面粗糙度 Ra 通常 10-20μm，表面缺口 Kt 可达 2-3",
            "as-built 试样中裂纹几乎全部从表面起裂",
            "抛光后疲劳寿命显著提升说明表面粗糙度是关键因素",
        ],
        "counter": [
            "当表面改善后，内部大孔隙（√area > 50μm）仍可主导起裂",
            "部分研究显示 as-built 试样中也有近表面孔隙起裂案例",
            "HIP 改善内部孔隙但 as-built 粗糙度不变，寿命仍有提升",
        ],
        "condition_dependent": [
            "表面粗糙度主导权取决于 Ra 是否超过临界值（约 Ra > 3-5μm）",
            "孔隙尺寸很大时（√area > 100μm），即使 as-built 表面也竞争不过",
            "VHCF 中表面效应减弱，内部效应增强",
        ],
    },
    "hip_eliminates_pore_effect": {
        "supporting": [
            "HIP 可将孔隙率从 >0.5% 降至 <0.05%，显著降低缺陷密度",
            "HIP 后疲劳极限提升 30-50%，接近锻件水平",
            "HIP 闭合内部孔隙，起裂源从孔隙转向表面特征",
        ],
        "counter": [
            "HIP 不能完全消除表面连通孔隙和超大未熔合缺陷",
            "部分 HIP 试样中仍从残留孔隙起裂",
            "HIP 改善效果受初始缺陷尺寸和位置影响",
        ],
        "condition_dependent": [
            "初始孔隙率低时 HIP 改善幅度有限",
            "as-built+HIP 表面粗糙度仍然存在，寿命提升不如 polished+HIP",
        ],
    },
    "pore_size_determines_fatigue_life": {
        "supporting": [
            "Murakami √area 模型预测疲劳极限与实测值吻合较好（±15%）",
            "大孔隙（√area > 50μm）试样的 Nf 系统性低于小孔隙试样",
        ],
        "counter": [
            "孔隙位置（距表面距离）的影响有时大于孔隙尺寸",
            "表面粗糙度较高时孔隙尺寸对 Nf 的解释力降低",
            "多个小孔隙的交互作用可能比单个大孔隙更危险",
        ],
        "condition_dependent": [
            "polished 表面下 pore_size 的解释力更强",
            "高应力幅下 pore_size 效应减弱",
        ],
    },
}

# 证据平衡枚举
EVIDENCE_BALANCE_TYPES = [
    "support_dominant",
    "counter_dominant",
    "condition_dependent",
    "insufficient_evidence",
    "mixed_evidence",
]

EVIDENCE_BALANCE_CN = {
    "support_dominant": "支持占主导",
    "counter_dominant": "反证占主导",
    "condition_dependent": "条件依赖",
    "insufficient_evidence": "证据不足",
    "mixed_evidence": "混合证据",
}


def search_counter_evidence(
    hypothesis: str,
    hypothesis_key: str = "",
    literature_db: Optional[pd.DataFrame] = None,
    evidence_df: Optional[pd.DataFrame] = None,
    rag_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    对单个假设进行反向证据检索。

    对每个假设，主动检索：
    1. 支持证据（supporting_evidence）
    2. 反向证据（counter_evidence）
    3. 条件依赖证据（condition_dependent_evidence）
    4. 缺失证据（missing_evidence）
    5. 证据平衡判断（evidence_balance）

    Args:
        hypothesis: 假设文本
        hypothesis_key: 预设假设的 key（在 COUNTER_EVIDENCE_KNOWLEDGE 中）
        literature_db: 文献数据
        evidence_df: 证据片段数据
        rag_chunks: RAG 文本块列表，每个 chunk 含 text / source 字段

    Returns:
        {
            "hypothesis": str,
            "supporting_evidence": List[str],
            "counter_evidence": List[str],
            "condition_dependent_evidence": List[str],
            "missing_evidence": List[str],
            "evidence_balance": str,
            "needs_downgrade": bool,
            "needed_literature_types": List[str],
        }
    """
    # 尝试从知识库匹配
    if hypothesis_key and hypothesis_key in COUNTER_EVIDENCE_KNOWLEDGE:
        kb = COUNTER_EVIDENCE_KNOWLEDGE[hypothesis_key]
        supporting = list(kb["supporting"])
        counter = list(kb["counter"])
        condition_dep = list(kb["condition_dependent"])
    else:
        # 基于文本关键词的模糊匹配
        hyp_lower = hypothesis.lower()
        supporting = []
        counter = []
        condition_dep = []

        for key, kb in COUNTER_EVIDENCE_KNOWLEDGE.items():
            key_terms = key.replace("_", " ").lower()
            # 检查是否关键词匹配
            if any(term in hyp_lower for term in key_terms.split()):
                supporting.extend(kb["supporting"])
                counter.extend(kb["counter"])
                condition_dep.extend(kb["condition_dependent"])

        # 如果证据库有数据，补充搜索
        if evidence_df is not None and not evidence_df.empty:
            for _, row in evidence_df.iterrows():
                claim = str(row.get("extracted_claim", "") or "")
                ev_type = str(row.get("evidence_type", "") or "")
                mech = str(row.get("mechanism", "") or "")

                if not claim:
                    continue

                # 相关性判断
                if any(term in claim.lower() for term in hyp_lower.split()[:3]):
                    if "conflict" in ev_type or "conflict" in claim.lower():
                        counter.append(claim[:150])
                    else:
                        supporting.append(claim[:150])

        # 从 RAG 文本块中检索
        if rag_chunks is not None:
            hyp_terms = [t for t in hyp_lower.split() if len(t) > 3]
            for chunk in rag_chunks:
                text = str(chunk.get("text", "") or "")
                source = str(chunk.get("source_file", chunk.get("source", "RAG")))[:40]
                if not text:
                    continue
                # 关键词匹配
                match_count = sum(1 for t in hyp_terms if t in text.lower())
                if match_count >= 2:
                    snippet = text[:200].strip()
                    prefix = f"[RAG:{source}] "
                    # 判断方向：包含负面/矛盾关键词视为反证
                    neg_kw = ["however", "but", "conflict", "contradict",
                              "相反", "矛盾", "不同", "差异", "然而"]
                    if any(kw in text.lower() for kw in neg_kw):
                        counter.append(prefix + snippet)
                    else:
                        supporting.append(prefix + snippet)

    # 去重
    supporting = list(dict.fromkeys(supporting))
    counter = list(dict.fromkeys(counter))
    condition_dep = list(dict.fromkeys(condition_dep))

    # 证据平衡判断
    n_support = len(supporting)
    n_counter = len(counter)
    n_cond = len(condition_dep)

    if n_support == 0 and n_counter == 0 and n_cond == 0:
        evidence_balance = "insufficient_evidence"
    elif n_counter == 0 and n_cond == 0:
        evidence_balance = "support_dominant"
    elif n_support == 0 and n_cond == 0:
        evidence_balance = "counter_dominant"
    elif n_cond > max(n_support, n_counter):
        evidence_balance = "condition_dependent"
    elif abs(n_support - n_counter) <= 2 or (n_support > 0 and n_counter > 0):
        evidence_balance = "mixed_evidence"
    elif n_support > n_counter:
        evidence_balance = "support_dominant"
    else:
        evidence_balance = "counter_dominant"

    # 假设是否需要降级
    needs_downgrade = evidence_balance in ("counter_dominant", "insufficient_evidence")
    if evidence_balance == "condition_dependent":
        needs_downgrade = False  # 条件依赖假设仍有价值，但需明确条件
    if evidence_balance == "mixed_evidence" and n_counter >= n_support:
        needs_downgrade = True

    # 还需要补充的文献类型
    needed_literature = []
    if n_support < 3:
        needed_literature.append("支持该假设的直接实验文献")
    if n_counter < 2:
        needed_literature.append("对比/反驳该假设的文献")
    if n_cond < 2:
        needed_literature.append("不同条件下的系统对比文献")
    if evidence_balance == "insufficient_evidence":
        needed_literature.append("对该假设的任何形式证据文献")

    return {
        "hypothesis": hypothesis,
        "hypothesis_key": hypothesis_key,
        "supporting_evidence": supporting[:5],
        "counter_evidence": counter[:5],
        "condition_dependent_evidence": condition_dep[:5],
        "missing_evidence": needed_literature[:5],
        "evidence_balance": evidence_balance,
        "evidence_balance_cn": EVIDENCE_BALANCE_CN.get(evidence_balance, evidence_balance),
        "needs_downgrade": needs_downgrade,
        "needed_literature_types": needed_literature,
    }


def batch_search_counter_evidence(
    gaps: List[Dict[str, Any]],
    evidence_df: Optional[pd.DataFrame] = None,
    rag_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    对所有研究空白/假设批量进行反向证据检索。

    Args:
        gaps: 研究空白列表
        evidence_df: 证据片段数据
        rag_chunks: RAG 文本块列表

    Returns:
        每个 gap 的反向证据检索结果列表
    """
    results = []

    # 假设关键词到预设知识库的映射
    hypothesis_keywords = {
        "near_surface_pore": "near_surface_pore_dominates",
        "pore dominate": "near_surface_pore_dominates",
        "surface roughness dominate": "surface_roughness_dominates",
        "surface粗糙度主导": "surface_roughness_dominates",
        "hip消除": "hip_eliminates_pore_effect",
        "hip闭合": "hip_eliminates_pore_effect",
        "孔隙尺寸决定": "pore_size_determines_fatigue_life",
        "pore size determine": "pore_size_determines_fatigue_life",
    }

    for gap in gaps:
        gap_id = gap.get("gap_id", "")
        gap_title = gap.get("gap_title", "")
        hypothesis = gap.get("candidate_hypothesis", "")
        combined = (gap_title + " " + hypothesis).lower()

        # 查找匹配的预设假设
        matched_key = ""
        for kw, key in hypothesis_keywords.items():
            if kw in combined:
                matched_key = key
                break

        # 执行反向证据检索
        result = search_counter_evidence(
            hypothesis=f"{gap_title}: {hypothesis}" if hypothesis else gap_title,
            hypothesis_key=matched_key,
            evidence_df=evidence_df,
            rag_chunks=rag_chunks,
        )
        result["gap_id"] = gap_id
        result["gap_title"] = gap_title
        results.append(result)

    return results


def format_counter_evidence_markdown(result: Dict[str, Any]) -> str:
    """
    将反向证据检索结果格式化为 Markdown。
    """
    lines = []
    lines.append("## 🔄 反向证据检索\n\n")

    # 假设
    lines.append(f"**假设**: {result.get('hypothesis', '')}\n\n")

    # 证据平衡标签
    balance = result.get("evidence_balance", "insufficient_evidence")
    balance_cn = result.get("evidence_balance_cn", balance)
    balance_icon = {
        "support_dominant": "🟢", "counter_dominant": "🔴",
        "condition_dependent": "🟡", "insufficient_evidence": "⚪",
        "mixed_evidence": "🟠",
    }.get(balance, "⚪")

    lines.append(f"**证据平衡**: {balance_icon} {balance_cn}\n\n")

    if result.get("needs_downgrade"):
        lines.append("> ⚠️ **警告**: 反向证据占优或证据不足，**建议该假设降级处理**\n\n")

    # 支持证据
    support_ev = result.get("supporting_evidence", [])
    lines.append("### ✅ 支持该假设的证据\n\n")
    if support_ev:
        for ev in support_ev:
            lines.append(f"- {ev}\n")
    else:
        lines.append("- 当前未检索到直接支持证据\n")
    lines.append("\n")

    # 反向证据
    counter_ev = result.get("counter_evidence", [])
    lines.append("### ❌ 可能反驳该假设的证据\n\n")
    if counter_ev:
        for ev in counter_ev:
            lines.append(f"- {ev}\n")
    else:
        lines.append("- 当前未检索到明显反驳证据\n")
    lines.append("\n")

    # 条件依赖证据
    cond_ev = result.get("condition_dependent_evidence", [])
    lines.append("### 🟡 条件依赖证据\n\n")
    if cond_ev:
        for ev in cond_ev:
            lines.append(f"- {ev}\n")
    else:
        lines.append("- 未发现条件依赖证据\n")
    lines.append("\n")

    # 缺失证据
    missing = result.get("missing_evidence", [])
    if missing:
        lines.append("### 📋 还需要补充的文献类型\n\n")
        for m in missing:
            lines.append(f"- {m}\n")
        lines.append("\n")

    return "".join(lines)


def format_counter_evidence_batch_markdown(results: List[Dict[str, Any]]) -> str:
    """批量格式化反向证据检索结果为 Markdown。"""
    lines = []
    lines.append("# 反向证据检索报告\n\n")
    lines.append(f"共检索 {len(results)} 个假设/研究空白\n\n")
    lines.append("---\n\n")

    for r in results:
        lines.append(format_counter_evidence_markdown(r))
        lines.append("---\n\n")

    return "".join(lines)

    return str(plan_path)
