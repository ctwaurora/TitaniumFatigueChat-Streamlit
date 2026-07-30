"""Evidence-aware candidate hypothesis presentation helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


CANDIDATE_MODEL_NOTICE = (
    "该公式是系统提出的待验证候选模型，不是当前文献已经证明的普适疲劳定律。"
)

CANDIDATE_EQUATION = (
    "log10(Nf) = β0 - β1 log10(stress_amplitude) "
    "- β2 log10(sqrt_area) + β3 D* + β4 surface_state "
    "+ β5(D* × surface_state) + β6 stress_ratio_R + ε"
)

MODEL_COMPARISONS = [
    ("Model A", "stress_amplitude"),
    ("Model B", "stress_amplitude + sqrt_area"),
    ("Model C", "stress_amplitude + sqrt_area + D*"),
    (
        "Model D",
        "stress_amplitude + sqrt_area + D* + surface_state "
        "+ D*×surface_state",
    ),
]

SUPPORT_CRITERIA = [
    "D*、surface_state 及交互项的系数置信区间与研究问题所要求的方向一致，且关键区间不跨越 0。",
    "加入 D* 后，重复交叉验证或嵌套交叉验证误差在不同划分中稳定改善。",
    "Model C/D 相对简化模型具有更高的 adjusted R²，并在 AIC、BIC 的复杂度惩罚下仍得到支持。",
    "在独立论文验证或留一文献验证中，效应方向和预测改进能够重复。",
]

FALSIFICATION_CRITERIA = [
    "D* 和 D*×surface_state 交互项的系数无法与 0 区分。",
    "加入 D* 后预测误差没有稳定改善。",
    "D* 或交互项的效应方向在重复划分、独立论文或留一文献验证中无法重复。",
    "表面状态、缺陷尺寸和应力幅已经能够解释结果，加入 D* 不再提供独立信息。",
]

FORMULA_VARIABLES = {
    "Nf": "疲劳失效循环数。",
    "stress_amplitude": "疲劳加载应力幅；必须使用文献或用户输入中的实际单位。",
    "sqrt_area": "缺陷投影面积平方根；必须保留原文定义和单位。",
    "D*": "distance_to_surface / sqrt_area，无量纲归一化距表面距离。",
    "surface_state": "文献实际报告的表面状态分类变量，不预设具体类别。",
    "stress_ratio_R": "文献或用户输入报告的应力比，不预设数值。",
    "β0...β6": "待数据拟合的回归系数；数据不足时不得生成数值。",
    "ε": "模型残差项。",
}


def _extract_user_conditions(question: str) -> List[str]:
    """Return only conditions explicitly present in the user question."""
    text = question or ""
    conditions: List[str] = []
    patterns = (
        r"\bR\s*=\s*[-+]?\d+(?:\.\d+)?",
        r"\b[-+]?\d+(?:\.\d+)?\s*(?:Hz|kHz)\b",
        r"\b[-+]?\d+(?:\.\d+)?\s*(?:°C|℃|K)\b",
        r"\b(?:as-built|machined|polished|ground)\b",
        r"\b(?:HIP|annealed|heat-treated|stress-relieved)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = " ".join(match.group(0).split())
            if value not in conditions:
                conditions.append(value)
    return conditions


def _format_evidence_refs(evidence: Optional[Iterable[Mapping[str, Any]]]) -> List[str]:
    refs: List[str] = []
    for item in evidence or []:
        evidence_id = str(item.get("evidence_id") or "").strip()
        paper_id = str(item.get("paper_id") or "").strip()
        page = item.get("page_number") or item.get("page")
        claim = str(item.get("claim") or item.get("text") or "").strip()
        provenance = ", ".join(
            part
            for part in (
                f"evidence_id={evidence_id}" if evidence_id else "",
                f"paper_id={paper_id}" if paper_id else "",
                f"page={page}" if page else "",
            )
            if part
        )
        if provenance and claim:
            refs.append(f"{provenance}: {claim}")
        elif provenance:
            refs.append(provenance)
    return refs


def build_candidate_hypothesis(
    question: str,
    ind_var: Optional[str] = None,
    dep_var: Optional[str] = None,
    supporting_evidence: Optional[Iterable[Mapping[str, Any]]] = None,
    counter_evidence: Optional[Iterable[Mapping[str, Any]]] = None,
    fit_data_available: bool = False,
) -> Dict[str, Any]:
    """Build the D* main-case candidate without inventing conditions or fit results."""
    explicit_conditions = _extract_user_conditions(question)
    support_refs = _format_evidence_refs(supporting_evidence)
    counter_refs = _format_evidence_refs(counter_evidence)
    condition_boundary = (
        "；".join(explicit_conditions)
        if explicit_conditions
        else "用户未指定固定工况；分析时必须按文献真实工况分层或匹配，不能预设条件。"
    )
    iv = ind_var or "D*、sqrt_area、stress_amplitude"
    dv = dep_var or "log10(Nf)"
    candidate: Dict[str, Any] = {
        "hypothesis_statement": (
            "在控制缺陷尺寸和应力幅后，归一化距表面距离 D* 可能为疲劳寿命提供独立解释；"
            "surface_state 可能调节 D* 与 log10(Nf) 的关系。该关系需要跨论文验证。"
        ),
        "independent_variables": [iv, "D* = distance_to_surface / sqrt_area"],
        "dependent_variables": [dv],
        "control_variables": [
            "material_and_process",
            "specimen_geometry",
            "loading_mode",
            "heat_treatment",
            "test_environment",
        ],
        "moderating_variables": ["surface_state", "stress_ratio_R"],
        "condition_boundary": condition_boundary,
        "mechanism_chain": (
            "distance_to_surface 与 sqrt_area 共同决定 D* → 自由表面邻近效应与缺陷局部应力集中耦合 "
            "→ 裂纹起裂驱动力变化 → log10(Nf) 响应；surface_state 可能改变该耦合强度。"
        ),
        "supporting_evidence": support_refs or [
            "当前候选尚未绑定可直接支持该完整关系的 evidence_id、paper_id 和真实页码。"
        ],
        "counter_evidence": counter_refs or [
            "当前候选尚未绑定可直接反驳该完整关系的逐页证据。"
        ],
        "missing_evidence": [
            "同一样本或可比文献中的 stress_amplitude、sqrt_area、distance_to_surface、surface_state、stress_ratio_R 与 Nf 配对数据。",
            "能够区分表面状态、缺陷尺寸和 D* 独立贡献的跨论文数据。",
            "独立论文验证或留一文献验证结果。",
        ],
        "prediction_direction": (
            "在其他变量受控时，D* 对 log10(Nf) 的方向由拟合系数及其置信区间判定；"
            "系统不预设效应大小或固定阈值。"
        ),
        "support_criteria": list(SUPPORT_CRITERIA),
        "falsification_criteria": list(FALSIFICATION_CRITERIA),
        "candidate_equation": CANDIDATE_EQUATION,
        "formula_variables": dict(FORMULA_VARIABLES),
        "model_comparisons": list(MODEL_COMPARISONS),
        "model_notice": CANDIDATE_MODEL_NOTICE,
        "evidence_status": (
            "EVIDENCE_LINKED_CANDIDATE" if support_refs else "EVIDENCE_LINK_REQUIRED"
        ),
        "manual_review_requirements": [
            "核对每条支持与反向证据的 paper_id、真实页码和原文。",
            "确认变量定义、单位、表面状态编码和应力比在不同论文间可比。",
            "审核模型设定、共线性、缺失值处理和跨论文验证方案。",
            "在数据充分前不得填写 β 系数、置信区间或性能指标。",
        ],
        "fit_status": (
            "FIT_DATA_AVAILABLE_REVIEW_REQUIRED"
            if fit_data_available
            else "STRUCTURE_ONLY_INSUFFICIENT_DATA"
        ),
        "fit_results": None,
        "explicit_user_conditions": explicit_conditions,
        "support_evidence_bound": bool(support_refs),
        "counter_evidence_bound": bool(counter_refs),
    }
    candidate["score"] = calculate_dynamic_score(candidate)
    return candidate


def calculate_dynamic_score(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Calculate a transparent field-completeness score instead of storing a total."""
    dimensions = {
        "variables": min(
            5,
            sum(
                bool(candidate.get(key))
                for key in (
                    "independent_variables",
                    "dependent_variables",
                    "control_variables",
                    "moderating_variables",
                )
            )
            + 1,
        ),
        "condition_boundary": 5 if candidate.get("explicit_user_conditions") else 2,
        "mechanism_chain": 5 if candidate.get("mechanism_chain") else 0,
        "supporting_evidence": 5 if candidate.get("support_evidence_bound") else 1,
        "counter_evidence": 5 if candidate.get("counter_evidence_bound") else 1,
        "missing_evidence": 5 if candidate.get("missing_evidence") else 0,
        "prediction_and_criteria": 5
        if candidate.get("prediction_direction") and candidate.get("support_criteria")
        else 0,
        "falsifiability": 5 if candidate.get("falsification_criteria") else 0,
        "candidate_model": 5
        if candidate.get("candidate_equation") and candidate.get("formula_variables")
        else 0,
        "manual_review": 5 if candidate.get("manual_review_requirements") else 0,
    }
    return {
        "dimensions": dimensions,
        "total": sum(dimensions.values()),
        "maximum": 5 * len(dimensions),
        "method": "DYNAMIC_FIELD_COMPLETENESS",
    }


def format_candidate_hypothesis(candidate: Mapping[str, Any]) -> str:
    """Render every required candidate field as auditable Markdown."""
    lines = ["### 候选假设：D* 与表面状态对疲劳寿命的候选调节模型\n\n"]
    lines.append(f"**假设陈述**：{candidate['hypothesis_statement']}\n\n")
    for label, key in (
        ("自变量", "independent_variables"),
        ("因变量", "dependent_variables"),
        ("控制变量", "control_variables"),
        ("调节变量", "moderating_variables"),
    ):
        lines.append(f"**{label}**：{'；'.join(candidate[key])}\n\n")
    lines.append(f"**条件边界**：{candidate['condition_boundary']}\n\n")
    lines.append(f"**机制链**：{candidate['mechanism_chain']}\n\n")

    for label, key in (
        ("支持证据", "supporting_evidence"),
        ("反向证据", "counter_evidence"),
        ("缺失证据", "missing_evidence"),
    ):
        lines.append(f"**{label}**：\n")
        lines.extend(f"- {item}\n" for item in candidate[key])
        lines.append("\n")

    lines.append(f"**预测方向**：{candidate['prediction_direction']}\n\n")
    lines.append("**支持判据**：\n")
    lines.extend(f"- {item}\n" for item in candidate["support_criteria"])
    lines.append("\n**推翻判据**：\n")
    lines.extend(f"- {item}\n" for item in candidate["falsification_criteria"])

    lines.append("\n**候选公式**：\n\n")
    lines.append(f"> **{candidate['model_notice']}**\n\n")
    lines.append("```text\nD* = distance_to_surface / sqrt_area\n\n")
    lines.append(candidate["candidate_equation"] + "\n```\n\n")

    lines.append("**对比模型**：\n\n| 模型 | 结构 |\n|---|---|\n")
    for name, structure in candidate["model_comparisons"]:
        lines.append(f"| {name} | {structure} |\n")

    lines.append("\n**公式变量解释**：\n\n| 变量 | 解释 |\n|---|---|\n")
    for name, explanation in candidate["formula_variables"].items():
        lines.append(f"| {name} | {explanation} |\n")

    lines.append(f"\n**证据状态**：`{candidate['evidence_status']}`\n\n")
    lines.append(f"**拟合状态**：`{candidate['fit_status']}`\n\n")
    if not candidate.get("fit_results"):
        lines.append(
            "> 当前数据不足以拟合：不生成 β 系数和性能指标，只显示模型结构。\n\n"
        )

    lines.append("**人工审核要求**：\n")
    lines.extend(f"- {item}\n" for item in candidate["manual_review_requirements"])

    score = candidate["score"]
    lines.append(
        f"\n**动态评分**：{score['total']}/{score['maximum']} "
        f"（`{score['method']}`）\n\n"
    )
    lines.append("| 分项 | 得分 |\n|---|---:|\n")
    for name, value in score["dimensions"].items():
        lines.append(f"| {name} | {value}/5 |\n")
    return "".join(lines)


def build_candidate_hypothesis_markdown(
    question: str,
    ind_var: Optional[str] = None,
    dep_var: Optional[str] = None,
    supporting_evidence: Optional[Iterable[Mapping[str, Any]]] = None,
    counter_evidence: Optional[Iterable[Mapping[str, Any]]] = None,
) -> str:
    candidate = build_candidate_hypothesis(
        question=question,
        ind_var=ind_var,
        dep_var=dep_var,
        supporting_evidence=supporting_evidence,
        counter_evidence=counter_evidence,
    )
    return format_candidate_hypothesis(candidate)
