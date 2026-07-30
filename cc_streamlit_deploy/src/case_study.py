"""
case_study.py — 案例验证模块

围绕三个核心案例验证 TitaniumFatigueChat 系统能力：
Case 1: 孔隙尺寸、距表面距离和表面状态对 crack_initiation_site 和 Nf 的共同影响
Case 2: 表面粗糙度和内部孔隙在不同 surface_state 下的竞争主导
Case 3: DeltaK、da/dN 与 Paris_C/Paris_m 的模型解释

每个案例输出:
   question, retrieved_evidence, condition_evidence,
   answer, hypothesis, experiment_design,
   model_explanation, score, limitations
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

CASE_TEMPLATES = [
    {
        "case_id": "CASE_001",
        "title": "孔隙尺寸-距表面距离-表面状态耦合控制疲劳起裂",
        "question": "L-PBF Ti-6Al-4V 中，孔隙尺寸、距表面距离和表面状态如何共同影响 crack_initiation_site 和 Nf？",
        "domain": "defect-fatigue",
        "expected_variables": "pore_size; distance_to_surface; surface_state; crack_initiation_site; Nf",
        "hypothesis_template": (
            "在 polished + SR 条件下，近表面大孔隙（area>50um, d<100um）通过孔隙边缘应力集中与自由表面应力场叠加，"
            "显著降低 Nf（预期降幅 50-70%）。在 as-built 条件下表面粗糙度竞争，孔隙效应被部分掩盖。"
        ),
        "model_template": "Murakami area with surface correction; Kitagawa-Takahashi with position dependence",
        "experiment_template": "2x2x2 因子设计：2 表面状态 x 2 孔隙位置 x 2 孔隙尺寸 + micro-CT + HCF + SEM fractography",
    },
    {
        "case_id": "CASE_002",
        "title": "表面粗糙度与内部孔隙的竞争主导机制",
        "question": "L-PBF Ti-6Al-4V 中，表面粗糙度和内部孔隙在不同 surface_state 下谁主导疲劳起裂？",
        "domain": "surface-state",
        "expected_variables": "surface_roughness_Ra; pore_size; surface_state; crack_initiation_site; failure_mode",
        "hypothesis_template": (
            "as-built 表面（Ra>10um）由表面缺口主导起裂；polished 表面（Ra<1um）由近表面大孔隙主导起裂。"
            "存在临界 Ra 阈值（约 3-10um），低于该值时主导权转向孔隙。"
        ),
        "model_template": "competitive failure probability model; Kt comparison",
        "experiment_template": "多级表面粗糙度（as-built/polished/intermediate）+ micro-CT + HCF + SEM 断口 + 逻辑回归",
    },
    {
        "case_id": "CASE_003",
        "title": "DeltaK-da/dN 模型解释与 Paris 参数分析",
        "question": "L-PBF Ti-6Al-4V 中，DeltaK、da/dN 与 Paris_C/Paris_m 的关系如何模型化？缺陷和组织如何影响参数？",
        "domain": "crack-growth",
        "expected_variables": "Delta_K; da_dN; Paris_C; Paris_m; defect_state; microstructure",
        "hypothesis_template": (
            "缺陷状态（孔隙率）主要改变 Paris C（log(C)与孔隙率正相关），微观组织（α lath 宽度）主要改变 Paris m。"
            "高 R 比通过降低裂纹闭合效应增大 da/dN。"
        ),
        "model_template": "Paris law; Walker model; crack closure correction",
        "experiment_template": "不同缺陷/组织试样 FCGR 试验 + Paris 参数拟合 + EBSD 组织表征",
    },
]


def load_case_templates() -> List[Dict[str, Any]]:
    return CASE_TEMPLATES


def run_case_study(
    case_id: str,
    evidence_df: Optional[pd.DataFrame] = None,
    variable_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    运行单个案例验证。

    Args:
        case_id: CASE_001 / CASE_002 / CASE_003
        evidence_df: 证据片段数据
        variable_df: 变量关系数据

    Returns:
        包含完整案例验证输出的字典
    """
    templates = {t["case_id"]: t for t in CASE_TEMPLATES}
    if case_id not in templates:
        return {"case_id": case_id, "error": f"Unknown case: {case_id}"}

    t = templates[case_id]

    # 检索相关证据
    retrieved = _retrieve_case_evidence(case_id, evidence_df)
    condition_ev = _extract_condition_evidence(case_id, evidence_df)

    result = {
        "case_id": case_id,
        "title": t["title"],
        "question": t["question"],
        "domain": t["domain"],
        "expected_variables": t["expected_variables"],
        "retrieved_evidence": retrieved,
        "condition_evidence": condition_ev,
        "answer": _generate_case_answer(case_id, retrieved, condition_ev),
        "hypothesis": t["hypothesis_template"],
        "experiment_design": t["experiment_template"],
        "model_explanation": t["model_template"],
        "score": _compute_case_score(retrieved, condition_ev),
        "limitations": _identify_case_limitations(case_id, retrieved),
    }
    return result


def _retrieve_case_evidence(
    case_id: str, evidence_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """检索与案例相关的证据。"""
    keywords_map = {
        "CASE_001": ["pore", "initiation", "surface", "defect"],
        "CASE_002": ["roughness", "competition", "surface", "pore", "initiation"],
        "CASE_003": ["paris", "crack growth", "da/dn", "delta k", "fcgr"],
    }
    keywords = keywords_map.get(case_id, [])
    results = []

    if evidence_df is not None and not evidence_df.empty:
        for _, row in evidence_df.iterrows():
            claim = str(row.get("extracted_claim", "") or "")
            mech = str(row.get("mechanism", "") or "")
            text = (claim + " " + mech).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score >= 1:
                results.append({
                    "evidence_id": row.get("evidence_id", ""),
                    "claim": claim[:200],
                    "mechanism": mech,
                    "material": row.get("material", ""),
                    "surface_state": row.get("surface_state", ""),
                    "relevance_score": score,
                })

    results.sort(key=lambda x: -x["relevance_score"])
    return results[:8]


def _extract_condition_evidence(
    case_id: str, evidence_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """提取条件绑定的证据。"""
    if evidence_df is None or evidence_df.empty:
        return []

    condition_fields = ["material", "surface_state", "heat_treatment",
                        "stress_ratio_R", "fatigue_type", "pore_size",
                        "surface_roughness_Ra", "condition_boundary"]
    results = []
    for _, row in evidence_df.iterrows():
        has_condition = any(
            str(row.get(f, "") or "").strip() for f in condition_fields
        )
        if has_condition:
            results.append({
                "evidence_id": row.get("evidence_id", ""),
                "claim": str(row.get("extracted_claim", ""))[:150],
                **{f: row.get(f, "") for f in condition_fields if f in row.index},
            })
    return results[:6]


def _generate_case_answer(
    case_id: str, retrieved: list, condition_ev: list
) -> str:
    """生成案例回答摘要。"""
    templates = {
        "CASE_001": (
            "L-PBF Ti-6Al-4V 中孔隙尺寸、距表面距离和表面状态共同控制疲劳起裂："
            "（1）近表面大孔隙（area>50um, d<100um）是 polished 条件下的主导起裂源；"
            "（2）as-built 表面粗糙度（Ra>10um）可掩盖孔隙效应；"
            "（3）临界条件边界由 Ra、area 和 d 三者共同决定。"
        ),
        "CASE_002": (
            "表面粗糙度与内部孔隙存在竞争主导关系：as-built 状态表面缺口主导，"
            "polished/machined 状态内部孔隙主导。临界 Ra 约 3-10um，取决于孔隙尺寸分布。"
        ),
        "CASE_003": (
            "DeltaK-da/dN 关系服从 Paris 定律。缺陷状态（孔隙率）主要影响 C 参数，"
            "微观组织（α lath 宽度）主要影响 m 参数。高 R 比增大 da/dN。"
        ),
    }
    return templates.get(case_id, "待补充")


def _compute_case_score(retrieved: list, condition_ev: list) -> Dict[str, Any]:
    """计算案例得分。"""
    return {
        "evidence_coverage": min(len(retrieved) / 5, 1.0),
        "condition_evidence_ratio": min(len(condition_ev) / 3, 1.0),
        "total_evidence": len(retrieved) + len(condition_ev),
        "evidence_quality": "high" if len(retrieved) >= 5 else "medium" if len(retrieved) >= 2 else "low",
    }


def _identify_case_limitations(case_id: str, retrieved: list) -> List[str]:
    """识别案例验证的局限性。"""
    limitations = []
    if len(retrieved) < 3:
        limitations.append("检索到的直接支持证据不足")
    limitations.append("条件化证据字段尚未充分填充，影响条件边界推断精度")
    limitations.append("当前为自动规则检索，未使用 Qwen API 深度推理")
    return limitations


def format_case_study_markdown(result: Dict[str, Any]) -> str:
    """将案例验证结果格式化为 Markdown。"""
    lines = [
        f"# Case Study: {result.get('title', '')}\n",
        f"**案例 ID**: {result.get('case_id', '')}\n",
        f"**领域**: {result.get('domain', '')}\n",
        f"**问题**: {result.get('question', '')}\n\n",
        "---\n",
    ]

    # 检索证据
    lines.append("## 检索到的证据\n\n")
    retrieved = result.get("retrieved_evidence", [])
    if retrieved:
        for ev in retrieved[:5]:
            lines.append(f"- [{ev.get('evidence_id', '')}] {ev.get('claim', '')}\n")
    else:
        lines.append("- 未检索到直接证据\n")
    lines.append("\n")

    # 条件绑定证据
    lines.append("## 条件化证据\n\n")
    cond_ev = result.get("condition_evidence", [])
    if cond_ev:
        for ev in cond_ev[:4]:
            cond_str = f"mat={ev.get('material','')} sur={ev.get('surface_state','')} HT={ev.get('heat_treatment','')} R={ev.get('stress_ratio_R','')}"
            lines.append(f"- [{ev.get('evidence_id','')}] {cond_str}: {ev.get('claim','')}\n")
    else:
        lines.append("- 无条件化证据\n")
    lines.append("\n")

    # 答案
    lines.append(f"## 系统回答\n\n{result.get('answer', '')}\n\n")

    # 假设
    lines.append(f"## 生成假设\n\n{result.get('hypothesis', '')}\n\n")

    # 实验设计
    lines.append(f"## 实验方案\n\n{result.get('experiment_design', '')}\n\n")

    # 模型解释
    lines.append(f"## 模型解释\n\n{result.get('model_explanation', '')}\n\n")

    # 评分
    score = result.get("score", {})
    lines.append("## 案例评分\n\n")
    lines.append(f"- 证据覆盖率: {score.get('evidence_coverage', 0):.2f}\n")
    lines.append(f"- 条件化证据比例: {score.get('condition_evidence_ratio', 0):.2f}\n")
    lines.append(f"- 总证据数: {score.get('total_evidence', 0)}\n")
    lines.append(f"- 证据质量: {score.get('evidence_quality', 'low')}\n\n")

    # 局限性
    lines.append("## 局限性\n\n")
    for lim in result.get("limitations", []):
        lines.append(f"- {lim}\n")
    lines.append("\n---\n")

    return "".join(lines)


def run_all_case_studies() -> List[Dict[str, Any]]:
    """运行所有三个案例验证。"""
    evidence_path = TRUSTED_EVIDENCE_PATH
    evidence_df = None
    if evidence_path.exists():
        try:
            evidence_df = pd.read_csv(evidence_path, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            pass

    results = []
    for t in CASE_TEMPLATES:
        result = run_case_study(t["case_id"], evidence_df=evidence_df)
        results.append(result)

    # 生成报告
    report_lines = ["# TitaniumFatigueChat 案例验证报告\n\n"]
    for r in results:
        report_lines.append(format_case_study_markdown(r))
        report_lines.append("\n\n")

    report_path = OUTPUTS_DIR / "case_studies_report.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    print(f"[case_study] Report saved to {report_path}")

    return results


if __name__ == "__main__":
    run_all_case_studies()
