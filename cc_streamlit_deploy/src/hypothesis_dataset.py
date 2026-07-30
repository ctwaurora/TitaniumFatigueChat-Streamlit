"""
hypothesis_dataset.py — 假设数据集管理模块

每个系统生成的假设都保存到 data/hypothesis_dataset.csv，
包含 32 字段的完整 schema，并支持 evidence trace 链接。

字段清单:
  hypothesis_id, source_question, hypothesis_type,
  hypothesis_statement, independent_variable, dependent_variable,
  moderating_variables, controlled_variables, specific_condition,
  mechanism_chain, possible_equation_or_model, required_data,
  supporting_evidence_ids, conflicting_evidence_ids, missing_evidence,
  experimental_design, characterization_methods, data_analysis_method,
  support_criteria, falsification_condition,
  specificity_score, evidence_grounding_score, mechanistic_plausibility_score,
  parameter_awareness_score, testability_score, falsifiability_score,
  novelty_score, paper_potential_score, total_score, priority_level,
  created_time, last_updated

假设类型:
  "evidence_supported" — 有直接证据支持的假设
  "search_guided_candidate" — 基于搜索发现的候选假设，缺少直接证据
  "conflict_resolution" — 基于文献冲突解决的假设
  "gap_filling" — 填补明确研究空白的假设
  "methodological" — 方法学假设
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

HYPOTHESIS_TYPES = [
    "evidence_supported",
    "search_guided_candidate",
    "conflict_resolution",
    "gap_filling",
    "methodological",
]

FIELD_NAMES = [
    "hypothesis_id",
    "source_question",
    "hypothesis_type",
    "hypothesis_statement",
    "independent_variable",
    "dependent_variable",
    "moderating_variables",
    "controlled_variables",
    "specific_condition",
    "mechanism_chain",
    "possible_equation_or_model",
    "required_data",
    "supporting_evidence_ids",
    "conflicting_evidence_ids",
    "missing_evidence",
    "experimental_design",
    "characterization_methods",
    "data_analysis_method",
    "support_criteria",
    "falsification_condition",
    "specificity_score",
    "evidence_grounding_score",
    "mechanistic_plausibility_score",
    "parameter_awareness_score",
    "testability_score",
    "falsifiability_score",
    "novelty_score",
    "paper_potential_score",
    "total_score",
    "priority_level",
    "created_time",
    "last_updated",
]


def _next_hypothesis_id() -> str:
    """生成新 hypothesis_id。"""
    path = DATA_DIR / "hypothesis_dataset.csv"
    if not path.exists():
        return "HYP_0001"
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            max_num = 0
            for row in reader:
                hid = row.get("hypothesis_id", "")
                if hid.startswith("HYP_"):
                    try:
                        num = int(hid.split("_")[1])
                        max_num = max(max_num, num)
                    except (IndexError, ValueError):
                        pass
        return f"HYP_{max_num + 1:04d}"
    except Exception:
        return "HYP_0001"


def load_hypotheses() -> List[Dict[str, str]]:
    """加载所有假设。"""
    path = DATA_DIR / "hypothesis_dataset.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_hypothesis_by_id(hypothesis_id: str) -> Optional[Dict[str, str]]:
    """按 ID 查找假设。"""
    hypotheses = load_hypotheses()
    for h in hypotheses:
        if h.get("hypothesis_id") == hypothesis_id:
            return h
    return None


def get_hypotheses_by_type(h_type: str) -> List[Dict[str, str]]:
    """按类型筛选假设。"""
    if h_type not in HYPOTHESIS_TYPES:
        return []
    return [h for h in load_hypotheses() if h.get("hypothesis_type") == h_type]


def _compute_priority(total_score: float) -> str:
    """根据总分计算优先级。"""
    if total_score >= 80:
        return "critical"
    elif total_score >= 65:
        return "high"
    elif total_score >= 50:
        return "medium"
    elif total_score >= 35:
        return "low"
    else:
        return "exploratory"


def _compute_priority_level(scores: Dict[str, float]) -> Tuple[float, str]:
    """计算加权总分和优先级。"""
    weights = {
        "specificity_score": 0.10,
        "evidence_grounding_score": 0.15,
        "mechanistic_plausibility_score": 0.15,
        "parameter_awareness_score": 0.10,
        "testability_score": 0.10,
        "falsifiability_score": 0.15,
        "novelty_score": 0.15,
        "paper_potential_score": 0.10,
    }
    total = sum(
        scores.get(k, 0) * w for k, w in weights.items()
    )
    priority = _compute_priority(total)
    return total, priority


def create_hypothesis(
    source_question: str,
    hypothesis_type: str,
    hypothesis_statement: str,
    independent_variable: str,
    dependent_variable: str,
    moderating_variables: str = "",
    controlled_variables: str = "",
    specific_condition: str = "",
    mechanism_chain: str = "",
    possible_equation_or_model: str = "",
    required_data: str = "",
    supporting_evidence_ids: str = "",
    conflicting_evidence_ids: str = "",
    missing_evidence: str = "",
    experimental_design: str = "",
    characterization_methods: str = "",
    data_analysis_method: str = "",
    support_criteria: str = "",
    falsification_condition: str = "",
    scores: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """创建新假设并保存到 CSV。"""
    if hypothesis_type not in HYPOTHESIS_TYPES:
        raise ValueError(f"Invalid hypothesis_type: {hypothesis_type}. Must be one of {HYPOTHESIS_TYPES}")

    hyp_id = _next_hypothesis_id()
    now = datetime.now().isoformat()

    scores = scores or {}
    total_score, priority = _compute_priority_level(scores)

    record = {
        "hypothesis_id": hyp_id,
        "source_question": source_question,
        "hypothesis_type": hypothesis_type,
        "hypothesis_statement": hypothesis_statement,
        "independent_variable": independent_variable,
        "dependent_variable": dependent_variable,
        "moderating_variables": moderating_variables,
        "controlled_variables": controlled_variables,
        "specific_condition": specific_condition,
        "mechanism_chain": mechanism_chain,
        "possible_equation_or_model": possible_equation_or_model,
        "required_data": required_data,
        "supporting_evidence_ids": supporting_evidence_ids,
        "conflicting_evidence_ids": conflicting_evidence_ids,
        "missing_evidence": missing_evidence,
        "experimental_design": experimental_design,
        "characterization_methods": characterization_methods,
        "data_analysis_method": data_analysis_method,
        "support_criteria": support_criteria,
        "falsification_condition": falsification_condition,
        "specificity_score": str(scores.get("specificity_score", 0)),
        "evidence_grounding_score": str(scores.get("evidence_grounding_score", 0)),
        "mechanistic_plausibility_score": str(scores.get("mechanistic_plausibility_score", 0)),
        "parameter_awareness_score": str(scores.get("parameter_awareness_score", 0)),
        "testability_score": str(scores.get("testability_score", 0)),
        "falsifiability_score": str(scores.get("falsifiability_score", 0)),
        "novelty_score": str(scores.get("novelty_score", 0)),
        "paper_potential_score": str(scores.get("paper_potential_score", 0)),
        "total_score": f"{total_score:.1f}",
        "priority_level": priority,
        "created_time": now,
        "last_updated": now,
    }

    # 追加写入
    path = DATA_DIR / "hypothesis_dataset.csv"
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"[HYPOTHESIS] Created {hyp_id}: [{priority}] {hypothesis_statement[:80]}...")
    return record


def update_hypothesis(hypothesis_id: str, updates: Dict[str, str]):
    """更新假设字段。"""
    hypotheses = load_hypotheses()
    updated = False
    for h in hypotheses:
        if h.get("hypothesis_id") == hypothesis_id:
            h.update(updates)
            h["last_updated"] = datetime.now().isoformat()

            # 重新计算总分
            score_keys = [
                "specificity_score", "evidence_grounding_score",
                "mechanistic_plausibility_score", "parameter_awareness_score",
                "testability_score", "falsifiability_score",
                "novelty_score", "paper_potential_score",
            ]
            scores = {}
            for k in score_keys:
                try:
                    scores[k] = float(h.get(k, 0))
                except (ValueError, TypeError):
                    scores[k] = 0
            total, priority = _compute_priority_level(scores)
            h["total_score"] = f"{total:.1f}"
            h["priority_level"] = priority
            updated = True
            break

    if updated:
        _save_all(hypotheses)
        print(f"[HYPOTHESIS] Updated {hypothesis_id}")
    else:
        print(f"[WARNING] Hypothesis {hypothesis_id} not found")


def _save_all(hypotheses: List[Dict[str, str]]):
    """保存所有假设到 CSV。"""
    path = DATA_DIR / "hypothesis_dataset.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        for h in hypotheses:
            writer.writerow(h)


def get_hypothesis_statistics() -> Dict[str, Any]:
    """获取假设统计。"""
    hypotheses = load_hypotheses()
    stats = {
        "total": len(hypotheses),
        "by_type": {t: 0 for t in HYPOTHESIS_TYPES},
        "by_priority": {"critical": 0, "high": 0, "medium": 0, "low": 0, "exploratory": 0},
        "evidence_supported": 0,
        "search_guided": 0,
        "lacks_experimental_validation": 0,
    }
    for h in hypotheses:
        h_type = h.get("hypothesis_type", "")
        if h_type in stats["by_type"]:
            stats["by_type"][h_type] += 1
        priority = h.get("priority_level", "")
        if priority in stats["by_priority"]:
            stats["by_priority"][priority] += 1
        if h_type == "evidence_supported":
            stats["evidence_supported"] += 1
        if h_type == "search_guided_candidate":
            stats["search_guided"] += 1
        if not h.get("falsification_condition", "").strip():
            stats["lacks_experimental_validation"] += 1

    return stats


def export_hypothesis_for_sci() -> str:
    """导出假设统计为论文可引用的文本。"""
    stats = get_hypothesis_statistics()
    lines = [
        "## Hypothesis Dataset Statistics",
        "",
        f"Total hypotheses generated: {stats['total']}",
        "",
        "### By Hypothesis Type",
    ]
    for h_type, count in stats["by_type"].items():
        lines.append(f"- {h_type}: {count}")
    lines.append("")
    lines.append("### By Priority Level")
    for priority, count in stats["by_priority"].items():
        lines.append(f"- {priority}: {count}")
    lines.append("")
    lines.append(f"Evidence-supported hypotheses: {stats['evidence_supported']}")
    lines.append(f"Search-guided candidate hypotheses: {stats['search_guided']}")
    lines.append(f"Hypotheses lacking experimental validation: {stats['lacks_experimental_validation']}")
    return "\n".join(lines)
