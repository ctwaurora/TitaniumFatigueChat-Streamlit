"""
research_gap_dataset.py — 研究空白数据集管理模块

每个研究空白保存到 data/research_gap_dataset.csv，
包含 28 字段的完整 schema。

禁止泛泛输出：
  ❌ "需要进一步研究"
  ❌ "机制尚不清楚"
  ❌ "影响疲劳性能"
  ❌ "文献存在争议"

必须具体到：
  ✅ 哪个变量
  ✅ 哪个疲劳指标
  ✅ 哪个条件
  ✅ 缺哪种证据
  ✅ 可提出什么假设
  ✅ 需要什么实验验证
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── 空白类型 ──

GAP_TYPES = [
    "parameter_gap",          # 缺少关键方程参数
    "mechanism_gap",          # 机制理解不足
    "conflict_gap",           # 文献冲突未解决
    "validation_gap",         # 缺乏定量验证
    "condition_gap",          # 特定条件未覆盖
    "interaction_gap",        # 多因素交互未研究
    "methodological_gap",     # 方法学缺失
]

# ── 泛泛表达检测 ──

VAGUE_PATTERNS = [
    r"进一步研究",
    r"有待探索",
    r"需要更多实验",
    r"需要进一步",
    r"值得深入研究",
    r"有待进一步",
    r"尚需更多",
    r"有待更深入",
    r"需要更系统的",
    r"应开展更多",
    r"建议进一步",
    r"机制尚不清",
    r"影响疲劳性能",
    r"文献存在争议",
    r"needs further study",
    r"mechanism is unclear",
    r"requires more investigation",
    r"remains to be explored",
    r"needs more research",
]

GAP_FIELD_NAMES = [
    "gap_id",
    "gap_type",
    "gap_title",
    "gap_statement",
    "related_variables",
    "target_indicators",
    "existing_evidence_summary",
    "supporting_evidence_ids",
    "conflicting_evidence_ids",
    "missing_evidence",
    "why_it_matters",
    "candidate_hypothesis",
    "possible_equation_or_model",
    "experimental_design",
    "required_data_fields",
    "falsification_condition",
    "scientific_value_score",
    "novelty_score",
    "testability_score",
    "feasibility_score",
    "parameter_potential_score",
    "conflict_usefulness_score",
    "paper_potential_score",
    "total_priority_score",
    "priority_rank",
    "priority_level",
    "created_time",
    "last_updated",
]


def _next_gap_id() -> str:
    path = DATA_DIR / "research_gap_dataset.csv"
    if not path.exists():
        return "GAP_0001"
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            max_num = 0
            for row in reader:
                gid = row.get("gap_id", "")
                if gid.startswith("GAP_"):
                    try:
                        num = int(gid.split("_")[1])
                        max_num = max(max_num, num)
                    except (IndexError, ValueError):
                        pass
        return f"GAP_{max_num + 1:04d}"
    except Exception:
        return "GAP_0001"


def validate_gap_statement(gap_statement: str) -> List[str]:
    """
    检查 gap 是否存在泛泛表达。
    返回所有违规模式列表（空列表表示通过）。
    """
    import re
    violations = []
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, gap_statement, re.IGNORECASE):
            violations.append(f"Contains vague pattern: '{pattern}'")
    return violations


def validate_gap_quality(gap: Dict[str, str]) -> List[str]:
    """全面检查 gap 质量。"""
    issues = []

    # 必须指定变量
    if not gap.get("related_variables", "").strip():
        issues.append("related_variables is empty — must specify which variable")
    if not gap.get("target_indicators", "").strip():
        issues.append("target_indicators is empty — must specify which fatigue indicator")
    if not gap.get("missing_evidence", "").strip():
        issues.append("missing_evidence is empty — must specify what evidence is lacking")
    if not gap.get("candidate_hypothesis", "").strip():
        issues.append("candidate_hypothesis is empty — must propose a concrete hypothesis")
    if not gap.get("experimental_design", "").strip():
        issues.append("experimental_design is empty — must describe what experiment is needed")
    if not gap.get("falsification_condition", "").strip():
        issues.append("falsification_condition is empty — must specify what would disprove this")

    # 禁止泛泛表达
    statement = gap.get("gap_statement", "")
    violations = validate_gap_statement(statement)
    issues.extend(violations)

    return issues


def _compute_priority(scores: Dict[str, float]) -> Tuple[float, str]:
    """计算优先级分数和等级。"""
    weights = {
        "scientific_value_score": 0.18,
        "novelty_score": 0.15,
        "testability_score": 0.15,
        "feasibility_score": 0.12,
        "parameter_potential_score": 0.14,
        "conflict_usefulness_score": 0.10,
        "paper_potential_score": 0.16,
    }
    total = sum(
        scores.get(k, 0) * w for k, w in weights.items()
    )

    if total >= 80:
        level = "critical"
    elif total >= 65:
        level = "high"
    elif total >= 50:
        level = "medium"
    elif total >= 35:
        level = "low"
    else:
        level = "exploratory"

    return total, level


def create_gap(
    gap_type: str,
    gap_title: str,
    gap_statement: str,
    related_variables: str,
    target_indicators: str,
    existing_evidence_summary: str = "",
    supporting_evidence_ids: str = "",
    conflicting_evidence_ids: str = "",
    missing_evidence: str = "",
    why_it_matters: str = "",
    candidate_hypothesis: str = "",
    possible_equation_or_model: str = "",
    experimental_design: str = "",
    required_data_fields: str = "",
    falsification_condition: str = "",
    scores: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """创建新研究空白。"""

    # 类型验证
    if gap_type not in GAP_TYPES:
        raise ValueError(f"Invalid gap_type: {gap_type}. Must be one of {GAP_TYPES}")

    # 质量检查
    temp_record = {
        "gap_statement": gap_statement,
        "related_variables": related_variables,
        "target_indicators": target_indicators,
        "missing_evidence": missing_evidence,
        "candidate_hypothesis": candidate_hypothesis,
        "experimental_design": experimental_design,
        "falsification_condition": falsification_condition,
    }
    issues = validate_gap_quality(temp_record)
    if issues:
        for issue in issues:
            print(f"[WARNING] Gap quality issue: {issue}")

    # 生成 ID
    gap_id = _next_gap_id()
    now = datetime.now().isoformat()

    # 计分
    scores = scores or {}
    total_score, priority_level = _compute_priority(scores)

    record = {
        "gap_id": gap_id,
        "gap_type": gap_type,
        "gap_title": gap_title,
        "gap_statement": gap_statement,
        "related_variables": related_variables,
        "target_indicators": target_indicators,
        "existing_evidence_summary": existing_evidence_summary,
        "supporting_evidence_ids": supporting_evidence_ids,
        "conflicting_evidence_ids": conflicting_evidence_ids,
        "missing_evidence": missing_evidence,
        "why_it_matters": why_it_matters,
        "candidate_hypothesis": candidate_hypothesis,
        "possible_equation_or_model": possible_equation_or_model,
        "experimental_design": experimental_design,
        "required_data_fields": required_data_fields,
        "falsification_condition": falsification_condition,
        "scientific_value_score": str(scores.get("scientific_value_score", 0)),
        "novelty_score": str(scores.get("novelty_score", 0)),
        "testability_score": str(scores.get("testability_score", 0)),
        "feasibility_score": str(scores.get("feasibility_score", 0)),
        "parameter_potential_score": str(scores.get("parameter_potential_score", 0)),
        "conflict_usefulness_score": str(scores.get("conflict_usefulness_score", 0)),
        "paper_potential_score": str(scores.get("paper_potential_score", 0)),
        "total_priority_score": f"{total_score:.1f}",
        "priority_rank": "",
        "priority_level": priority_level,
        "created_time": now,
        "last_updated": now,
    }

    # 追加写入
    path = DATA_DIR / "research_gap_dataset.csv"
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_FIELD_NAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"[GAP] Created {gap_id}: [{priority_level}] {gap_title}")
    return record


def load_gaps() -> List[Dict[str, str]]:
    """加载所有研究空白。"""
    path = DATA_DIR / "research_gap_dataset.csv"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def get_gap_by_id(gap_id: str) -> Optional[Dict[str, str]]:
    gaps = load_gaps()
    for g in gaps:
        if g.get("gap_id") == gap_id:
            return g
    return None


def get_gaps_by_priority(level: str) -> List[Dict[str, str]]:
    return [g for g in load_gaps() if g.get("priority_level") == level]


def get_gap_statistics() -> Dict[str, Any]:
    """获取研究空白统计。"""
    gaps = load_gaps()
    stats = {
        "total": len(gaps),
        "by_type": {t: 0 for t in GAP_TYPES},
        "by_priority": {"critical": 0, "high": 0, "medium": 0, "low": 0, "exploratory": 0},
        "has_experimental_design": 0,
        "has_falsification": 0,
    }
    for g in gaps:
        g_type = g.get("gap_type", "")
        if g_type in stats["by_type"]:
            stats["by_type"][g_type] += 1
        priority = g.get("priority_level", "")
        if priority in stats["by_priority"]:
            stats["by_priority"][priority] += 1
        if g.get("experimental_design", "").strip():
            stats["has_experimental_design"] += 1
        if g.get("falsification_condition", "").strip():
            stats["has_falsification"] += 1

    return stats


def re_rank_gaps():
    """重新排序所有 gap 的 priority_rank。"""
    gaps = load_gaps()
    gaps.sort(
        key=lambda g: float(g.get("total_priority_score", 0)),
        reverse=True,
    )
    for i, g in enumerate(gaps):
        g["priority_rank"] = str(i + 1)

    path = DATA_DIR / "research_gap_dataset.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        for g in gaps:
            writer.writerow(g)

    print(f"[GAP] Re-ranked {len(gaps)} gaps")


def export_gaps_for_sci() -> str:
    """导出研究空白统计为论文可引用文本。"""
    stats = get_gap_statistics()
    re_rank_gaps()
    gaps = load_gaps()

    lines = [
        "## Research Gap Dataset",
        "",
        f"Total research gaps identified: {stats['total']}",
        "",
        "### By Gap Type",
    ]
    for g_type, count in stats["by_type"].items():
        lines.append(f"- {g_type}: {count}")
    lines.append("")
    lines.append("### By Priority Level")
    for priority, count in stats["by_priority"].items():
        lines.append(f"- {priority}: {count}")
    lines.append("")
    lines.append(f"Gaps with experimental design: {stats['has_experimental_design']}")
    lines.append(f"Gaps with falsification condition: {stats['has_falsification']}")
    lines.append("")

    # 显示 Top-5 gaps
    lines.append("### Top-5 Research Gaps (by priority score)")
    lines.append("")
    lines.append("| Rank | Gap ID | Title | Type | Priority | Score |")
    lines.append("|------|--------|-------|------|----------|-------|")
    for g in gaps[:5]:
        lines.append(
            f"| {g.get('priority_rank', '')} "
            f"| {g.get('gap_id', '')} "
            f"| {g.get('gap_title', '')[:60]} "
            f"| {g.get('gap_type', '')} "
            f"| {g.get('priority_level', '')} "
            f"| {g.get('total_priority_score', '')} |"
        )

    return "\n".join(lines)
