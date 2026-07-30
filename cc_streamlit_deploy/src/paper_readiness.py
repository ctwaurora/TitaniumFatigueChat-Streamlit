"""
paper_readiness.py — 研究完整度评估 (论文级)

12 维度评分 → 5 级完整度判定：

评分维度:
  1. 文献库完整性
  2. 文献核验可靠性
  3. 结构化证据数量
  4. 变量关系覆盖度
  5. 方程参数数据覆盖度
  6. 假设具体性
  7. 假设可验证性
  8. 证据可追踪性
  9. baseline 是否完成
  10. ablation 是否完成
  11. 定量验证是否完成
  12. 论文图表材料是否准备完成

输出等级:
  not_ready (0-29)
  early_prototype (30-49)
  data_accumulation (50-64)
  paper_candidate_after_validation (65-79)
  paper_ready_candidate (80-100)
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.stage1_store import TRUSTED_EVIDENCE_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

READINESS_LEVELS = [
    (0, 29, "not_ready", "Not ready for publication submission"),
    (30, 49, "early_prototype", "Early prototype; significant data collection needed"),
    (50, 64, "data_accumulation", "Data accumulation phase; needs more evidence"),
    (65, 79, "paper_candidate_after_validation", "Paper candidate after further validation"),
    (80, 100, "paper_ready_candidate", "Paper ready for submission"),
]

DIMENSION_WEIGHTS = {
    "literature_database_completeness": 0.10,
    "literature_verification_reliability": 0.08,
    "structured_evidence_volume": 0.10,
    "variable_relation_coverage": 0.10,
    "equation_parameter_coverage": 0.08,
    "hypothesis_specificity": 0.10,
    "hypothesis_testability": 0.08,
    "evidence_traceability": 0.10,
    "baseline_completed": 0.08,
    "ablation_completed": 0.08,
    "quantitative_validation_completed": 0.05,
    "paper_figures_tables_prepared": 0.05,
}


def _load_csv_count(path: Path) -> int:
    """返回 CSV 数据行数（不计表头）。"""
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1  # 减去表头
    except Exception:
        return 0


def _load_csv_field_count(path: Path, field: str) -> int:
    """返回 CSV 中某字段非空行数。"""
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(field, "").strip():
                    count += 1
    except Exception:
        pass
    return count


def score_literature_database_completeness() -> Tuple[float, str]:
    """1. 文献库完整性：目标 30+ 篇。"""
    paper_count = _load_csv_count(DATA_DIR / "literature_database.csv")
    candidate_count = _load_csv_count(DATA_DIR / "candidate_papers.csv")

    if paper_count >= 30:
        return (100, f"{paper_count} papers in database (target: 30+)")
    elif paper_count >= 20:
        return (70, f"{paper_count} papers + {candidate_count} candidates (target: 30)")
    elif paper_count >= 10:
        return (40, f"{paper_count} papers (need ≥30)")
    else:
        return (10, f"Only {paper_count} papers (critical shortage)")


def score_verification_reliability() -> Tuple[float, str]:
    """2. 文献核验可靠性。"""
    path = DATA_DIR / "literature_database.csv"
    if not path.exists():
        return (0, "No literature database")

    verified = _load_csv_field_count(path, "verification_status")
    total = _load_csv_count(path)
    if total == 0:
        return (0, "No papers")

    ratio = verified / total
    if ratio >= 0.9:
        return (100, f"{verified}/{total} papers verified ({ratio:.0%})")
    elif ratio >= 0.7:
        return (70, f"{verified}/{total} verified")
    elif ratio >= 0.5:
        return (50, f"Only {verified}/{total} verified")
    else:
        return (20, f"Most papers unverified ({verified}/{total})")


def score_evidence_volume() -> Tuple[float, str]:
    """3. 结构化证据数量：目标 100+ 条。"""
    count = _load_csv_count(TRUSTED_EVIDENCE_PATH)
    if count >= 100:
        return (100, f"{count} evidence snippets (target: 100+)")
    elif count >= 50:
        return (70, f"{count} evidence snippets")
    elif count >= 20:
        return (40, f"{count} evidence snippets (need ≥50)")
    else:
        return (10, f"Only {count} evidence snippets")


def score_variable_relation_coverage() -> Tuple[float, str]:
    """4. 变量关系覆盖度。"""
    # 检查 variable_relation_dataset.csv
    vr_path = DATA_DIR / "variable_relation_dataset.csv"
    ev_path = TRUSTED_EVIDENCE_PATH

    vr_count = _load_csv_count(vr_path)

    # 也检查 evidence_snippets 中有多少独立变量
    unique_iv = set()
    unique_dv = set()
    try:
        with open(ev_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("independent_variable", "").strip():
                    unique_iv.add(row["independent_variable"].strip())
                if row.get("dependent_variable", "").strip():
                    unique_dv.add(row["dependent_variable"].strip())
    except Exception:
        pass

    total_vars = len(unique_iv) + len(unique_dv)

    if vr_count >= 15 and total_vars >= 10:
        return (100, f"{vr_count} relations, {total_vars} unique variables")
    elif vr_count >= 8 and total_vars >= 6:
        return (65, f"{vr_count} relations, {total_vars} variables")
    elif vr_count >= 3:
        return (35, f"{vr_count} relations")
    else:
        return (10, f"Minimal variable relations ({vr_count})")


def score_equation_coverage() -> Tuple[float, str]:
    """5. 方程参数覆盖度。"""
    eq_path = DATA_DIR / "equation_parameter_dataset.csv"
    eq_count = _load_csv_count(eq_path)

    # Paris law 参数数量
    paris_count = 0
    try:
        with open(DATA_DIR / "literature_database.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Paris_C", "").strip() or row.get("Paris_m", "").strip():
                    paris_count += 1
    except Exception:
        pass

    if eq_count >= 15 and paris_count >= 5:
        return (100, f"{eq_count} equation records, {paris_count} with Paris params")
    elif eq_count >= 8 or paris_count >= 3:
        return (65, f"{eq_count} equation records, {paris_count} Paris params")
    elif eq_count >= 3:
        return (35, f"{eq_count} equation records")
    else:
        return (10, f"Minimal equation data ({eq_count})")


def score_hypothesis_specificity() -> Tuple[float, str]:
    """6. 假设具体性：检查是否有 falsification_condition。"""
    hyp_path = DATA_DIR / "hypothesis_dataset.csv"
    if not hyp_path.exists():
        return (0, "No hypothesis dataset")

    total = _load_csv_count(hyp_path)
    with_falsif = _load_csv_field_count(hyp_path, "falsification_condition")
    with_exp = _load_csv_field_count(hyp_path, "experimental_design")

    if total == 0:
        return (0, "No hypotheses")

    falsif_ratio = with_falsif / total if total > 0 else 0
    exp_ratio = with_exp / total if total > 0 else 0

    score = 50 * (falsif_ratio + exp_ratio)

    if score >= 80:
        return (100, f"{total} hypotheses, {with_falsif} falsifiable, {with_exp} with experimental design")
    elif score >= 50:
        return (65, f"{total} hypotheses, falsif_ratio={falsif_ratio:.0%}")
    else:
        return (30, f"Hypotheses lack specificity ({total} total)")


def score_hypothesis_testability() -> Tuple[float, str]:
    """7. 假设可验证性。"""
    hyp_path = DATA_DIR / "hypothesis_dataset.csv"
    total = _load_csv_count(hyp_path)
    with_support = _load_csv_field_count(hyp_path, "supporting_evidence_ids")
    with_support_criteria = _load_csv_field_count(hyp_path, "support_criteria")

    if total == 0:
        return (0, "No hypotheses")

    supported_ratio = with_support / total if total > 0 else 0
    criteria_ratio = with_support_criteria / total if total > 0 else 0

    score = 60 * supported_ratio + 40 * criteria_ratio
    return (score, f"supported_ratio={supported_ratio:.0%}, criteria_ratio={criteria_ratio:.0%}")


def score_evidence_traceability() -> Tuple[float, str]:
    """8. 证据可追踪性。"""
    ev_path = TRUSTED_EVIDENCE_PATH
    ev_count = _load_csv_count(ev_path)
    ev_with_paper_id = _load_csv_field_count(ev_path, "paper_id")

    # 还有假设关联
    hyp_path = DATA_DIR / "hypothesis_dataset.csv"
    hyp_with_evidence = _load_csv_field_count(hyp_path, "supporting_evidence_ids")

    if ev_count == 0:
        return (0, "No evidence snippets")

    paper_trace_ratio = ev_with_paper_id / ev_count if ev_count > 0 else 0

    if paper_trace_ratio >= 0.95 and hyp_with_evidence > 0:
        return (100, f"{ev_count} evidence, all paper-traced, {hyp_with_evidence} hypotheses link evidence")
    elif paper_trace_ratio >= 0.8:
        return (70, f"{ev_count} evidence, {paper_trace_ratio:.0%} paper-traced")
    else:
        return (35, f"Evidence traceability limited ({ev_count} total)")


def score_baseline_completed() -> Tuple[float, str]:
    """9. Baseline 是否完成。"""
    path = DATA_DIR / "baseline_comparison.csv"
    count = _load_csv_count(path)
    if count >= 15:
        return (100, f"Baseline completed ({count} comparison entries)")
    elif count >= 10:
        return (70, f"Partial baseline ({count} entries)")
    elif count >= 5:
        return (40, f"Minimal baseline ({count} entries)")
    else:
        return (0, "Baseline not completed")


def score_ablation_completed() -> Tuple[float, str]:
    """10. Ablation 是否完成。"""
    path = DATA_DIR / "ablation_results.csv"
    count = _load_csv_count(path)
    if count >= 12:
        return (100, f"Ablation completed ({count} ablation entries)")
    elif count >= 8:
        return (75, f"Partial ablation ({count} entries)")
    elif count >= 4:
        return (40, f"Minimal ablation ({count} entries)")
    else:
        return (0, "Ablation not completed")


def score_quantitative_validation() -> Tuple[float, str]:
    """11. 定量验证是否完成。"""
    paris_path = DATA_DIR / "paris_law_validation_dataset.csv"
    retro_path = DATA_DIR / "retrospective_validation_results.csv"

    paris_count = _load_csv_count(paris_path)
    retro_count = _load_csv_count(retro_path)

    if paris_count >= 20 and retro_count >= 5:
        return (100, f"Paris validation ({paris_count}) + retrospective ({retro_count}) done")
    elif paris_count >= 10 or retro_count >= 3:
        return (65, f"Partial validation: Paris({paris_count}), Retro({retro_count})")
    elif paris_count >= 5:
        return (40, f"Limited validation: Paris({paris_count})")
    else:
        return (10, "Quantitative validation not completed")


def score_paper_figures_tables() -> Tuple[float, str]:
    """12. 论文图表材料准备情况。"""
    # 检查 outputs 目录中的相关文件
    paper_files = [
        OUTPUTS_DIR / "methods_draft.md",
        OUTPUTS_DIR / "results_tables.md",
        OUTPUTS_DIR / "discussion_claims.md",
        OUTPUTS_DIR / "figure_plan.md",
    ]
    # Also check old sci_ prefixed files for backward compatibility
    for old_name, new_name in [
        ("sci_methods_draft.md", "methods_draft.md"),
        ("sci_results_tables.md", "results_tables.md"),
        ("sci_discussion_claims.md", "discussion_claims.md"),
        ("sci_figure_plan.md", "figure_plan.md"),
    ]:
        old_path = OUTPUTS_DIR / old_name
        new_path = OUTPUTS_DIR / new_name
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)

    existing = sum(1 for f in paper_files if f.exists())

    if existing >= 4:
        return (100, f"All paper materials prepared ({existing}/4)")
    elif existing >= 2:
        return (60, f"Partial paper materials ({existing}/4)")
    elif existing >= 1:
        return (30, f"Minimal paper materials ({existing}/4)")
    else:
        return (0, "No paper materials prepared")


def compute_readiness() -> Dict:
    """计算完整就绪度评分。"""
    score_funcs = {
        "literature_database_completeness": score_literature_database_completeness,
        "literature_verification_reliability": score_verification_reliability,
        "structured_evidence_volume": score_evidence_volume,
        "variable_relation_coverage": score_variable_relation_coverage,
        "equation_parameter_coverage": score_equation_coverage,
        "hypothesis_specificity": score_hypothesis_specificity,
        "hypothesis_testability": score_hypothesis_testability,
        "evidence_traceability": score_evidence_traceability,
        "baseline_completed": score_baseline_completed,
        "ablation_completed": score_ablation_completed,
        "quantitative_validation_completed": score_quantitative_validation,
        "paper_figures_tables_prepared": score_paper_figures_tables,
    }

    details = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for dim, func in score_funcs.items():
        score, detail = func()
        weight = DIMENSION_WEIGHTS.get(dim, 0.05)
        weighted_sum += score * weight
        total_weight += weight
        details[dim] = {
            "score": score,
            "weight": weight,
            "weighted_contribution": score * weight / total_weight if total_weight > 0 else 0,
            "detail": detail,
        }

    overall_score = weighted_sum / total_weight if total_weight > 0 else 0

    # 等级判定
    level = "not_ready"
    level_label = "Not ready for publication submission"
    for low, high, lvl, lbl in READINESS_LEVELS:
        if low <= overall_score <= high:
            level = lvl
            level_label = lbl
            break

    # 如果有 critical 缺失，降级
    critical_missing = []
    if details.get("literature_database_completeness", {}).get("score", 0) < 20:
        critical_missing.append("literature_database")
    if details.get("structured_evidence_volume", {}).get("score", 0) < 20:
        critical_missing.append("evidence_volume")
    if details.get("baseline_completed", {}).get("score", 0) == 0:
        critical_missing.append("baseline")
    if details.get("ablation_completed", {}).get("score", 0) == 0:
        critical_missing.append("ablation")

    if critical_missing:
        overall_score = min(overall_score, 64)
        level = "data_accumulation"
        level_label = f"Missing critical components: {', '.join(critical_missing)}"

    if overall_score < 50:
        level = "early_prototype"
        level_label = "Early prototype; significant data collection needed"

    return {
        "overall_score": round(overall_score, 1),
        "level": level,
        "level_label": level_label,
        "dimensions": details,
        "critical_missing": critical_missing,
    }


def run_paper_readiness_gate(verbose: bool = True) -> Dict:
    """主入口：运行论文就绪度门禁。"""
    if verbose:
        print("=" * 60)
        print("  Paper Readiness Gate (论文就绪度门禁)")
        print("=" * 60)

    result = compute_readiness()

    if verbose:
        print(f"\n  Overall Score: {result['overall_score']:.1f}/100")
        print(f"  Level: {result['level']}")
        print(f"  Description: {result['level_label']}")
        print()
        print(f"  {'Dimension':<45} {'Score':>8}")
        print(f"  {'-'*45} {'-'*8}")
        for dim, info in result["dimensions"].items():
            label = dim.replace("_", " ").title()
            print(f"  {label:<45} {info['score']:>8.1f}")
        print()
        if result["critical_missing"]:
            print(f"  ⚠ Critical missing: {', '.join(result['critical_missing'])}")
        print()

    return result


if __name__ == "__main__":
    run_paper_readiness_gate()
