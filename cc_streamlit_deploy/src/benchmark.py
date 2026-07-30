"""
benchmark.py — TitaniumFatigueBench 评测模块

比较三种系统配置：
1. Direct Qwen (无 RAG)
2. RAG only (检索+直接回答)
3. TitaniumFatigueChat full system (完整系统)

评价指标：
answer_accuracy, variable_identification, evidence_grounding,
mechanism_depth, experiment_design_quality, formula_model_matching,
hypothesis_specificity, falsifiability

保存到 data/benchmark_results.csv
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BENCHMARK_FIELDS = [
    "run_id", "timestamp", "system_config", "question_id", "question",
    "question_type", "difficulty",
    "answer_accuracy", "variable_identification",
    "evidence_grounding", "mechanism_depth",
    "experiment_design_quality", "formula_model_matching",
    "hypothesis_specificity", "falsifiability",
    "overall_score", "model_used", "answer_text",
    "error_message",
]

SYSTEM_CONFIGS = [
    "direct_qwen",
    "rag_only",
    "titanium_fatigue_chat_full",
]


def load_benchmark_questions() -> pd.DataFrame:
    path = DATA_DIR / "titanium_fatigue_bench.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")


def create_condition_evidence_dataset():
    """创建条件化证据数据集 (condition_evidence_dataset.csv)"""
    path = DATA_DIR / "condition_evidence_dataset.csv"
    fieldnames = [
        "evidence_id", "paper_id", "title",
        "condition_id", "condition_description",
        "material", "process", "heat_treatment", "surface_state",
        "surface_roughness_Ra", "surface_roughness_Rz",
        "stress_ratio_R", "fatigue_type", "temperature", "environment",
        "defect_type", "pore_size", "sqrt_area", "distance_to_surface",
        "pore_location", "porosity",
        "claim", "mechanism", "fatigue_indicator",
        "evidence_strength", "source_sentence",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
    print(f"[benchmark] Created {path}")


def create_hypothesis_dataset():
    """创建假设数据集 (hypothesis_dataset.csv)"""
    path = DATA_DIR / "hypothesis_dataset.csv"
    fieldnames = [
        "hypothesis_id", "gap_id",
        "hypothesis_statement", "null_hypothesis",
        "independent_variable", "dependent_variable",
        "moderating_variables", "controlled_variables",
        "mechanism_chain", "prediction_direction",
        "expected_model", "support_criteria",
        "falsification_condition", "condition_boundary",
        "evidence_summary", "evidence_balance",
        "hypothesis_score", "status",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
    print(f"[benchmark] Created {path}")


def create_experiment_design_dataset():
    """创建实验方案数据集 (experiment_design_dataset.csv)"""
    path = DATA_DIR / "experiment_design_dataset.csv"
    fieldnames = [
        "design_id", "hypothesis_id", "question",
        "experiment_objective", "scientific_hypothesis",
        "independent_variable", "dependent_variable",
        "controlled_variables", "moderating_variables",
        "prediction_direction", "support_criteria",
        "falsification_conditions",
        "sample_groups", "testing_method",
        "characterization_method", "data_analysis_method",
        "confounding_variables_warning", "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
    print(f"[benchmark] Created {path}")


def create_benchmark_results():
    """创建基准评测结果文件 (benchmark_results.csv)"""
    path = DATA_DIR / "benchmark_results.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_FIELDS, extrasaction="ignore")
        writer.writeheader()
    print(f"[benchmark] Created {path}")


def run_benchmark_evaluation(
    system_config: str = "titanium_fatigue_chat_full",
    questions: Optional[List[Dict[str, Any]]] = None,
    model_used: str = "qwen-plus",
) -> List[Dict[str, Any]]:
    """
    运行一次基准评测。

    Args:
        system_config: 系统配置 (direct_qwen / rag_only / full)
        questions: 问题列表（未指定则全部加载）
        model_used: 使用的模型

    Returns:
        评测结果列表
    """
    if questions is None:
        df = load_benchmark_questions()
        if df.empty:
            print("[benchmark] No benchmark questions found.")
            return []
        questions = df.to_dict("records")

    results = []
    run_id = f"BR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()

    for q in questions:
        qid = q.get("question_id", "?")
        qtext = q.get("question", "")
        qtype = q.get("question_type", "")
        difficulty = q.get("difficulty", "")

        result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "system_config": system_config,
            "question_id": qid,
            "question": qtext,
            "question_type": qtype,
            "difficulty": difficulty,
            "answer_accuracy": "",
            "variable_identification": "",
            "evidence_grounding": "",
            "mechanism_depth": "",
            "experiment_design_quality": "",
            "formula_model_matching": "",
            "hypothesis_specificity": "",
            "falsifiability": "",
            "overall_score": "",
            "model_used": model_used,
            "answer_text": "",
            "error_message": "",
        }
        results.append(result)

    return results


def save_benchmark_results(results: List[Dict[str, Any]]):
    """保存评测结果到 CSV。"""
    path = DATA_DIR / "benchmark_results.csv"
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in BENCHMARK_FIELDS})
    print(f"[benchmark] Saved {len(results)} results to {path}")


def generate_benchmark_report(results: List[Dict[str, Any]]) -> str:
    """生成评测报告 Markdown。"""
    if not results:
        return "# Benchmark Report\n\nNo results to report.\n"

    lines = [
        "# TitaniumFatigueBench 评测报告\n",
        f"评测时间: {datetime.now().isoformat()}",
        f"问题数: {len(results)}\n",
        "---\n",
        "## 评测配置\n",
        "### 系统配置对比\n\n",
        "| 配置 | 说明 |\n",
        "|---|---|\n",
        "| direct_qwen | 直接调用 Qwen 回答，无 RAG |\n",
        "| rag_only | RAG 检索 + 直接回答 |\n",
        "| titanium_fatigue_chat_full | 完整系统（RAG+证据链+假设生成+实验辅助） |\n\n",
        "### 评价指标\n\n",
        "| 指标 | 说明 | 评分范围 |\n",
        "|---|---|---|\n",
        "| answer_accuracy | 回答准确性 | 0-5 |\n",
        "| variable_identification | 变量识别正确性 | 0-5 |\n",
        "| evidence_grounding | 证据支撑程度 | 0-5 |\n",
        "| mechanism_depth | 机制分析深度 | 0-5 |\n",
        "| experiment_design_quality | 实验设计质量 | 0-5 |\n",
        "| formula_model_matching | 公式模型匹配 | 0-5 |\n",
        "| hypothesis_specificity | 假设具体性 | 0-5 |\n",
        "| falsifiability | 可推翻性 | 0-5 |\n",
        "| overall_score | 总分（8 项平均） | 0-5 |\n\n",
        "---\n",
    ]

    # Group by system config
    configs = set(r.get("system_config", "") for r in results)
    for cfg in sorted(configs):
        cfg_results = [r for r in results if r.get("system_config") == cfg]
        lines.append(f"## {cfg}\n")
        lines.append(f"问题数: {len(cfg_results)}\n\n")

        # Average scores
        metrics = ["answer_accuracy", "variable_identification", "evidence_grounding",
                   "mechanism_depth", "experiment_design_quality", "formula_model_matching",
                   "hypothesis_specificity", "falsifiability"]
        lines.append("| 指标 | 平均分 |\n|---|---|\n")
        for m in metrics:
            scores = []
            for r in cfg_results:
                v = r.get(m, "")
                try:
                    scores.append(float(v))
                except (ValueError, TypeError):
                    pass
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"| {m} | {avg:.2f} |\n")
        lines.append("\n---\n")

    return "".join(lines)


def init_benchmark_data():
    """初始化所有基准数据文件。"""
    create_condition_evidence_dataset()
    create_hypothesis_dataset()
    create_experiment_design_dataset()
    create_benchmark_results()
    print("[benchmark] All benchmark data files initialized.")


if __name__ == "__main__":
    init_benchmark_data()
