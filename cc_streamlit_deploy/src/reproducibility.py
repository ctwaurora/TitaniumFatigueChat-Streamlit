"""
reproducibility.py — Reproducibility Manifest 模块

生成：
- outputs/13_reproducibility_manifest.md
- competition_package/run_command.txt (更新版)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from skills.library_skill import get_all_papers

OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
COMP_DIR = BASE_DIR / "competition_package"


def run_reproducibility() -> Dict[str, Any]:
    """Generate the reproducibility manifest and update run_command.txt.

    Returns:
        Dict with paths.
    """
    papers = get_all_papers()
    n_papers = len(papers)

    # Collect output file list
    output_files = _collect_output_files()
    data_files = _collect_data_files()

    # Generate manifest
    manifest_path = _write_manifest(n_papers, output_files, data_files)

    # Update run_command.txt
    run_path = _write_run_command()

    return {
        "manifest": str(manifest_path),
        "run_command": str(run_path),
        "output_files": len(output_files),
        "data_files": len(data_files),
    }


def _collect_output_files() -> List[Dict[str, str]]:
    """Collect output markdown files with descriptions."""
    output_mapping = [
        ("01_evidence_map.md", "Literature evidence map"),
        ("02_gap_diagnosis.md", "Research gap diagnosis"),
        ("03_hypothesis_summary.md", "Scientific hypothesis summary"),
        ("04_baseline_comparison.md", "Baseline comparison against direct Qwen"),
        ("05_scientific_hypothesis_plan.md", "Full research plan with validation"),
        ("06_competition_readiness.md", "Competition readiness self-check"),
        ("07_retrospective_validation.md", "Historical lookback validation"),
        ("08_ablation_study.md", "Ablation study"),
        ("09_evidence_trace_report.md", "Evidence snippet trace report"),
        ("10_award_readiness_scorecard.md", "Award readiness scorecard"),
        ("11_qwen_usage_report.md", "Qwen usage report"),
        ("12_evidence_quality_gate.md", "Evidence quality gate report"),
        ("13_reproducibility_manifest.md", "Reproducibility manifest"),
        ("final_demo_report.md", "Final demo run report"),
    ]
    result = []
    for name, desc in output_mapping:
        path = OUTPUTS_DIR / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        result.append({
            "name": name,
            "description": desc,
            "path": str(path),
            "exists": exists,
            "size_bytes": size,
        })
    return result


def _collect_data_files() -> List[Dict[str, str]]:
    """Collect data CSV files with descriptions."""
    data_mapping = [
        ("literature_database.csv", "Structured literature cards (13+ fields)"),
        ("evidence_snippets.csv", "Evidence snippet trace data"),
        ("ablation_results.csv", "Ablation study scores"),
        ("retrospective_validation_pairs.csv", "Retrospective validation pairs"),
        ("minimum_validation_dataset_schema.csv", "Minimum validation dataset schema"),
        ("variable_mechanism.csv", "Variable-property-mechanism relationship table"),
        ("coverage_matrix.csv", "Evidence coverage matrix"),
        ("qwen_call_log.csv", "Qwen API call log"),
    ]
    result = []
    for name, desc in data_mapping:
        path = DATA_DIR / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        result.append({
            "name": name,
            "description": desc,
            "path": str(path),
            "exists": exists,
            "size_bytes": size,
        })
    return result


def _write_manifest(n_papers: int, output_files: List[Dict], data_files: List[Dict]) -> Path:
    """Write 13_reproducibility_manifest.md"""
    # Check Python version
    import platform
    py_version = platform.python_version()

    # Check requirements file
    req_path = BASE_DIR / "requirements.txt"
    req_exists = req_path.exists()

    # Count PDF directories
    pdf_dirs_info = []
    for d in ["papers", "early_papers", "followup_papers"]:
        d_path = BASE_DIR / d
        if d_path.exists():
            pdf_count = len(list(d_path.glob("*.pdf")))
            pdf_dirs_info.append(f"  - {d}/: {pdf_count} PDFs")
        else:
            pdf_dirs_info.append(f"  - {d}/: (not found)")

    existing_outputs = [f for f in output_files if f["exists"]]
    existing_data = [f for f in data_files if f["exists"]]

    lines = [
        "# Reproducibility Manifest（可复现性声明）",
        "",
        "> **目的**: 确保评审者可以从零开始完整复现本系统的所有输出。",
        f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. 环境要求",
        "",
        "### Python 版本",
        "",
        f"Python {py_version}",
        "",
        "### 依赖",
        "",
        f"- `requirements.txt`: {'✅ 存在' if req_exists else '❌ 未找到'}",
        "- 核心依赖：typer, rich, PyMuPDF, pandas, requests",
        "- 安装命令：`pip install -r requirements.txt`",
        "",
        "### API Key",
        "",
        "- 需要阿里云 Qwen API Key（以 sk- 开头）",
        "- 保存到 `qwen_key.txt`",
        "- 如果没有 API Key，系统仍可运行但不会调用 Qwen（部分输出为空）",
        "",
        "---",
        "",
        "## 2. 输入",
        "",
        "### PDF 文献",
        "",
        "将钛合金疲劳相关的 PDF 文件放入以下目录：",
        "",
    ]
    lines.extend(pdf_dirs_info)
    lines.extend([
        "",
        f"当前文献库: {n_papers} 篇去重文献",
        "",
        "### 目录结构说明",
        "",
        "- `papers/`: 核心文献（纳入全部分析）",
        "- `early_papers/`: 早期文献（用于历史回溯验证）",
        "- `followup_papers/`: 后续文献（用于历史回溯验证）",
        "",
        "---",
        "",
        "## 3. 运行命令",
        "",
        "### 标准运行（完整流程）",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python app.py demo",
        "```",
        "",
        "### 分步运行",
        "",
        "```bash",
        "python app.py ingest     # Step 1: 构建文献库",
        "python app.py discover   # Step 2: 发现研究空白",
        "python app.py validate   # Step 3: 验证假设",
        "python app.py demo       # Step 4: 导出比赛包",
        "```",
        "",
        "### 可选命令",
        "",
        "```bash",
        "python app.py check           # 环境检查",
        "python app.py collect-papers  # 自动下载 OA 文献",
        "python app.py auto            # 自动采集 + 完整 Pipeline",
        "```",
        "",
        "> **注意**: `python app.py auto` 是可选功能，不是 demo 必需步骤。",
        "> demo 命令使用本地已有 PDF（papers/, early_papers/, followup_papers/）。",
        "",
        "---",
        "",
        "## 4. 输出文件列表",
        "",
        f"共 {len(existing_outputs)} / {len(output_files)} 个输出文件：",
        "",
        "| File | Description | Size |",
        "|------|-------------|------|",
    ])
    for f in output_files:
        status = "✅" if f["exists"] else "❌"
        size_str = f"{f['size_bytes']:,} B" if f["exists"] else "-"
        lines.append(f"| {status} {f['name']} | {f['description']} | {size_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. 数据文件列表",
        "",
        f"共 {len(existing_data)} / {len(data_files)} 个数据文件：",
        "",
        "| File | Description | Size |",
        "|------|-------------|------|",
    ])
    for f in data_files:
        status = "✅" if f["exists"] else "❌"
        size_str = f"{f['size_bytes']:,} B" if f["exists"] else "-"
        lines.append(f"| {status} {f['name']} | {f['description']} | {size_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. 随机种子",
        "",
        "本系统主要运行确定性流程：",
        "- 文献卡片提取（数据结构化）：使用 Qwen，temperature=0.2（低随机性）",
        "- 覆盖矩阵和缺失证据检测：规则驱动，无需种子",
        "- 消融实验评分：使用 Qwen，temperature=0.2",
        "- 其他分析（证据追溯、历史回溯等）：规则驱动，确定性的",
        "",
        "> 由于 Qwen 调用存在低随机性，多次运行结果可能有微小差异。",
        "> 核心输出（evidence_map, gap_diagnosis 等）在相同输入下应保持稳定。",
        "",
        "---",
        "",
        "## 7. 如何复现各模块",
        "",
        "### 复现 Baseline 对比",
        "",
        "Baseline 对比（04_baseline_comparison.md）由 validate 阶段自动生成。",
        "比较的是本系统 vs 直接 Qwen vs 摘要 Qwen 三种方式在同一任务上的输出。",
        "",
        "### 复现 Ablation 实验",
        "",
        "消融实验（08_ablation_study.md）由 ablation_study.py 生成。",
        "使用 Qwen 一次性对 6 个版本 × 12 维度评分。",
        "回退策略使用模板评分。",
        "",
        "### 复现 Retrospective Validation",
        "",
        "历史回溯验证（07_retrospective_validation.md）由 retrospective_validation.py 生成。",
        "将文献按年份分为 early 和 followup 两组，从 early 生成空白，用 followup 验证。",
        "",
        "### 复现 Evidence Trace",
        "",
        "证据追溯（09_evidence_trace_report.md + evidence_snippets.csv）由 evidence_trace.py 生成。",
        "使用关键词匹配从文献数据库中提取证据片段，是纯规则驱动。",
        "",
        "---",
        "",
        "## 8. 从零运行完整流程",
        "",
        "```bash",
        "# 1. 克隆/解压项目",
        "# 2. 安装依赖",
        "pip install -r requirements.txt",
        "",
        "# 3. 设置 API Key",
        "# 将 Qwen API Key 写入 qwen_key.txt",
        "echo 'sk-your-key' > qwen_key.txt",
        "",
        "# 4. 放入 PDF",
        "# 将钛合金疲劳 PDF 放入 papers/, early_papers/, followup_papers/",
        "",
        "# 5. 运行完整流程",
        "python app.py demo",
        "",
        "# 6. （可选）环境检查",
        "python app.py check",
        "```",
        "",
        "---",
        "",
        "## 9. 比赛包结构",
        "",
        "```",
        "competition_package/",
        "├─ 01_evidence_map.md          # Literature evidence map",
        "├─ 02_gap_diagnosis.md         # Research gap diagnosis",
        "├─ 03_hypothesis_summary.md    # Scientific hypothesis",
        "├─ 04_baseline_comparison.md   # Baseline vs direct Qwen",
        "├─ 05_scientific_hypothesis_plan.md  # Full research plan",
        "├─ 06_competition_readiness.md # Task completion self-check",
        "├─ 07_retrospective_validation.md   # Historical validation",
        "├─ 08_ablation_study.md        # Ablation study",
        "├─ 09_evidence_trace_report.md # Evidence trace",
        "├─ 10_award_readiness_scorecard.md  # Award scorecard",
        "├─ 11_qwen_usage_report.md     # Qwen usage report",
        "├─ 12_evidence_quality_gate.md # Evidence quality gate",
        "├─ 13_reproducibility_manifest.md  # Reproducibility manifest",
        "├─ README_FOR_REVIEWERS.md     # Reviewer guide",
        "├─ run_command.txt             # Run instructions",
        "├─ final_demo_report.md        # Final demo report",
        "└─ data_preview/              # Data preview CSVs",
        "```",
        "",
        "---",
        "",
        "> This manifest is auto-generated by `python app.py demo`.",
        "",
    ])

    out_path = OUTPUTS_DIR / "13_reproducibility_manifest.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_run_command() -> Path:
    """Write updated run_command.txt to competition_package."""
    content = f"""# TitaniumFatigueChat — 运行命令

## 标准运行（完整流程）

pip install -r requirements.txt
python app.py demo

## 分步运行

python app.py ingest       # Step 1: 构建文献库
python app.py discover     # Step 2: 发现研究空白
python app.py validate     # Step 3: 验证假设
python app.py demo         # Step 4: 导出比赛包

## 辅助命令

python app.py check        # 环境检查
python app.py --help       # 查看全部命令

## 可选命令（非 demo 必需）

python app.py collect-papers  # 自动搜索并下载开放获取（Open Access）PDF
python app.py auto            # 自动采集 + 完整 Pipeline

## 前置条件

1. Python 3.9+
2. pip install -r requirements.txt
3. 在 qwen_key.txt 中写入阿里云 Qwen API Key（以 sk- 开头）
4. 将钛合金疲劳 PDF 放入 papers/、early_papers/、followup_papers/

## 输入文件夹

- papers/         — 核心文献（纳入全部分析）
- early_papers/   — 早期文献（用于历史回溯验证）
- followup_papers/ — 后续文献（用于历史回溯验证）

## 输出文件

所有核心报告生成在 outputs/ 目录：
- 01_evidence_map.md             文献证据地图
- 02_gap_diagnosis.md            研究空白诊断
- 03_hypothesis_summary.md       科学假设摘要
- 04_baseline_comparison.md      基线对比（直接 Qwen）
- 05_scientific_hypothesis_plan.md 研究计划
- 06_competition_readiness.md    完成度自检
- 07_retrospective_validation.md 历史回溯验证
- 08_ablation_study.md           消融实验
- 09_evidence_trace_report.md    证据片段追溯
- 10_award_readiness_scorecard.md 奖项完成度评分卡
- 11_qwen_usage_report.md        Qwen 使用报告
- 12_evidence_quality_gate.md    证据质量门禁
- 13_reproducibility_manifest.md 可复现性声明

数据文件生成在 data/ 目录：
- literature_database.csv        文献数据库
- evidence_snippets.csv          证据片段
- ablation_results.csv           消融实验结果
- retrospective_validation_pairs.csv 历史回溯验证对
- minimum_validation_dataset_schema.csv 最低验证数据集
- qwen_call_log.csv              Qwen 调用日志

## 比赛包

competition_package/ 包含全部核心报告 + 数据预览 + README_FOR_REVIEWERS.md
运行 python app.py demo 后自动生成。

## 注意事项

1. python app.py auto 是可选功能（联网采集 OA 文献），不是 demo 必需步骤
2. demo 命令仅使用本地已有 PDF
3. 所有输出在相同输入和 API key 下可复现
4. 如无 API Key，部分 Qwen 依赖的输出为空

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    run_path = COMP_DIR / "run_command.txt"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(content, encoding="utf-8")
    return run_path
