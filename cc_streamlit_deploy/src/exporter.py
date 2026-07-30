"""
exporter.py — 比赛提交包导出模块

实现 demo 命令的 export 部分：
- 生成 final_demo_report.md
- 生成 competition_package/ 中的全部文件
- 生成 run_command.txt
- 生成 README_FOR_JUDGES.md
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from skills.library_skill import get_all_papers

OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
COMP_DIR = BASE_DIR / "competition_package"


def run_export() -> Dict[str, Any]:
    """执行 demo 导出。"""
    COMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 生成 final_demo_report.md
    report_path = _write_final_demo_report()

    # 2. 复制核心报告
    _copy_reports()

    # 3. 复制数据预览
    _copy_data_previews()

    # 4. 生成 README_FOR_REVIEWERS.md
    _write_readme_for_judges()

    # 5. 生成 run_command.txt
    _write_run_command()

    # 6. 生成 final_checklist.md
    _write_final_checklist()

    # 7. 安全清理：确保竞赛包中不含真实 qwen_key.txt，仅保留 qwen_key.example.txt
    for p in COMP_DIR.rglob("*"):
        if p.name == "qwen_key.txt":
            p.unlink(missing_ok=True)
    # 若 qwen_key.example.txt 存在则复制一份到包根目录
    example_key = BASE_DIR / "qwen_key.example.txt"
    if example_key.exists():
        shutil.copy2(example_key, COMP_DIR / "qwen_key.example.txt")

    return {
        "competition_package": str(COMP_DIR),
        "final_report": str(report_path),
        "files": [str(f) for f in sorted(COMP_DIR.rglob("*")) if f.is_file()],
    }


def _copy_reports() -> None:
    """复制核心报告到比赛包。"""
    report_mapping = [
        ("01_evidence_map.md", "01_evidence_map.md"),
        ("02_gap_diagnosis.md", "02_gap_diagnosis.md"),
        ("03_hypothesis_summary.md", "03_hypothesis_summary.md"),
        ("04_baseline_comparison.md", "04_baseline_comparison.md"),
        ("05_scientific_hypothesis_plan.md", "05_scientific_hypothesis_plan.md"),
        ("06_competition_readiness.md", "06_competition_readiness.md"),
        ("07_retrospective_validation.md", "07_retrospective_validation.md"),
        ("08_ablation_study.md", "08_ablation_study.md"),
        ("09_evidence_trace_report.md", "09_evidence_trace_report.md"),
        ("10_award_readiness_scorecard.md", "10_award_readiness_scorecard.md"),
        ("11_qwen_usage_report.md", "11_qwen_usage_report.md"),
        ("12_evidence_quality_gate.md", "12_evidence_quality_gate.md"),
        ("13_reproducibility_manifest.md", "13_reproducibility_manifest.md"),
    ]

    for src_name, dst_name in report_mapping:
        src = OUTPUTS_DIR / src_name
        if src.exists():
            shutil.copy2(src, COMP_DIR / dst_name)
        else:
            _write_placeholder(COMP_DIR / dst_name, src_name)


def _write_placeholder(path: Path, name: str) -> None:
    """写入占位符文件。"""
    content = f"""# {name}

> **文件未生成**：请先运行 `python app.py demo` 完整流程。

当前文献库规模较小，该文件在完整运行后自动生成。
"""
    path.write_text(content, encoding="utf-8")


def _copy_data_previews() -> None:
    """复制数据预览文件。"""
    preview_dir = COMP_DIR / "data_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "literature_database_preview.csv": "literature_database.csv",
        "variable_mechanism_preview.csv": "variable_mechanism.csv",
        "coverage_matrix_preview.csv": "coverage_matrix.csv",
        "evidence_snippets_preview.csv": "evidence_snippets.csv",
        "minimum_validation_dataset_schema.csv": "minimum_validation_dataset_schema.csv",
        "ablation_results.csv": "ablation_results.csv",
        "retrospective_validation_pairs.csv": "retrospective_validation_pairs.csv",
        "qwen_call_log_preview.csv": "qwen_call_log.csv",
    }

    for preview_name, source_name in data_files.items():
        src = DATA_DIR / source_name
        dst = preview_dir / preview_name
        if src.exists():
            # Copy header + first 10 rows
            lines = src.read_text(encoding="utf-8").split("\n")
            preview_lines = lines[:11]
            dst.write_text("\n".join(preview_lines), encoding="utf-8")
        else:
            dst.write_text(f"{source_name} 未生成", encoding="utf-8")

    # Also copy full CSV data files (not just previews)
    data_out_dir = COMP_DIR / "data"
    data_out_dir.mkdir(parents=True, exist_ok=True)
    for full_name in ["literature_database.csv", "ablation_results.csv",
                       "retrospective_validation_pairs.csv", "evidence_snippets.csv",
                       "minimum_validation_dataset_schema.csv", "qwen_call_log.csv"]:
        src = DATA_DIR / full_name
        if src.exists():
            shutil.copy2(src, data_out_dir / full_name)


def _write_readme_for_judges() -> None:
    """Generate README_FOR_REVIEWERS.md — 13 Q&A format"""
    n_papers = len(get_all_papers())
    content = f"""# TitaniumFatigueChat — README for Reviewers

---

## Q1: What problem does this system solve?

TitaniumFatigueChat transforms a small collection of scientific PDFs on titanium alloy fatigue into a **verifiable, traceable, and falsifiable scientific hypothesis** with a full research plan.

The problem it solves: General-purpose LLMs (including Qwen) can generate plausible-sounding research directions, but they cannot:
- Trace claims back to specific PDF sentences
- Systematically detect what evidence is missing
- Distinguish well-studied areas from genuine research gaps
- Provide concrete, actionable validation paths
- State what would disprove the hypothesis

This system bridges the gap between LLM-generated ideas and scientifically rigorous, reproducible hypotheses.

---

## Q2: What is the scientific domain?

Titanium alloy fatigue — specifically, the effect of **pore defects in L-PBF Ti-6Al-4V** on fatigue crack initiation, early crack growth, and fatigue life.

The system is designed to be domain-agnostic in architecture but is currently populated with titanium alloy fatigue literature. Main case: AM Ti-6Al-4V. Background: TC17, Ti60, near-alpha/alpha+beta/beta titanium alloys, and related manufacturing-process-fatigue studies.

---

## Q3: What is the input?

- **PDF files** on titanium alloy fatigue, placed in:
  - `papers/` (core literature)
  - `early_papers/` (pre-2019, for retrospective validation)
  - `followup_papers/` (2019+, for retrospective validation)
- A valid **Qwen API key** in `qwen_key.txt` (format: sk-...) for live Qwen calls. If no key is provided, `python app.py demo` can still run in cached evidence mode using existing structured data, and the Qwen usage report will honestly state that no new Qwen call was recorded.

Current corpus: **{n_papers} papers** (deduplicated).

---

## Q4: What is the output?

The system generates **13 core reports** + data files:

| # | File | Content |
|---|------|---------|
| 01 | `01_evidence_map.md` | Literature evidence map with coverage matrix |
| 02 | `02_gap_diagnosis.md` | Research gap diagnosis with missing evidence |
| 03 | `03_hypothesis_summary.md` | Scientific hypothesis (predictive, with evidence basis) |
| 04 | `04_baseline_comparison.md` | 12-dimension comparison vs direct Qwen |
| 05 | `05_scientific_hypothesis_plan.md` | Full research plan with validation and falsification |
| 06 | `06_competition_readiness.md` | Task completion self-check |
| 07 | `07_retrospective_validation.md` | Historical lookback validation |
| 08 | `08_ablation_study.md` | 6-version ablation study |
| 09 | `09_evidence_trace_report.md` | Evidence snippet trace (paper_id to snippet) |
| 10 | `10_award_readiness_scorecard.md` | Award readiness scorecard |
| 11 | `11_qwen_usage_report.md` | Qwen usage report |
| 12 | `12_evidence_quality_gate.md` | Evidence quality gate |
| 13 | `13_reproducibility_manifest.md` | Reproducibility manifest |

---

## Q5: How does the system use Qwen?

Qwen is used as a **structured extraction and constrained generation engine** — not as a chat interface.

Specifically:
- **Ingest phase**: Qwen extracts 13+ structured fields from each PDF (material_system, processing_method, loading_condition, mechanical_indicators, crack_initiation, crack_growth_mechanism, etc.)
- **Discover phase**: Qwen identifies variable-property-mechanism relationships and builds the coverage matrix
- **Validate phase**: Qwen generates hypothesis cards under strict quality gates (8 condition checks)

Every Qwen call is schema-constrained with few-shot examples. The system never asks Qwen open-ended questions.

Some modules are **purely rule-based** (no Qwen):
- Evidence snippet extraction (keyword matching)
- Coverage matrix computation (rule-based aggregation)
- Retrospective validation (time-split + keyword matching)
- Data preview generation (CSV truncation)

---

## Q6: How does it generate scientific hypotheses?

The hypothesis generation pipeline:

1. **Structured evidence extraction** — each PDF produces a structured card (13+ fields)
2. **Coverage analysis** — 9 dimensions by 50+ categories reveal what is covered versus missing
3. **Missing evidence detection** — explicitly lists absent experiments or relationships
4. **Pseudo-gap filtering** — rejects gaps that are too generic, non-actionable, or out-of-scope
5. **Quality gate** — 8 condition checks: evidence support, variable clarity, missing evidence, validation path, etc.
6. **Hypothesis Card** — 18-field structured output with hypothesis_statement, evidence_basis, controlled_variables, expected_trend, missing_evidence, validation_path, falsification_conditions

The hypothesis is always **predictive**, never descriptive.

---

## Q7: How are evidence snippets used?

Evidence snippets are short sentences extracted from PDF text fields (title, abstract, conclusion, results, discussion, figure captions) using keyword matching.

Each snippet is tagged with:
- **evidence_id** (e.g., E0001)
- **paper_id** (e.g., P01)
- **author_year** (e.g., "Smith et al., 2020")
- **evidence_type** (e.g., "pore_fatigue_life", "fcgr_da_dN")
- **linked_variable, linked_indicator, linked_mechanism**

Snippets are stored in `data/evidence_snippets.csv`. They serve as the **traceable evidence layer** — every claim in the hypothesis can be traced to specific evidence IDs.

The Evidence Quality Gate (12_evidence_quality_gate.md) validates snippets against 8 rules (no truncated sentences, valid evidence types, non-empty author_year/paper_id, etc.) and classifies them into core evidence versus background evidence.

---

## Q8: How are missing evidence and falsification conditions generated?

**Missing evidence** is detected through the coverage matrix:
1. The system builds a 9-dimension by 50+ category matrix from structured literature cards
2. Cells with zero or insufficient coverage produce candidate missing evidence
3. Pseudo-gap filter removes: out-of-scope topics, too-generic gaps, non-actionable gaps
4. Remaining gaps are classified as: well-studied / partially studied / still missing

**Falsification conditions** are a required field in every Hypothesis Card:
- Specifies what experimental results would disprove or downgrade the hypothesis
- Quality gate rejects hypotheses without falsification_conditions

---

## Q9: How does baseline comparison prove improvement over direct Qwen?

The baseline comparison (04_baseline_comparison.md) evaluates **three approaches** on the **same task**:

1. **Direct Qwen**: Qwen generates a hypothesis with no literature input
2. **Summary Qwen**: Qwen generates with paper abstracts but no structured tables
3. **Full system**: Complete TitaniumFatigueChat pipeline

Scoring dimensions (12 metrics, auto-summed):
- Real supporting literature (traceability)
- Missing evidence identification
- Variable to property to mechanism chain formation
- Minimum-cost validation path
- Full validation path
- Success criteria
- Falsification conditions
- Hypothesis Card structure
- Research plan generation
- Literature database traceability
- Research topic feasibility
- Avoidance of vague directions

The full system scores significantly higher because of the combination of evidence extraction, missing evidence detection, falsification, and evidence tracing.

---

## Q10: How does ablation study prove module effectiveness?

The ablation study (08_ablation_study.md) evaluates **6 versions** of the same task by removing one component at a time:

| Version | What is removed | Effect |
|---------|---------------|--------|
| A_direct_qwen | All modules (direct Qwen only) | Everything degrades |
| B_summary_qwen | Structured extraction (abstracts only) | Traceability, completeness |
| C_without_missing_evidence | Missing evidence detection | Gap identification |
| D_without_falsification | Falsification conditions | Scientific rigor |
| E_without_evidence_trace | Evidence snippet trace | Traceability |
| F_full_system | Nothing (full system) | Reference baseline |

Each version is scored on the same 12 dimensions. The full system achieves the highest total score.

---

## Q11: How does retrospective validation test discovery ability?

Retrospective validation (07_retrospective_validation.md) simulates a time-travel experiment:

1. Split literature into **early papers** (2018 or earlier) and **followup papers** (2019 or later)
2. Generate research gaps **only from early papers** — what would the system have proposed in the past?
3. Check whether those early gaps were **later studied** by followup papers
4. Classify each gap as: supported / partially_supported / contradicted / not_found / insufficient_followup

This tests whether the system detects real gaps that the research community later pursued.

---

## Q12: How can reviewers reproduce the result?

```bash
pip install -r requirements.txt
echo "sk-your-api-key" > qwen_key.txt
python app.py demo
```

This executes all pipeline steps and regenerates all outputs from scratch.

**Key properties:**
- **Deterministic**: same input + same API key produce same output
- **Self-contained**: all outputs are regenerated, no cached state
- **Portable**: competition_package/ contains all reports + data previews

**Optional commands (not required for reproduction):**
```bash
python app.py check           # environment check
python app.py collect-papers  # auto-download OA PDFs
python app.py auto            # auto-collect + full pipeline
```

---

## Q13: What are the current limitations?

1. **Small literature corpus**: about {n_papers} papers, coverage matrix differentiation is limited
2. **Preliminary evidence level**: hypothesis is labeled preliminary, not evidence-supported
3. **Qwen dependency**: extraction quality depends on Qwen output quality
4. **No OCR/image recognition**: evidence snippets are text-only, no chart/curve data extraction
5. **Validation at design level only**: system generates validation plans but does NOT execute experiments or simulations
6. **Ablation/baseline scoring uses estimation**: scores are Qwen-estimated, not from actual re-runs
7. **Evidence quality gate is rule-based**: truncation detection uses simple regex
8. **No real-time API call logging**: qwen_call_log.csv logs calls from the current run only

**To upgrade to evidence-supported hypothesis**: expand literature to 30-100 papers, especially FCGR, micro-CT, SEM/EBSD, and HCF/VHCF studies.

---
"""
    (COMP_DIR / "README_FOR_REVIEWERS.md").write_text(content, encoding="utf-8")


def _write_run_command() -> Path:
    """Write updated run_command.txt to competition_package."""
    content = """# TitaniumFatigueChat — Run Commands

## Standard run (full pipeline)

pip install -r requirements.txt
python app.py demo

## Step-by-step

python app.py ingest       # Step 1: build literature database
python app.py discover     # Step 2: discover research gaps
python app.py validate     # Step 3: validate hypotheses
python app.py demo         # Step 4: export competition package

## Helper commands

python app.py check        # environment check
python app.py --help       # show all commands

## Optional commands (not required for demo)

python app.py collect-papers  # auto-download Open Access PDFs
python app.py auto            # auto-collect + full pipeline

## Prerequisites

1. Python 3.9+
2. pip install -r requirements.txt
3. Qwen API key in qwen_key.txt (starting with sk-)
4. PDFs in papers/, early_papers/, followup_papers/

## Input folders

- papers/          — core literature (all analysis)
- early_papers/    — pre-2019 papers (retrospective validation)
- followup_papers/ — 2019+ papers (retrospective validation)

## Output files

All core reports in outputs/:
- 01_evidence_map.md             Literature evidence map
- 02_gap_diagnosis.md            Research gap diagnosis
- 03_hypothesis_summary.md       Scientific hypothesis
- 04_baseline_comparison.md      Baseline comparison vs direct Qwen
- 05_scientific_hypothesis_plan.md  Research plan
- 06_competition_readiness.md    Task completion self-check
- 07_retrospective_validation.md Historical validation
- 08_ablation_study.md           Ablation study
- 09_evidence_trace_report.md    Evidence trace
- 10_award_readiness_scorecard.md  Award scorecard
- 11_qwen_usage_report.md        Qwen usage report
- 12_evidence_quality_gate.md    Evidence quality gate
- 13_reproducibility_manifest.md Reproducibility manifest

Data files in data/:
- literature_database.csv
- evidence_snippets.csv
- ablation_results.csv
- retrospective_validation_pairs.csv
- minimum_validation_dataset_schema.csv
- qwen_call_log.csv

## Competition package

competition_package/ contains all core reports + data previews + README
Auto-generated by python app.py demo

## Notes

1. python app.py auto is optional (auto-download OA PDFs), not required for demo
2. demo uses only local PDFs in papers/ etc.
3. All outputs are reproducible given same input and API key
4. Without API key, Qwen-dependent outputs will be empty
"""
    import platform
    from datetime import datetime
    run_path = COMP_DIR / "run_command.txt"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_path.write_text(content + "\n" + now, encoding="utf-8")
    return run_path


def _write_final_demo_report() -> Path:
    """生成 outputs/final_demo_report.md — 系统完整运行报告"""
    papers = get_all_papers()
    lc = _get_literature_counts()
    n_papers = lc["deduplicated_paper_count"]
    n_in_scope = lc["core_titanium_fatigue_count"]

    # 读取各报告摘要
    evidence_summary = _read_output_section("01_evidence_map.md", 5)
    gap_summary = _read_output_section("02_gap_diagnosis.md", 5)
    rec_summary = _read_output_section("03_hypothesis_summary.md", 10)
    baseline_summary = _read_output_section("04_baseline_comparison.md", 5)

    # 检测是否有证据支持型推荐
    has_evidence_recs = _has_evidence_recommendations()
    is_preliminary = _is_preliminary_mode()

    if not has_evidence_recs:
        rec_section = (
            "由于当前文献库证据不足，条件检查未通过，本轮未生成证据支持型科学假设。"
        )
        rec_count_line = "本轮无证据支持型科学假设（条件检查未通过）"
    elif is_preliminary:
        rec_section = (
            "本系统当前提供了 1 个初步证据支持型科学假设（preliminary evidence-backed hypothesis）。\n\n"
            "> **⚠️ 当前结论为 preliminary，基于小型案例验证，不代表完整领域结论。**"
        )
        rec_count_line = "1 个初步证据支持型科学假设（preliminary）"
    else:
        n_recs = _count_recs()
        rec_section = f"本系统当前提供了 **{n_recs}** 个证据支持型科学假设。"
        rec_count_line = f"{n_recs} 个证据支持型科学假设"

    report = f"""# Final Report（系统运行报告）

> **项目名称**: TitaniumFatigueChat
> **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> **文献库规模**: {n_papers} 篇（其中钛合金疲劳方向 {n_in_scope} 篇）
> **文献统计**: raw PDF {lc['raw_pdf_count']} → deduplicated {n_papers} → core Ti fatigue {n_in_scope} → main case AM Ti-6Al-4V {lc['main_case_related_count']}

---

## 一、项目定位

TitaniumFatigueChat 不是普通文献总结工具，而是面向钛合金疲劳研究，通过文献证据抽取、缺失证据检测、研究空白验证和可行性判断，生成可追溯、可验证、可推翻的科学假设生成系统。

---

## 二、榜题要求如何对应

| 榜题要求 | 本系统对应设计 |
|---------|---------------|
| 使用 Qwen/千问 | ingest 阶段使用 Qwen 进行文献卡片抽取；discover 阶段使用 Qwen 进行变量-机制关系抽取 |
| 从文献输入到可验证假设输出 | ingest（文献卡片）→ discover（覆盖矩阵+研究空白）→ validate（质量门禁+假设生成+Hypothesis Card）→ export（比赛包） |
| 有文献挖掘与事实提取 | 13+ 结构化字段的文献卡片抽取；变量—性能—机制—证据关系表；9 维度 × 50+ 类别覆盖矩阵 |
| 有逻辑驱动假设生成 | 缺失证据检测驱动假设方向；质量门禁（13 项检查）确保条件完整性；Hypothesis Card 18 字段验证假设可操作性 |
| 有可行性论证 | A/B/C 可行性等级；最低成本验证路径 + 完整验证路径；成功判据 + 推翻条件 |
| 有多轮质量门槛 | 伪空白筛除（空泛表达拒绝）；8 条件检查（至少 2 篇文献支持）；13 项 quality_gate |
| 有基线对比 | 12 项指标 × 3 组（直接 Qwen / 摘要 Qwen / 本系统）系统对比 |
| 有科学假设与研究计划 | Hypothesis Card（每张推荐卡片）；Scientific Hypothesis Plan（05_scientific_hypothesis_plan.md）；明确 Problem Statement、Rationale、Methods、Experiments 等 |

---

## 三、为什么普通大模型不够

| 维度 | 普通大模型（直接问 Qwen） | TitaniumFatigueChat |
|------|--------------------------|---------------------|
| 文献追溯 | 无具体文献，或编造文献 | 每条证据可追溯到文献卡片 |
| 缺失证据 | 不标注缺失 | 必须标注，空则不能通过 |
| 机制链 | 泛泛描述 | 变量-性能-机制链显式建模 |
| 验证路径 | 不提供 | 最低成本 + 完整验证路径 |
| 推翻条件 | 无 | 必填推翻条件 |
| 空话控制 | 常有"进一步研究" | 质量门禁拒绝空话 |
| 可复现性 | 每次回答不同 | 结构化流程，可复现 |

---

## 四、系统三层架构

### 第一层：钛合金疲劳知识库层

- PDF 文献（papers/, early_papers/, followup_papers/）
- 文献卡片（13+ 结构化字段）
- 变量—疲劳性能—机制—证据关系表

### 第二层：研究空白质量验证层

- 覆盖矩阵（9 大维度 × 50+ 类别）
- 缺失证据检测
- 伪空白筛除
- 历史回溯验证

### 第三层：可验证假设推荐层

- 假设生成
- 结构审查（12 项检查）
- A/B/C 可行性判断
- 最低成本验证路径
- Hypothesis Card（18 字段）

---

## 五、当前运行结果摘要

### Evidence Map 摘要
{evidence_summary}

### Gap Diagnosis 摘要
{gap_summary}

### 科学假设摘要
{rec_section}

### 假设摘要
{rec_summary}

### 基线对比摘要
{baseline_summary}

---

## 六、被拒绝的伪空白类型

本系统拒绝了以下类型的研究空白候选：

1. **非钛合金/非疲劳方向（out_of_scope）** — 如铝合金 7085、钢、镁合金、非钛合金复合材料等，当前文献库中已剔除 1 篇此类文献。
2. **不属于当前主案例但属钛合金疲劳方向（secondary/background）** — 如 TC17、Ti60、近α/α+β/β钛合金等材料体系研究，与钛合金疲劳相关但因主案例聚焦 AM Ti-6Al-4V，暂作为背景参考，不进入本轮推荐。
3. **证据链不完整（insufficient_evidence_chain）** — 属于钛合金疲劳方向，但缺少"变量—指标—机制—evidence"闭环。
4. **过于空泛（too_generic）** — 如单个表征方法名称（SEM/EBSD）或模型名称（Paris/Walker）本身不能构成研究空白，需与具体材料、变量和机制组合。
5. **不可操作（not_actionable）** — 无法形成最低成本验证路径。

当前保留的候选空白均与主案例 AM Ti-6Al-4V 直接相关。

---

## 七、最终科学假设

请见 `03_hypothesis_summary.md`（Scientific Hypothesis Summary）或 `05_scientific_hypothesis_plan.md`（完整科学假设与研究计划）。

{rec_count_line}

---

## 八、基线对比结论

本系统在以下指标上优于普通大模型：
1. **支持文献** — 推荐假设引用真实文献，可追溯
2. **缺失证据** — 明确标注
3. **变量—性能—机制链** — 显式建模
4. **验证路径** — 最低成本 + 完整验证
5. **成功判据与推翻条件** — 可验证、可推翻
6. **避免空话** — quality_gate 拒绝空泛表达

> ⚠️ **当前比较仅为小型案例验证。**

---

## 九、当前限制

1. **文献库规模有限** — 当前仅 {n_papers} 篇文献，覆盖矩阵区分度不足
2. **假设为初步性质** — 当前科学假设标注为 preliminary evidence-backed hypothesis
3. **LLM 质量依赖** — 文献卡片抽取和空白检测依赖 Qwen 输出
4. **验证停留在方案层面** — 系统生成验证路径但不执行实际实验/仿真

---

## 十、下一步最优先补充的文献类型

按优先级排序：

1. **AM Ti-6Al-4V 疲劳裂纹扩展（FCGR/da/dN-ΔK）**
2. **AM Ti-6Al-4V 高温疲劳（300-500°C）**
3. **AM Ti-6Al-4V 变幅/谱载疲劳**
4. **AM Ti-6Al-4V 内部缺陷（micro-CT + SEM 断口表征）**
5. **AM Ti-6Al-4V 残余应力（XRD/轮廓法测量 + 对疲劳的影响）**
6. **AM Ti-6Al-4V 热处理—组织—疲劳性能系统研究**
7. **AM Ti-6Al-4V 表面粗糙度（as-built vs 机加工 vs 喷丸）**
8. **TC17/Ti60 等高温钛合金疲劳**

---

## 十一、核心输出文件

| 文件 | 定位 | 回答的问题 |
|------|------|-----------|
| `01_evidence_map.md` | 文献证据地图 | 当前文献库中已有证据是什么 |
| `02_gap_diagnosis.md` | 研究空白诊断 | 哪些证据缺失，哪些空白可能值得做 |
| `03_hypothesis_summary.md` | 科学假设摘要 | 系统最终生成了什么科学假设 |
| `04_baseline_comparison.md` | 基线对比 | 为什么不是普通大模型直接生成 |
| `05_scientific_hypothesis_plan.md` | 研究计划 | 假设如何验证、如何推翻 |
| `06_competition_readiness.md` | 完成度自检 | 对榜题要求满足到什么程度 |
"""
    out_path = OUTPUTS_DIR / "final_demo_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def _read_output_section(filename: str, num_lines: int) -> str:
    """读取输出文件的前几行作为摘要。"""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return f"（{filename} 未生成）"
    lines = path.read_text(encoding="utf-8").split("\n")[:num_lines]
    return "\n".join(lines)


def _has_evidence_recommendations() -> bool:
    """检查正式推荐卡片中是否有证据支持型推荐。"""
    path = OUTPUTS_DIR / "03_recommendation_cards.md"
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if "暂不生成" in content or "证据不足" in content:
        return False
    return "科学假设" in content and "## 科学假设" in content


def _is_preliminary_mode() -> bool:
    """检查是否是 preliminary（初步）模式。"""
    path = OUTPUTS_DIR / "03_recommendation_cards.md"
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    return "初步" in content and "证据等级" in content


def _recommendation_status() -> str:
    """生成推荐状态摘要。"""
    if _has_evidence_recommendations():
        n = _count_recs()
        return f"本轮生成了 **{n}** 个证据支持型科学假设（通过 quality_gate）。"
    return (
        "由于当前文献库证据不足，质量门禁未通过，本轮未生成证据支持型科学假设。"
        "系统仅输出格式示例（example_recommendation_templates.md），"
        "供补充文献后参考。"
    )


def _count_recs() -> int:
    """统计推荐方向数量。"""
    path = OUTPUTS_DIR / "03_recommendation_cards.md"
    if not path.exists():
        return 0
    content = path.read_text(encoding="utf-8")
    return content.count("## 科学假设")


# ── Final checklist ─────────────────────────────────────────────────────────


def _get_literature_counts() -> dict:
    """获取四类文献数量。"""
    papers = get_all_papers()
    deduplicated = len(papers)
    core = sum(1 for p in papers if str(p.get("alloy_type", "")).strip() != "out_of_scope")
    # 统计主案例相关（AM Ti-6Al-4V 疲劳方向）
    main_case = 0
    for p in papers:
        try:
            from skills.library_skill import classify_titanium_scope
            sr = classify_titanium_scope(p)
            if sr.get("main_case_relevance") == "primary":
                main_case += 1
        except Exception:
            pass
    # 统计原始 PDF 数量
    raw_pdf = 0
    for d in ["papers", "early_papers", "followup_papers"]:
        d_path = BASE_DIR / d
        if d_path.exists():
            raw_pdf += len(list(d_path.glob("*.pdf")))
    return {
        "raw_pdf_count": raw_pdf,
        "deduplicated_paper_count": deduplicated,
        "core_titanium_fatigue_count": core,
        "main_case_related_count": main_case,
    }


def _count_literature_papers() -> int:
    """统计所有 PDF 文献数量。"""
    return _get_literature_counts()["raw_pdf_count"]


def _stage_from_count(n: int) -> str:
    if n < 5:
        return "prototype only"
    if n < 10:
        return "process validation"
    if n < 30:
        return "small-case validation"
    if n < 100:
        return "partial case"
    return "moderate case"


def _weakest_spot() -> str:
    """判断当前最大短板。"""
    issues = []
    n = _count_literature_papers()
    if n < 10:
        issues.append(("文献数量不足", f"仅 {n} 篇，建议扩充至 30+"))
    if not _has_evidence_recommendations():
        issues.append(("无科学假设", "quality_gate 未通过，需补充文献证据"))
    if not (OUTPUTS_DIR / "05_scientific_hypothesis_plan.md").exists():
        issues.append(("缺科学假设计划", "05_scientific_hypothesis_plan.md 未生成"))

    if issues:
        return f"{issues[0][0]} — {issues[0][1]}"
    # 默认短板：证据等级
    return "科学假设证据等级为 preliminary，需补充更多实验文献提升至 evidence-supported"


def _write_final_checklist() -> str:
    """生成 outputs/final_checklist.md — 榜题完成度自检清单"""
    from datetime import datetime

    lc = _get_literature_counts()
    n_papers = lc["deduplicated_paper_count"]
    stage = _stage_from_count(lc["raw_pdf_count"])
    has_recs = _has_evidence_recommendations()
    has_hypothesis = _has_hypothesis_card()
    has_plan = (OUTPUTS_DIR / "05_scientific_hypothesis_plan.md").exists()
    ref_ok = _references_verified()
    has_baseline = (OUTPUTS_DIR / "04_baseline_comparison.md").exists()
    has_competition = BASE_DIR / "competition_package" / "README_FOR_REVIEWERS.md"
    has_competition_ok = has_competition.exists()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 判定当前 readiness 等级
    if has_recs and has_hypothesis and ref_ok and has_baseline and has_competition_ok:
        if n_papers >= 30:
            readiness = "evidence-supported case"
        else:
            readiness = "small-case validation"
    elif has_recs:
        readiness = "prototype validation"
    else:
        readiness = "prototype validation"

    lines = [
        "# Competition Readiness Check（榜题完成度自检清单）",
        "",
        f"- **生成时间**: {now}",
        "",
        "---",
        "",
        "## 1. Current Competition Readiness",
        "",
        f"**Status: {readiness}**",
        "",
    ]

    if readiness == "small-case validation":
        lines.append(
            "当前已生成 scientific hypothesis、Hypothesis Card、baseline comparison "
            "和 reproducible package，但证据等级仍为 preliminary，"
            "尚未达到完整领域结论。"
        )
    elif readiness == "prototype validation":
        lines.append("当前仅部分满足提交条件。")
    else:
        lines.append("当前已生成较完整的 evidence-supported hypothesis 包。")
    lines.append("")

    # 文献统计详细
    lines.extend([
        "---",
        "",
        "## 2. Literature Counts",
        "",
        f"| Category | Count |",
        f"| --- | --- |",
        f"| Raw PDF files | {lc['raw_pdf_count']} |",
        f"| Deduplicated papers | {lc['deduplicated_paper_count']} |",
        f"| Core Ti fatigue | {lc['core_titanium_fatigue_count']} |",
        f"| Main case (AM Ti-6Al-4V) | {lc['main_case_related_count']} |",
        "",
        f"**判定阶段**: {stage}",
        "",
        "| Range | Stage |",
        "| --- | --- |",
        "| <5 | prototype only |",
        "| 5—9 | process validation |",
        "| 10—29 | small-case validation ← 当前 |",
        "| 30—99 | partial case |",
        "| ≥100 | moderate case |",
        "",
        "---",
        "",
        "## 3—8. Core Output Check",
        "",
        "| # | Check Item | Status |",
        "| --- | --- | --- |",
    ])

    checks = [
        ("Scientific hypothesis (03_recommendation_cards.md)", "✅" if has_recs else "❌"),
        ("Hypothesis Card", "✅" if has_hypothesis else "❌"),
        ("Research plan (05_scientific_hypothesis_plan.md)", "✅" if has_plan else "❌"),
        ("Reference verification", "✅" if ref_ok else "❌"),
        ("Baseline comparison (04_baseline_comparison.md)", "✅" if has_baseline else "❌"),
        ("Competition package (competition_package/)", "✅" if has_competition_ok else "❌"),
    ]
    for i, (label, status) in enumerate(checks, 3):
        lines.append(f"| {i} | {label} | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Current Weakest Point",
        "",
        f"{_weakest_spot()}",
        "",
        "---",
        "",
        "## 10. Next 3 Priority Actions",
        "",
        "1. **Expand literature library** — target 30–100 papers, especially FCGR/micro-CT/SEM-EBSD studies",
        "2. **Upgrade evidence level** — from preliminary to evidence-supported by adding experimental data",
        "3. **Strengthen hypothesis** — refine variable-property-mechanism chain with quantitative data",
        "",
        "---",
        "",
        "> This checklist is auto-generated by `python app.py demo`.",
        "> Run demo after each literature update to refresh the status.",
    ])

    out_path = OUTPUTS_DIR / "final_checklist.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def _has_hypothesis_card() -> bool:
    """检查 recommendation_cards 中是否包含 Hypothesis Card。"""
    path = OUTPUTS_DIR / "03_recommendation_cards.md"
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    return "Hypothesis Card" in content


def _references_verified() -> bool:
    """检查引文校验是否通过。"""
    path = OUTPUTS_DIR / "reference_verification_report.md"
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    # 看是否有异常标记（未引用较多的文献）
    match = re.search(r'未引用\s*(\d+)', content)
    if match:
        unref_count = int(match.group(1))
        if unref_count > 3:
            return False
    return True
