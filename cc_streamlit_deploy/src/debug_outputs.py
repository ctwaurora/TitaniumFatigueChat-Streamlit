"""
debug_outputs.py — 调试输出生成模块

生成：
- outputs/debug_scope_report.md
- outputs/debug_extraction_quality.md
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from skills.library_skill import get_all_papers, CSV_PATH
from src.validation import classify_titanium_scope

OUTPUTS_DIR = BASE_DIR / "outputs"


def write_debug_scope_report() -> str:
    """生成 outputs/debug_scope_report.md

    内容：
    - 每篇文献的 title
    - material_scope
    - fatigue_scope
    - include_in_core_analysis
    - main_case_relevance
    - exclude_reason
    """
    papers = get_all_papers()

    lines = [
        "# Debug: Scope Report（文献范围分类调试报告）",
        "",
        f"生成时间: 文献库共 {len(papers)} 篇",
        "",
        "| # | Title (前80字) | material_scope | fatigue_scope | include_in_core | main_case_relevance | exclude_reason |",
        "|---|---------------|---------------|--------------|----------------|---------------------|---------------|",
    ]

    for i, p in enumerate(papers):
        title = str(p.get("title", ""))[:80]
        alloy_type = str(p.get("alloy_type", ""))

        if alloy_type == "out_of_scope":
            # 对 out_of_scope 没有 classify 结果
            lines.append(
                f"| {i} | {title[:75]} | N/A | N/A | False | out_of_scope | "
                f"非钛合金疲劳方向（已有标记） |"
            )
        else:
            sr = classify_titanium_scope(p)
            lines.append(
                f"| {i} | {title[:75]} | "
                f"{sr.get('material_scope','?')[:30]} | "
                f"{sr.get('fatigue_scope','?')[:30]} | "
                f"{sr.get('include_in_core_analysis',False)} | "
                f"{sr.get('main_case_relevance','?')[:20]} | "
                f"{sr.get('exclude_reason','')[:30]} |"
            )

    lines.extend([
        "",
        "## 分类说明",
        "",
        "- **include_in_core_analysis**: 是否纳入核心钛合金疲劳分析",
        "- **main_case_relevance** 取值说明:",
        "  - `primary`: 主案例（AM Ti-6Al-4V FCG）直接相关",
        "  - `secondary_case`: 属钛合金疲劳方向，但不是当前主案例（如TC17、Ti60）",
        "  - `background`: 钛合金疲劳背景文献",
        "  - `out_of_scope`: 非钛合金疲劳方向",
        "- **exclude_reason**: 排除原因（空字符串表示无排除理由）",
        "",
        "## 修复说明",
        "",
        "修复后：",
        "- TC17/Ti60/近α钛合金/α+β钛合金/β钛合金 → 不再被标记为'非钛合金疲劳方向'",
        "- 真正 out_of_scope: aluminum alloy, steel, magnesium alloy 等",
        "- 如果某文献是 TC17/Ti60 但不是主案例 → main_case_relevance = secondary_case",
        "",
    ])

    out_path = OUTPUTS_DIR / "debug_scope_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def write_debug_extraction_quality() -> str:
    """生成 outputs/debug_extraction_quality.md

    内容：
    - 每篇文献抽取到的 material_system, manufacturing_process, fatigue_type,
      mechanical_indicators, mechanism, methods
    - 哪些字段缺失
    - 哪些字段影响 quality_gate
    """
    papers = get_all_papers()

    lines = [
        "# Debug: Extraction Quality（文献抽取质量调试报告）",
        "",
        f"生成时间: 文献库共 {len(papers)} 篇",
        "",
    ]

    key_fields = [
        "material_system", "processing_method", "loading_condition",
        "mechanical_indicators", "crack_initiation",
        "crack_growth_mechanism", "characterization_methods",
        "model_or_method", "temperature_environment",
        "heat_treatment", "microstructure", "stress_ratio_R",
        "limitations", "possible_innovation", "doi",
    ]

    # 总体统计
    total_cards = len(papers)
    field_completeness = {}
    for f in key_fields:
        filled = sum(1 for p in papers if p.get(f) and str(p.get(f, "")).strip())
        field_completeness[f] = (filled, total_cards)

    lines.append("## 字段填充率统计\n")
    lines.append("| 字段 | 已填充 | 总数 | 填充率 |")
    lines.append("|------|--------|------|--------|")
    for f in key_fields:
        filled, total = field_completeness[f]
        pct = f"{filled/total*100:.0f}%" if total > 0 else "N/A"
        lines.append(f"| {f} | {filled} | {total} | {pct} |")

    lines.append("")

    # 每篇文献详情
    lines.append("## 逐篇抽取详情\n")

    for i, p in enumerate(papers):
        title = str(p.get("title", ""))[:100]
        alloy_type = str(p.get("alloy_type", ""))

        lines.append(f"### {i}. {title}")
        lines.append(f"- **alloy_type**: {alloy_type}")
        lines.append("")

        # 抽取到的字段
        extracted_fields = []
        missing_fields = []
        for f in key_fields:
            val = p.get(f, "")
            if val and str(val).strip():
                val_str = str(val).strip()[:80]
                extracted_fields.append(f"  - {f}: {val_str}")
            else:
                missing_fields.append(f)

        if extracted_fields:
            lines.append("**已抽取字段:**")
            lines.extend(extracted_fields)
        else:
            lines.append("**已抽取字段:** (无)")

        lines.append("")

        if missing_fields:
            lines.append(f"**缺失字段({len(missing_fields)}):** {', '.join(missing_fields)}")
            lines.append("")
        else:
            lines.append("**缺失字段:** (无)")
            lines.append("")

        # quality_gate 影响分析
        qg_impact = []
        if not p.get("material_system") or not str(p.get("material_system", "")).strip():
            qg_impact.append("material_system → 影响 research_object 检查")
        if not p.get("loading_condition") or not str(p.get("loading_condition", "")).strip():
            qg_impact.append("loading_condition → 影响 fatigue_type 检查")
        if not p.get("mechanical_indicators") or not str(p.get("mechanical_indicators", "")).strip():
            qg_impact.append("mechanical_indicators → 影响 property_metric 检查")
        if not p.get("crack_growth_mechanism") or not str(p.get("crack_growth_mechanism", "")).strip():
            qg_impact.append("crack_growth_mechanism → 影响 mechanism_chain 检查")
        if not p.get("limitations") or not str(p.get("limitations", "")).strip():
            qg_impact.append("limitations → 影响 missing_evidence 检查")

        if qg_impact:
            lines.append("**quality_gate 影响:**")
            for impact in qg_impact:
                lines.append(f"- {impact}")
            lines.append("")

    out_path = OUTPUTS_DIR / "debug_extraction_quality.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
