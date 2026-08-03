"""Evidence-bound scientific analysis Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, missing_evidence, primary_citation, traceable_cards
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "scientific_analysis_skill"
TASK_DEFINITION = "综合正式文献，回答具体结论、机制、条件边界、冲突、公式和未知项。"
QUALITY_METRICS = ("citation_truthfulness", "condition_completeness", "counter_evidence_coverage", "formula_applicability", "unsupported_claim_rate")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 scientific_analysis_skill，只生成这一项任务的最终中文科研回答。
问题：{value.user_query}
只能使用下方 EvidenceBundle。不得报告证据数量，不得使用模糊占位词，不得补造参数、阈值或文献。
先直接回答，再依次写：具体机制；实验条件与适用边界；文献冲突与反向观点；公式或数学关系；当前不能确定；科研意义或验证建议。
每个关键结论紧邻标注 Evidence ID 与页码。真实公式必须逐字显示 Formula ID、变量、单位和适用条件；没有可信公式时明确说明。
用户未问孔隙时，孔隙只能作为有证据的次要混杂因素。参考文献由系统在答案末尾统一附加，不要自造参考文献。
EvidenceBundle：
{bundle_prompt(value)}"""


def build_repair_prompt(value: SkillInput, *, draft: str, failures: list[str]) -> str:
    return f"""修复 scientific_analysis_skill 草稿。失败项：{'；'.join(failures)}。
保留有来源的具体结论，删除证据计数和无关孔隙主线，补齐机制、条件边界、反向观点、公式适用性和未知项。
只能引用 EvidenceBundle 中存在的 Evidence ID、页码和 Formula ID。
草稿：{draft}
EvidenceBundle：{bundle_prompt(value)}"""


def _fallback(value: SkillInput) -> tuple[str, str]:
    iv, dv = entity_labels(value)
    bundle = value.evidence_bundle or {}
    synthesis = bundle.get("synthesis") or {}
    consensus = synthesis.get("consensus") or []
    conflicts = synthesis.get("conflicts") or []
    mismatches = synthesis.get("condition_mismatches") or []
    formulas = bundle.get("formulas") or []
    if not consensus:
        return "本地正式证据不足，无法对该变量关系给出可追溯结论。", "当前缺少可核验的直接证据、条件与反向结果。"
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    direct = f"现有正式证据支持在已报告条件内分析{iv or '所问自变量'}对{dv or '所问结果'}的作用，但不能跨越未匹配条件直接外推。{consensus[0]} {support_cite}"
    formula_text = "未检索到通过可信结构检查的文献原公式；现有证据支持趋势，但不足以确定具体函数形式。"
    if formulas:
        item = formulas[0]
        formula_text = f"{item['equation']}（Formula ID：{item['formula_id']}，页码：{item['page_number']}；适用条件：{item.get('applicable_conditions') or '原文未完整报告'}）"
    reasoning = f"""### 具体机制
{consensus[1] if len(consensus) > 1 else '证据未充分分离各机制贡献，不能把相关性写成单因素因果。'}

### 实验条件和适用边界
{'；'.join(mismatches) if mismatches else '只能适用于证据卡中明确报告的材料、处理、表面和加载条件。'}

### 文献冲突及反向观点
{(conflicts[0] + ' ' + counter_cite) if conflicts else '当前检索结果未提供可明确复核的相反结论；这不等于不存在反向证据。'}

### 公式或数学关系
{formula_text}

### 当前不能确定的内容
{'；'.join(synthesis.get('unsupported_conclusions') or ['未报告条件下的效应方向和定量幅度不能确定。'])}

### 科研意义或验证建议
采用同批材料和匹配载荷条件的对照设计，分别测量所问变量、裂纹阶段指标与关键混杂因素。"""
    return direct, reasoning


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning = _fallback(value)
    complete = synthesis.strip()
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["evidence_sufficient"] = bool(value.support_evidence)
    gate["passed"] = gate["passed"] and gate["evidence_sufficient"]
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=complete or direct,
        structured_reasoning="" if complete else reasoning,
        uncertainty="结论只覆盖EvidenceBundle中已报告且可比较的条件。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing_evidence(value),
        specific_fields={"complete_answer": complete, "quality_metrics": QUALITY_METRICS},
        trace={"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]},
    )


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    if complete:
        return str(complete)
    return f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
