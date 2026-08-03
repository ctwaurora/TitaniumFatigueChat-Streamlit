"""Independent evidence-bound scientific analysis skill."""

from __future__ import annotations

from src.research_skills.common import (
    base_quality_gate,
    entity_labels,
    evidence_counts,
    evidence_prompt_block,
    formula_lines,
    missing_evidence,
    traceable_cards,
)
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "scientific_analysis_skill"
TASK_DEFINITION = "解释具体变量关系、机制、冲突、条件边界和证据不足。"
QUALITY_METRICS = (
    "citation_truthfulness",
    "page_accuracy",
    "condition_completeness",
    "counter_evidence_coverage",
    "formula_applicability",
    "unsupported_claim_rate",
)


def build_prompt(value: SkillInput) -> str:
    iv, dv = entity_labels(value)
    return (
        "你正在执行 scientific_analysis_skill。只做科学关系分析，不生成研究空白、候选假设或实验方案。"
        "必须逐项核对支持、反向和条件依赖证据，说明机制、实验条件、冲突与适用边界。"
        "没有原文公式时不得补造公式或参数。回答不得出现泛化占位词。\n"
        f"问题：{value.user_query}\n自变量：{iv}\n因变量：{dv}\n证据：\n{evidence_prompt_block(value)}"
    )


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    iv, dv = entity_labels(value)
    support, counter, conditional = evidence_counts(value)
    missing = missing_evidence(value)
    specific = bool(value.parsed_entities.get("specific"))
    evidence_sufficient = bool(value.retrieved_evidence and value.support_evidence)
    if not specific:
        direct = "本地系统无法从问题中可靠识别完整变量关系，因此拒绝生成泛化科研结论。"
        mechanism = "请明确材料、具体自变量和可测因变量后再分析。"
    elif not evidence_sufficient:
        direct = "本地正式可信 RAG 缺少直接支持证据，因此拒绝把机制推断写成科研结论。"
        mechanism = "只能确认当前证据不足，不能根据领域常识补写无来源机制。"
    else:
        direct = synthesis.strip() or (
            f"关于 {iv} 对 {dv} 的影响，本地正式库提供 {support} 条支持证据、"
            f"{counter} 条反向证据和 {conditional} 条条件依赖证据。"
        )
        mechanism = (
            f"{iv} 可能通过局部应力、组织演化或损伤起始路径改变 {dv}；"
            "具体方向必须按证据卡片中的材料、表面状态、热处理和载荷条件分别判断。"
        )
    formulas = formula_lines(value)
    reasoning = f"""**结论**

{direct}

**具体机制**

{mechanism}

**实验条件与适用边界**

只把条件记录中明确报告的材料、制造工艺、建造方向、热处理、表面状态、应力比和疲劳区间作为结论边界。

**文献冲突与反向观点**

本次独立核对 {counter} 条反向证据和 {conditional} 条条件依赖证据。跨材料、跨应力比或跨表面状态的差异不直接视为同条件矛盾。

**必要公式**

{chr(10).join(formulas) if formulas else '本次检索未召回可追溯原文公式，不输出无来源公式或参数。'}

**证据不足部分**

{('；'.join(missing)) if missing else '当前检索范围内未发现结构性缺项，但仍需遵守文献条件边界。'}"""
    combined = direct + "\n" + reasoning
    gate = base_quality_gate(
        value,
        combined,
        skill_name=SKILL_NAME,
        required_terms=("具体机制", "实验条件与适用边界", "文献冲突与反向观点"),
    )
    gate["evidence_sufficient"] = evidence_sufficient
    gate["passed"] = gate["passed"] and evidence_sufficient
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=direct,
        structured_reasoning=reasoning,
        uncertainty="证据充分度取决于直接证据、反证和条件记录是否覆盖同一实验空间。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing,
        specific_fields={
            "mechanism": mechanism,
            "boundary_policy": "MATCHED_REPORTED_CONDITIONS_ONLY",
            "formula_assessment": formulas,
            "quality_metrics": QUALITY_METRICS,
        },
        trace={
            "dataset_version": value.dataset_version,
            "previous_skill": getattr(value.previous_output, "skill_name", None),
            "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)],
        },
    )


def render_output(output: SkillOutput) -> str:
    return f"""## 第一部分：直接回答

{output.direct_answer}

## 第二部分：科研分析

{output.structured_reasoning}

## 第三部分：结论边界

- 可直接支持：原文、页码和实验条件一致的结论。
- 系统推断：机制串联与跨文献归纳。
- 本地证据不足：{('；'.join(output.missing_evidence)) if output.missing_evidence else '未发现结构性缺项。'}
"""
