"""Independent falsifiable-hypothesis generation skill."""

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


SKILL_NAME = "hypothesis_generation_skill"
TASK_DEFINITION = "生成具体、公式约束、可证伪且能拟合验证的候选假设。"
QUALITY_METRICS = (
    "hypothesis_falsifiability", "formula_applicability", "condition_completeness",
    "counter_evidence_coverage", "unsupported_claim_rate", "citation_truthfulness",
)


def build_prompt(value: SkillInput) -> str:
    iv, dv = entity_labels(value)
    return (
        "你正在执行 hypothesis_generation_skill。输出必须是候选假设，不得声称已证明。"
        "明确自变量、因变量、控制变量、中介机制、预测关系、至少两条证伪判据和参数拟合计划。"
        "只能引用证据中的原文公式；自建模型必须标注候选模型且不得伪造系数。\n"
        f"问题：{value.user_query}\n自变量：{iv}\n因变量：{dv}\n证据：\n{evidence_prompt_block(value)}"
    )


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    iv, dv = entity_labels(value)
    support, counter, conditional = evidence_counts(value)
    missing = missing_evidence(value)
    specific = bool(value.parsed_entities.get("specific"))
    evidence_sufficient = bool(value.retrieved_evidence and value.support_evidence)
    if specific and evidence_sufficient:
        statement = (
            f"候选假设：在材料批次、组织、表面状态和载荷条件匹配时，{iv} 的受控变化会导致 {dv} 出现可重复的方向性变化；"
            "当反向证据对应条件出现时，该方向允许发生转换。"
        )
        direct = synthesis.strip() or statement
    elif not specific:
        statement = "变量关系不完整，无法形成可证伪候选假设。"
        direct = "hypothesis_generation_skill 拒绝根据模糊问题生成假设。"
    else:
        statement = "本地正式可信 RAG 缺少直接支持证据，无法形成有来源的候选假设。"
        direct = "hypothesis_generation_skill 拒绝把证据不足包装成可证伪假设。"
    formulas = formula_lines(value)
    formula_text = (
        "\n".join(formulas)
        if formulas
        else "未召回可追溯原文公式。候选模型只保留符号关系，不设定系数：DV = f(IV, controls)。"
    )
    falsifiers = [
        f"匹配条件后，{iv} 对 {dv} 的效应不显著且置信区间排除预注册最小效应。",
        f"独立批次中效应方向稳定反转，或 {iv} 的贡献被单一已测混杂变量完全解释。",
    ]
    reasoning = f"""**具体候选假设**

{statement}

**自变量**：{iv}。

**因变量**：{dv}。

**控制变量**：材料批次、制造窗口、试样几何、表面状态、热处理、应力比、频率、温度和环境。

**中介机制**

{iv} 改变局部应力或微观损伤演化，再通过起裂位置或裂纹扩展路径影响 {dv}。

**文献原公式或候选模型**

{formula_text}

**参数、单位和适用范围**

原文未报告的参数不补值；应力、长度、循环数和 da/dN 单位必须统一，拟合范围不得跨越不同失效机制。

**预测关系**

采用 {support} 条支持证据中的主方向作为预注册候选预测，并用 {counter} 条反向证据和 {conditional} 条条件依赖证据限定边界。

**明确证伪判据**

1. {falsifiers[0]}
2. {falsifiers[1]}

**参数拟合与实验验证方案**

先做先导试验估计方差和效应量，再使用预注册模型拟合交互项；以留出批次验证方向、参数稳定性和预测区间。"""
    combined = direct + "\n" + reasoning
    gate = base_quality_gate(
        value, combined, skill_name=SKILL_NAME,
        required_terms=("明确证伪判据", "参数拟合与实验验证方案", "中介机制"),
    )
    gate["falsification_criteria_count"] = len(falsifiers)
    gate["evidence_sufficient"] = evidence_sufficient
    gate["passed"] = gate["passed"] and len(falsifiers) >= 2 and evidence_sufficient
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=direct,
        structured_reasoning=reasoning,
        uncertainty="该结果是 evidence-bound candidate hypothesis，不是已证实结论。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing,
        specific_fields={
            "hypothesis": statement,
            "independent_variable": iv,
            "dependent_variable": dv,
            "mediator": "local stress / micro-damage / crack path",
            "formula_or_candidate_model": formula_text,
            "falsification_criteria": falsifiers,
            "fit_and_validation_plan": "preregistered interaction model plus held-out build",
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

- 假设性质：候选、可证伪，尚未被证明。
- 原文公式与候选模型已分开标注。
- 本地证据不足：{('；'.join(output.missing_evidence)) if output.missing_evidence else '未发现结构性缺项。'}
"""
