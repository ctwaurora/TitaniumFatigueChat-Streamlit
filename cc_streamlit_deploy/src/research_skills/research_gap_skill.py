"""Independent evidence-bound research gap skill."""

from __future__ import annotations

from src.research_skills.common import (
    base_quality_gate,
    entity_labels,
    evidence_counts,
    evidence_prompt_block,
    missing_evidence,
    traceable_cards,
)
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "research_gap_skill"
TASK_DEFINITION = "区分已解决与未解决问题，压制研究空白误报并提出可证伪问题。"
QUALITY_METRICS = (
    "research_gap_false_positive_rate",
    "counter_evidence_coverage",
    "condition_completeness",
    "citation_truthfulness",
    "testability",
)


def build_prompt(value: SkillInput) -> str:
    iv, dv = entity_labels(value)
    return (
        "你正在执行 research_gap_skill。禁止把文献数量少直接称为研究空白。"
        "先证明已有研究不能在匹配条件下回答具体变量关系，再给出证据缺口矩阵。"
        "必须检索可能反驳该空白的证据，并给出具体研究问题、可证伪假设和最低成本验证。\n"
        f"问题：{value.user_query}\n待审计关系：{iv} -> {dv}\n证据：\n{evidence_prompt_block(value)}"
    )


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    iv, dv = entity_labels(value)
    support, counter, conditional = evidence_counts(value)
    missing = missing_evidence(value)
    specific = bool(value.parsed_entities.get("specific"))
    evidence_sufficient = bool(value.retrieved_evidence and value.support_evidence)
    title = (
        f"{iv}—{dv} 在匹配组织、表面和载荷条件下的边界缺口"
        if specific and evidence_sufficient
        else "无法建立具体研究空白"
    )
    if not specific:
        direct = "变量关系不完整，research_gap_skill 拒绝输出泛化研究空白。"
        unresolved = "需要先明确可测自变量、因变量和目标材料。"
    elif not evidence_sufficient:
        direct = "本地正式可信 RAG 缺少直接支持证据，无法区分真正空白与检索缺失，因此拒绝宣称存在研究空白。"
        unresolved = "只能记录本地证据不足，不能据此生成研究空白结论。"
    else:
        direct = synthesis.strip() or (
            f"当前证据不能在完全匹配的材料批次、组织、表面状态和载荷条件下确定 {iv} 对 {dv} 的独立效应边界。"
        )
        unresolved = f"尚未解决 {iv} 与 {dv} 在关键条件交互下何时改变效应方向或主导机制。"
    matrix = (
        "| 缺口维度 | 当前状态 | 判定 |\n|---|---:|---|\n"
        f"| 支持证据 | {support} | 仅支持已报告条件 |\n"
        f"| 反向证据 | {counter} | 用于缩小或取消空白 |\n"
        f"| 条件依赖证据 | {conditional} | 需要匹配实验空间 |"
    )
    falsifiable = f"在材料批次、组织和载荷匹配后，改变 {iv} 仍会导致 {dv} 出现可重复的方向性变化。"
    reasoning = f"""### 研究空白标题

{title}

**已解决问题**

正式文献提供 {support} 条支持证据，只说明该关系在部分已报告条件中可观察。

**未解决的具体问题**

{unresolved}

**为什么已有研究不能回答**

跨论文的材料批次、组织、表面状态、应力比和疲劳区间并不完全匹配，不能把跨论文差异直接解释为单变量因果效应。

**证据缺口矩阵**

{matrix}

**反证检索**

已重新核对 {counter} 条反向证据与 {conditional} 条条件依赖证据；若这些证据覆盖目标条件，空白必须缩小或取消。

**具体研究问题**

1. 匹配组织和载荷后，{iv} 对 {dv} 的效应方向是否保持？
2. 哪一组条件组合触发主导机制转换？

**可证伪假设**

{falsifiable}

**最低成本验证**

复用同批次试样，补齐关键条件测量，完成小规模配对疲劳试验和盲法断口溯源。"""
    combined = direct + "\n" + reasoning
    gate = base_quality_gate(
        value,
        combined,
        skill_name=SKILL_NAME,
        required_terms=("证据缺口矩阵", "反证检索", "可证伪假设", "最低成本验证"),
    )
    gate["false_positive_guard"] = counter > 0 or conditional > 0
    gate["evidence_sufficient"] = evidence_sufficient
    gate["passed"] = gate["passed"] and gate["false_positive_guard"] and evidence_sufficient
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=direct,
        structured_reasoning=reasoning,
        uncertainty="若反向证据已经覆盖目标条件，则该空白应取消而非继续放大。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing,
        specific_fields={
            "gap_title": title,
            "gap_matrix": matrix,
            "candidate_questions": [f"{iv} 对 {dv} 的效应边界是什么？"],
            "falsifiable_hypothesis": falsifiable,
            "minimum_validation": "matched-pair pilot plus blinded fractography",
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

- 研究空白成立条件：已有证据不能在匹配实验空间内回答问题。
- 可能取消空白的证据：反向或条件依赖证据覆盖目标条件。
- 本地证据不足：{('；'.join(output.missing_evidence)) if output.missing_evidence else '未发现结构性缺项。'}
"""
