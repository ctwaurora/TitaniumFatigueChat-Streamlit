"""Independent experimental-validation design skill."""

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


SKILL_NAME = "experiment_design_skill"
TASK_DEFINITION = "把证据约束的候选假设转化为分组、测量、拟合和证伪方案。"
QUALITY_METRICS = (
    "experiment_design_completeness", "hypothesis_falsifiability",
    "condition_completeness", "formula_applicability", "confounder_control",
)


def build_prompt(value: SkillInput) -> str:
    iv, dv = entity_labels(value)
    previous = getattr(value.previous_output, "specific_fields", {}) or {}
    prior_hypothesis = previous.get("hypothesis") or "未提供上一步候选假设"
    return (
        "你正在执行 experiment_design_skill。必须重新核对原始证据，不能把上一步文本当作文献事实。"
        "输出自变量水平、因变量测量、对照、样本依据、载荷、表征、拟合、支持与推翻结果、"
        "最低成本和完整方案。不得把固定样本量冒充通用文献结论。\n"
        f"问题：{value.user_query}\n自变量：{iv}\n因变量：{dv}\n上一步候选假设：{prior_hypothesis}\n"
        f"重新核对的原始证据：\n{evidence_prompt_block(value)}"
    )


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    iv, dv = entity_labels(value)
    support, counter, conditional = evidence_counts(value)
    missing = missing_evidence(value)
    specific = bool(value.parsed_entities.get("specific"))
    evidence_sufficient = bool(value.retrieved_evidence and value.support_evidence)
    if specific and evidence_sufficient:
        core_hypothesis = f"在匹配材料、组织和载荷条件后，{iv} 的受控变化仍会改变 {dv}。"
        direct = synthesis.strip() or f"建议采用分层对照试验验证：{core_hypothesis}"
    elif not specific:
        core_hypothesis = "变量关系不完整，无法注册实验假设。"
        direct = "experiment_design_skill 拒绝为模糊问题生成看似完整的实验方案。"
    else:
        core_hypothesis = "本地正式可信 RAG 缺少直接支持证据，无法注册有依据的实验假设。"
        direct = "experiment_design_skill 拒绝把证据不足扩展成看似完整的实验方案。"
    formulas = formula_lines(value)
    formula_fit = (
        "；".join(formulas)
        if formulas
        else "未召回可追溯原文公式；只预注册模型结构，不预填系数。"
    )
    supports = f"{iv} 的组间效应在控制协变量后方向一致，并能在留出批次预测 {dv}。"
    falsifies = f"{iv} 的效应消失或稳定反转，且结果由已测混杂变量完全解释。"
    reasoning = f"""**研究对象**

与问题材料和制造路线一致的钛合金疲劳试样；具体牌号、粉末批次和工艺窗口必须登记。

**核心假设**

{core_hypothesis}

**自变量水平**

设置 {iv} 的基准、低、中、高或问题定义的离散工艺水平；先导试验后冻结水平。

**因变量及测量方法**

{dv}；使用疲劳寿命/极限测量、裂纹复制或原位监测、SEM断口溯源，并按问题补充 micro-CT、EBSD或残余应力测量。

**控制变量**

材料批次、制造窗口、几何、热处理、表面状态、应力比、频率、温度和环境。

**分组与对照**

同批次基准组、{iv} 多水平组和关键条件交互组；保留盲法断口判定。

**样本建议**

先导阶段每组不少于5件仅作为方差估计起点；正式样本量由预注册最小效应、方差、功效和删失率计算。

**载荷条件**

应力幅、应力比、频率、环境、run-out和HCF/VHCF区间预注册，不跨机制混合拟合。

**表征方法**

按问题选用3D表面轮廓、micro-CT、残余应力、EBSD、SEM断口与裂纹路径表征。

**数据和公式拟合方法**

{formula_fit} 使用混合效应或生存模型检验主效应与交互项，并报告不确定性区间和留出批次验证。

**支持假设的结果**

{supports}

**推翻假设的结果**

{falsifies}

**最低成本方案**

复用现有同批次试样，补做关键条件测量、小规模配对疲劳和盲法断口复核。

**完整验证方案**

跨批次全因子试验，结合无损缺陷追踪、组织/残余应力表征、预注册统计模型和独立验证批次。"""
    combined = direct + "\n" + reasoning
    required = (
        "自变量水平", "因变量及测量方法", "分组与对照", "样本建议",
        "支持假设的结果", "推翻假设的结果", "最低成本方案", "完整验证方案",
    )
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME, required_terms=required)
    gate["design_section_count"] = sum(term in combined for term in required)
    gate["original_evidence_rechecked"] = bool(value.retrieved_evidence)
    gate["evidence_sufficient"] = evidence_sufficient
    gate["passed"] = (
        gate["passed"]
        and gate["original_evidence_rechecked"]
        and evidence_sufficient
    )
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=direct,
        structured_reasoning=reasoning,
        uncertainty="精确样本量、载荷水平和安全边界必须由先导方差、设备能力和伦理/安全审查确认。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing,
        specific_fields={
            "research_object": "question-matched titanium fatigue specimens",
            "core_hypothesis": core_hypothesis,
            "independent_variable_levels": f"controlled levels of {iv}",
            "dependent_measurement": dv,
            "formula_fit": formula_fit,
            "supports_hypothesis_if": supports,
            "falsifies_hypothesis_if": falsifies,
            "minimum_cost_plan": "matched pilot using existing specimens",
            "complete_plan": "cross-build factorial validation",
            "quality_metrics": QUALITY_METRICS,
            "evidence_counts": {"support": support, "counter": counter, "conditional": conditional},
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

- 方案参数：样本量和载荷水平需经先导试验冻结。
- 证据核对：实验 Skill 已重新读取原始证据卡，而非继承上一步文本。
- 本地证据不足：{('；'.join(output.missing_evidence)) if output.missing_evidence else '未发现结构性缺项。'}
"""
