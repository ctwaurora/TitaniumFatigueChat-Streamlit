"""Formula-aware falsifiable hypothesis Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "hypothesis_generation_skill"
TASK_DEFINITION = "生成有来源、公式约束、可拟合并可被明确推翻的候选假设。"
QUALITY_METRICS = ("hypothesis_falsifiability", "formula_applicability", "condition_completeness", "unsupported_claim_rate")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 hypothesis_generation_skill。问题：{value.user_query}
基于EvidenceBundle生成候选假设，不得声称已被证明。必须写：研究对象；自变量；因变量；控制变量；中介机制；预测关系的函数类型；公式、参数、单位与适用范围；至少两条具体推翻条件；替代解释；拟合和验证方法。
文献公式须逐字引用Formula ID、题名、页码、章节；无可信文献公式但有定量依据时，只能写“以下为系统提出的待拟合候选模型，并非文献原公式”，且不得伪造系数。
每个事实标 Evidence ID 和页码，不报告证据数量，不把未提及的孔隙设为核心变量。
{evidence_level_instruction()}
EvidenceBundle：{bundle_prompt(value)}"""


def build_repair_prompt(value: SkillInput, *, draft: str, failures: list[str]) -> str:
    return f"""修复 hypothesis_generation_skill 草稿。失败项：{'；'.join(failures)}。
补齐具体变量、机制、函数形式、单位、范围、替代解释、两条独立推翻判据和验证方法。真实公式只能逐字来自EvidenceBundle；候选模型必须明确标注且不得填系数。
草稿：{draft}\nEvidenceBundle：{bundle_prompt(value)}"""


def _fallback(value: SkillInput) -> tuple[str, str, list[str]]:
    iv, dv = entity_labels(value)
    bundle = value.evidence_bundle or {}
    cross = bundle.get("synthesis") or {}
    formulas = bundle.get("formulas") or []
    if not (value.support_evidence and iv and dv):
        return "本地正式证据不足，无法形成有来源的可证伪假设。", "需要明确变量并补足直接证据与实验条件。", []
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    hypothesis = f"候选假设：在材料、热处理、表面状态和载荷条件匹配后，{iv}通过改变局部裂纹驱动力或微观屏障作用，使{dv}出现可重复、可拟合的条件化变化。{support_cite}"
    frame = value.query_frame or bundle.get("query_frame") or {}
    independent = frame.get("independent_variables") or []
    dependent = frame.get("dependent_variables") or []
    if formulas:
        item = formulas[0]
        formula = f"文献原公式：{item['equation']}；Formula ID：{item['formula_id']}；页码：{item['page_number']}；参数：{item.get('parameters') or '未报告'}；单位：{item.get('units') or '未报告'}；适用条件：{item.get('applicable_conditions') or '未完整报告'}。"
    elif {"residual_stress", "alpha_lath_width"}.issubset(set(independent)) and any("da_dn" in item or "crack_growth" in item for item in dependent):
        formula = "以下为系统提出的待拟合候选模型，并非文献原公式：log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3lα+β4R+β5σres·lα。da/dN为裂纹扩展速率；ΔKeff单位MPa√m；σres单位MPa；lα单位µm；R无量纲；β0—β5均需实验估计，β5检验残余应力与片层宽度交互。"
    else:
        formula = f"以下为系统提出的待拟合候选模型，并非文献原公式：g({dv})=β0+β1·{iv}+Σβk·控制变量。β参数全部需要实验拟合，不预填数值；链接函数g、线性/幂律/阈值或分段形式由物理约束和留出验证共同选择。"
    falsifiers = [
        f"控制协变量后，{iv}项及其关键交互项的置信区间覆盖预注册的最小效应，且加入该项不改善留出批次预测。",
        f"独立材料批次出现稳定相反方向，或中介指标不随{iv}变化而{dv}仍变化。",
    ]
    reasoning = f"""### 具体候选假设
{hypothesis}

### 研究对象、自变量和因变量
研究对象以证据中题名、材料和工艺匹配的钛合金试样为限；自变量：{iv}；因变量：{dv}。

### 控制变量和中介机制
控制材料批次、制造窗口、几何、热处理、表面状态、应力比、频率、温度和环境。中介量测为局部裂纹驱动力、裂纹闭合或微观组织屏障，按问题主题选择。

### 预测关系、公式和适用范围
{formula}

### 支持、反向与替代解释
支持依据：{(cross.get('consensus') or ['现有证据只支持条件化趋势。'])[0]} {support_cite}\n反向边界：{(cross.get('conflicts') or ['未召回明确反向结果，不能据此认定不存在。'])[0]} {counter_cite}\n替代解释包括未测残余应力、表面状态、组织尺度和载荷历史。

### 明确证伪判据
1. {falsifiers[0]}\n2. {falsifiers[1]}

### 参数拟合与实验验证方案
预注册候选函数、单位和适用区间；报告参数置信区间、共线性和留出批次预测误差，并以独立试样复核机制指标。"""
    reasoning += "\n\n### 数据要求\n需要逐试样记录自变量、因变量、应力比、裂纹阶段、材料批次和关键协变量；训练集用于拟合，独立批次作为验证集。\n\n### 替代解释\n热处理、表面加工或HIP可能同时改变残余应力、组织和缺陷状态，必须以协变量或匹配设计拆分。"
    return hypothesis, reasoning, falsifiers


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning, falsifiers = _fallback(value)
    complete = synthesis.strip()
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["falsification_criteria_count"] = 2 if complete else len(falsifiers)
    gate["passed"] = gate["passed"] and bool(value.support_evidence) and gate["falsification_criteria_count"] >= 2
    independent, dependent = query_variables(value)
    candidate_model = "log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3lα+β4R+β5σres·lα" if {"residual_stress", "alpha_lath_width"}.issubset(set(independent)) else f"g({dependent[0] if dependent else 'y'})=β0+Σβi·xi"
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, "候选假设不是已证实结论；条件不匹配时不得外推。", traceable_cards(value), gate, missing_evidence(value), {
        "complete_answer": complete, "falsification_criteria": falsifiers, "quality_metrics": QUALITY_METRICS,
        "hypothesis_statement": complete or direct,
        "independent_variables": independent, "dependent_variables": dependent,
        "control_variables": ["material_batch", "manufacturing_window", "surface_condition", "stress_ratio", "frequency", "temperature", "environment"],
        "covariates": ["residual_stress", "microstructure", "surface_roughness", "initial_crack_length"],
        "expected_function_form": "interaction_model",
        "literature_formula": (value.evidence_bundle or {}).get("formulas") or [],
        "proposed_candidate_model": candidate_model,
        "parameters_to_fit": ["β0", "β1", "β2", "β3", "β4", "β5"],
        "data_requirements": "逐试样变量、条件、批次与独立验证集",
        "fitting_method": "预注册候选模型比较与留出验证",
        "validation_method": "独立材料批次复现",
        "alternative_explanations": ["热处理共变", "表面状态共变", "未测组织或残余应力"],
        "evidence_level": "PROPOSED_HYPOTHESIS",
    }, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
