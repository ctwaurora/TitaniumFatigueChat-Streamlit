"""Formula-aware falsifiable hypothesis Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, extract_falsification_criteria, joint_short_crack_candidate_models, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
from src.research_skills.contracts import SkillInput, SkillOutput
from src.research_skills.domain_profiles import profile_prompt_block, select_domain_profile


SKILL_NAME = "hypothesis_generation_skill"
PROMPT_VERSION = "hypothesis-generation-v1.0.0"
TASK_DEFINITION = "生成有来源、公式约束、可拟合并可被明确推翻的候选假设。"
QUALITY_METRICS = ("hypothesis_falsifiability", "formula_applicability", "condition_completeness", "unsupported_claim_rate")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 hypothesis_generation_skill。问题：{value.user_query}
基于EvidenceBundle生成候选假设，不得声称已被证明。必须写：研究对象；自变量；因变量；控制变量；中介机制；预测关系的函数类型；可拟合公式、参数、单位与适用范围；预测符号；至少两条具体推翻条件；替代解释；拟合和验证方法。
文献公式须逐字引用Formula ID、题名、页码、章节；无可信文献公式但有定量依据时，只能写“以下为系统提出的待拟合候选模型，并非文献原公式”，且不得伪造系数。
每个事实标 Evidence ID 和页码，不报告证据数量，不把未提及的孔隙设为核心变量。
{evidence_level_instruction()}
{profile_prompt_block(value)}
EvidenceBundle：{bundle_prompt(value)}"""


def build_repair_prompt(value: SkillInput, *, draft: str, failures: list[str]) -> str:
    return f"""修复 hypothesis_generation_skill 草稿。失败项：{'；'.join(failures)}。
补齐具体变量、机制、函数形式、单位、范围、替代解释、两条独立推翻判据和验证方法。真实公式只能逐字来自EvidenceBundle；候选模型必须明确标注且不得填系数。
草稿：{draft}\nEvidenceBundle：{bundle_prompt(value)}"""


def _fallback(value: SkillInput) -> tuple[str, str, list[str]]:
    iv, dv = entity_labels(value)
    bundle = value.evidence_bundle or {}
    cross = bundle.get("synthesis") or {}
    formulas = usable_formulas(value)
    frame = value.query_frame or bundle.get("query_frame") or {}
    profile = select_domain_profile(value)
    joint_profile = profile.key == "residual_microstructure_short_crack"
    provisional_joint = joint_profile and bool(value.retrieved_evidence) and bool(iv and dv)
    if not ((value.support_evidence and iv and dv) or provisional_joint):
        return "本地正式证据不足，无法形成有来源的可证伪假设。", "需要明确变量并补足直接证据与实验条件。", []
    support_cite = primary_citation(value)
    alternative_cite = primary_citation(value, "ALTERNATIVE_MECHANISM")
    counter_cite = primary_citation(value, "COUNTER")
    hypothesis = f"候选假设：在材料、热处理、表面状态和载荷条件匹配后，{iv}通过改变局部裂纹驱动力或微观屏障作用，使{dv}出现可重复、可拟合的条件化变化。{support_cite}"
    if not frame.get("requested_formulas"):
        formulas = []
    independent = frame.get("independent_variables") or []
    dependent = frame.get("dependent_variables") or []
    variables = set(independent) | set(dependent)
    subject = str(frame.get("alloy_grade") or "题目限定的钛合金")
    if profile.key != "defect_size_life":
        hypothesis = (
            f"候选假设：针对{subject}，{profile.direct_claim}"
            f"该命题仅是证据约束的待验证假设。{support_cite}"
        )
    pore_life = profile.key == "defect_size_life"
    pore_candidate_model = (
        "log10(Nf)=β0−β1log10(σa)−β2log10(√area)+β3(d/√area)+β4R+β5σres"
        "+β6[log10(√area)×d/√area]+ε"
    )
    if pore_life:
        hypothesis = (
            "核心科学假设：孔隙尺寸对疲劳寿命的影响不是固定单变量效应，而受归一化缺陷深度调节；"
            "缺陷越接近自由表面，相同√area引起的寿命下降幅度越大。该关系还受应力幅、应力比、"
            f"残余应力和表面状态约束。{support_cite}"
        )
        formula = (
            "待拟合候选模型，并非现有文献已经确定的公式：\n\n"
            f"`{pore_candidate_model}`\n\n"
            "Nf为疲劳寿命，单位cycle；σa为应力幅，单位MPa；√area为缺陷投影面积平方根，单位µm；"
            "d为缺陷距自由表面的距离，单位µm；d/√area为归一化缺陷深度，无量纲；R为应力比，无量纲；"
            "σres为缺陷附近残余应力，单位MPa；β0—β6为待实验数据拟合的参数；ε为未解释误差项。"
        )
    elif joint_profile:
        formula = joint_short_crack_candidate_models()
    elif formulas:
        item = formulas[0]
        formula = f"文献原公式：{item['equation']}；Formula ID：{item['formula_id']}；页码：{item['page_number']}；参数：{item.get('parameters') or '未报告'}；单位：{item.get('units') or '未报告'}；适用条件：{item.get('applicable_conditions') or '未完整报告'}。"
    else:
        formula = (
            f"以下为系统提出的待拟合候选模型，并非文献原公式：`{profile.model}`。"
            f"{profile.variables_and_units} {profile.model_note} 所有β参数都由实验估计，不预填数值。"
        )
    if pore_life:
        falsifiers = [
            "控制应力幅、表面状态、残余应力和组织后，√area项置信区间覆盖0，且加入该项不改善外部验证误差。",
            "尺寸×位置交互项置信区间覆盖0，或其方向在独立制造批次中无法复现。",
            "仅含应力幅和表面粗糙度的基线模型达到与完整模型相当的交叉验证性能。",
            "XCT与断口SEM不支持较大或近表面缺陷更易成为主裂纹源。",
        ]
    elif joint_profile:
        falsifiers = [
            "匹配ΔKeff后，σres主效应以及σres·lα、σres·T交互项均不显著，且M4不改善留出误差。",
            "加入lα和T后，原先归因于残余应力的效应被组织变量完全解释，M1不再优于M2。",
            "σres·lα或σres·T的预注册方向在独立制造批次中不能复现。",
        ]
    else:
        falsifiers = list(profile.falsifiers)
    prediction_text = (
        "在上述符号约定下，β1与β2预计大于0，因此应力幅或√area增大使log10(Nf)下降；"
        "β3预计大于0，表示缺陷更深时寿命相对提高；若近表面缺陷中的尺寸效应更强，β6预计大于0，"
        "使随d/√area增大时尺寸惩罚减弱。以拉应力为正时β5预计小于0。"
        "这些只是待检验方向，不是已知系数。"
        if pore_life else
        f"{profile.predictions} {profile.title}的预测方向必须结合{'、'.join(profile.independent)}的证据与预实验预注册；"
        f"涉及{'、'.join(profile.dependent)}的候选机制不能写成已证实因果。"
    )
    model_comparison = (
        "| 模型 | 变量 | 目的 | 比较指标 |\n"
        "|---|---|---|---|\n"
        "| 基线模型 | σa、表面状态及预注册控制变量 | 建立不含缺陷指标的基准 | 交叉验证RMSE/MAE、AIC/BIC、调整R² |\n"
        "| 尺寸模型 | 基线+√area | 检验缺陷尺寸增量解释力 | ΔAIC/BIC、Bootstrap系数区间 |\n"
        "| 尺寸—位置模型 | 基线+√area+d/√area | 检验位置独立贡献 | 外部验证误差、调整R² |\n"
        "| 交互模型 | 基线+√area+d/√area+尺寸×位置 | 检验近表面效应增强 | 似然比检验、交叉验证、Bootstrap区间 |"
        if pore_life else
        "| 模型 | 变量 | 回答的问题 | 比较方法 |\n"
        "|---|---|---|---|\n"
        "| M0 | ΔKeff、试样随机效应 | 有效驱动力基线能解释多少变异 | AIC/BIC、留出RMSE/MAE |\n"
        "| M1 | M0+σres、a、R | 残余应力是否有增量解释力 | M0/M1似然比、交叉验证 |\n"
        "| M2 | M0+lα、T、a | 可测组织变量是否有增量解释力 | M0/M2似然比、交叉验证 |\n"
        "| M3 | M0+σres+lα+T+a+R | 两类主效应是否可加 | 与M1/M2比较AIC/BIC |\n"
        "| M4 | M3+σres·lα+σres·T | 是否存在可复现交互 | M3/M4似然比、留出批次与预注册方向 |"
        if joint_profile else
        "| 模型层级 | 主题变量 | 目的 | 评价指标 |\n"
        "|---|---|---|---|\n"
        f"| {profile.title}基线 | {profile.confounders} | 建立主题基准 | {profile.title}留出误差与信息准则 |\n"
        f"| {profile.title}主效应 | {'、'.join(profile.independent)} | 检验增量解释力 | {profile.title}参数区间与交叉验证 |\n"
        f"| {profile.title}交互 | `{profile.model}` | 检验{profile.gap_focus} | {profile.title}外部批次误差 |"
    )
    extra_falsifiers = ""
    if len(falsifiers) > 2:
        extra_falsifiers = "".join(
            f"\n{index}. {criterion}"
            for index, criterion in enumerate(falsifiers[2:], start=3)
        )
    support_finding = (cross.get("consensus") or ["现有证据只支持条件化趋势。"]) [0]
    support_line = (
        f"{profile.title}直接支持依据：{support_finding} {support_cite}"
        if support_cite else
        f"【证据不足】{profile.title}尚无可支持联合主张的直接证据。"
    )
    counter_finding = (cross.get("conflicts") or [""])[0]
    if not counter_finding or counter_finding.startswith("不同文献报告的结果方向不一致"):
        counter_finding = (
            f"{profile.title}的反向结果必须按{profile.boundary}比较；"
            f"条件不一致时只能归为{'、'.join(profile.dependent)}的条件依赖证据。"
        )
    reasoning = f"""### 具体候选假设
{hypothesis}

### 研究对象、自变量和因变量
研究对象以证据中题名、材料和工艺匹配的钛合金试样为限；自变量：{'、'.join(profile.independent)}；因变量：{'、'.join(profile.dependent)}。

### 控制变量和中介机制
控制{profile.confounders}。中介机制为：{profile.mechanism_chain}

### 预测关系、公式和适用范围
{formula}

### 预测方向
{prediction_text}

### 支持、反向与替代解释
{support_line}\n{profile.title}反向边界：{counter_finding} {counter_cite}\n替代机制证据：{profile.confounders}。{alternative_cite}

### 交互证据边界
{"当前正式文献库尚未找到在同试样、同载荷条件下直接量化残余应力—微观组织交互的证据；M4只能作为待验证候选模型。" if joint_profile else "是否加入交互项由问题变量和直接证据共同决定。"}

### 明确证伪判据
1. 证伪判据一：{falsifiers[0]}\n2. 证伪判据二：{falsifiers[1]}{extra_falsifiers}

### 参数拟合与实验验证方案
{model_comparison}

针对{profile.title}预注册候选函数、{profile.variables_and_units}及适用区间；报告{profile.title}参数区间、共线性和留出误差，并用{profile.measurements}复核机制。"""
    reasoning += (
        f"\n\n### 数据要求\n{profile.title}需要逐试样记录{'、'.join(profile.independent)}、"
        f"{'、'.join(profile.dependent)}以及{profile.confounders}；单位约定为：{profile.variables_and_units} "
        "参数拟合集与独立制造批次验证集必须分开。"
        f"\n\n### 替代解释\n本主题的替代解释为{profile.confounders}，需按{profile.boundary}通过匹配、分层或协变量调整拆分。"
    )
    return hypothesis, reasoning, falsifiers


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning, falsifiers = _fallback(value)
    complete = synthesis.strip()
    rendered_falsifiers = extract_falsification_criteria(complete) if complete else falsifiers
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["falsification_criteria_count"] = len(rendered_falsifiers)
    profile = select_domain_profile(value)
    evidence_basis = bool(value.support_evidence) or (
        profile.key == "residual_microstructure_short_crack" and bool(value.retrieved_evidence)
    )
    gate["passed"] = gate["passed"] and evidence_basis and gate["falsification_criteria_count"] >= 2
    gate["provisional_evidence_basis"] = not bool(value.support_evidence) and evidence_basis
    independent, dependent = query_variables(value)
    variables = set(independent) | set(dependent)
    if profile.key == "defect_size_life":
        candidate_model = "log10(Nf)=β0−β1log10(σa)−β2log10(√area)+β3(d/√area)+β4R+β5σres+β6[log10(√area)×d/√area]+ε"
    elif profile.key == "residual_microstructure_short_crack":
        candidate_model = joint_short_crack_candidate_models()
    else:
        candidate_model = profile.model
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, "候选假设不是已证实结论；条件不匹配时不得外推。", traceable_cards(value), gate, missing_evidence(value), {
        "complete_answer": complete, "falsification_criteria": rendered_falsifiers, "quality_metrics": QUALITY_METRICS,
        "hypothesis_statement": complete or direct,
        "independent_variables": independent, "dependent_variables": dependent,
        "control_variables": ["material_batch", "manufacturing_window", "surface_condition", "stress_ratio", "frequency", "temperature", "environment"],
        "covariates": ["residual_stress", "microstructure", "surface_roughness", "initial_crack_length"],
        "expected_function_form": "interaction_model",
        "literature_formula": (value.evidence_bundle or {}).get("formulas") or [],
        "proposed_candidate_model": candidate_model,
        "parameters_to_fit": [f"β{index}" for index in range(9)] if profile.key == "residual_microstructure_short_crack" else ["β0", "β1", "β2", "β3", "β4", "β5"],
        "data_requirements": "逐试样变量、条件、批次与独立验证集",
        "fitting_method": "预注册候选模型比较与留出验证",
        "validation_method": "独立材料批次复现",
        "alternative_explanations": ["热处理共变", "表面状态共变", "未测组织或残余应力"],
        "evidence_level": "PROPOSED_HYPOTHESIS",
    }, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
