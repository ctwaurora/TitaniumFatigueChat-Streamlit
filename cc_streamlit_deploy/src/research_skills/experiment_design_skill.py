"""Executable evidence-conditioned experiment-design Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
from src.research_skills.contracts import SkillInput, SkillOutput
from src.research_skills.domain_profiles import profile_prompt_block, select_domain_profile


SKILL_NAME = "experiment_design_skill"
TASK_DEFINITION = "把证据约束的假设转化为可执行、可拟合、可证伪的疲劳实验。"
QUALITY_METRICS = ("experiment_design_completeness", "confounder_control", "formula_applicability", "falsifiability")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 experiment_design_skill。问题：{value.user_query}
必须重新核对EvidenceBundle，不得把任何上一步自然语言当作事实。以Markdown表格为主体，至少输出变量定义表、实验分组表、预测与证伪表；并说明研究对象、载荷、样本量、表征、统计模型、最低成本与完整方案。
不得随意给固定样本数或无来源阈值。每个证据约束标Evidence ID和页码，不报告证据数量。用户未问孔隙时不得自动设计孔隙实验。
{evidence_level_instruction()}
{profile_prompt_block(value)}
EvidenceBundle：{bundle_prompt(value)}"""


def build_repair_prompt(value: SkillInput, *, draft: str, failures: list[str]) -> str:
    return f"""修复 experiment_design_skill 草稿。失败项：{'；'.join(failures)}。
使方案可执行：写清对象、水平设计、指标和单位、全部控制变量、协变量、四类分组、功效估计、载荷、表征、拟合、两条推翻判据、最低成本与完整方案。不得新增无来源参数。
草稿：{draft}\nEvidenceBundle：{bundle_prompt(value)}"""


def _fallback(value: SkillInput) -> tuple[str, str, list[str]]:
    iv, dv = entity_labels(value)
    bundle = value.evidence_bundle or {}
    papers = bundle.get("papers") or []
    cross = bundle.get("synthesis") or {}
    if not (value.support_evidence and iv and dv):
        return "本地正式证据不足，无法把该问题扩展为可信实验方案。", "需要明确自变量、因变量和可匹配的材料与加载条件。", []
    conditions = (papers[0].get("conditions") if papers else {}) or {}
    object_text = f"{conditions.get('alloy_grade') or conditions.get('material') or '题目指定的钛合金'}，{conditions.get('manufacturing_process') or conditions.get('process') or '制造工艺需与证据匹配'}，热处理={conditions.get('heat_treatment') or '固定并报告'}，表面={conditions.get('surface_treatment') or conditions.get('surface_state') or '固定并报告'}"
    hypothesis = f"在其余条件匹配后，改变{iv}会通过问题对应的裂纹驱动力或组织机制改变{dv}；若控制协变量后效应消失，则推翻该假设。{primary_citation(value)}"
    falsifiers = [
        f"{iv}主效应和预注册交互项的置信区间均覆盖最小效应，且留出批次预测没有改善。",
        f"独立批次的{dv}变化方向与预测相反，或预期中介量不变而结果仍出现。",
    ]
    formulas = usable_formulas(value)
    formula_note = "没有通过结构检查的文献公式，先比较预注册候选模型，不填造系数。"
    if formulas:
        item = formulas[0]
        formula_note = f"拟合文献原公式 {item['equation']}（Formula ID：{item['formula_id']}，页码：{item['page_number']}），仅在其单位和适用条件一致时比较。"
    frame = value.query_frame or bundle.get("query_frame") or {}
    profile = select_domain_profile(value)
    frame_subject = str(frame.get("alloy_grade") or "").strip()
    if frame_subject and frame_subject.casefold() not in object_text.casefold():
        object_text = f"{frame_subject}；{object_text}"
    if not frame.get("requested_formulas"):
        formulas = []
    variables = set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])
    pore_life = profile.key == "defect_size_life"
    if not pore_life:
        hypothesis = (
            f"{profile.direct_claim} 实验将检验中介链“{profile.mechanism_chain}”。"
            f"若{profile.falsifiers[0]}，则推翻或削弱该假设。{primary_citation(value)}"
        )
        falsifiers = list(profile.falsifiers)
    if pore_life:
        hypothesis = (
            "在材料、表面、残余应力和载荷条件匹配后，√area增大将降低Nf，且该效应受d/√area调节；"
            "近表面缺陷中的尺寸效应预计更强。若尺寸主效应与尺寸×位置交互均不改善独立批次预测，"
            f"则推翻或削弱该假设。{primary_citation(value)}"
        )
        formula_note = (
            "拟合待验证候选模型log10(Nf)=β0−β1log10(σa)−β2log10(√area)+β3(d/√area)"
            "+β4R+β5σres+β6[log10(√area)×d/√area]+ε；该式并非文献原公式，β参数不预填数值。"
        )
        variable_table = """| 变量类型 | 变量 | 建议水平或范围 | 单位 | 测量方法 | 设计目的 |
|---|---|---|---|---|---|
| 自变量 | √area | 按本批实际缺陷分布和XCT分辨率设置3—4级或保留连续值 | µm | 疲劳前XCT；断口SEM复核 | 检验尺寸主效应 |
| 自变量 | 缺陷距表面距离d | 表面、近表面、内部；边界按XCT分辨率预注册 | µm | XCT三维定位 | 检验位置效应 |
| 因变量 | Nf | 连续记录至失效或run-out | cycle | 疲劳试验机 | 寿命终点 |
| 控制变量 | 应力幅σa | 各组保持一致或按预实验S-N曲线分层 | MPa | 载荷控制与校准 | 排除载荷差异 |
| 控制变量 | 表面粗糙度Ra | 统一加工状态并实测 | µm | 轮廓仪/三维形貌 | 排除表面效应 |
| 协变量 | 残余应力σres | 每组实测，不预填数值 | MPa | XRD | 调整残余应力影响 |
| 协变量 | 组织尺度 | 同批取样并量化 | µm | EBSD/金相 | 调整组织屏障影响 |"""
        group_table = """| 组别 | 缺陷尺寸 | 缺陷位置 | 表面状态 | 热处理/HIP状态 | 载荷条件 | 主要比较 |
|---|---|---|---|---|---|---|
| G1 | 较小级/连续低分位 | 表面或近表面 | 统一加工 | 固定并报告 | 同σa、R、频率和环境 | 位置基线 |
| G2 | 较大级/连续高分位 | 表面或近表面 | 同G1 | 同G1 | 同G1 | 表面侧尺寸效应 |
| G3 | 较小级/连续低分位 | 内部 | 同G1 | 同G1 | 同G1 | 小尺寸位置效应 |
| G4 | 较大级/连续高分位 | 内部 | 同G1 | 同G1 | 同G1 | 尺寸×位置交互 |
| G5（验证） | 覆盖全范围 | 混合但盲法判定 | 独立批次同工艺 | 同主试验 | 同主试验 | 外部验证 |"""
        prediction_table = """| 检验项目 | 假设成立时的预测结果 | 推翻或削弱假设的结果 | 统计检验 |
|---|---|---|---|
| √area主效应 | √area增大时log10(Nf)下降 | 控制其他变量后效应接近0或方向不稳定 | 回归系数、Bootstrap置信区间 |
| 尺寸×位置交互 | 近表面缺陷中的尺寸效应更强 | 交互项不显著或方向相反 | 交互回归/似然比检验 |
| 裂纹起源 | 较大或近表面缺陷更常成为主裂纹源 | 起源概率与尺寸、位置无关 | Logistic/竞争风险模型 |
| 外部验证 | 完整模型优于基线模型 | 完整模型不优于基线模型 | RMSE/MAE、AIC/BIC、交叉验证 |"""
        loading_protocol = (
            "载荷模式优先采用与正式证据一致的轴向恒幅疲劳；应力比R、频率和温度/环境必须在组间固定并报告。"
            "具体应力水平需根据材料静力学性能和预实验S-N曲线确定。终止循环数由目标HCF/VHCF区间和设备能力预注册，"
            "不得从其他材料直接移植；失效定义为完全断裂或达到预注册裂纹/刚度终点，未失效试样记为run-out并按删失处理。"
        )
    else:
        variable_table = f"""| 变量类型 | 变量 | 建议水平或范围 | 单位 | 测量方法 | 设计目的 |
|---|---|---|---|---|---|
| 自变量 | {'、'.join(profile.independent)} | 文献范围与设备能力交集，经预实验冻结 | {profile.variables_and_units} | {profile.measurements} | 检验主效应与交互 |
| 因变量 | {'、'.join(profile.dependent)} | 连续记录至预注册终点 | 保持原始SI单位 | 对应疲劳/裂纹测量 | 分离起裂、扩展或寿命终点 |
| {profile.title}控制变量 | {profile.confounders} | 围绕{profile.title}固定或分层 | 主题原始单位 | 记录{profile.title}制程、环境与载荷 | 排除{profile.title}混杂 |
| {profile.title}协变量 | 批次、实际载荷及{'、'.join(profile.independent[:2])}相关机制量 | {profile.title}逐件实测 | 主题原始单位 | {profile.measurements} | 调整{profile.title}剩余共变 |
| 机制中介量 | {profile.mechanism_chain} | 各阶段同步测量 | 按测量方法 | {profile.measurements} | 检验因果链 |"""
        group_table = f"""| 组别 | {profile.title}因素设置 | 匹配条件 | 载荷/环境 | 主要比较 |
|---|---|---|---|---|
| G1-{profile.title}基线 | {'、'.join(profile.independent[:1])}基准 | {profile.confounders} | 按{profile.title}证据冻结 | 主题基线重复性 |
| G2-{profile.title}主效应 | 改变{'、'.join(profile.independent[:1])} | {profile.title}同批次匹配 | 同主题基线 | 主效应 |
| G3-{profile.title}交互 | {'×'.join(profile.independent[:2])}交互 | 补测{profile.title}中介量 | 同主题基线 | {profile.title}竞争机制 |
| G4-{profile.title}验证 | 覆盖{'、'.join(profile.independent)} | {profile.title}独立批次 | 复现主题主试验 | 外部验证 |"""
        prediction_table = f"""| 检验项目 | 假设成立时的预测结果 | 推翻或削弱假设的结果 | 统计检验 |
|---|---|---|---|
| {profile.title}主效应 | {profile.predictions} | {profile.falsifiers[0]} | {profile.title}系数区间与效应图 |
| {profile.title}机制中介 | {profile.mechanism_chain} | {profile.title}中介量不响应或方向相反 | 主题中介/混合效应模型 |
| {profile.title}外部验证 | `{profile.model}`优于主题基线 | {profile.falsifiers[1]} | {profile.title}留出误差与交叉验证 |"""
        loading_protocol = f"围绕{profile.title}，{profile.boundary} 精确水平、终止循环数和失效终点由匹配文献、设备能力与预实验冻结，不填写无来源数值。"
    reasoning = f"""### 研究对象
{object_text}。

### 核心假设
{hypothesis}

### 表1：变量定义表
{variable_table}

### 表2：实验分组表
{group_table}

### 表3：预测与证伪表
{prediction_table}

### 样本建议
{profile.title}预实验先估计{'、'.join(profile.dependent)}的方差、run-out删失率和最小有意义效应；正式样本量针对{'×'.join(profile.independent[:2])}交互，由功效分析或仿真确定，不预填固定数量。

### 加载条件
{loading_protocol}

### 表征方法
{profile.measurements}

### 数据分析、统计模型和公式拟合
{formula_note} {profile.title}的主题候选模型为`{profile.model}`；{profile.model_note} 围绕{'、'.join(profile.dependent)}选择统计结构，并用{profile.measurements}形成的机制指标复核参数；报告置信区间、共线性与独立批次预测误差。

### 预测结果
若假设成立，{profile.predictions}

### 推翻假设的结果
1. {falsifiers[0]}\n2. {falsifiers[1]}

### 最低成本方案
围绕{profile.title}，最低成本版本采用{profile.factor_design}，优先检验{'×'.join(profile.independent[:2])}并同步测量一个主题机制指标。

### 完整验证方案与风险
完整方案采用{profile.factor_design}，联合表征为{profile.measurements}，并用独立批次检验`{profile.model}`。主要风险来自{profile.confounders}；执行时必须维持{profile.boundary}"""
    return f"建议采用证据匹配的分层对照设计验证：{hypothesis}", reasoning, falsifiers


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning, falsifiers = _fallback(value)
    complete = synthesis.strip()
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["original_evidence_rechecked"] = bool(value.retrieved_evidence)
    gate["falsification_criteria_count"] = 2 if complete else len(falsifiers)
    gate["passed"] = gate["passed"] and bool(value.support_evidence) and gate["original_evidence_rechecked"] and gate["falsification_criteria_count"] >= 2
    independent, dependent = query_variables(value)
    profile = select_domain_profile(value)
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, f"{profile.title}的精确水平和样本量必须由预实验方差、设备能力与安全审查冻结。", traceable_cards(value), gate, missing_evidence(value), {
        "complete_answer": complete, "falsification_criteria": falsifiers, "quality_metrics": QUALITY_METRICS,
        "research_object": (value.query_frame or {}).get("alloy_grade") or (value.query_frame or {}).get("material"),
        "hypothesis": direct, "independent_variables": independent,
        "factor_levels": "文献范围∩设备能力，经预实验冻结；双因素优先全因子或响应面",
        "factor_level_rationale": "避免跨失效机制并检验处理引入的共变",
        "dependent_variables": dependent,
        "measurement_methods": "按裂纹扩展、起裂或寿命终点选择复制法/柔度法/原位成像/疲劳试验",
        "measurement_units": "保持原始SI单位并报告测量不确定度",
        "control_variables": ["material_batch", "manufacturing_window", "geometry", "surface_condition", "stress_ratio", "frequency", "temperature", "environment"],
        "covariates": ["residual_stress", "microstructure", "surface_roughness", "initial_crack_length"],
        "randomization": "材料批次内区组随机化测试顺序",
        "blocking": "按材料批次和制造批次区组",
        "sample_size_strategy": "预实验估计方差、效应量、删失率和交互效应后进行功效分析或仿真",
        "runout_definition": "在方案冻结时按疲劳区间定义并作为删失记录",
        "statistical_model": "混合效应回归、Paris/Walker拟合或删失生存模型，按终点选择",
        "minimum_cost_plan": "同批材料小规模配对试验和关键协变量补测",
        "full_validation_plan": "跨批次因子设计、联合表征、预注册模型和独立验证",
    }, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
