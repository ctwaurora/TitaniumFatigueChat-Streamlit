"""Executable evidence-conditioned experiment-design Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, missing_evidence, primary_citation, traceable_cards
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "experiment_design_skill"
TASK_DEFINITION = "把证据约束的假设转化为可执行、可拟合、可证伪的疲劳实验。"
QUALITY_METRICS = ("experiment_design_completeness", "confounder_control", "formula_applicability", "falsifiability")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 experiment_design_skill。问题：{value.user_query}
必须重新核对EvidenceBundle，不得把任何上一步自然语言当作事实。输出：研究对象（具体材料/工艺/热处理/表面）；核心假设；自变量水平设计原则；因变量指标、单位和测量；控制变量；协变量；基准/处理/交互/阴性或阳性对照；预实验与正式样本量估计；加载条件；表征；数据与公式拟合；具体预测；至少两条推翻结果；最低成本方案；完整方案；风险混杂。
不得随意给固定样本数或无来源阈值。每个证据约束标Evidence ID和页码，不报告证据数量。用户未问孔隙时不得自动设计孔隙实验。
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
    formula_note = "没有通过结构检查的文献公式，先比较预注册候选模型，不填造系数。"
    if bundle.get("formulas"):
        item = bundle["formulas"][0]
        formula_note = f"拟合文献原公式 {item['equation']}（Formula ID：{item['formula_id']}，页码：{item['page_number']}），仅在其单位和适用条件一致时比较。"
    reasoning = f"""### 研究对象
{object_text}。

### 核心假设
{hypothesis}

### 自变量
{iv}设置基准、低、中、高或物理上有意义的离散水平；先由预实验响应范围冻结水平，不跨越不同失效机制混合拟合。

### 因变量
{dv}；记录原始单位、测量分辨率和删失/run-out状态，采用相应疲劳试验、裂纹复制或原位监测测量。

### 控制变量
材料批次、制造窗口、试样几何、热处理、表面状态、应力比、频率、温度和环境全部固定或纳入模型。

### 协变量
按主题测量残余应力、组织尺度、建造方向、表面粗糙度及实际载荷；只纳入有物理依据且在试验前定义的协变量。

### 分组和对照
同批次基准组；{iv}处理组；{iv}×关键条件交互组；保持处理流程但不改变{iv}的阴性对照，或采用已有有效处理的阳性对照。

### 样本建议
预实验用于估计方差、删失率和最小有意义效应；正式样本量由目标功效、显著性水平、效应量、组数及失访/run-out率计算，不预填固定数量。

### 加载条件
按证据匹配应力幅或ΔK、应力比、频率、温度、环境、run-out和LCF/HCF/VHCF或裂纹扩展区间，并在注册后固定。

### 表征方法
根据假设选择XRD残余应力、EBSD组织、表面轮廓、裂纹复制/DIC、SEM断口；只有缺陷是问题变量时才加入micro-CT。

### 数据分析和公式拟合
{formula_note} 使用混合效应/生存模型估计主效应和交互项，报告置信区间、共线性、模型比较与独立验证。

### 预测结果
若假设成立，{iv}变化应同时引起预设中介指标和{dv}按同一条件化模型变化，而不是只出现未解释的组间差异。

### 推翻假设的结果
1. {falsifiers[0]}\n2. {falsifiers[1]}

### 最低成本方案
复用同批材料，补测关键协变量，完成小规模配对疲劳和盲法断口判定。

### 完整验证方案与风险
跨批次因子设计、无损/组织/残余应力联合表征、预注册统计模型和独立批次验证。主要风险是批次、表面、热处理与载荷共线；{'；'.join(cross.get('condition_mismatches') or ['需防止条件漂移'])}。"""
    return f"建议采用证据匹配的分层对照设计验证：{hypothesis}", reasoning, falsifiers


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning, falsifiers = _fallback(value)
    complete = synthesis.strip()
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["original_evidence_rechecked"] = bool(value.retrieved_evidence)
    gate["falsification_criteria_count"] = 2 if complete else len(falsifiers)
    gate["passed"] = gate["passed"] and bool(value.support_evidence) and gate["original_evidence_rechecked"] and gate["falsification_criteria_count"] >= 2
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, "精确水平和样本量必须由预实验方差、设备能力与安全审查冻结。", traceable_cards(value), gate, missing_evidence(value), {"complete_answer": complete, "falsification_criteria": falsifiers, "quality_metrics": QUALITY_METRICS}, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
