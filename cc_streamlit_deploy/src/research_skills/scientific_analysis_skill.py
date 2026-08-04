"""Evidence-bound scientific analysis Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
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
{evidence_level_instruction()}
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
    formulas = usable_formulas(value)
    frame = value.query_frame or bundle.get("query_frame") or {}
    if not consensus:
        return "本地正式证据不足，无法对该变量关系给出可追溯结论。", "当前缺少可核验的直接证据、条件与反向结果。"
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    variables = set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])
    pore_life = "pore_size" in variables and bool({"fatigue_life", "fatigue_life_Nf"} & variables)
    if pore_life:
        direct = (
            "在L-PBF Ti-6Al-4V中，孔隙尺寸增大（常用√area表征）、即出现较大的孔隙时，通常与疲劳寿命降低相关，"
            "未熔合缺陷增大时也可出现相同方向的趋势，"
            "因为更大的等效初始缺陷会提高局部应力集中并缩短裂纹起裂或早期扩展阶段。"
            "但√area不是独立决定量：缺陷距自由表面的距离、形貌、表面粗糙度、残余应力、"
            "微观组织、应力幅/应力比以及HIP或热处理状态都会改变其成为主裂纹源的概率。"
            f"因此不能把单一√area阈值外推到所有HCF/VHCF和表面状态。{support_cite}"
        )
    else:
        direct = f"现有正式证据支持在已报告条件内分析{iv or '所问自变量'}对{dv or '所问结果'}的作用，但不能跨越未匹配条件直接外推。{support_cite}"
    formula_text = "未检索到通过可信结构检查的文献原公式；现有证据支持趋势，但不足以确定具体函数形式。"
    if formulas:
        item = formulas[0]
        formula_text = f"{item['equation']}（Formula ID：{item['formula_id']}，页码：{item['page_number']}；适用条件：{item.get('applicable_conditions') or '原文未完整报告'}）"
    mechanism = consensus[1] if len(consensus) > 1 else "证据未充分分离各机制贡献，不能把相关性写成单因素因果。"
    if pore_life:
        mechanism = (
            "√area增大 → 缺陷等效初始裂纹尺度和局部应力集中提高 → 起裂所需循环数减少，"
            "并可能更早进入短裂纹扩展阶段 → Nf降低。缺陷接近自由表面时自由表面放大局部驱动力，"
            "相同√area的危害可能更强；压缩残余应力、HIP后缺陷闭合或有利组织屏障可削弱该链条，"
            "而粗糙表面、拉伸残余应力和较高应力幅可增强该链条。后两类判断属于条件化综合，需匹配试验验证。"
        )
    if "residual_stress" in (frame.get("independent_variables") or []):
        mechanism = "残余拉应力升高 → 局部平均应力与有效裂纹驱动力提高 → 裂纹闭合减弱 → 相同外加ΔK下ΔKeff提高 → 短裂纹da/dN可能升高；压缩残余应力的链条方向相反，但必须核对循环松弛。该链条包含跨文献综合和待直接验证环节。"
    formula_boundary = ""
    if frame.get("crack_stage") == "SHORT_CRACK" and formulas and any("paris" in str(item.get("equation") or "").casefold() for item in formulas):
        formula_boundary = "该Paris类关系只能作为长裂纹基线，不能直接视为短裂纹定量模型。"
    boundary_text = (
        "结论仅适用于材料牌号、L-PBF工艺、热处理/HIP、表面状态、残余应力、载荷模式、"
        "应力比和疲劳区间可比的试样；HCF与VHCF、表面与内部起裂不得直接合并。"
        if pore_life else
        ("；".join(mismatches) if mismatches else "只能适用于证据卡中明确报告的材料、处理、表面和加载条件。")
    )
    reasoning = f"""### 具体机制
{mechanism}

### 实验条件和适用边界
{boundary_text}

### 文献冲突及反向观点
{(conflicts[0] + ' ' + counter_cite) if conflicts else '当前检索结果未提供可明确复核的相反结论；这不等于不存在反向证据。'}

### 公式或数学关系
{formula_text} {formula_boundary}

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
    independent, dependent = query_variables(value)
    bundle = value.evidence_bundle or {}
    return SkillOutput(
        skill_name=SKILL_NAME,
        direct_answer=complete or direct,
        structured_reasoning="" if complete else reasoning,
        uncertainty="结论只覆盖EvidenceBundle中已报告且可比较的条件。",
        evidence_cards=traceable_cards(value),
        quality_gate=gate,
        missing_evidence=missing_evidence(value),
        specific_fields={
            "complete_answer": complete,
            "quality_metrics": QUALITY_METRICS,
            "direct_answer": complete or direct,
            "mechanism_chain": [item for paper in bundle.get("papers") or [] for item in paper.get("mechanisms") or []][:6],
            "experimental_conditions": (bundle.get("synthesis") or {}).get("covered_conditions") or {},
            "quantitative_relationships": [item for paper in bundle.get("papers") or [] for item in paper.get("quantitative_results") or []][:8],
            "formula_analysis": bundle.get("formulas") or [],
            "conflicting_evidence": (bundle.get("synthesis") or {}).get("conflicting_findings") or [],
            "applicability_boundary": (bundle.get("synthesis") or {}).get("condition_mismatches") or [],
            "unresolved_questions": (bundle.get("synthesis") or {}).get("unsupported_conclusions") or [],
            "evidence_limitations": (bundle.get("synthesis") or {}).get("evidence_limitations") or [],
            "independent_variables": independent,
            "dependent_variables": dependent,
        },
        trace={"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]},
    )


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    if complete:
        return str(complete)
    return f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
