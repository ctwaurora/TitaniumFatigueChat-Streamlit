"""Evidence-bound scientific analysis Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
from src.research_skills.contracts import SkillInput, SkillOutput
from src.research_skills.domain_profiles import profile_prompt_block, select_domain_profile


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
{profile_prompt_block(value)}
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
    profile = select_domain_profile(value)
    if not frame.get("requested_formulas"):
        formulas = []
    if not consensus:
        return "本地正式证据不足，无法对该变量关系给出可追溯结论。", "当前缺少可核验的直接证据、条件与反向结果。"
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    variables = set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])
    pore_life = "pore_size" in variables and bool({"fatigue_life", "fatigue_life_Nf"} & variables)
    subject = str(frame.get("alloy_grade") or "").strip()
    subject_prefix = f"针对{subject}，" if subject and subject.casefold() not in profile.direct_claim.casefold() else ""
    entity_focus = ""
    if iv and iv not in profile.direct_claim or dv and dv not in profile.direct_claim:
        entity_focus = f" 该判断直接针对{iv or '题目自变量'}与{dv or '题目因变量'}。"
    direct = f"{subject_prefix}{profile.direct_claim}{entity_focus}{support_cite}"
    formula_text = (
        f"当前EvidenceBundle没有涉及{'、'.join(profile.independent[:2])}与"
        f"{'、'.join(profile.dependent)}的可逐字核验原公式；不能据此确定函数形式。"
    )
    if formulas:
        item = formulas[0]
        formula_text = f"{item['equation']}（Formula ID：{item['formula_id']}，页码：{item['page_number']}；适用条件：{item.get('applicable_conditions') or '原文未完整报告'}）"
    mechanism = profile.mechanism_chain
    formula_boundary = ""
    if frame.get("crack_stage") == "SHORT_CRACK" and formulas and any("paris" in str(item.get("equation") or "").casefold() for item in formulas):
        formula_boundary = "该Paris类关系只能作为长裂纹基线，不能直接视为短裂纹定量模型。"
    boundary_text = profile.boundary
    if mismatches:
        boundary_text += f" 当前EvidenceBundle中的条件尚不能同时满足上述{profile.title}比较边界。"
    missing_names = synthesis.get("missing_conditions") or []
    unknown_text = (
        f"{profile.title}目前不能确定的是“{profile.gap_focus}”。"
        f"仍缺少与该主题匹配的{'、'.join(missing_names[:5]) or profile.confounders}，"
        f"因此不能把{profile.direct_claim}外推到未覆盖状态。"
    )
    conflict_text = conflicts[0] if conflicts else ""
    if not conflict_text or conflict_text.startswith("不同文献报告的结果方向不一致"):
        conflict_text = (
            f"{profile.title}的支持结果与反向/条件依赖结果不能仅按方向归并；"
            f"需要依据{profile.boundary}复核是否为真正反证。"
        )
    reasoning = f"""### 具体机制
{mechanism}

### 实验条件和适用边界
{boundary_text}

### 文献冲突及反向观点
{conflict_text} {counter_cite}

### 公式或数学关系
{formula_text} {formula_boundary}
若用于后续验证，可预注册`{profile.model}`；{profile.model_note}

### 当前不能确定的内容
{unknown_text}

### 科研意义或验证建议
针对{profile.title}，采用{profile.factor_design}，并使用{profile.measurements}核对机制与结果。"""
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
        uncertainty=f"{select_domain_profile(value).title}的结论只覆盖EvidenceBundle中已报告且可比较的条件。",
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
