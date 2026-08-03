"""Counter-evidence-first research-gap Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "research_gap_skill"
TASK_DEFINITION = "在反证核验后识别具体、未被现有匹配条件研究解决的空白。"
QUALITY_METRICS = ("gap_false_positive_rate", "counter_evidence_coverage", "condition_completeness", "citation_truthfulness")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 research_gap_skill。问题：{value.user_query}
先核对EvidenceBundle中的COUNTER和CONDITION_DEPENDENT记录，判断问题是否已被部分或全部解决。不得为了产出空白而忽略反证。
输出：具体研究空白标题；已有研究解决了什么；尚缺哪个由2-4个变量组成的匹配条件组合；为何现有研究不能直接比较；反证和可能取消空白的证据；证据缺口矩阵；1-3个具体研究问题；可证伪假设；最低成本验证；本地库覆盖不足项。
必须将结论分类为CONFIRMED_GAP、CANDIDATE_GAP、LOCAL_LIBRARY_GAP、ALREADY_ADDRESSED或INSUFFICIENT_EVIDENCE，并用自然中文表达。{evidence_level_instruction()}
正文不得报告证据数量。每项关键判断标 Evidence ID 和页码。用户未问孔隙时不得把孔隙设为主空白。
EvidenceBundle：{bundle_prompt(value)}"""


def build_repair_prompt(value: SkillInput, *, draft: str, failures: list[str]) -> str:
    return f"""修复 research_gap_skill 草稿。失败项：{'；'.join(failures)}。
重新以反证优先核验空白，写清对象、疲劳阶段、核心变量、缺失条件组合和最低成本验证。若现有证据已解决问题，应取消或缩小空白。
不得新增EvidenceBundle之外的引用。草稿：{draft}\nEvidenceBundle：{bundle_prompt(value)}"""


def _fallback(value: SkillInput) -> tuple[str, str]:
    iv, dv = entity_labels(value)
    bundle = value.evidence_bundle or {}
    cross = bundle.get("synthesis") or {}
    consensus = cross.get("consensus") or []
    conflicts = cross.get("conflicts") or []
    missing = cross.get("missing_conditions") or []
    formulas = bundle.get("formulas") or []
    if not consensus or not value.counter_evidence:
        return "当前只能判定为候选证据缺口，尚不足以确认为研究空白。", "当前正式文献库不足以判断该问题是否构成真实研究空白；需要补足直接证据、反向证据与匹配实验条件。"
    title = f"{iv or '所问变量'}与{dv or '疲劳结果'}在匹配材料-处理-载荷条件下的效应边界"
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    formula_text = "现有EvidenceBundle未提供可逐字核验的文献原公式。"
    if formulas:
        formula = formulas[0]
        formula_text = (
            f"{formula['equation']}（Formula ID：{formula['formula_id']}；"
            f"页码：{formula['page_number']}；章节：{formula.get('section') or '未报告'}；"
            f"适用条件：{formula.get('applicable_conditions') or '原文未完整报告'}）"
        )
    direct = f"当前只能判定为候选证据缺口：{title}。现有结果尚不能在同一实验空间内分离变量贡献，且反向检索要求进一步缩小空白边界。{support_cite}"
    reasoning = f"""### 具体研究空白标题
{title}

### 已有研究解决了什么
{consensus[0]}

### 尚未解决的具体问题
缺少在同一材料批次、相同表面状态和相同疲劳阶段下，同时控制{iv or '自变量'}并测量{dv or '因变量'}的匹配比较。

### 为什么已有研究不能直接回答
{'；'.join(cross.get('condition_mismatches') or ['材料、处理或载荷条件没有形成可直接比较的交集。'])}

### 反证检索
{(conflicts[0] + ' ' + counter_cite) if conflicts else '未召回足以取消空白的明确反向结果，但不能据此断言其不存在。'}

### 证据缺口矩阵
| 维度 | 状态 | 需要补足 |\n|---|---|---|\n| 匹配条件 | 不完整 | {'、'.join(missing[:4]) or '关键控制变量'} |\n| 因果分离 | 未完成 | 同批次对照与交互项 |\n| 外部验证 | 未完成 | 独立批次复现 |

### 相关公式与适用性
{formula_text}

### 具体研究问题与可证伪假设
在匹配条件后，{iv or '自变量'}是否仍能独立解释{dv or '因变量'}？若效应置信区间包含预注册最小效应且独立批次无法复现，则该假设被推翻。

### 最低成本验证
复用同批次试样，补测缺失条件，开展小规模配对疲劳试验并盲法判定起裂位置。"""
    return direct, reasoning


def generate(value: SkillInput, synthesis: str = "") -> SkillOutput:
    direct, reasoning = _fallback(value)
    complete = synthesis.strip()
    combined = complete or f"{direct}\n{reasoning}"
    gate = base_quality_gate(value, combined, skill_name=SKILL_NAME)
    gate["false_positive_guard"] = bool(value.counter_evidence)
    gate["passed"] = gate["passed"] and bool(value.support_evidence) and gate["false_positive_guard"]
    independent, dependent = query_variables(value)
    cross = (value.evidence_bundle or {}).get("synthesis") or {}
    gap_status = "CANDIDATE_GAP" if value.support_evidence and value.counter_evidence else "LOCAL_LIBRARY_GAP"
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, "空白只在匹配条件内成立，反证覆盖后必须缩小或取消。", traceable_cards(value), gate, missing_evidence(value), {
        "complete_answer": complete, "quality_metrics": QUALITY_METRICS,
        "gap_title": f"{' × '.join(independent[:2]) or '所问变量'}对{' / '.join(dependent[:1]) or '目标疲劳结果'}的匹配条件缺口",
        "scientific_object": (value.query_frame or {}).get("alloy_grade") or (value.query_frame or {}).get("material"),
        "target_outcome": dependent,
        "core_variables": independent[:4],
        "what_is_known": cross.get("consistent_findings") or [],
        "what_has_been_tested": [paper.get("canonical_id") for paper in (value.evidence_bundle or {}).get("papers") or []],
        "missing_matched_conditions": cross.get("missing_condition_combinations") or [],
        "conflicting_evidence": cross.get("conflicting_findings") or [],
        "counter_evidence_result": "真实反证已核验" if value.counter_evidence else "本地库未检得足够反证",
        "gap_status": gap_status,
        "gap_confidence": "MEDIUM" if gap_status == "CANDIDATE_GAP" else "LOW",
        "minimum_validation": "同批材料、匹配表面和载荷条件的最小因子试验。",
    }, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
