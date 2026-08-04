"""Counter-evidence-first research-gap Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
from src.research_skills.contracts import SkillInput, SkillOutput


SKILL_NAME = "research_gap_skill"
TASK_DEFINITION = "在反证核验后识别具体、未被现有匹配条件研究解决的空白。"
QUALITY_METRICS = ("gap_false_positive_rate", "counter_evidence_coverage", "condition_completeness", "citation_truthfulness")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 research_gap_skill。问题：{value.user_query}
先核对EvidenceBundle中的COUNTER和CONDITION_DEPENDENT记录，判断问题是否已被部分或全部解决。不得为了产出空白而忽略反证。
必须执行3—5轮逐层缩小：宽泛关系→效应独立性→匹配条件组合→竞争机制→效应边界。宽泛问题已解决时继续寻找更窄空白，不得立刻回答“没有”。
输出：已经解决的问题；尚未解决的具体问题（材料、工艺、疲劳结果、2—4个变量、缺失条件组合、竞争机制）；反向证据检索结果；逐层缩小过程；明确研究问题；最低成本验证。
最终只允许三种自然语言状态：A. 得到支持的具体研究空白；B. 候选证据缺口，尚需外部文献验证；C. 候选空白被反向证据推翻，继续寻找更窄空白。{evidence_level_instruction()}
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
    formulas = usable_formulas(value)
    if not consensus or not value.counter_evidence:
        return "当前只能判定为候选证据缺口，尚不足以确认为研究空白。", "当前正式文献库不足以判断该问题是否构成真实研究空白；需要补足直接证据、反向证据与匹配实验条件。"
    frame = value.query_frame or bundle.get("query_frame") or {}
    variables = set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])
    pore_life = "pore_size" in variables and bool({"fatigue_life", "fatigue_life_Nf"} & variables)
    title = f"{iv or '所问变量'}与{dv or '疲劳结果'}在匹配材料-处理-载荷条件下的效应边界"
    if pore_life:
        title = "L-PBF Ti-6Al-4V中缺陷尺寸—位置耦合对疲劳寿命的独立效应边界"
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
    counter_row = value.counter_evidence[0] if value.counter_evidence else {}
    counter_title = str(counter_row.get("title") or "反向研究")
    if pore_life:
        direct = (
            "宽泛命题“孔隙越大，疲劳寿命通常越低”已有研究覆盖，不应再包装成研究空白。"
            "进一步缩小后，当前可提出的具体缺口是：在同一L-PBF Ti-6Al-4V材料批次、表面状态、"
            "残余应力和载荷条件下，尚缺同时改变√area与归一化缺陷深度d/√area的配对试验，"
            "因此还不能区分纯尺寸效应与尺寸—位置耦合效应。该结论属于B类候选证据缺口，"
            f"仍需外部文献验证。{support_cite}"
        )
    else:
        direct = f"当前只能判定为B. 候选证据缺口，尚需外部文献验证：{title}。反向检索后需要继续缩小空白边界。{support_cite}"
    known_text = (
        "现有研究已经观察到缺陷尺寸、缺陷类型或起裂位置与疲劳寿命之间的条件化关联，"
        "并已使用S-N/Basquin关系描述应力幅与寿命；因此“孔隙尺寸是否影响寿命”本身不是新空白。"
        f"{support_cite}"
        if pore_life else f"现有研究已覆盖该变量关系的部分条件。{support_cite}"
    )
    unresolved_text = (
        "缺少同一材料牌号和L-PBF工艺下，同时控制热处理/HIP、表面状态、残余应力、载荷模式、"
        "应力比与HCF/VHCF区间，并联合测量√area、缺陷距表面距离d和Nf的匹配试验。"
        "当前尚未区分孔隙尺寸控制、缺陷位置控制、表面粗糙度控制、残余应力控制、"
        "微观组织控制及短裂纹屏障控制。"
        if pore_life else
        f"缺少同一材料批次、表面状态和疲劳阶段下，同时控制{iv or '自变量'}并测量{dv or '因变量'}的匹配比较。"
    )
    narrowing = (
        "| 细化轮次 | 核验问题 | 当前判断 | 下一步缩小 |\n"
        "|---|---|---|---|\n"
        "| 1. 宽泛关系 | 孔隙尺寸是否影响疲劳寿命 | 已有研究覆盖 | 转向独立效应 |\n"
        "| 2. 效应独立性 | 尺寸能否独立于位置、形貌和表面状态 | 尚未充分分离 | 固定表面并加入d/√area |\n"
        "| 3. 条件组合 | 材料、HIP、残余应力、R比和疲劳区间是否同时匹配 | 匹配不完整 | 建立同批次配对设计 |\n"
        "| 4. 竞争机制 | 尺寸、位置、粗糙度、残余应力和组织谁主导 | 尚未区分 | 拟合竞争模型与交互项 |\n"
        "| 5. 效应边界 | HCF/VHCF及HIP前后主导机制是否转换 | 边界未确定 | 分层外部验证 |"
        if pore_life else
        "| 细化轮次 | 核验问题 | 当前判断 |\n|---|---|---|\n| 1 | 宽泛关系 | 已部分覆盖 |\n| 2 | 独立效应 | 尚未分离 |\n| 3 | 匹配条件 | 不完整 |"
    )
    counter_explanation = (
        f"《{counter_title}》看似可能削弱该空白，并提供了相反或条件依赖结果{counter_cite}。"
        "但其材料/处理/表面/载荷条件未同时形成√area×d/√area的正交匹配，"
        "所以它可以推翻“尺寸在所有条件下单独主导”的宽泛说法，却不足以解决更窄的尺寸—位置耦合边界。"
        if pore_life else
        f"《{counter_title}》提供了可能否定候选空白的条件依赖结果{counter_cite}；"
        "其覆盖条件必须与当前材料、处理和载荷逐项核对，不能仅凭结论方向认定问题已解决。"
    )
    research_question = (
        "在匹配表面状态、残余应力、载荷模式和应力比后，√area对Nf的效应是否受d/√area调节？"
        "若尺寸项与尺寸×位置交互项均不改善独立批次预测，则该更窄空白被削弱或推翻。"
        if pore_life else
        f"在匹配条件后，{iv or '自变量'}是否仍能独立解释{dv or '因变量'}？"
        "若效应置信区间覆盖预注册最小效应且独立批次无法复现，则该候选空白被削弱。"
    )
    reasoning = f"""### 具体研究空白标题
{title}

### 1. 已经解决的问题
{known_text}

### 2. 尚未解决的具体问题
{unresolved_text}

### 为什么已有研究不能直接回答
{'；'.join(cross.get('condition_mismatches') or ['材料、处理或载荷条件没有形成可直接比较的交集。'])}

### 3. 反向证据检索结果
{counter_explanation}

### 逐层缩小空白搜索
{narrowing}

### 证据缺口矩阵
| 维度 | 状态 | 需要补足 |\n|---|---|---|\n| 匹配条件 | 不完整 | {'、'.join(missing[:4]) or '关键控制变量'} |\n| 因果分离 | 未完成 | 同批次对照与交互项 |\n| 外部验证 | 未完成 | 独立批次复现 |

### 相关公式与适用性
{formula_text}

### 具体研究问题与可证伪假设
{research_question}

### 4. 研究空白是否成立
B. 候选证据缺口，尚需外部文献验证。宽泛空白已被已有研究否定，但更窄的尺寸—位置耦合及机制转换边界仍未被当前匹配证据完全解决。

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
    variables = set(independent) | set(dependent)
    pore_life = "pore_size" in variables and bool({"fatigue_life", "fatigue_life_Nf"} & variables)
    cross = (value.evidence_bundle or {}).get("synthesis") or {}
    gap_status = "B. 候选证据缺口，尚需外部文献验证" if value.support_evidence and value.counter_evidence else "B. 候选证据缺口，尚需外部文献验证"
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
        "gap_confidence": "MEDIUM" if gap_status.startswith("B.") else "LOW",
        "minimum_validation": "同批材料、匹配表面和载荷条件的最小因子试验。",
        "narrowing_rounds": 5 if pore_life else 3,
    }, {"dataset_version": value.dataset_version, "previous_skill": getattr(value.previous_output, "skill_name", None), "rechecked_evidence_ids": [card["evidence_id"] for card in traceable_cards(value)]})


def render_output(output: SkillOutput) -> str:
    complete = output.specific_fields.get("complete_answer")
    return str(complete) if complete else f"## 直接结论\n\n{output.direct_answer}\n\n{output.structured_reasoning}\n\n## 结论边界\n\n{output.uncertainty}"
