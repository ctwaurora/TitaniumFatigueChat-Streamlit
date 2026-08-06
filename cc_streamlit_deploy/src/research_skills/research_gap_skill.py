"""Counter-evidence-first research-gap Skill."""

from __future__ import annotations

from src.research_skills.common import base_quality_gate, bundle_prompt, entity_labels, evidence_level_instruction, facet_citation, has_direct_interaction_evidence, missing_evidence, primary_citation, query_variables, traceable_cards, usable_formulas
from src.research_skills.contracts import SkillInput, SkillOutput
from src.research_skills.domain_profiles import profile_prompt_block, select_domain_profile


SKILL_NAME = "research_gap_skill"
TASK_DEFINITION = "在反证核验后识别具体、未被现有匹配条件研究解决的空白。"
QUALITY_METRICS = ("gap_false_positive_rate", "counter_evidence_coverage", "condition_completeness", "citation_truthfulness")


def build_prompt(value: SkillInput) -> str:
    return f"""你正在独立执行 research_gap_skill。问题：{value.user_query}
先核对EvidenceBundle中的COUNTER和CONDITION_DEPENDENT记录，判断问题是否已被部分或全部解决。不得为了产出空白而忽略反证。
必须执行3—5轮逐层缩小：宽泛关系→效应独立性→匹配条件组合→竞争机制→效应边界。宽泛问题已解决时继续寻找更窄空白，不得立刻回答“没有”。
输出：已经解决的问题；尚未解决的具体问题（材料、工艺、疲劳结果、2—4个变量、缺失条件组合、竞争机制）；反向证据检索结果；逐层缩小过程；明确研究问题；最低成本验证。
最终只允许三种自然语言状态：A. 得到支持的具体研究空白；B. 候选证据缺口，尚需外部文献验证；C. 候选空白被反向证据推翻，继续寻找更窄空白。{evidence_level_instruction()}
{profile_prompt_block(value)}
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
    frame = value.query_frame or bundle.get("query_frame") or {}
    profile = select_domain_profile(value)
    if profile.key == "residual_microstructure_short_crack":
        interaction_direct = has_direct_interaction_evidence(value)
        residual_cite = facet_citation(value, "residual_stress_evidence_query")
        micro_cite = facet_citation(value, "microstructure_evidence_query")
        status = "coverage_gap" if interaction_direct else "candidate_evidence_gap"
        direct = (
            f"本问题应归为 {status}（B. 候选证据缺口，尚需外部文献验证），"
            "不能仅凭本地库未检得记录就宣称为已确认的新颖研究空白。"
        )
        counter_text = (
            f"检出可比条件下的真正反证，空白必须继续缩小。{primary_citation(value, 'COUNTER')}"
            if value.counter_evidence else
            "没有检出满足同材料、同载荷和明确否定条件的真正反证；这不等于外部文献中不存在反证。"
        )
        alternative_text = (
            f"已检出组织取向、滑移或裂纹路径等替代机制。{primary_citation(value, 'ALTERNATIVE_MECHANISM')}"
            if value.alternative_mechanism_evidence else
            "当前替代机制覆盖仍有限。"
        )
        reasoning = f"""### 具体研究空白标题
同试样匹配条件下残余应力—α/α′组织对短裂纹驱动力及转变判据的联合量化缺口

### 1. 已经解决的问题
现有正式库分别包含短裂纹、残余应力、组织/织构与裂纹路径的相关研究。残余应力侧直接引文：{residual_cite or '当前无满足完整主张的直接引文'}；组织侧直接引文：{micro_cite or '当前无满足L-PBF匹配条件的直接引文'}。

### 2. 尚未解决的具体问题
当前尚未证明在同一L-PBF Ti-6Al-4V试样、同一R与ΔK历史下，σres、α片层宽度lα和织构指标T如何共同解释短裂纹da/dN、ΔKeff以及短—长裂纹转变。
当前缺少的不是宽泛相关性描述，而是同试样、同载荷下的联合测量与独立批次复现。

### 为什么已有研究不能直接回答
分别讨论残余应力或组织的论文不能证明交互项；材料制备、热处理、表面状态、裂纹长度、R和测量方法不匹配时也不能横向合并。

### 3. 真正反证与反向证据检索结果
{counter_text}

### 替代机制核验
{alternative_text} 替代机制用于解释竞争路径，不标为真正反证。

### 逐层缩小空白搜索
| 细化层级 | 核验内容 | 当前判断 |
|---|---|---|
| 宽泛关系 | 残余应力和组织是否分别与裂纹行为有关 | 本地库已有部分覆盖 |
| 条件匹配 | 是否为L-PBF Ti-6Al-4V且表面、R、ΔK历史一致 | 覆盖不完整 |
| 联合测量 | 是否在同试样同步测量σres、lα/T、ΔKeff和da/dN | 未形成可确认覆盖 |
| 转变判据 | 是否给出裂纹长度/组织尺度、闭合建立或长裂纹关系适用点 | 未统一 |
| 外部验证 | 是否在独立制造批次复现 | 尚需验证 |

### 证据缺口矩阵
| 维度 | 已覆盖 | 缺失的匹配比较 |
|---|---|---|
| 残余应力 | 相关背景或单因素研究 | σres—开闭载荷—ΔKeff—da/dN同步链 |
| 微观组织 | 组织/织构与裂纹路径研究 | lα、T与同一裂纹轨迹的定量匹配 |
| 联合作用 | 宽泛共同影响表述 | σres·lα和σres·T交互项 |
| 转变 | 短裂纹和长裂纹概念 | 可复现的转变判据 |

### 具体研究问题与可证伪假设
在表面、R、ΔK历史和裂纹长度匹配后，加入σres·lα与σres·T交互项是否能稳定改善短裂纹da/dN和转变点预测？若交互项在独立批次不显著或方向不能复现，则共同作用假设被削弱。

### 4. 研究空白分类
{status}。需要外部系统检索和专家复核后，才可能升级为confirmed_gap；若找到同条件联合测量研究，则应降为false_gap或更窄的coverage_gap。

### 最低成本验证
先复用同批次试样，统一表面后补测XRD残余应力、EBSD织构/片层尺度与复制法短裂纹轨迹；精确R、ΔK和样本量由直接文献与预实验冻结。"""
        return direct, reasoning
    if not consensus or not value.counter_evidence:
        return "当前只能判定为候选证据缺口，尚不足以确认为研究空白。", "当前正式文献库不足以判断该问题是否构成真实研究空白；需要补足直接证据、反向证据与匹配实验条件。"
    if not frame.get("requested_formulas"):
        formulas = []
    variables = set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])
    subject = str(frame.get("alloy_grade") or "题目限定的钛合金")
    pore_life = "pore_size" in variables and bool({"fatigue_life", "fatigue_life_Nf"} & variables)
    title = f"{profile.title}：{profile.gap_focus}"
    support_cite = primary_citation(value)
    counter_cite = primary_citation(value, "COUNTER")
    formula_text = f"{profile.title}的当前EvidenceBundle未提供可逐字核验的文献原公式；{profile.model_note}"
    if formulas:
        formula = formulas[0]
        formula_text = (
            f"{formula['equation']}（Formula ID：{formula['formula_id']}；"
            f"页码：{formula['page_number']}；章节：{formula.get('section') or '未报告'}；"
            f"适用条件：{formula.get('applicable_conditions') or '原文未完整报告'}）"
        )
    counter_row = value.counter_evidence[0] if value.counter_evidence else {}
    counter_title = str(counter_row.get("title") or "反向研究")
    mismatch_text = (
        f"{profile.title}需要围绕{'、'.join(profile.independent)}与{'、'.join(profile.dependent)}核对条件。"
        f"当前召回研究尚未同时满足以下可比边界：{profile.boundary}"
    )
    direct = (
        f"针对{subject}，已有研究已经覆盖“{profile.title}存在关联”这一宽泛事实，不能把它重新包装为新空白。"
        f"反向检索后，当前只能把“{profile.gap_focus}”列为B. 候选证据缺口，尚需外部文献验证。{support_cite}"
    )
    known_text = f"现有研究已覆盖{profile.direct_claim}{support_cite}"
    unresolved_text = (
        f"尚未解决的是：{profile.gap_focus} 必须联合观测{'、'.join(profile.independent)}与"
        f"{'、'.join(profile.dependent)}，并控制{profile.confounders}。"
    )
    narrowing = (
        "| 细化轮次 | 核验问题 | 当前判断 | 下一步缩小 |\n"
        "|---|---|---|---|\n"
        f"| 1. 宽泛关系 | {profile.title}是否存在 | 已有研究部分覆盖 | 转向独立效应 |\n"
        f"| 2. 效应独立性 | {'、'.join(profile.independent[:2])}能否分离 | 尚未充分分离 | 加入交互项 |\n"
        f"| 3. 条件组合 | {profile.boundary} | 匹配不完整 | 同批次配对 |\n"
        f"| 4. 竞争机制 | {profile.mechanism_chain} | 贡献未拆分 | 同步测量中介量 |\n"
        f"| 5. 效应边界 | {profile.gap_focus} | 边界未确定 | 独立批次验证 |"
    )
    counter_explanation = (
        f"《{counter_title}》为{profile.title}提供相反或条件依赖结果{counter_cite}。"
        f"判断其是否构成直接反证时，要比较{'、'.join(profile.independent)}及{'、'.join(profile.dependent)}的匹配程度；"
        f"该记录可否定{profile.title}的无条件泛化，但仍不足以解决更窄的“{profile.gap_focus}”。"
    )
    research_question = (
        f"在{profile.boundary}的前提下，{'、'.join(profile.independent)}是否能独立解释"
        f"{'、'.join(profile.dependent)}？若{profile.falsifiers[0]}，则该候选空白被削弱。"
    )
    reasoning = f"""### 具体研究空白标题
{title}

### 1. 已经解决的问题
{known_text}

### 2. 尚未解决的具体问题
{unresolved_text}

缺少的匹配条件包括：{profile.boundary}

### 为什么已有研究不能直接回答
{mismatch_text}

### 3. 反向证据检索结果
{counter_explanation}

### 逐层缩小空白搜索
{narrowing}

### 证据缺口矩阵
| 维度 | 状态 | 需要补足 |\n|---|---|---|\n| {profile.title}匹配条件 | 不完整 | {'、'.join(missing[:4]) or profile.boundary} |\n| {'×'.join(profile.independent[:2])}因果分离 | 未完成 | {profile.factor_design} |\n| {profile.title}外部验证 | 未完成 | 以独立批次复现{'、'.join(profile.dependent)} |

### 相关公式与适用性
{formula_text}

### 具体研究问题与可证伪假设
{research_question}

### 4. 研究空白是否成立
B. 候选证据缺口，尚需外部文献验证。宽泛关联已被已有研究覆盖，但更窄的“{profile.gap_focus}”未被当前匹配证据完全解决。

### 最低成本验证
{profile.factor_design} 最低成本时优先检验{'×'.join(profile.independent[:2])}，同步记录{'、'.join(profile.dependent)}并盲法判定终点。"""
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
    profile = select_domain_profile(value)
    return SkillOutput(SKILL_NAME, complete or direct, "" if complete else reasoning, f"{profile.title}的空白只在匹配条件内成立；反证覆盖后必须缩小或取消。", traceable_cards(value), gate, missing_evidence(value), {
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
