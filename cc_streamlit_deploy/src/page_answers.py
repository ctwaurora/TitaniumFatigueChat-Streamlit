"""
page_answers.py — 页面专属回答生成器

每个页面类型有自己的回答模板，不再使用统一 10 板块骨架。
页面优先级高于意图分类。
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variable_mapper import extract_variable_pair


def normalize_claim(text: str) -> str:
    """标准化证据文本用于去重。"""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def dedup_evidence(rows: list, max_count: int = 5) -> list:
    """证据去重：同一 paper_id + 相似 claim 只保留一条。"""
    seen = set()
    result = []
    for row in rows:
        paper_id = str(row.get("paper_id", row.get("evidence_id", "")))
        claim = normalize_claim(str(row.get("extracted_claim", row.get("claim", "")) or ""))
        cond = normalize_claim(str(row.get("condition_summary", row.get("surface_state", "")) or ""))
        # 去重键
        key = f"{paper_id}|{claim[:60]}|{cond[:30]}"
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result[:max_count]
from src.condition_mechanism_map import generate_condition_mechanism_map, format_condition_map_markdown
from src.formula_comparison import compare_candidate_models, format_model_comparison_markdown
from src.research_gap_discovery import search_counter_evidence, format_counter_evidence_markdown
from src.equation_engine import extract_variables_from_query
from src.data_cache import load_evidence_snippets


# ═══════════════════════════════════════════════════════════════════════
# 路由核心：页面优先 + 用户意图补充
# ═══════════════════════════════════════════════════════════════════════

def route_research_task(question: str, page_context: str) -> Dict[str, Any]:
    """
    路由决策：页面优先，用户意图补充。

    Args:
        question: 用户问题
        page_context: 当前页面标识 (research_analysis / experiment_design / research_gap / formula_explain / hypothesis_generation)

    Returns:
        {
            "primary_task": str,       # 主要任务
            "secondary_tasks": list,   # 次要任务（最多 1-2 个）
            "full_report": bool,       # 是否全面分析
        }
    """
    q = question.lower()

    # 检查是否明确要求全面分析
    full_triggers = ["全面分析", "完整分析", "系统分析", "全面", "完整报告",
                     "full analysis", "complete analysis", "all modules",
                     "全部模块"]
    if any(t in q for t in full_triggers):
        return {"primary_task": "full_report", "secondary_tasks": [], "full_report": True}

    # 从问题中检测次要意图
    secondary = []
    if any(kw in q for kw in ["机制", "机理", "competition", "哪个.*更", "vs", "versus"]):
        secondary.append("mechanism")
    if any(kw in q for kw in ["实验", "验证", "方案", "怎么.*做", "design", "verify"]):
        secondary.append("experiment")
    if any(kw in q for kw in ["空白", "gap", "missing", "还缺", "不足"]):
        secondary.append("gap")
    if any(kw in q for kw in ["公式", "模型", "公式", "paris", "murakami", "equation"]):
        secondary.append("formula")
    if any(kw in q for kw in ["反证", "反驳", "counter", "争议", "相反"]):
        secondary.append("counter")

    # 页面优先决定主要任务
    page_task_map = {
        "experiment_design": "experiment_design",
        "research_gap": "research_gap",
        "formula_explain": "formula_explanation",
        "hypothesis_generation": "hypothesis_generation",
        "research_analysis": "research_analysis",
    }
    primary = page_task_map.get(page_context, "research_analysis")

    # 过滤不合理的 secondary（页面自身已经有对应内容时不去重加）
    # 实验设计页面可以补充机制说明
    # 公式页面不补充实验设计
    if primary == "formula_explanation":
        secondary = [s for s in secondary if s not in ("experiment", "gap")]
    if primary == "research_gap":
        secondary = [s for s in secondary if s not in ("experiment", "formula")]
    if primary == "experiment_design":
        # 可以保留 mechanism 用于说明待验证机制
        secondary = [s for s in secondary if s not in ("experiment",)]

    return {
        "primary_task": primary,
        "secondary_tasks": secondary[:2],
        "full_report": False,
    }


# ═══════════════════════════════════════════════════════════════════════
# A. 实验验证设计页面
# ═══════════════════════════════════════════════════════════════════════

def answer_experiment_design(question: str, ind_var: str, dep_var: str) -> str:
    iv = (ind_var or "").lower()
    dv = (dep_var or "").lower()
    lines = ["## 实验验证方案\n"]

    # 1. 待验证假设（简短）
    lines.append("### 1. 待验证假设\n")
    if "pore" in iv or "defect" in iv:
        lines.append(f"在 polished + SR L-PBF Ti-6Al-4V 中，近表面大孔隙（{ind_var} 增大且 distance_to_surface < 100μm）比深部同等尺寸孔隙更可能成为疲劳裂纹起裂源，导致 Nf 降低。\n\n")
    elif "roughness" in iv or "surface" in iv:
        lines.append(f"as-built 表面（Ra>10μm）由表面缺口主导起裂；polished 表面（Ra<1μm）由内部孔隙主导起裂。{ind_var} 对 {dep_var} 的影响受 surface_state 调节。\n\n")
    else:
        lines.append(f"{ind_var or '目标变量'} 对 {dep_var or '疲劳指标'} 的影响需通过控制变量实验验证。\n\n")

    # 2. 实验目标
    lines.append("### 2. 实验目标\n")
    lines.append(f"验证 {ind_var or '自变量'} 对 {dep_var or '因变量'} 的影响，确定变量关系及其条件边界。\n\n")

    # 3. 变量定义表
    lines.append("### 3. 变量定义\n")
    lines.append("| 变量类型 | 具体变量 | 测量方式 | 说明 |\n")
    lines.append("|---|---|---|---|\n")
    if "pore" in iv:
        lines.append(f"| 自变量 (IV) | {ind_var} / √area | micro-CT | 核心缺陷特征 |\n")
        lines.append(f"| 自变量 (IV) | distance_to_surface | micro-CT 三维定位 | 区分近表面/内部 |\n")
    elif "roughness" in iv:
        lines.append(f"| 自变量 (IV) | {ind_var} / Rz | 3D profiler | 核心自变量 |\n")
        lines.append("| 自变量 (IV) | surface_state | 分组控制 | as-built / polished / machined |\n")
    else:
        lines.append(f"| 自变量 (IV) | {ind_var or '目标变量'} | 待确定 | — |\n")
    lines.append(f"| 因变量 (DV) | {dep_var or '疲劳寿命 Nf / 起裂位置'} | 疲劳试验 / SEM | 主要输出 |\n")
    lines.append("| 控制变量 (CV) | Ti-6Al-4V Grade 23 | 供应商认证 | 所有组保持一致 |\n")
    lines.append("| 控制变量 (CV) | L-PBF 工艺参数 | 工艺记录 | — |\n")
    lines.append("| 控制变量 (CV) | stress_ratio R | 试验设定 | R=0.1 |\n")
    lines.append("| 调节变量 (MV) | surface_state / heat_treatment | 分组控制 | 评估条件依赖 |\n\n")

    # 4. 样品分组
    lines.append("### 4. 样品分组\n")
    if "pore" in iv:
        lines.append("| 组别 | 表面 | 热处理 | 孔隙特征 | 说明 |\n")
        lines.append("|------|------|--------|---------|------|\n")
        lines.append("| A | polished | SR | 近表面大孔隙 | 验证距离效应 |\n")
        lines.append("| B | polished | SR | 深部大孔隙 | 对照 |\n")
        lines.append("| C | polished | SR | 低孔隙 | 基准 |\n")
        lines.append("| D | as-built | SR | 自然分布 | 评估粗糙度竞争 |\n\n")
        lines.append("> ⚠️ 组 D 与 A-C 比较时表面粗糙度与孔隙同时变化，存在混杂变量！\n\n")
    elif "roughness" in iv:
        lines.append("| 组别 | 表面状态 | 热处理 | Ra 目标 | 说明 |\n")
        lines.append("|------|---------|-------|--------|------|\n")
        lines.append("| A | as-built | SR | 10-15μm | 高粗糙度基准 |\n")
        lines.append("| B | polished | SR | <1μm | 去除表面缺口 |\n")
        lines.append("| C | machined | SR | 1-3μm | 中等粗糙度 |\n\n")
        lines.append("> ⚠️ 如需评估 HIP 效果，应避免同时改变表面状态和热处理。\n\n")
    else:
        lines.append("| 组别 | 条件 | 说明 |\n")
        lines.append("|------|------|------|\n")
        lines.append("| A | 基准组 | 标准工艺 |\n")
        lines.append("| B | 实验组 | 改变目标变量 |\n\n")

    # 5. 实验条件
    lines.append("### 5. 实验条件\n")
    lines.append("- 材料: Ti-6Al-4V Grade 23\n")
    lines.append("- 工艺: L-PBF\n")
    lines.append("- 加载: 轴向拉-拉, R=0.1\n")
    lines.append("- 温度: 室温 (25°C)\n")
    lines.append("- 频率: 15-20 Hz\n\n")

    # 6. 测试与表征
    lines.append("### 6. 测试与表征方法\n")
    if "pore" in iv:
        lines.append("- micro-CT: 疲劳前扫描，提取孔隙三维特征（√area、位置、长宽比）\n")
        lines.append("- HCF 试验: 多应力水平（至少 4 级），每级 3-5 试样\n")
        lines.append("- SEM fractography: 确认起裂源与 micro-CT 定位孔隙的对应关系\n")
        lines.append("- EBSD（可选）: 起裂源周围晶粒取向分析\n\n")
    elif "roughness" in iv:
        lines.append("- 3D 光学 profiler / 接触式轮廓仪: Ra/Rz 测量\n")
        lines.append("- HCF 试验: 多应力水平（至少 4 级）\n")
        lines.append("- SEM fractography: 确认起裂源类型（表面粗糙峰 vs 内部孔隙）\n")
        lines.append("- micro-CT（辅助）: 排除内部孔隙差异\n\n")
    else:
        lines.append("- HCF 试验: 多应力水平\n")
        lines.append("- SEM: 断口分析\n")
        lines.append("- micro-CT: 缺陷表征\n\n")

    # 7. 预测方向
    lines.append("### 7. 预测方向\n")
    if "pore" in iv:
        lines.append("1. 近表面大孔隙组 Nf 低于深部大孔隙组 (p<0.05)\n")
        lines.append("2. SEM 起裂源对应 micro-CT 近表面孔隙（对应率 >70%）\n")
        lines.append("3. polished 后 pore_size / distance_to_surface 对 Nf 的解释力增强\n")
        lines.append("4. Nf 与 √area 负相关，与 distance_to_surface 正相关\n\n")
    elif "roughness" in iv:
        lines.append("1. as-built 试样表面起裂率 >80%，抛光试样 <30%\n")
        lines.append("2. 抛光后 Nf 提升 2-10 倍\n")
        lines.append("3. Ra/Rz 与 Nf 负相关 (|r|>0.6)\n\n")

    # 8. 支持判据
    lines.append("### 8. 支持判据\n")
    lines.append("- 统计显著: p<0.05\n")
    lines.append("- 效应量: Cohen's d>0.8\n")
    lines.append("- SEM/micro-CT 表征与预测机制一致\n")
    lines.append("- Murakami/Paris 模型预测与实测偏差 <20%\n")
    lines.append("- 同条件变异系数 <15%\n\n")

    # 9. 推翻判据
    lines.append("### 9. 推翻判据\n")
    if "pore" in iv:
        lines.append("1. 起裂源与孔隙对应率 <30% → 推翻 H1\n")
        lines.append("2. Nf 方差主要由 Ra/Rz 解释 (R²>0.5) → 降级\n")
        lines.append("3. 近表面组与深部组 Nf 无显著差异 (p>0.05) → 推翻\n")
        lines.append("4. 所有组起裂源均为表面特征 → 推翻\n\n")
    elif "roughness" in iv:
        lines.append("1. 抛光后 Nf 提升 <20% → 推翻 H1\n")
        lines.append("2. as-built 和 polished 组起裂源均为孔隙 → 推翻\n")
        lines.append("3. Ra/Rz 与 Nf 无稳定相关 (|r|<0.3) → 推翻\n\n")

    # 10. 数据分析
    lines.append("### 10. 数据分析方法\n")
    if "pore" in iv:
        lines.append("- S-N 曲线分组拟合: log(Nf) = A - B·log(σa)\n")
        lines.append("- Murakami √area 验证: σw = C·(HV+120)/(√area)^{1/6}\n")
        lines.append("- 多因素回归: Nf = β₀ + β₁·log(√area) + β₂·d + β₃·(√area×d)\n")
        lines.append("- 断口统计分析: 起裂源类型分布\n\n")
    elif "roughness" in iv:
        lines.append("- S-N 分组拟合: as-built vs polished vs machined\n")
        lines.append("- 回归分析: Nf vs Ra/Rz + pore_size\n")
        lines.append("- 断口起裂源分布统计\n\n")

    # 11. 混杂变量
    lines.append("### 11. 潜在混杂变量\n")
    if "pore" in iv:
        lines.append("- 表面粗糙度: as-built 试样中表面缺口可能掩盖孔隙效应\n")
        lines.append("- 残余应力: L-PBF 工艺固有，影响裂纹闭合\n")
        lines.append("- 微观组织差异: 不同打印位置组织不均匀\n\n")
    elif "roughness" in iv:
        lines.append("- 内部孔隙: 表面改善后孔隙成为主要竞争因素\n")
        lines.append("- 残余应力: 去除表面层时残余应力可能释放\n\n")

    # 12. 需要补充的数据
    lines.append("### 12. 当前缺少的关键数据\n")
    if "pore" in iv:
        lines.append("- 同时包含 √area + distance_to_surface + Nf 的结构化配对数据\n")
        lines.append("- 系统控制表面状态的 pore-Nf 对比实验\n")
    elif "roughness" in iv:
        lines.append("- 同时包含 Ra/Rz + pore_size + Nf 的系统对比数据\n")
        lines.append("- 临界 Ra 阈值的定量实验数据\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# B. 研究空白页面
# ═══════════════════════════════════════════════════════════════════════

def answer_research_gap(question: str, ind_var: str, dep_var: str) -> str:
    lines = ["## 研究空白分析\n"]

    # 1. 已知结论
    lines.append("### 1. 已知结论\n")
    if "pore" in question.lower() or "defect" in question.lower():
        lines.append("- 孔隙缺陷降低 L-PBF Ti-6Al-4V 疲劳寿命，Murakami √area 模型可预测疲劳极限\n")
        lines.append("- 近表面孔隙比深部孔隙更危险（自由表面应力场叠加）\n")
        lines.append("- as-built 表面粗糙度（Ra>10μm）可掩盖内部孔隙效应\n")
        lines.append("- HIP 可闭合大部分内部孔隙，但表面连通孔隙可能残留\n\n")
    elif "roughness" in question.lower() or "surface" in question.lower():
        lines.append("- as-built 表面粗糙度（Ra 10-20μm）通过表面缺口效应降低 Nf\n")
        lines.append("- 抛光（Ra<1μm）后疲劳寿命可提升 2-10 倍\n")
        lines.append("- 表面粗糙度与内部孔隙存在竞争主导关系\n\n")
    else:
        lines.append("- 当前文献已有部分关于变量关系的独立研究，但缺乏系统整合\n\n")

    # 2. 支持证据（已去重）
    lines.append("### 2. 已有支持证据\n")
    ev_df = load_evidence_snippets()
    q = question.lower()
    keywords = [kw for kw in ["pore", "roughness", "surface", "defect", "孔隙", "粗糙", "表面"] if kw in q]
    if not ev_df.empty and keywords:
        matched = []
        for _, row in ev_df.iterrows():
            claim = str(row.get("extracted_claim", "") or "").lower()
            if any(kw in claim for kw in keywords):
                matched.append(row)
        # 去重
        matched = dedup_evidence(matched, max_count=4)
        if matched:
            for row in matched:
                lines.append(f"- {str(row.get('extracted_claim', '') or '')[:120]}\n")
            if len(matched) < 3:
                lines.append(f"\n> 当前仅检索到 {len(matched)} 条有效证据。\n")
        else:
            lines.append("- 本地文献库中无直接匹配的证据片段\n\n")
    else:
        lines.append("- 本地文献库中无直接匹配的证据片段\n\n")

    # 3. 反向证据
    lines.append("### 3. 反向证据\n")
    try:
        hyp_text = f"{ind_var or ''} affects {dep_var or ''} in L-PBF Ti-6Al-4V"
        result = search_counter_evidence(hypothesis=hyp_text)
        counter = result.get("counter_evidence", [])
        if counter:
            for ev in counter[:3]:
                lines.append(f"- {ev}\n")
        else:
            lines.append("- 当前文献库未检索到直接反向证据\n")
            lines.append("- ⚠️ 未检索到反向证据 ≠ 该假设已被证实\n\n")
    except Exception:
        lines.append("- 反向证据检索暂不可用\n\n")

    # 4. 条件依赖证据
    lines.append("### 4. 条件依赖证据\n")
    if "pore" in q:
        lines.append("- polished 条件下孔隙效应显著；as-built 条件下表面粗糙度掩盖孔隙效应\n")
        lines.append("- 高应力幅下孔隙效应相对减弱；低应力幅（长寿命）下孔隙效应更突出\n")
        lines.append("- VHCF 区（>10⁷ cycles）内部孔隙可能通过鱼眼机制主导起裂\n\n")
    elif "roughness" in q:
        lines.append("- as-built 状态表面缺口主导；polished/machined 状态孔隙转为主导\n")
        lines.append("- 存在临界 Ra（约 3-10μm），主导权随 Ra 降低切换\n\n")

    # 5. 缺失证据
    lines.append("### 5. 缺失的关键证据\n")
    if "pore" in q:
        lines.append("- 同时包含 √area + distance_to_surface + surface_state + Nf 的系统数据\n")
        lines.append("- 近表面孔隙 vs 深部孔隙的定量对比（控制其他变量后）\n")
        lines.append("- HIP 后残留缺陷的尺寸/位置分布与疲劳寿命的定量关系\n\n")
    elif "roughness" in q:
        lines.append("- 同时包含 Ra/Rz + pore_size + Nf + crack_initiation_site 的系统对比数据\n")
        lines.append("- 不同 Ra 等级下的起裂源分布统计\n")
        lines.append("- 临界 Ra 值的定量测定数据\n\n")

    # 6. 为什么是 Gap
    lines.append("### 6. 研究价值\n")
    if "pore" in q:
        lines.append("- 确定孔隙尺寸和距表面距离的耦合效应有助于建立更准确的缺陷容限判据\n")
        lines.append("- 区分表面粗糙度和孔隙的竞争主导关系可指导后处理工艺选择\n\n")
    elif "roughness" in q:
        lines.append("- 明确表面粗糙度 vs 内部孔隙的主导权切换条件\n")
        lines.append("- 为 as-built vs machined vs polished 的工艺选择提供量化依据\n\n")

    # 7. 候选假设
    lines.append("### 7. 候选假设\n")
    if "pore" in q:
        lines.append("**H1 (D* 调控起裂风险)**: D* = distance_to_surface / √area 越小，孔隙越可能成为起裂源。\n")
        lines.append("**H2 (粗糙度掩盖)**: as-built 表面粗糙度（Ra>10μm）掩盖内部孔隙对 Nf 的解释力。\n\n")
    elif "roughness" in q:
        lines.append("**H1 (竞争主导)**: as-built 表面缺口主导起裂；polished 后内部孔隙转为主导。\n\n")

    # 8. 优先级评分
    lines.append("### 8. 优先级评分\n")
    lines.append("| 维度 | 评分 |\n|---|---|\n")
    lines.append("| 科学价值 | 17/20 |\n")
    lines.append("| 新颖性 | 12/15 |\n")
    lines.append("| 可验证性 | 16/20 |\n")
    lines.append("| 工程意义 | 14/15 |\n\n")

    # 9. 简要验证路径
    lines.append("### 9. 简要验证路径\n")
    lines.append("micro-CT（疲劳前表征孔隙）→ 控制表面状态（as-built / polished）→ HCF 试验 → SEM 断口确认起裂源 → 统计分析。\n\n")

    # 10. 证据不足说明
    lines.append("### 10. 本地证据说明\n")
    lines.append("> 当前本地文献库规模有限，以上分析为 search-guided candidate。实际研究空白需结合完整文献检索和领域专家判断。\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# C. 公式模型解释页面
# ═══════════════════════════════════════════════════════════════════════

def answer_formula_explanation(question: str, ind_var: str, dep_var: str) -> str:
    lines = ["## 公式模型解释\n"]

    # 1. 当前问题涉及的变量
    lines.append("### 1. 问题与变量\n")
    lines.append(f"**问题**: {question}\n")
    lines.append(f"**自变量 (IV)**: {ind_var or '待识别'}\n")
    lines.append(f"**因变量 (DV)**: {dep_var or '待识别'}\n\n")

    # 2-8. 模型对比
    iv = (ind_var or "").lower()
    dv = (dep_var or "").lower()
    has_pore = "pore" in iv or "defect" in iv or "area" in iv
    has_delta = "delta" in iv or "crack" in iv or "da" in dv or "dk" in iv
    has_roughness = "roughness" in iv or "surface" in iv

    # 尝试调用正式模型对比
    try:
        detected = extract_variables_from_query(question)
        if not detected:
            detected = [v for v in [ind_var, dep_var] if v]
        if detected:
            result = compare_candidate_models(question=question, detected_variables=detected)
            if result.get("comparisons"):
                lines.append("### 2. 候选模型比较\n")
                lines.append(format_model_comparison_markdown(result))
                lines.append("\n### 3. 模型选择建议\n")
                lines.append("根据问题涉及的变量关系：\n")
                if has_pore:
                    lines.append("- **优先推荐**: Murakami √area（预测疲劳极限）+ S-N/Basquin（完整寿命曲线）\n")
                    lines.append("- **补充**: Kitagawa-Takahashi（缺陷容限分析）\n")
                    lines.append("- **不优先**: Paris law（除非关注裂纹扩展）\n\n")
                elif has_delta:
                    lines.append("- **优先推荐**: Paris law（da/dN = C·ΔKᵐ）+ Walker（多 R 比修正）\n")
                    lines.append("- **补充**: Kitagawa-Takahashi（门槛值预测）\n\n")
                elif has_roughness:
                    lines.append("- **优先推荐**: S-N/Basquin + surface correction\n")
                    lines.append("- **补充**: Murakami 表面等效缺陷方法\n\n")
                return "".join(lines)
    except Exception:
        pass

    # 回退：手动构建相关模型表
    lines.append("### 2. 候选模型比较\n\n")
    lines.append("| 模型 | 适用关系 | 输入参数 | 适用条件 | 推荐等级 |\n")
    lines.append("|---|---|---|---|---|\n")
    if has_pore:
        lines.append("| S-N / Basquin | σa → Nf | σa, Nf | 固定 R, HCF | 🟢 推荐 |\n")
        lines.append("| Murakami √area | √area → σw | HV, √area | 含缺陷材料 | 🟢 推荐 |\n")
        lines.append("| Kitagawa-Takahashi | a → Δσth | ΔKth, a | 缺陷容限 | 🟡 条件 |\n")
        lines.append("| El Haddad | 小裂纹修正 | ΔKth, σw, a | 小缺陷区 | 🟡 条件 |\n")
    if has_delta:
        lines.append("| Paris law | ΔK → da/dN | ΔK, C, m | Region II | 🟢 推荐 |\n")
        lines.append("| Walker | ΔK+R → da/dN | ΔK, R, C, m, p | 多 R 比 | 🟡 条件 |\n")
    if has_roughness:
        lines.append("| S-N / Basquin | σa → Nf | σa, Nf | 固定 R | 🟢 推荐 |\n")
        lines.append("| Murakami (表面) | Ra → σw | HV, 等效√area | 表面缺陷 | 🟡 条件 |\n")

    lines.append("\n### 3. 参数说明\n")
    if has_pore:
        lines.append("- **Murakami公式**: σw = C·(HV+120)/(√area)^{1/6}，C 取决于缺陷位置（表面 1.43 / 近表面 1.41 / 内部 1.56）\n")
        lines.append("- **Basquin公式**: σa = σf'·(2Nf)ᵇ，σf' 为疲劳强度系数，b 为疲劳强度指数\n\n")
    if has_delta:
        lines.append("- **Paris公式**: da/dN = C·(ΔK)ᵐ，C 和 m 为材料参数，缺陷增大 C，组织影响 m\n")
        lines.append("- **Walker修正**: da/dN = C·(ΔK·(1-R)^{p-1})ᵐ，p 为 Walker 参数\n\n")

    lines.append("### 4. 当前数据是否足够\n")
    lines.append("| 数据 | 状态 |\n")
    lines.append("|---|---|\n")
    if has_pore:
        lines.append("| Nf 数据 | 有限 |\n| HV 硬度 | 未提取 |\n| √area 数据 | 有限 |\n| 疲劳极限 σw | 有限 |\n\n")
    elif has_delta:
        lines.append("| Paris C/m | 有限 |\n| ΔKth | 有限 |\n| 多 R 比数据 | 缺少 |\n\n")

    lines.append("### 5. 需要补充的参数\n")
    if has_pore:
        lines.append("- 缺陷尺寸 √area（micro-CT 或断口测量）\n")
        lines.append("- 维氏硬度 HV\n")
        lines.append("- 升降法测定的疲劳极限 σw\n\n")
    elif has_delta:
        lines.append("- 至少 3 个试样的 FCGR 数据\n")
        lines.append("- Paris C 和 m 拟合值\n")
        lines.append("- 至少 2 个 R 比下的数据（用于 Walker 模型）\n\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# D. 科研分析页面（围绕问题组织内容）
# ═══════════════════════════════════════════════════════════════════════

def answer_research_analysis(question: str, ind_var: str, dep_var: str) -> str:
    q = question.lower()
    lines = ["## 科研分析\n"]

    # 1. 问题理解
    lines.append("### 1. 问题理解\n")
    lines.append(f"**问题**: {question}\n")
    lines.append(f"**自变量 (IV)**: {ind_var or '待识别'}\n")
    lines.append(f"**因变量 (DV)**: {dep_var or '待识别'}\n\n")

    # 2. 直接结论
    lines.append("### 2. 直接结论\n")
    iv = (ind_var or "").lower()
    dv = (dep_var or "").lower()
    if "pore" in iv and "fatigue_life" in dv:
        lines.append("孔隙尺寸 √area 增大 → Nf 降低。该关系受 distance_to_surface 和 surface_state 调节。近表面孔隙（d<100μm）比深部孔隙更危险，as-built 表面粗糙度可掩盖该效应。\n\n")
    elif "roughness" in iv and "fatigue_life" in dv:
        lines.append("表面粗糙度 Ra/Rz 增大 → Nf 降低。该关系受 surface_state 调节。as-built (Ra>10μm) 时表面缺口主导，polished (Ra<1μm) 时内部孔隙转为为导。结论条件依赖。\n\n")
    elif "delta" in iv or "da" in dv:
        lines.append("ΔK 增大 → da/dN 增大（Paris 定律）。高 R 比增大 da/dN。缺陷增大 Paris C，组织影响 Paris m。\n\n")
    else:
        lines.append(f"{ind_var or '自变量'} 对 {dep_var or '因变量'} 的影响受实验条件调节，需结合具体条件判断。\n\n")

    # 3. 条件化证据（已去重）
    lines.append("### 3. 条件化证据\n")
    ev_df = load_evidence_snippets()
    keywords = [kw for kw in ["pore", "roughness", "surface", "defect", "crack", "paris", "da/dn",
                               "孔隙", "粗糙", "表面", "缺陷", "裂纹", "疲劳", "距离"] if kw in q]
    if not ev_df.empty and keywords:
        matched = []
        for _, row in ev_df.iterrows():
            text = str(row.get("extracted_claim", "") or "").lower()
            if any(kw in text for kw in keywords):
                matched.append(row)
        matched = dedup_evidence(matched, max_count=5)
        if matched:
            for row in matched:
                cond_parts = []
                for f in ["material", "surface_state", "heat_treatment", "stress_ratio_R"]:
                    v = str(row.get(f, "") or "").strip()
                    if v:
                        cond_parts.append(f"{f}={v}")
                cond_str = "; ".join(cond_parts) if cond_parts else "条件未提取"
                claim = str(row.get("extracted_claim", "") or "")[:100]
                lines.append(f"- {claim} | 条件: {cond_str}\n")
            if len(matched) < 2:
                lines.append(f"> 当前仅检索到 {len(matched)} 条有效条件化证据。\n")
        else:
            lines.append("- 本地文献库中无直接匹配证据\n")
    else:
        lines.append("- 本地文献库中无直接匹配证据\n")
    lines.append("\n")

    # 4. 机制分析（根据问题类型选择深度）
    lines.append("### 4. 机制分析\n")
    if "pore" in iv or "defect" in iv:
        lines.append("**机制链**:\n")
        lines.append("```\npore / defect (√area ↑) → 局部应力集中 (Kt ↑)\n  → 裂纹在孔隙边缘起裂\n  → 近表面时自由表面应力场叠加，起裂驱动力增强\n  → 表面粗糙度高时表面缺口效应掩盖孔隙效应\n```\n\n")
    elif "roughness" in iv:
        lines.append("**机制链**:\n")
        lines.append("```\nas-built 表面 (Ra ↑) → 表面粗糙峰根部应力集中 (Kt≈2-3)\n  → 表面裂纹优先起裂 (vs 内部孔隙)\n  → 表面改善后 (Ra ↓) → 内部孔隙转为主导\n```\n\n")
    elif "delta" in iv or "da" in dv:
        lines.append("**机制链**:\n")
        lines.append("```\nΔK ↑ → 裂纹尖端塑性区尺寸 ↑ → 损伤累积速率 ↑ → da/dN ↑\n  缺陷 (孔隙) → C ↑ (da/dN-ΔK 曲线整体上移)\n  组织 (α lath) → m ↑ (曲线斜率增大)\n```\n\n")

    # 5. 多机制竞争（如果问题涉及比较）
    if any(kw in q for kw in ["哪个", "竞争", "vs", "versus", "主导", "还是", "或"]):
        lines.append("### 5. 多机制竞争\n")
        try:
            cm_map = generate_condition_mechanism_map(question=q)
            if cm_map.get("entries"):
                lines.append(format_condition_map_markdown(cm_map))
                lines.append("\n")
        except Exception:
            pass

    # 6. 候选拟合模型（如果问题涉及变量关系）
    has_candidate_vars = ("pore" in iv or "roughness" in iv or "defect" in iv or "distance" in iv or "fatigue" in q or "nf" in q)
    if has_candidate_vars:
        lines.append("### 6. 候选拟合模型\n")
        lines.append(
            "> **该公式是系统提出的待验证候选模型，不是当前文献已经证明的普适疲劳定律。**\n\n"
        )
        if "pore" in iv or "defect" in iv:
            lines.append("```text\n")
            lines.append("D* = distance_to_surface / sqrt_area\n\n")
            lines.append(
                "log10(Nf) = β0 - β1 log10(stress_amplitude) "
                "- β2 log10(sqrt_area) + β3 D* + β4 surface_state "
                "+ β5(D* × surface_state) + β6 stress_ratio_R + ε\n"
            )
            lines.append("```\n\n")
            lines.append("**建议模型比较**:\n")
            lines.append("| 模型 | 变量 |\n")
            lines.append("|---|---|\n")
            lines.append("| Model A | stress_amplitude |\n")
            lines.append("| Model B | stress_amplitude + sqrt_area |\n")
            lines.append("| Model C | stress_amplitude + sqrt_area + D* |\n")
            lines.append("| Model D | stress_amplitude + sqrt_area + D* + surface_state + D*×surface_state |\n\n")
            lines.append(
                "**评价方法**: 系数置信区间、交叉验证误差、adjusted R²、AIC、BIC，"
                "以及独立论文或留一文献验证。\n\n"
            )
            lines.append(
                "**推翻条件**: D* 和交互项系数无法与 0 区分；加入 D* 后预测误差没有稳定改善；"
                "效应方向无法重复；或表面状态、缺陷尺寸和应力幅已经能够解释结果。\n\n"
            )
            lines.append(
                "> 当前数据不足以拟合：不生成 β 系数和性能指标，只显示模型结构。\n\n"
            )
        elif "roughness" in iv:
            lines.append("基于当前变量关系，提出以下候选模型：\n\n")
            lines.append("**log(Nf) = β₀ + β₁·log(Ra) + β₂·log(√area_max) + β₃·I_polished + β₄·log(Ra)×log(√area_max) + ε**\n\n")
            lines.append("预期 as-built 组 β₁ 显著 β₂ 不显著，polished 组 β₂ 显著 β₁ 不显著。\n\n")

    # 7. 证据缺口
    lines.append("### 7. 证据缺口\n")
    if "pore" in iv or "defect" in iv:
        lines.append("- 缺少同时包含 √area + distance_to_surface + Nf 的结构化配对数据\n")
        lines.append("- 缺少系统控制表面状态的 pore-Nf 对比实验\n")
    elif "roughness" in iv:
        lines.append("- 缺少同时包含 Ra/Rz + pore_size + Nf 的系统对比数据\n")
        lines.append("- 缺少临界 Ra 阈值的定量实验数据\n")
    elif "delta" in iv or "da" in dv:
        lines.append("- 缺少不同缺陷状态下的 Paris C/m 系统比较\n")
    else:
        lines.append("- 缺少目标变量对的直接实验数据\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 主分发函数
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# E. 假设生成页面
# ═══════════════════════════════════════════════════════════════════════

def answer_hypothesis_generation(question: str, ind_var: str, dep_var: str) -> str:
    """Generate one complete, auditable candidate without fixed conditions."""
    from src.hypothesis_candidate import build_candidate_hypothesis_markdown

    return "## 候选科学假设\n\n" + build_candidate_hypothesis_markdown(
        question=question,
        ind_var=ind_var,
        dep_var=dep_var,
    )


PAGE_ANSWER_MAP = {
    "experiment_design": answer_experiment_design,
    "research_gap": answer_research_gap,
    "formula_explain": answer_formula_explanation,
    "research_analysis": answer_research_analysis,
    "hypothesis_generation": answer_hypothesis_generation,
}


def generate_page_answer(question: str, page_context: str) -> str:
    """
    根据页面类型生成专属回答。

    Args:
        question: 用户问题
        page_context: 页面标识 (experiment_design / research_gap / formula_explain / research_analysis / hypothesis_generation)

    Returns:
        页面类型的定制回答
    """
    ind_var, dep_var, _ = extract_variable_pair(question)
    # 回退：从问题文本中提取变量
    if not ind_var and not dep_var:
        try:
            from src.equation_engine import extract_variables_from_query
            vars_list = extract_variables_from_query(question)
            if len(vars_list) >= 2:
                ind_var, dep_var = vars_list[0], vars_list[1]
            elif len(vars_list) == 1:
                ind_var = vars_list[0]
        except Exception:
            pass

    # 路由决策
    route = route_research_task(question, page_context)
    primary = route["primary_task"]
    secondary = route["secondary_tasks"]
    full_report = route["full_report"]

    # 全面分析 → 调用旧版 build_paper_level_answer
    if full_report:
        from src.unified_answer import build_paper_level_answer
        return build_paper_level_answer(question)

    # 获取主回答
    answer_func = PAGE_ANSWER_MAP.get(primary)
    if answer_func is None:
        answer_func = answer_research_analysis

    main_answer = answer_func(question, ind_var, dep_var)

    # 补充二次任务（简短引用）
    supplement = ""
    if "mechanism" in secondary and primary != "research_analysis":
        # 简短机制说明
        try:
            from src.condition_mechanism_map import generate_condition_mechanism_map, format_condition_map_markdown
            cm = generate_condition_mechanism_map(question="")
            if cm.get("entries"):
                supplement += "\n\n---\n### 相关机制参考\n"
                supplement += format_condition_map_markdown(cm)
        except Exception:
            pass

    if "counter" in secondary and primary not in ("research_gap", "research_analysis"):
        try:
            from src.research_gap_discovery import search_counter_evidence, format_counter_evidence_markdown
            result = search_counter_evidence(hypothesis=f"{ind_var} affects {dep_var} in L-PBF Ti-6Al-4V")
            if result.get("supporting_evidence") or result.get("counter_evidence"):
                supplement += "\n\n---\n" + format_counter_evidence_markdown(result)
        except Exception:
            pass

    return main_answer + supplement
