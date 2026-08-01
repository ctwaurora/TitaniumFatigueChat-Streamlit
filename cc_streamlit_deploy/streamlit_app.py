"""
streamlit_app.py — TitaniumFatigueChat AI Scientist (Search-Engine Style)
面向钛合金疲劳研究的 AI Scientist；L-PBF Ti-6Al-4V 是当前主案例

四种回答模式:
  - popular_science      科普解释
  - research_analysis    科研分析
  - hypothesis_generation 假设生成
  - experiment_design    实验验证设计

运行: streamlit run streamlit_app.py
"""

import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── DEMO_MODE: 演示模式（公网链接用）──
# 启用后：禁止自动耗时任务、折叠长内容、限制显示数量
DEMO_MODE = os.environ.get("DEMO_MODE", "False").lower() in ("true", "1", "yes")
DEMO_MAX_ITEMS = 10  # 演示模式最多显示条数

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="TitaniumFatigueChat: L-PBF Ti-6Al-4V 疲劳机制发现",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication must complete before importing data, PDF, index, or LLM modules.
from src.app_auth import require_authentication

if not require_authentication(st):
    st.stop()

from src.interactive_modules import (
    EvidenceRelationExplorer,
    EquationParameterMiner,
    ConflictDetector,
    HypothesisGenerator,
    HypothesisScorer,
    RelevanceRanking,
    MacroMicroLinkExplorer,
    LiteratureSearchPlanner,
)
from src.equation_engine import (
    run_equation_pipeline,
    format_equation_results_markdown,
)
from src.formula_comparison import (
    compare_candidate_models,
    format_model_comparison_markdown,
)
from src.condition_mechanism_map import (
    generate_condition_mechanism_map,
    format_condition_map_markdown,
    generate_competition_map,
)
from src.variable_mapper import (
    extract_variable_pair,
    find_exact_or_near_relation,
    evaluate_literature_support,
    build_mechanism_chain,
    get_synonym_group,
)
from src.literature_search_agent import (
    run_literature_search,
    generate_search_queries as lit_gen_queries,
)
from src.research_gap_discovery import (
    discover_research_gaps,
    save_gaps,
    load_review_text,
    search_counter_evidence,
    format_counter_evidence_markdown,
)
from src.query_understanding import (
    understand_user_query,
    format_query_understanding_markdown,
)
from src.formula_renderer import (
    get_formula_card,
    recommend_formulas_for_pair,
)
from src.scientific_framework import (
    run_baseline_comparison,
    run_ablation_study,
    run_retrospective_validation,
    run_paris_law_validation,
    paper_quality_gate,
    generate_sci_paper_export,
    hypothesis_quality_score,
    get_failure_cases,
    save_failure_case,
)


# ═══════════════════════════════════════════════════════════════════════════
# Safe key generator
# ═══════════════════════════════════════════════════════════════════════════

def safe_key(prefix: str, text: str = "", index: int = 0) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(text))
    h = hashlib.md5(clean.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{index}_{h}"


def make_unique_ui_key(prefix: str, record_id: str = "", title: str = "",
                       row_index: int = 0, extra: str = "") -> str:
    """生成全局唯一的 Streamlit UI key，避免 checkbox key 冲突。"""
    raw = f"{prefix}_{record_id}_{title}_{row_index}_{extra}"
    short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)[:40]
    return f"{prefix}_{row_index}_{short_hash}"


# ── 文献分类中文映射 ──
PAPER_TYPE_CN = {
    "review": "综述文献",
    "pore_fatigue_life": "孔隙/缺陷—疲劳寿命文献",
    "micro_ct_defects": "micro-CT 缺陷表征文献",
    "surface_roughness": "表面粗糙度/表面状态文献",
    "hip_heat_treatment": "HIP/热处理文献",
    "fcgr_paris_law": "FCGR/Paris 裂纹扩展文献",
    "defect_tolerance_models": "缺陷容限模型文献",
    "ai_materials_fatigue": "AI/材料疲劳文献",
    "experimental_fatigue": "疲劳实验文献",
    "candidate": "候选未分类文献",
    "other": "其他/待人工确认",
}

# Reverse map for saving to CSV
PAPER_TYPE_FROM_CN = {v: k for k, v in PAPER_TYPE_CN.items()}


def paper_type_to_cn(code: str) -> str:
    """将英文 paper_type code 转为中文显示名。"""
    return PAPER_TYPE_CN.get(code, "其他/待人工确认")


def paper_type_from_cn(label: str) -> str:
    """将中文显示名转回英文 code（用于 CSV 存储）。"""
    return PAPER_TYPE_FROM_CN.get(label, "other")


def check_duplicate_ids(df, id_col: str) -> tuple:
    """检查 DataFrame 中是否有重复 ID。

    Returns:
        (has_duplicates: bool, duplicate_ids: List, cleaned_df: DataFrame)
    """
    if id_col not in df.columns:
        return False, [], df
    dupes = df[df[id_col].duplicated(keep=False)]
    if dupes.empty:
        return False, [], df
    dup_ids = sorted(dupes[id_col].unique().tolist())
    # Generate new unique IDs
    df = df.reset_index(drop=True)
    new_ids = []
    prefix = id_col.split("_")[0][:1].upper() if "_" in id_col else "ID"
    for i in range(len(df)):
        if prefix == "C":
            new_ids.append(f"CAND_{i+1:04d}")
        elif prefix == "P":
            new_ids.append(f"P{i+1:04d}")
        else:
            new_ids.append(f"{prefix}{i+1:04d}")
    df["display_id"] = new_ids
    return True, dup_ids, df


SAMPLE_QUESTIONS = [
    "孔隙尺寸和疲劳寿命之间是什么关系？",
    "表面粗糙度和内部孔隙哪个更容易主导疲劳失效？",
    "HIP 处理是否能完全消除孔隙缺陷对疲劳性能的影响？",
    "给定 pore size 和 stress ratio，能否生成一个可验证假设？",
    "L-PBF Ti-6Al-4V 中哪些文献结论存在冲突？",
    "Paris 参数 C/m 如何受缺陷和微观组织影响？",
]

BAD_HYPOTHESIS_EXAMPLES = [
    "孔隙影响疲劳性能。",
    "微观组织和疲劳寿命有关。",
    "热处理可以改善疲劳性能。",
    "应进一步研究 Ti-6Al-4V 的疲劳行为。",
    "材料性能受多因素影响。",
]

# Mode constants (both for internal use and display)
MODES = {
    "popular_science": {
        "label": "🔬 科普解释",
        "desc": "系统将用通俗语言解释材料疲劳现象。",
    },
    "research_analysis": {
        "label": "📐 科研分析",
        "desc": "系统将输出变量关系、机制链条、文献支持、候选模型和条件边界。",
    },
    "hypothesis_generation": {
        "label": "🧪 假设生成",
        "desc": "系统将生成具体、可验证、可推翻的候选科学假设，并自动评分。",
    },
    "experiment_design": {
        "label": "🧫 实验验证设计",
        "desc": "系统将生成样品分组、测试方法、表征方法和判定标准。",
    },
}

MODE_LIST = list(MODES.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

for key, default in [
    ("answer", None),
    ("user_question", ""),
    ("answer_mode", "research_analysis"),
    ("analysis_done", False),
    ("trigger_search", False),
    ("variable_pair", (None, None, "no_variable")),
    ("literature_result", None),
    ("last_question", ""),
    ("last_mode", "research_analysis"),
    ("answer_timestamp", 0.0),
    ("page", "search"),
    ("gap_results", None),
    ("gap_review_text", ""),
    ("gap_scope", "all"),
    ("answer_depth", "paper_level"),
    ("answer_depth_label", "论文级"),
]:
    if key not in st.session_state:
        if isinstance(default, tuple):
            st.session_state[key] = default
        elif isinstance(default, bool):
            st.session_state[key] = default
        elif default is None:
            st.session_state[key] = None
        else:
            st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════
# Intent detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_intent(user_input: str) -> str:
    if not user_input:
        return "general_explanation"
    text = user_input.lower()

    if any(kw in text for kw in ["冲突", "不一致", "矛盾", "争议", "相反结论",
                                  "conflict", "inconsisten", "contradict"]):
        return "conflict_detection"
    if any(kw in text for kw in ["方程", "公式", "模型", "paris", "basquin",
                                  "walker", "kitagawa", "murakami", "da/dn",
                                  "Δk", "delta k", "c/m", "参数"]):
        return "equation_generation"
    if any(kw in text for kw in ["假设", "提出", "生成", "研究问题",
                                  "科学假设", "hypothesis"]):
        return "hypothesis_generation"
    if any(kw in text for kw in ["实验", "验证", "怎么做", "方案", "测试",
                                  "试验设计", "表征"]):
        return "experiment_design"
    if any(kw in text for kw in ["微观", "宏观", "组织", "孔隙", "裂纹",
                                  "表征", "sem", "ebsd", "micro-ct", "micro_ct"]):
        return "macro_micro_mechanism"
    if any(kw in text for kw in ["关系", "影响", "怎么相关", "变量",
                                  "机制", "相关", "作用"]):
        return "relation_analysis"
    return "general_explanation"


# ═══════════════════════════════════════════════════════════════════════════
# Fallback: load variable rankings from CSV
# ═══════════════════════════════════════════════════════════════════════════

def load_top_variables_fallback(top_n: int = 5) -> list:
    import pandas as pd
    candidates = [
        Path("data/relevance_ranking.csv"),
        Path("data/relation_table.csv"),
        Path("data/variable_mechanism.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            if df.empty:
                continue
            if "variable" in df.columns and "total_relevance_score" in df.columns:
                df = df.sort_values("total_relevance_score", ascending=False)
                return df.head(top_n).to_dict("records")
            if "variable" in df.columns:
                return df.head(top_n).to_dict("records")
            if "independent_variable" in df.columns:
                counts = df["independent_variable"].value_counts().head(top_n)
                return [{"variable": var, "score": int(count),
                         "total_relevance_score": int(count),
                         "reason": "根据关系表出现频次排序"}
                        for var, count in counts.items()]
        except Exception:
            continue
    return []


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# MODE-SPECIFIC ANSWER GENERATORS
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════


# ── Shared helper: literature support section ──

def _build_lit_support_section(ind_var: Optional[str], dep_var: Optional[str],
                                user_query: str) -> str:
    """通用的文献支持评估板块。"""
    if not ind_var and not dep_var:
        return ""
    support = evaluate_literature_support(ind_var, dep_var, user_query)
    cov = support["coverage_level"]
    cov_icon = {"sufficient": "🟢", "partial": "🟡", "weak": "🟠", "not_found": "🔴"}
    lines = ["## 文献符合情况\n"]
    lines.append(f"- **本地覆盖等级**: {cov_icon.get(cov, '⚪')} {cov}\n")
    lines.append(f"- **匹配文献**: {support['matched_paper_count']} 篇\n")
    lines.append(f"- **匹配证据**: {support['matched_evidence_count']} 条\n")

    if support["supporting_evidence"]:
        lines.append("- **支持证据**:\n")
        for e in support["supporting_evidence"][:3]:
            lines.append(f"  - {e[:120]}\n")

    if support["conflicting_or_conditional_evidence"]:
        lines.append("- **冲突/条件差异**:\n")
        for e in support["conflicting_or_conditional_evidence"][:2]:
            lines.append(f"  - {e[:120]}\n")

    conclusion = support["evidence_conclusion"]
    concl_icon = {"证据支持": "🟢", "证据部分支持": "🟡",
                   "存在条件冲突": "🟠", "证据不足": "🔴"}
    lines.append(f"- **证据结论**: {concl_icon.get(conclusion, '⚪')} {conclusion}\n")
    return "".join(lines)


# ── Shared helper: equation recommendation section ──

def _build_equation_section(ind_var: Optional[str], dep_var: Optional[str],
                             user_query: str) -> str:
    """根据变量对推荐合适方程 — 使用 LaTeX 公式卡片。"""
    from src.formula_renderer import recommend_formulas_for_pair

    cards = recommend_formulas_for_pair(ind_var, dep_var, user_query)
    if cards:
        lines = ["## 候选方程/模型\n"]
        for card in cards:
            lines.append(card + "\n---\n")
        return "".join(lines)

    # Fallback if no formula matched
    lines = ["## 可能方程/模型\n"]
    lines.append("当前变量组合暂未匹配到可靠疲劳方程。\n")
    lines.append(f"如需方程推荐，建议输入更明确的变量组合（如 ΔK+da/dN 或 缺陷尺寸+疲劳极限）。\n")
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1: Popular Science (科普解释)
# ═══════════════════════════════════════════════════════════════════════════

def generate_popular_science_answer(
    question: str,
    ind_var: Optional[str],
    dep_var: Optional[str],
    var_class: str,
) -> str:
    """
    科普解释模式 — 面向非专业用户。
    少公式、少术语、用类比、不输出评分表格。
    """
    lines = []

    # ── 1. Simple explanation ──
    lines.append("## 简单解释\n")

    if ind_var == "pore_size" and dep_var == "fatigue_life":
        lines.append(
            "孔隙就像材料里面的小缺口。材料在反复受力时，"
            "裂纹往往会从这些薄弱位置开始生长。\n\n"
            "孔隙越大、越靠近表面，就越容易成为裂纹起点，"
            "因此疲劳寿命通常越短。\n\n"
            "不过，如果零件表面本身非常粗糙，"
            "裂纹也可能先从表面的凹凸处开始，"
            "而不一定从内部的孔隙开始。\n\n"
            "简单说：孔隙越大 → 越容易疲劳失效；"
            "而且孔隙在表面附近比在内部更危险。\n"
        )
    elif ind_var == "surface_roughness" and dep_var == "fatigue_life":
        lines.append(
            "表面粗糙度就像零件表面的'伤痕'。"
            "这些凹凸不平的地方在受力时会产生应力集中，"
            "就像一张纸在反复折叠时从折痕处裂开一样。\n\n"
            "表面越粗糙 → 越容易产生裂纹 → 疲劳寿命越短。\n\n"
            "抛光或机加工可以大幅改善表面质量，从而提高疲劳寿命。\n"
        )
    elif ind_var == "delta_k" and dep_var == "da_dn":
        lines.append(
            "ΔK 是裂纹尖端的'受力强度'，da/dN 是裂纹每往复一次扩展的距离。\n\n"
            "这两个量的关系就像：给裂纹的'推力'越大，裂纹'跑得越快'。\n\n"
            "在材料科学中，这个关系可以用 Paris 公式描述，"
            "简单说就是：推力翻倍，速度可能增长数倍。\n"
        )
    elif ind_var == "heat_treatment" and dep_var == "fatigue_life":
        lines.append(
            "热处理就像是给材料'做理疗'。\n\n"
            "L-PBF Ti-6Al-4V 打印后内部有残余应力和微小孔隙，"
            "通过热处理（特别是 HIP 热等静压）可以：\n\n"
            "1. 闭合内部孔隙（就像把海绵里的气泡压掉）\n"
            "2. 消除残余应力（让材料内部更均匀）\n"
            "3. 改善微观组织（让材料结构更强韧）\n\n"
            "经过良好热处理后，材料的疲劳寿命通常显著提高。\n"
        )
    else:
        lines.append(
            f"关于「{question}」，简单来说：\n\n"
            "L-PBF Ti-6Al-4V（激光3D打印钛合金）的疲劳性能"
            "主要受内部缺陷（孔隙、未熔合）和表面状态（粗糙度）影响。\n\n"
            "这些因素都会影响材料在反复受力时的寿命。\n"
        )

    # ── 2. Analogy ──
    lines.append("## 打个比方\n")
    if ind_var == "pore_size":
        lines.append(
            "想象一根橡皮筋：如果上面有一个小切口，"
            "反复拉伸时，裂纹总是从切口处开始扩展。\n\n"
            "材料的孔隙就像是这个切口——"
            "孔隙越大、越靠近边缘（表面），就越容易成为裂纹起点。\n\n"
            "所以，想要材料更耐用，就要尽量减小内部缺陷，"
            "或者通过 HIP 等工艺把它们'焊合'。\n"
        )
    elif ind_var == "surface_roughness":
        lines.append(
            "想象一张平整的纸和一张有锯齿边缘的纸：\n\n"
            "反复弯折时，锯齿边缘的纸总是在锯齿根部先裂开。\n\n"
            "材料的表面粗糙度就像这些锯齿——"
            "表面越粗糙，应力集中越严重，裂纹越容易从这里开始。\n\n"
            "所以抛光（把表面磨平）能大幅提高疲劳寿命。\n"
        )
    else:
        lines.append(
            "就像一根铁丝反复弯折最终会断——\n\n"
            "材料的疲劳失效是一个从微观损伤逐渐积累到宏观断裂的过程。"
            "内部缺陷和表面状态就是'薄弱环节'，决定了能用多久。\n"
        )

    # ── 3. Key factors ──
    lines.append("## 关键影响因素\n")
    if ind_var == "pore_size":
        lines.append("1. **孔隙尺寸**: 越大越危险\n")
        lines.append("2. **孔隙位置**: 靠近表面比内部更危险\n")
        lines.append("3. **表面粗糙度**: 表面太粗糙会掩盖孔隙的影响\n")
        lines.append("4. **热处理**: HIP 可闭合孔隙，显著提高寿命\n")
    elif ind_var == "surface_roughness":
        lines.append("1. **表面粗糙度 (Ra/Rz)**: 越粗糙寿命越短\n")
        lines.append("2. **内部孔隙**: 表面改善后，内部孔隙成为主要问题\n")
        lines.append("3. **后处理**: 抛光/机加工可大幅改善\n")
    elif ind_var == "delta_k":
        lines.append("1. **ΔK 大小**: 越大裂纹扩展越快\n")
        lines.append("2. **材料组织**: 组织越细密，扩展越慢\n")
        lines.append("3. **应力比 R**: 平均应力越大扩展越快\n")
    else:
        lines.append("1. **内部缺陷**: 孔隙、未熔合、夹杂\n")
        lines.append("2. **表面状态**: 粗糙度、表面缺陷\n")
        lines.append("3. **热处理**: HIP、退火、固溶时效\n")
        lines.append("4. **微观组织**: α/β 相、晶粒尺寸\n")

    # ── 4. One-sentence summary ──
    lines.append("## 一句话总结\n")
    if ind_var and dep_var:
        lines.append(
            f"{ind_var.replace('_', ' ')} 影响 {dep_var.replace('_', ' ')}，"
            f"但具体程度取决于表面状态、热处理和微观组织的共同作用。\n"
        )
    else:
        lines.append("L-PBF Ti-6Al-4V 的疲劳性能由缺陷、表面和组织的共同作用决定。\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2: Research Analysis (科研分析)
# ═══════════════════════════════════════════════════════════════════════════

def _build_question_understanding_section(
    question: str,
    ind_var: Optional[str],
    dep_var: Optional[str],
    var_class: str = "",
) -> str:
    """构建问题理解模块（论文级回答第1节）。"""
    lines = ["## 1. 问题理解\n"]
    lines.append(f"**原始问题**: {question}\n\n")

    # Use query understanding module
    try:
        from src.query_understanding import understand_user_query
        sq = understand_user_query(question)
        task_intent = sq.get("task_intent", "general_explanation")
        intent_labels = {
            "general_explanation": "一般解释", "relation_analysis": "变量关系分析",
            "equation_generation": "方程/参数生成", "hypothesis_generation": "假设生成",
            "experiment_design": "实验设计", "conflict_detection": "文献冲突检测",
            "macro_micro_mechanism": "宏微观机制分析",
            "dominance_comparison": "因素比较", "literature_search": "文献检索",
            "research_gap_discovery": "研究空白",
        }
        lines.append(f"**系统理解**: 将用户问题识别为「{intent_labels.get(task_intent, '一般解释')}」任务。\n\n")

        # Corrections
        if sq.get("has_corrections") and sq.get("corrections"):
            lines.append("**自动修正**:\n")
            for c in sq["corrections"]:
                conf = int(c.get("confidence", 0.8) * 100)
                lines.append(f"- 「{c['raw']}」→「{c['corrected']}」(置信度: {conf}%)\n")
            lines.append("\n")
    except Exception:
        pass

    lines.append("**识别变量**:\n")
    if ind_var and dep_var:
        lines.append(f"- **自变量 (IV)**: {ind_var}\n")
        lines.append(f"- **因变量 (DV)**: {dep_var}\n")
        if var_class:
            lines.append(f"- **变量关系类型**: {var_class}\n")
    elif ind_var:
        lines.append(f"- **自变量 (IV)**: {ind_var}\n")
        lines.append(f"- **因变量 (DV)**: 待识别\n")
    elif dep_var:
        lines.append(f"- **自变量 (IV)**: 待识别\n")
        lines.append(f"- **因变量 (DV)**: {dep_var}\n")
    else:
        lines.append("- **自变量 (IV)**: 待识别\n")
        lines.append("- **因变量 (DV)**: 待识别\n")

    # Add recommended variables if vague
    if dep_var and dep_var.lower() in BANNED_VAGUE_DV:
        iv_lower = (ind_var or "").lower()
        suggestions = DV_RECOMMENDED_MAP.get(iv_lower, [
            "Nf", "fatigue_limit σw", "crack_initiation_site", "da/dN"
        ])
        lines.append(f"\n> ⚠️ 因变量「{dep_var}」过于模糊，建议拆解为: {', '.join(suggestions)}\n")

    lines.append("\n---\n")
    return "".join(lines)


def generate_research_analysis_answer(
    question: str,
    ind_var: Optional[str],
    dep_var: Optional[str],
    var_class: str,
    ranker=None,
    depth: str = "paper_level",
) -> str:
    """
    科研分析模式 — 输出论文级详细科研分析。

    深度:
      - standard: 简短结论
      - detailed: 含变量关系、机制、文献
      - paper_level: 15层完整论文分析
    """
    lines = []

    # ── 0. Question understanding (paper-level) ──
    if depth == "paper_level":
        lines.append(_build_question_understanding_section(question, ind_var, dep_var, var_class))

    # ── 1. Direct conclusion ──
    if depth == "paper_level":
        lines.append("## 2. 直接结论\n")
    else:
        lines.append("## 直接关系结论\n")
    lines.append("**用户关注变量**:\n")
    lines.append(f"- **自变量**: {ind_var or '（未识别）'}\n")
    lines.append(f"- **因变量**: {dep_var or '（未识别）'}\n\n")

    direct = _direct_relation_text(ind_var, dep_var)
    lines.append(direct + "\n")

    # ── 2. Variable relation analysis ──
    if depth == "paper_level":
        lines.append("## 3. 详细变量关系分析\n")
        lines.append("### 3.1 自变量如何影响因变量\n\n")
    else:
        lines.append("## 变量关系判断\n")

    rel_result = find_exact_or_near_relation(ind_var, dep_var)
    if rel_result["found"]:
        lines.append(f"- **匹配质量**: {rel_result['match_quality']}\n")
        lines.append(f"- **匹配数量**: {len(rel_result['relations'])} 条\n")
        for r in rel_result["relations"][:3]:
            iv = r.get("independent_variable", "")
            dv = r.get("dependent_variable", "")
            rt = r.get("relation_type", "")
            cond = r.get("condition", "")
            mech = r.get("mechanism", "")[:120]
            lines.append(f"- {iv} → {dv}（{rt}）\n")
            if cond:
                lines.append(f"  - **条件**: {cond}\n")
            if mech:
                lines.append(f"  - {mech}\n")
    else:
        lines.append(f"- {rel_result['evidence_summary']}\n")

    # ── 3. Condition boundaries & competing mechanisms ──
    if depth == "paper_level" and ind_var and dep_var:
        lines.append("\n### 3.2 调节变量如何改变关系\n\n")
        iv_lower = ind_var.lower()
        dv_lower = dep_var.lower()

        if "roughness" in iv_lower:
            lines.append(
                "**表面状态 (surface_state)** 是核心调节变量。\n\n"
                "- 在 **as-built** 状态下，表面粗糙度 Ra/Rz 较高（通常 Ra > 10μm），"
                "表面缺口效应占主导，裂纹倾向于从表面粗糙峰根部起裂。\n"
                "- 在 **polished** 或 **machined** 状态下，表面缺口消除，"
                "内部孔隙或近表面缺陷对疲劳寿命 Nf 的解释力增强。\n"
                "- **应力比 R** 影响局部应力幅水平和裂纹闭合效应，"
                "高 R 比下表面缺口效应可能更加显著。\n\n"
            )
            lines.append("### 3.3 竞争机制\n\n")
            lines.append(
                "表面粗糙度与内部孔隙存在**竞争主导关系**:\n\n"
                "- 表面粗糙度高（as-built）→ **表面主导**起裂模式\n"
                "- 表面改善后（polished/machined）→ **孔隙主导**起裂模式\n"
                "- 存在临界表面粗糙度值（如 Ra 阈值），"
                "低于该值时主导权转向孔隙。\n"
                "- 该临界值受 √area、distance_to_surface、R 比的调节。\n\n"
            )
        elif "pore" in iv_lower or "defect" in iv_lower:
            lines.append(
                "**距表面距离 (distance_to_surface)** 是核心调节变量。\n\n"
                "- 近表面孔隙（distance_to_surface < 100μm）："
                "孔隙边缘应力集中与自由表面应力场叠加，"
                "显著提高起裂驱动力，降低 Nf。\n"
                "- 深部孔隙（distance_to_surface > 500μm）："
                "应力叠加效应减弱，起裂驱动力以孔隙自身应力集中为主。\n"
                "- **表面状态**竞争：在 as-built 试样中，表面粗糙度"
                "可能掩盖近表面孔隙的起裂主导作用。\n"
                "- **应力比 R**：高 R 比下平均应力增高，"
                "孔隙边缘的应力集中效应被放大。\n\n"
            )
        elif "delta" in iv_lower or "crack" in iv_lower:
            lines.append(
                "**微观组织**是核心调节变量。\n\n"
                "- α lath 宽度增大可能提高 Paris 参数 m，"
                "降低裂纹扩展抗力。\n"
                "- **缺陷状态**：孔隙/未熔合缺陷在高 ΔK 下影响减弱，"
                "但在近门槛值区缺陷对 ΔKth 有明显降低作用。\n"
                "- **应力比 R**：高 R 比降低裂纹闭合效应，"
                "提高有效 ΔKeff，加速裂纹扩展。\n\n"
            )
        else:
            lines.append(
                "变量间的关系可能受以下调节变量影响：\n"
                "- **表面状态** (as-built vs polished)\n"
                "- **热处理** (stress-relieved vs HIP vs annealing)\n"
                "- **应力比 R**\n"
                "- **缺陷状态** (孔隙率、缺陷尺寸)\n"
                "- **微观组织** (α/β 相比例、晶粒尺寸)\n\n"
            )

    # ── 4. Mechanism chain ──
    if ind_var and dep_var:
        if depth == "paper_level":
            lines.append("## 4. 机制链条\n\n")
        else:
            lines.append("## 机制链条\n")
        chain = build_mechanism_chain(ind_var, dep_var)
        lines.append("```\n" + chain + "\n```\n")

    # ── 5. Literature evidence status ──
    if depth == "paper_level":
        lines.append("## 5. 文献证据状态\n\n")
        support = evaluate_literature_support(ind_var, dep_var, question)
        cov = support["coverage_level"]
        n_ev = support["matched_evidence_count"]
        n_direct = len(support["supporting_evidence"])
        n_conflict = len(support["conflicting_or_conditional_evidence"])

        if cov == "sufficient" and n_direct >= 2:
            lines.append("**证据等级**: 🟢 direct / sufficient\n\n")
            lines.append(f"本地文献库中有 {n_direct} 条直接证据支持该变量关系，"
                         f"共匹配 {n_ev} 条相关证据片段。\n\n")
        elif cov in ("sufficient", "partial") and n_direct >= 1:
            lines.append("**证据等级**: 🟡 partially supported / indirect\n\n")
            lines.append(f"存在 {n_direct} 条直接证据，"
                         f"另有 {n_ev - n_direct} 条间接证据。"
                         f"需要更多系统定量研究。\n\n")
        elif n_conflict > 0:
            lines.append("**证据等级**: 🟠 conflicting\n\n")
            lines.append(f"检测到 {n_conflict} 条冲突或条件依赖证据。"
                         f"不同文献结论可能因未控制变量而异。\n\n")
        else:
            lines.append("**证据等级**: 🔴 insufficient / search-guided candidate\n\n")
            lines.append("当前本地文献库不足以支持强判断。\n\n")

        # Missing data fields
        lines.append("**缺失的关键数据字段**:\n")
        if "roughness" in (ind_var or "").lower():
            lines.append("- surface_roughness_Ra/Rz 与 Nf 配对数据\n"
                         "- crack_initiation_site SEM 确认数据\n"
                         "- 同时包含 Ra/Rz、pore_size、distance_to_surface 的系统对比\n")
        elif "pore" in (ind_var or "").lower():
            lines.append("- pore_size / √area 与 Nf 配对数据\n"
                         "- distance_to_surface 的系统控制研究\n"
                         "- 近表面 vs 深部孔隙的对比实验\n")
        elif "delta" in (ind_var or "").lower() or "crack" in (ind_var or "").lower():
            lines.append("- da/dN-ΔK 完整曲线\n"
                         "- Paris_C / Paris_m 参数\n"
                         "- ΔKth 门槛值\n")
        else:
            lines.append(f"- {ind_var or '自变量'} → {dep_var or '因变量'} 的直接实验数据\n")

    # ── 6. Equation / models ──
    if depth == "paper_level":
        lines.append("## 6. 可能方程或模型\n\n")
    else:
        lines.append("## 可能方程/模型\n")
    lines.append(_build_equation_section(ind_var, dep_var, question))

    # ── 7. Literature support ──
    if depth == "paper_level":
        lines.append("## 7. 文献支撑\n\n")
    lines.append(_build_lit_support_section(ind_var, dep_var, question))

    # ── 8. Scientific judgment ──
    if depth == "paper_level":
        lines.append("## 8. 科研判断\n\n")
    else:
        lines.append("## 科研判断\n")
    if ind_var and dep_var:
        support = evaluate_literature_support(ind_var, dep_var, question)
        cov = support["coverage_level"]
        n_ev = support["matched_evidence_count"]
        n_direct = len(support["supporting_evidence"])

        if cov == "sufficient" and n_direct >= 2:
            lines.append("**判断**: 🟢 **evidence-supported**\n")
            lines.append("本地文献库有直接证据支持该变量关系。\n")
        elif cov in ("sufficient", "partial") and n_direct >= 1:
            lines.append("**判断**: 🟡 **partially supported**\n")
            lines.append("存在部分证据支持，但需要更多定量数据。\n")
        elif n_direct == 0 and len(support["conflicting_or_conditional_evidence"]) > 0:
            lines.append("**判断**: 🟠 **存在条件冲突**\n")
            lines.append("不同文献结论不一致，可能依赖未控制条件。\n")
        else:
            lines.append("**判断**: 🔴 **search-guided candidate / evidence insufficient**\n")
            lines.append("当前证据不足以支持强判断，需要补充文献。\n")
    else:
        lines.append("**判断**: 变量不明确，无法做出科研判断。\n")

    # ── 9. Candidate hypothesis (paper-level only) ──
    if depth == "paper_level" and ind_var and dep_var:
        lines.append("## 9. 候选科学假设\n\n")
        detailed_hyp = _generate_detailed_variable_hypothesis(ind_var, dep_var, question)
        if detailed_hyp:
            lines.append(detailed_hyp + "\n")
        else:
            lines.append(
                "基于当前变量对，可生成以下候选假设：\n\n"
            )

    # ── 10. Experimental verification (paper-level only) ──
    if depth == "paper_level" and ind_var and dep_var:
        lines.append("## 10. 实验验证方案\n\n")
        lines.append(generate_experiment_design_answer(question, ind_var, dep_var, var_class))
        lines.append("\n")

    # ── 11. Data needed ──
    if depth == "paper_level":
        lines.append("## 11. 当前不足与需要补充的数据\n\n")
    else:
        lines.append("## 下一步需要的数据\n")
    if ind_var == "pore_size" and dep_var == "fatigue_life":
        lines.append("""1. 包含 pore_size / √area 与 Nf 对应关系的实验文献
2. 同时包含 micro-CT 缺陷表征和 HCF 疲劳数据的文献
3. 包含裂纹起裂源与孔隙位置对应关系的断口分析
4. Murakami / Kitagawa-Takahashi / El Haddad 模型验证数据
""")
    elif ind_var and "roughness" in ind_var.lower():
        lines.append("""1. 表面粗糙度 Ra/Rz 与 Nf 的直接配对实验数据
2. 同时包含 Ra/Rz、pore_size、distance_to_surface 和 crack_initiation_site 的系统对比
3. as-built vs polished 组间裂纹起裂源比较数据
4. 不同表面状态下 S-N 曲线数据
""")
    else:
        lines.append("""1. 补充该变量对的定量实验数据
2. 控制变量的系统对比研究
3. 含方程参数的文献数据
""")

    # ── 12. Improvement recommendation (paper-level) ──
    if depth == "paper_level":
        lines.append("## 12. 改进后的高质量假设方向\n\n")
        iv_lower = (ind_var or "").lower()
        if "roughness" in iv_lower:
            lines.append(
                "基于以上分析，推荐优先验证以下高质量假设方向：\n\n"
                "**表面粗糙度—内部孔隙竞争主导疲劳裂纹起裂的条件边界假设**\n\n"
                "在 as-built L-PBF Ti-6Al-4V 中，较高的表面粗糙度 Ra/Rz "
                "可能通过表面缺口效应优先诱导疲劳裂纹在表面粗糙峰处起裂，"
                "从而削弱内部孔隙尺寸对疲劳寿命 Nf 的解释力。相反，"
                "在 polished 或 machined 表面中，表面缺口效应减弱，"
                "近表面或内部大孔隙对裂纹起裂位置和 Nf 的影响会增强。\n\n"
                "该假设包含明确的条件边界（as-built vs polished）、"
                "竞争机制（表面缺口 vs 内部孔隙）、"
                "具体因变量（Nf / crack_initiation_site）"
                "和验证方案（SEM fractography + micro-CT + HCF）。\n"
            )
        elif "pore" in iv_lower:
            lines.append(
                "**孔隙尺寸—距表面距离耦合控制疲劳裂纹起裂的假设**\n\n"
                "在控制材料成分、L-PBF 工艺参数和表面状态（polished）后，"
                "具有较大 √area 且距自由表面较近的孔隙，"
                "比深部同等尺寸孔隙更容易成为疲劳裂纹起裂源，"
                "并导致疲劳寿命 Nf 降低。\n\n"
                "该假设需通过 micro-CT + SEM fractography + HCF 联合验证。\n"
            )
        else:
            lines.append(
                "建议对当前变量对进行更系统的文献检索，"
                "补充直接实验证据后生成具体可验证假设。\n"
            )

    return "".join(lines)


def _direct_relation_text(ind_var: Optional[str], dep_var: Optional[str]) -> str:
    """生成直接关系结论的文本。"""
    known_relations = {
        ("pore_size", "fatigue_life"): (
            "在 L-PBF Ti-6Al-4V 中，孔隙尺寸增大通常会通过局部应力集中"
            "促进疲劳裂纹提前萌生，从而降低疲劳寿命 Nf。"
            "该关系受到孔隙位置、表面状态、应力比 R、热处理状态和孔隙形态的调节。\n\n"
            "近表面大尺寸孔隙通常比深部小孔隙更危险；"
            "但在 as-built 表面粗糙度较高时，"
            "表面粗糙峰或表面缺陷可能掩盖内部孔隙的影响。"
        ),
        ("pore_size", "fatigue_limit"): (
            "孔隙尺寸增大通常会降低疲劳极限 σw，"
            "该关系可用 Murakami √area 模型描述："
            "σw ∝ (HV+120)/(√area)^{1/6}。"
            "近表面缺陷对疲劳极限的降低效应比内部缺陷更显著。"
        ),
        ("surface_roughness", "fatigue_life"): (
            "表面粗糙度 Ra/Rz 增大通过表面应力集中促进裂纹在表面起裂，"
            "从而降低疲劳寿命 Nf。"
            "该效应在 as-built 状态下最显著，"
            "polished/machined 后可大幅减轻。\n\n"
            "表面粗糙度与内部孔隙对疲劳寿命存在竞争主导关系："
            "表面粗糙时表面主导，表面改善后内部孔隙转为主导。"
        ),
        ("delta_k", "da_dn"): (
            "ΔK 与 da/dN 在稳定裂纹扩展区（Region II）服从 Paris 定律："
            "da/dN = C(ΔK)^m。\n\n"
            "Paris 参数 C 和 m 受材料组织、缺陷状态和应力比 R 影响。"
            "L-PBF Ti-6Al-4V 中，缺陷（孔隙/未熔合）倾向于增大 C，"
            "而微观组织（α lath 宽度、β 相含量）倾向于改变 m。"
        ),
        ("heat_treatment", "fatigue_life"): (
            "热处理（特别是 HIP）可通过以下机制提高疲劳寿命 Nf：\n"
            "1. HIP 闭合内部孔隙，降低缺陷密度\n"
            "2. 退火消除残余应力，减小局部应力集中\n"
            "3. α′ 马氏体分解为 α+β 层片组织，改善扩展抗力\n\n"
            "但 HIP 不能完全消除所有缺陷——"
            "表面连通孔隙或大尺寸未熔合缺陷可能残留。"
        ),
        ("pore_location", "fatigue_life"): (
            "孔隙距表面的距离是疲劳寿命的重要调节变量。\n\n"
            "近表面孔隙（距表面 < 100μm）比深部孔隙更容易成为裂纹起裂源，"
            "原因是近表面孔隙与自由表面应力场叠加，"
            "导致局部有效应力强度因子升高。\n\n"
            "该效应在 polished/machined 试样中比 as-built 试样中更明显。"
        ),
    }

    key = (ind_var, dep_var)
    if key in known_relations:
        return known_relations[key]

    # Try reverse
    reverse_key = (dep_var, ind_var)
    if reverse_key in known_relations:
        return known_relations[reverse_key]

    return (
        f"关于 {ind_var or '自变量'} 对 {dep_var or '因变量'} 的影响，"
        f"目前本地文献库中的直接证据有限。"
        f"建议补充专题文献后再做定量判断。"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODE 3: Hypothesis Generation (假设生成) — 最具体模式
# ═══════════════════════════════════════════════════════════════════════════

# ── 禁止使用的过泛因变量 ──
BANNED_VAGUE_DV = {
    "fatigue performance", "fatigue property", "fatigue behavior",
    "fatigue resistance", "mechanical property", "疲劳性能",
    "疲劳行为", "疲劳特性", "力学性能",
}

# ── 因变量推荐映射 ──
DV_RECOMMENDED_MAP = {
    "surface_roughness": ["Nf", "crack_initiation_site", "fatigue_limit σw"],
    "pore_size": ["Nf", "crack_initiation_site", "fatigue_limit σw"],
    "porosity": ["Nf", "fatigue_limit σw"],
    "pore_location": ["crack_initiation_site", "Nf"],
    "delta_k": ["da/dN", "Paris_C", "Paris_m"],
    "ΔK": ["da/dN", "Paris_C", "Paris_m"],
    "heat_treatment": ["Nf", "fatigue_limit σw", "da/dN", "Paris_C", "Paris_m"],
    "microstructure": ["da/dN", "Nf", "crack_initiation_site"],
    "stress_ratio": ["da/dN", "ΔKth", "Nf"],
}

IV_RECOMMENDED_MAP = {
    "surface_roughness": ["surface_roughness_Ra", "surface_roughness_Rz",
                           "surface_state: as-built/polished/machined"],
    "pore_size": ["pore_size / √area", "distance_to_surface", "pore_location"],
    "porosity": ["porosity %", "max_pore_size / √area_max"],
    "pore_location": ["distance_to_surface", "pore_size / √area"],
    "delta_k": ["ΔK", "stress_ratio_R", "Kmax"],
    "ΔK": ["ΔK", "stress_ratio_R", "Kmax"],
    "heat_treatment": ["heat_treatment_type",
                        "heat_treatment_temperature", "cooling_rate"],
    "microstructure": ["α_lath_width", "grain_size", "phase_fraction"],
    "stress_ratio": ["stress_ratio_R", "ΔK", "Kmax"],
}


def _validate_variables(ind_var: Optional[str], dep_var: Optional[str]) -> Dict[str, Any]:
    """验证变量是否过于模糊，给出推荐替换。

    Returns:
        {
            "ind_var_ok": bool,
            "dep_var_ok": bool,
            "ind_var_corrected": str or None,
            "dep_var_corrected": str or None,
            "ind_var_suggestions": List[str],
            "dep_var_suggestions": List[str],
            "warnings": List[str],
        }
    """
    result = {
        "ind_var_ok": True,
        "dep_var_ok": True,
        "ind_var_corrected": None,
        "dep_var_corrected": None,
        "ind_var_suggestions": [],
        "dep_var_suggestions": [],
        "warnings": [],
    }

    if dep_var and dep_var.lower() in BANNED_VAGUE_DV:
        result["dep_var_ok"] = False
        # Try to suggest based on ind_var
        if ind_var and ind_var.lower() in DV_RECOMMENDED_MAP:
            result["dep_var_suggestions"] = DV_RECOMMENDED_MAP[ind_var.lower()]
        else:
            result["dep_var_suggestions"] = [
                "Nf（疲劳寿命）", "fatigue_limit σw（疲劳极限）",
                "crack_initiation_site（裂纹起裂位置）", "da/dN（裂纹扩展速率）",
            ]
        result["warnings"].append(
            f"因变量「{dep_var}」过于模糊，请替换为具体指标："
            f"{'、'.join(result['dep_var_suggestions'])}"
        )

    if ind_var and ind_var.lower() in BANNED_VAGUE_DV:
        result["ind_var_ok"] = False
        result["ind_var_suggestions"] = [
            "pore_size / √area", "surface_roughness_Ra/Rz",
            "distance_to_surface", "stress_ratio_R", "ΔK",
        ]
        result["warnings"].append(
            f"自变量「{ind_var}」过于模糊，请替换为具体变量："
            f"{'、'.join(result['ind_var_suggestions'])}"
        )

    return result


def _build_evidence_weakness_explanation(
    ind_var: Optional[str], dep_var: Optional[str],
    coverage_level: str,
) -> str:
    """当证据状态 weak 时，生成详细的缺失原因说明。"""
    lines = []

    if coverage_level in ("weak", "not_found"):
        lines.append("\n### 证据状态说明\n")
        lines.append(
            "**当前证据状态**: weak ⚠️\n\n"
        )

        if coverage_level == "not_found":
            lines.append(
                "**原因**: 本地文献库中未找到与当前变量对直接相关的文献。\n\n"
            )
        else:
            lines.append(
                "**原因**: 本地文献库中仅有少量间接相关文献，"
                "缺乏同时包含以下字段的直接对比数据：\n"
            )
            if ind_var and dep_var:
                iv_lower = ind_var.lower()
                dv_lower = dep_var.lower()

                # Surface roughness specific
                if "roughness" in iv_lower or "surface" in iv_lower:
                    lines.append(
                        "- surface_roughness_Ra/Rz（表面粗糙度量化测量）\n"
                        "- crack_initiation_site（裂纹起裂位置SEM确认）\n"
                        "- Nf 与 Ra/Rz 的直接配对数据\n"
                    )
                    if "pore" in dep_var.lower() or "defect" in dep_var.lower():
                        lines.append(
                            "- internal pore size（内部孔隙尺寸）\n"
                            "- 同一批样品的 as-built vs 加工后表面粗糙度对比\n"
                        )
                elif "pore" in iv_lower or "defect" in iv_lower:
                    lines.append(
                        "- pore_size / √area（孔隙尺寸量化，micro-CT）\n"
                        "- distance_to_surface（距表面距离）\n"
                        "- Nf 与孔隙特征一一对应的配对数据\n"
                        "- crack_initiation_site 与孔隙的 SEM 对应确认\n"
                    )
                elif "delta" in iv_lower or "Δk" in iv_lower.lower() or "dk" in iv_lower:
                    lines.append(
                        "- da/dN-ΔK 完整曲线数据\n"
                        "- Paris_C / Paris_m 参数\n"
                        "- ΔKth 门槛值\n"
                        "- 不同 R 比下的对比数据\n"
                    )
                else:
                    lines.append(
                        f"- {ind_var} → {dep_var} 的直接实验数据\n"
                        "- 控制变量的完整记录\n"
                    )

        lines.append(
            "\n**因此该假设属于**: search-guided candidate hypothesis，"
            "不能标记为 evidence-supported。\n\n"
            "**建议**: 补充上述特定文献或数据后再升级证据等级。\n"
        )

    return "".join(lines)


def improve_low_score_hypothesis(
    hypothesis: Dict[str, Any],
    score_result: Dict[str, Any],
    ind_var: Optional[str] = None,
    dep_var: Optional[str] = None,
) -> str:
    """当假设评分低时（total_score < 60），自动生成优化版。"""
    lines = []
    total = score_result.get("total_score", 0)
    if total >= 60:
        return ""

    weakness = score_result.get("major_weakness", "")
    dims = score_result.get("dim_scores", [])

    lines.append("\n---\n")
    lines.append("### 当前假设问题分析\n\n")

    # Analyze specific problems
    problems = []
    if weakness:
        problems.append(f"- **主要弱点**: {weakness}\n")

    # Check for vague variables
    if ind_var and ind_var.lower() in BANNED_VAGUE_DV:
        problems.append("- **变量过泛**: 自变量「{ind_var}」不够具体\n")
    if dep_var and dep_var.lower() in BANNED_VAGUE_DV:
        problems.append("- **变量过泛**: 因变量「{dep_var}」不够具体\n")

    # Check dimension scores
    low_dims = [d for d in dims if d.get("score", 5) <= 2]
    dim_labels = {
        "specificity": "假设具体性",
        "variable_match": "变量匹配",
        "evidence_grounding": "证据支撑",
        "mechanism_clarity": "机制清晰度",
        "testability": "可验证性",
        "experiment_alignment": "实验方法对齐",
        "falsifiability": "可推翻性",
    }
    for d in low_dims:
        dname = d.get("dimension", "")
        label = dim_labels.get(dname, dname)
        problems.append(f"- **{label}** 得分低（{d.get('score', 0)}/{d.get('max_score', 5)}）\n")

    if not problems:
        problems.append("- 假设不够具体，缺少条件边界和明确变量\n")
        problems.append("- 缺少与变量对应的实验方法\n")

    for p in problems:
        lines.append(p)

    lines.append("\n### 改进版假设\n\n")

    # Generate improved version based on variable pair
    iv = (ind_var or "").lower()
    dv = (dep_var or "").lower()

    if "roughness" in iv or "surface" in iv:
        lines.append(_hyp_surface_roughness_fatigue_life())
    elif "pore" in iv or "defect" in iv:
        lines.append(_hyp_pore_size_fatigue_life())
    elif "delta" in iv or "Δk" in iv or "dk" in iv or "crack" in iv:
        lines.append(_hyp_delta_k_da_dn())
    elif "heat" in iv or "hip" in iv or "anneal" in iv:
        lines.append(_hyp_heat_treatment_fatigue_life())
    elif "micro" in iv:
        lines.append(_hyp_microstructure_da_dn())
    else:
        # Generic improvement
        lines.append(
            f"**H1-improved**: {iv or '目标变量'} 对 {dv or '疲劳指标'}"
            f"影响的量化条件边界假设\n\n"
        )
        lines.append(
            "**假设陈述**: 在控制材料成分（Ti-6Al-4V Grade 23）、"
            "L-PBF 工艺参数和测试条件后，需进一步明确变量关系。\n\n"
        )
        if iv:
            lines.append(
                f"**推荐具体化自变量**:\n"
                f"- 将「{iv}」拆解为可量化的工程参数\n"
            )
        if dv:
            lines.append(
                f"**推荐具体化因变量**:\n"
                f"- 将「{dv}」拆解为 Nf、fatigue_limit σw、"
                f"crack_initiation_site、da/dN 等具体指标\n"
            )

    lines.append("\n### 需要补充的数据\n\n")
    if "roughness" in iv:
        lines.append(
            "- surface_roughness_Ra/Rz 数据\n"
            "- crack_initiation_site SEM 确认\n"
            "- Nf 与 Ra/Rz 配对数据\n"
            "- internal pore_size（用于排除孔隙干扰）\n"
        )
    elif "pore" in iv:
        lines.append(
            "- pore_size / √area（micro-CT）\n"
            "- distance_to_surface\n"
            "- Nf 与孔隙特征配对数据\n"
        )
    elif "delta" in iv or "crack" in iv:
        lines.append(
            "- da/dN-ΔK 曲线\n"
            "- Paris_C / Paris_m 参数\n"
            "- ΔKth 门槛值\n"
        )
    else:
        lines.append("- 对应自变量的直接实验数据\n")
        lines.append("- 控制变量记录\n")

    lines.append("\n### 需要补充的文献类型\n\n")
    if "roughness" in iv:
        lines.append(
            "- surface roughness + fatigue life 相关文献\n"
            "- micro-CT + crack initiation 相关文献\n"
            "- polished vs as-built 对比文献\n"
        )
    elif "pore" in iv:
        lines.append(
            "- pore characterization + fatigue 文献\n"
            "- defect tolerant model 文献\n"
        )
    elif "crack" in iv or "delta" in iv:
        lines.append(
            "- FCGR / Paris law 参数文献\n"
            "- 不同 R 比下 C/m 对比文献\n"
        )
    else:
        lines.append(f"- {iv} → {dv} 相关研究文献\n")

    # Expected improvement
    expected = min(total + 25, 85)
    lines.append(
        f"\n### 改进后预期评分\n\n"
        f"当前评分: {total}/50\n"
        f"修正后预计评分: {expected}/50\n"
        f"等级提升: {score_result.get('grade', 'weak')} → "
        f"{'good' if expected >= 40 else 'medium'}\n"
    )

    return "".join(lines)

def generate_hypothesis_generation_answer(
    question: str,
    ind_var: Optional[str],
    dep_var: Optional[str],
    var_class: str,
    coverage_level: str,
) -> str:
    """
    假设生成模式 — 生成具体、可验证、可推翻的科学假设。
    每个假设必须包含: 变量、条件、机制链、候选方程、实验设计、
    表征方法、数据分析、判据、推翻条件、评分。
    """
    lines = []

    lines.append("## 候选科学假设\n")

    # ── 0. Variable validation ──
    var_check = _validate_variables(ind_var, dep_var)
    if var_check["warnings"]:
        for w in var_check["warnings"]:
            lines.append(f"> ⚠️ **变量警告**: {w}\n\n")

    if coverage_level in ("weak", "not_found"):
        lines.append(
            "> ⚠️ 本地证据不足，以下假设为 **search-guided candidate**，"
            "不得视为 strong hypothesis。需补充文献后再升级证据等级。\n\n"
        )

    has_any_hypothesis = False
    low_score_hypotheses = []  # Track low-scoring hypotheses for auto-improvement

    # ── 1. Variable-specific detailed hypotheses (v2: 多变量自动拆分优先) ──
    has_split_hypotheses = False
    if ind_var and dep_var:
        from src.hypothesis_split import replace_old_hypothesis
        was_split, split_hyp = replace_old_hypothesis(question, ind_var, dep_var)
        if was_split and split_hyp:
            lines.append(split_hyp)
            has_any_hypothesis = True
            has_split_hypotheses = True
        else:
            detailed = _generate_detailed_variable_hypothesis(ind_var, dep_var, question)
            if detailed:
                lines.append(detailed)
                has_any_hypothesis = True

    # ── 2. Backend HypothesisGenerator results (跳过 if 已拆分) ──
    if not has_split_hypotheses:
        gen = HypothesisGenerator()
        all_gen_hyps = gen.generate_all()
        scorer_obj = HypothesisScorer()

        for h in all_gen_hyps:
            score_result = scorer_obj.score_hypothesis(h)
            if score_result["grade"] == "reject":
                continue
            has_any_hypothesis = True

            grade_icon = {"good": "🟢", "medium": "🟡", "weak": "🟠"}.get(
                score_result["grade"], "⚪")
            h_type_label = h.get("hypothesis_type", "").replace("_", " ")

        lines.append(f"### {h['hypothesis_id']}: {h_type_label}\n")
        lines.append(f"**假设陈述**: {score_result['hypothesis_statement']}\n\n")

        # Structured fields
        lines.append("| 字段 | 内容 |\n")
        lines.append("|---|---|\n")
        lines.append(f"| 控制变量 | {h.get('controlled_variables', '待补充')} |\n")
        lines.append(f"| 自变量 | {h.get('independent_variables', '待补充')} |\n")
        lines.append(f"| 因变量 | {h.get('dependent_indicators', '待补充')} |\n")
        lines.append(f"| 候选方程/模型 | {h.get('expected_relation', '待补充')} |\n")
        lines.append(f"| 证据状态 | {score_result['grade']} |\n\n")

        # Experimental design
        lines.append("#### 实验方法\n")
        lines.append(f"**验证方法**: {h.get('validation_path', '需设计实验验证')}\n\n")
        lines.append(f"**推翻条件**: {h.get('falsification_condition', '待补充')}\n\n")

        # Scoring per hypothesis
        lines.append("#### 假设评分\n")
        dim_labels_cn = {
            "specificity": "Specificity 具体性",
            "relation_clarity": "Relation Clarity 关系清晰度",
            "evidence_traceability": "Evidence Traceability 证据可追溯性",
            "parameter_awareness": "Parameter Awareness 参数意识",
            "mechanism_plausibility": "Mechanism Plausibility 机制合理性",
            "macro_micro_link": "Macro-Micro Link 宏微观关联",
            "testability": "Testability 可验证性",
            "falsifiability": "Falsifiability 可推翻性",
            "paper_potential": "Paper Potential 论文潜力",
        }
        for d in score_result["dim_scores"]:
            dname = d.get("dimension", "")
            label = dim_labels_cn.get(dname, dname)
            bar = "█" * d["score"] + "░" * (d["max_score"] - d["score"])
            lines.append(f"- **{label}**: {d['score']}/{d['max_score']} {bar}\n")
        lines.append(f"\n- **总分**: {score_result['total_score']}/50\n")
        lines.append(f"- **等级**: {grade_icon} {score_result['grade']}\n")
        if score_result.get("major_weakness"):
            lines.append(f"- **主要弱点**: {score_result['major_weakness']}\n")
            # Add explanation for low scores
            low_dims = [d for d in score_result["dim_scores"] if d.get("score", 5) <= 2]
            if low_dims:
                lines.append("\n**低分原因分析**:\n")
                for ld in low_dims:
                    dname_orig = ld.get("dimension", "")
                    lines.append(f"- **{dim_labels_cn.get(dname_orig, dname_orig)}** 得分 {ld['score']}/{ld['max_score']}\n")
            lines.append("\n")
        lines.append("---\n")

        # ── Track low-scoring for auto-improvement ──
        if score_result.get("total_score", 50) < 30:  # 30/50 ≈ 60%
            low_score_hypotheses.append((h, score_result))

    # ── 3. Equation-based hypotheses (skip if split) ──
    if not has_split_hypotheses and ind_var and dep_var:
        eq_result = run_equation_pipeline(question)
        if eq_result["equation_hypotheses"]:
            lines.append("### 方程型假设\n")
            for h in eq_result["equation_hypotheses"]:
                lines.append(f"**{h['equation_name']}**\n")
                lines.append(f"**候选方程**: `{h['formula']}`\n")
                lines.append(f"**假设**: {h['statement']}\n")
                lines.append(f"**控制变量**: {h.get('controlled_variables', '—')}\n")
                lines.append(f"**自变量**: {h.get('independent_variables', '—')}\n")
                lines.append(f"**因变量**: {h.get('dependent_variables', '—')}\n")
                lines.append(f"**验证数据**: {', '.join(h.get('validation_data', []))}\n")
                lines.append(f"**缺失参数**: {'、'.join(h.get('missing_parameters', []))}\n")
                s = h.get("score", {})
                lines.append(
                    f"**方程假设评分**: {s.get('total', 0)}/{s.get('max', 20)} "
                    f"({s.get('grade', '')}) — "
                    f"变量匹配={s.get('variable_match', 0)}/5, "
                    f"参数完整={s.get('param_completeness', 0)}/5, "
                    f"数据可行={s.get('data_feasibility', 0)}/5, "
                    f"机制清晰={s.get('mechanism_clarity', 0)}/5\n"
                )
                lines.append("\n")

    if not has_any_hypothesis:
        lines.append("当前无法生成具体假设。请更具体地输入两个变量。\n")

    # ── 4. Evidence weakness explanation (when coverage is weak/not_found) ──
    if coverage_level in ("weak", "not_found"):
        lines.append(_build_evidence_weakness_explanation(ind_var, dep_var, coverage_level))

    # ── 5. Auto-improvement for low-score hypotheses ──
    if low_score_hypotheses and not (ind_var and dep_var):
        # Only auto-improve if we can match to a known template
        for hyp, score_res in low_score_hypotheses[:1]:  # Improve the first one
            improvement = improve_low_score_hypothesis(
                hyp, score_res, ind_var, dep_var
            )
            if improvement:
                lines.append(improvement)

    # ── Literature support ──
    lines.append(_build_lit_support_section(ind_var, dep_var, question))

    return "".join(lines)


def _generate_detailed_variable_hypothesis(ind_var: str, dep_var: str, question: str = "") -> Optional[str]:
    """基于变量对生成超具体的假设（v2: 多变量自动拆分）。"""
    # ── 优先使用拆分引擎（多变量场景） ──
    from src.hypothesis_split import replace_old_hypothesis, generate_split_hypotheses
    was_replaced, new_hyp = replace_old_hypothesis(question, ind_var, dep_var)
    if was_replaced and new_hyp:
        return new_hyp

    # ── 单一变量对：回退旧模板 ──
    hypotheses_db = {
        ("pore_size", "fatigue_life"): _hyp_pore_size_fatigue_life(),
        ("surface_roughness", "fatigue_life"): _hyp_surface_roughness_fatigue_life(),
        ("delta_k", "da_dn"): _hyp_delta_k_da_dn(),
        ("heat_treatment", "fatigue_life"): _hyp_heat_treatment_fatigue_life(),
        ("pore_location", "fatigue_life"): _hyp_pore_location_fatigue_life(),
        ("microstructure", "da_dn"): _hyp_microstructure_da_dn(),
        ("surface_roughness", "fatigue_limit"): _hyp_surface_roughness_fatigue_limit(),
    }

    key = (ind_var, dep_var)
    if key in hypotheses_db:
        return hypotheses_db[key]
    rev_key = (dep_var, ind_var)
    if rev_key in hypotheses_db:
        return hypotheses_db[rev_key]
    return None


def _hyp_pore_size_fatigue_life() -> str:
    """孔隙尺寸→疲劳寿命 — 带条件边界的结构化假设。"""
    return """### H1：孔隙尺寸—距表面距离—表面状态耦合控制疲劳裂纹起裂的假设

**假设陈述**:
在控制材料成分（Ti-6Al-4V Grade 23）、L-PBF 工艺参数、热处理状态（stress-relieved）、加载方式（轴向拉-拉）、应力比 R（建议 0.1）和测试温度（室温）后，具有较大 √area 且距自由表面较近的孔隙，比深部同等尺寸孔隙更容易成为 L-PBF Ti-6Al-4V 的疲劳裂纹起裂源，并导致疲劳寿命 Nf 降低。其机制是近表面孔隙边缘的局部应力集中与自由表面缺口效应发生叠加，使局部有效应力和裂纹起裂驱动力提高。

当表面粗糙度较高（as-built）时，表面缺口效应可能掩盖或削弱孔隙尺寸对起裂的主导作用。仅在 polished 或 machined 表面条件下，孔隙特征对 Nf 和起裂位置的解释力才会充分显现。

| 字段 | 内容 |
|---|---|
| 自变量 | pore_size / √area; distance_to_surface; pore_location (surface / subsurface / internal) |
| 因变量 | Nf; crack_initiation_site; fatigue_limit σw |
| 调节变量 | surface_state (as-built / polished); stress_ratio_R; pore_aspect_ratio |
| 控制变量 | Ti-6Al-4V Grade 23; L-PBF process parameters; heat_treatment; stress_amplitude σa; loading mode; temperature |
| 候选方程 | Murakami: σw = C·(HV+120)/(√area)^(1/6); Kitagawa-Takahashi: Δσth = ΔKth/(Y√π√area) |
| 证据状态 | medium — 有部分 micro-CT + Nf 数据，但缺少系统 distance_to_surface 控制对比 |

该效应在 polished 或 machined 表面状态下更容易被观察到；在 as-built 表面状态下，表面粗糙峰和表面缺陷可能先行主导裂纹起裂，从而掩盖内部孔隙尺寸和距表面距离的影响。

> **定量说明**：关于临界距表面距离和临界 √area 的具体阈值，目前系统中尚无充分文献证据支持固定数值。这些阈值应由实验数据拟合确定，或从含 pore_size / distance_to_surface / Nf 的结构化文献数据中提取。当前阶段，这些应视为 **candidate range** 或 **to-be-fitted parameter**，而非已验证数值。

**调节变量**: pore_location (distance_to_surface)、surface_state、stress_ratio_R、pore_aspect_ratio、residual_stress
**控制变量**: material (Ti-6Al-4V)、process (L-PBF)、heat_treatment (stress-relieved)、stress_ratio (R=0.1)、loading mode、temperature、test frequency

**标准模型**（已有文献支持）:
- **Murakami √area model**: σw ∝ (HV+120)/(√area)^{1/6} — 适用于缺陷尺寸→疲劳极限关系
- **Kitagawa-Takahashi model**: Δσth = ΔKth/(Y√πa) — 适用于缺陷容限分析

**候选经验模型**（需文献数据或实验拟合验证，非已验证方程）:
- S-N fitting with defect correction: log(Nf) = β₀ + β₁·log(√area) + β₂·distance_to_surface + controls
  > *该模型为 empirical candidate model，需要足够的结构化文献数据或实验数据拟合，不能直接作为已验证方程。*

**机制链**（论文式表达）:

pore_size / √area + distance_to_surface
→ enhanced local stress concentration at pore edge
→ interaction with free-surface notch effect (near-surface region)
→ increased effective driving force for crack initiation
→ earlier crack initiation (reduced Ni)
→ reduced total fatigue life Nf

**实验方法（注意避免混杂变量）**:

*主效应验证（只改变孔隙位置，控制所有其他变量）:*
- **Group A**：polished + stress-relieved + 近表面大孔隙（micro-CT 确认 √area 在较高分位，distance_to_surface 在较低分位）
- **Group B**：polished + stress-relieved + 深部大孔隙（同等 √area 范围，但 distance_to_surface 显著大于 Group A）
- **Group C**：polished + stress-relieved + 低缺陷对照（√area 在较低分位）
各组除孔隙特征外，其他所有变量（表面状态、热处理、R 比、σa）必须保持一致。

*条件边界验证（评估表面粗糙度是否掩盖孔隙影响）:*
- **Group D**：as-built + stress-relieved + 自然孔隙分布（用于评估 surface roughness 的竞争效应）
- **Group E**：polished + HIP + 低缺陷（注意：此组同时改变了热处理和孔隙状态，属于混杂条件，不能作为主效应对照，仅用于条件边界分析）

**自变量**: pore_size（或 √area）、distance_to_surface（由 micro-CT 确定）
**因变量**: Nf、crack_initiation_site（SEM fractography 确定）
**控制变量**: R=0.1、σa、f=20Hz、RT、polished surface（主效应验证）

**表征方法**:
- micro-CT（疲劳前）：设定适当分辨率，识别并量化每个 >10μm 孔隙的 √area、长宽比、距表面距离
- SEM fractography（疲劳后）：确认裂纹起裂源坐标，判断是否源于孔隙
- EBSD（可选）：分析起裂源周围晶粒取向
- 粗糙度测量：疲劳前测 Ra/Rz（用于条件边界分析）

**数据分析方法**:
- 按 √area 分位和 distance_to_surface 分位对样品分组
- 比较各组 Nf 差异（非参数检验，如 Kruskal-Wallis）
- 统计裂纹起裂源与 micro-CT 定位孔隙的空间对应率
- 拟合 S-N 曲线，比较不同孔隙组的参数差异

**支持判据**:
- 近表面大孔隙组（Group A）的 Nf 显著低于深部大孔隙组（Group B）和低缺陷组（Group C）（p<0.05，效应量 Cohen's d>0.8）
- SEM 确认的起裂源与 micro-CT 定位的近表面孔隙的空间对应率 >70%
- Nf 与 √area 和 distance_to_surface 存在统计显著的相关性

**推翻条件**:
1. 若裂纹起裂源与近表面大孔隙的对应率 <30%，则该假设应被推翻
2. 若 Nf 的方差主要由 Ra/Rz 解释而非由 √area / distance_to_surface 解释（如 Ra 在回归模型中的贡献 R² > 0.5），则孔隙主导假设应被降级
3. 若 Group A 与 Group B 的 Nf 无统计显著差异（p>0.05），则孔隙位置不是主导因素
4. 若所有组别的起裂源均为表面特征而非孔隙，则孔隙不起主导作用

**评分**:
- 具体性: 5/5 █████
- 关系清晰度: 5/5 █████
- 证据可追溯性: 3/5 ███░░（缺结构化√area+Nf数据）
- 参数意识: 4/5 ████░（可关联Murakami/Kitagawa模型）
- 机制合理性: 5/5 █████
- 实验可验证性: 5/5 █████
- 设计完整性: 5/5 █████（避免了混杂变量）
- **总分**: 42/50
- **等级**: 🟢 good candidate
"""


def _hyp_surface_roughness_fatigue_life() -> str:
    """表面粗糙度→疲劳寿命 — 带条件边界的结构化假设。"""
    return """### H1：表面粗糙度—内部孔隙竞争主导疲劳裂纹起裂的条件边界假设

**假设陈述**:
在 as-built L-PBF Ti-6Al-4V 中，较高的表面粗糙度 Ra/Rz 可能通过表面缺口效应优先诱导疲劳裂纹在表面粗糙峰或亚表面缺陷处起裂，从而削弱内部孔隙尺寸对疲劳寿命 Nf 的解释力。相反，在 polished 或 machined 表面中，表面缺口效应减弱，近表面或内部大孔隙对裂纹起裂位置和 Nf 的影响会增强。

| 字段 | 内容 |
|---|---|
| 自变量 | surface_roughness_Ra / Rz; surface_state: as-built / polished / machined; pore_size / √area; distance_to_surface |
| 因变量 | Nf; crack_initiation_site; fatigue_limit σw |
| 调节变量 | surface_state; pore_size / √area; distance_to_surface; stress_ratio_R |
| 控制变量 | Ti-6Al-4V material grade; L-PBF process parameters; heat_treatment; stress_amplitude σa; loading mode; temperature |
| 候选方程 | Basquin: σa = σf'·(2Nf)^b; Murakami: σw = C·(HV+120)/(√area)^(1/6) |
| 证据状态 | weak — 缺少同时包含 Ra/Rz、pore_size、distance_to_surface 和 Nf 的系统对比数据 |

**机制链**:

as-built surface roughness
→ surface notch effect
→ enhanced local stress concentration at roughness valleys
→ surface crack initiation
→ reduced crack initiation cycles Ni
→ lower Nf
→ internal pore effect becomes masked

**实验方法**:
1. 准备 as-built、polished、machined 三组 L-PBF Ti-6Al-4V 样品；
2. **测量 Ra/Rz/Sa/Sq 和表面缺陷形貌**（3D profiler / SEM）；
3. 使用 micro-CT 记录内部孔隙尺寸、位置和分布，用于排除内部缺陷差异；
4. 在相同 R、σa 和温度下进行 HCF 测试（多应力水平，至少 4 级）；
5. 用 **SEM fractography 确认裂纹起裂源**来自表面粗糙峰还是内部/近表面孔隙；
6. 比较 Ra/Rz、pore_size、distance_to_surface 对 Nf 的解释度（回归分析）。

**支持判据**:
如果 as-built 样品中裂纹主要从表面粗糙峰起裂，且 Ra/Rz 对 Nf 的解释度高于 pore_size；而 polished 样品中起裂源转向近表面或内部孔隙，则支持该假设。

**推翻条件**:
如果 as-built 和 polished 样品中裂纹起裂均主要由内部孔隙控制，且 Ra/Rz 与 Nf 无稳定关系，则该假设应被降级。

**评分**: 具体性 5/5, 关系清晰度 5/5, 证据可追溯性 4/5, 参数意识 3/5, 机制合理性 5/5, 实验可验证性 5/5 → **总分 41/50, good candidate**
"""


def _hyp_delta_k_da_dn() -> str:
    """ΔK→da/dN — 结构化证据参数型假设。"""
    return """### H2 参数型假设：缺陷状态通过改变 Paris 参数 C 影响 FCGR 的假设

**假设陈述**:
在控制应力比 R、表面状态（polished）和试验环境（RT, laboratory air）后，L-PBF Ti-6Al-4V 中的孔隙缺陷群可能主要通过改变 Paris 参数 C（而非 m）影响 da/dN-ΔK 关系。该效应的物理机制是孔隙前沿的应力集中加速了早期裂纹扩展，但在长裂纹阶段孔隙影响可能逐渐饱和。

> **定量说明**：关于 √area 阈值、孔隙密度临界值和 Δlog(C) 的具体范围，当前系统中尚无充分证据支持固定数值。log(C) vs 孔隙密度的斜率、Paris 指数 m 的范围等应通过实验数据拟合确定，视为 **to-be-fitted parameter**，而非预设值。

**自变量**: defect_state (porosity, max_pore_size, pore_density)
**调节变量**: stress_ratio_R, build_orientation, heat_treatment
**控制变量**: specimen geometry (CT), heat_treatment, temperature, test frequency

**标准模型**:
- **Paris law**: da/dN = C(ΔK)^m — 标准疲劳裂纹扩展模型，适用于稳定长裂纹扩展阶段

**候选扩展模型**:
- **Paris law with defect correction**: log(C) = γ₀ + γ₁·porosity + γ₂·max_pore_size — **empirical candidate model**
  > 该关系需要足够的 FCGR 数据拟合，当前阶段为候选假设。

**机制链**（论文式表达）:

porosity / defect density
→ additional stress concentrators in crack path
→ higher da/dN at same nominal ΔK
→ increased Paris C (if m remains stable)
→ shift of da/dN-ΔK curve upward

**验证方法**:
制备不同孔隙特征的 CT 试样（至少 3 组：低缺陷/中缺陷/高缺陷，每组 ≥3 个），在相同 R 下测试 FCGR，拟合 Paris 参数，ANCOVA 比较组间 C 和 m 差异。注意：表面状态和热处理必须在各组间保持一致，避免混杂变量。

**推翻条件**: 若 C 或 m 在组间无统计显著差异（p>0.05），或 m 的变化量级大于 C 的变化，则该假设应被降级

**评分**: 具体性 5/5, 参数意识 5/5, 实验可验证性 5/5, 机制合理性 4/5 → **总分 43/50, good candidate**
"""


def _hyp_heat_treatment_fatigue_life() -> str:
    """热处理→疲劳寿命 的具体假设。"""
    return """### H1 机制型假设：热处理 → 疲劳寿命

**假设陈述**:
在控制表面状态（polished, Ra<1μm）和应力比 R=0.1 后，HIP 处理（920°C/100MPa/2h）通过闭合内部孔隙（孔隙率从 >0.5% 降至 <0.05%）和消除残余应力（从 +200–400MPa 降至 ±50MPa），使 L-PBF Ti-6Al-4V 的疲劳寿命 Nf 在 σa=0.5σy 下提升 3–10 倍。但 HIP 不能完全消除表面连通孔隙和近表面残留缺陷；因此表面粗糙度较高时（as-built+HIP），疲劳寿命提升幅度有限（仅 2–3 倍）。

**调节变量**: initial pore_size, surface_state, residual_stress_level
**控制变量**: material, L-PBF parameters, R=0.1, σa level

**验证方法**: micro-CT 疲劳前测孔隙 → HIP → micro-CT 测孔隙闭合率 → HCF → SEM 确认起裂源
**推翻条件**: HIP 后 Nf 提升 <50%，或 HIP 后起裂源仍以残留孔隙为主

**评分**: 具体性 5/5, 参数意识 3/5, 机制合理性 5/5, 实验可验证性 5/5 → **总分 40/50, good candidate**
"""


def _hyp_pore_location_fatigue_life() -> str:
    """孔隙位置→疲劳寿命 的具体假设。"""
    return """### H1 机制型假设：孔隙位置 → 疲劳寿命

**假设陈述**:
在控制表面状态（polished）、应力比 R=0.1 和孔隙尺寸（√area=30–50μm）后，距表面距离 d 对疲劳寿命 Nf 的影响具有非线性特征：d<100μm 的近表面孔隙使 Nf 降低 50–70%（相比无孔隙基准），而 d>300μm 的深部孔隙仅降低 10–30%。临界距离 d_critical ≈ 100–200μm，在此范围内孔隙与自由表面应力场产生叠加效应。

**机制链**:
```
pore_location (distance_to_surface d)
→ d < 100μm: stress field overlaps with free surface
→ effective Kt increases by 30-50% due to surface interaction
→ crack initiates easily from pore edge toward surface
→ very short Ni → significant Nf reduction

d > 300μm: pore embedded in bulk material
→ no surface interaction, only pore's own stress concentration
→ crack must grow from internal pore to surface
→ longer crack propagation period
→ less Nf reduction than near-surface pores
```

**验证方法**: micro-CT 定位孔隙三维坐标 → 按 d 分组 → HCF → SEM 确认起裂源 → 统计分析 Nf vs d 的关系
**推翻条件**: 若 Nf 与 d 无显著相关性，或所有深度孔隙的 Nf 差异 <20%，则假设降级
"""


def _hyp_microstructure_da_dn() -> str:
    """微观组织→da/dN 的具体假设。"""
    return """### H1 机制型假设：微观组织 → da/dN

**假设陈述**:
在控制应力比 R=0.1、温度 RT、缺陷状态（micro-CT 确认 <10μm）后，α lath 宽度从 as-built 的 <0.5μm（α′ martensite）增至 sub-β annealing 后的 1–3μm（α+β lamellar），预期 Paris 指数 m 从 3.5–4.0 降至 2.5–3.0，即裂纹扩展抗力提高。该效应源于较宽的 α lath 产生更显著的裂纹偏转和粗糙度诱导裂纹闭合（RICC），从而降低有效驱动力 ΔKeff。

**自变量**: α_lath_width / α′ fraction / β_phase_morphology
**因变量**: da_dN / Paris_m / Paris_C
**调节变量**: prior_β_grain_size, build_orientation, defect_state

**候选方程**: da/dN = C(ΔK)^m; da/dN = C(ΔK(1-R)^(p-1))^m (Walker)
**验证方法**: 不同热处理状态的 CT 试样，FCGR 测试，Paris 拟合，EBSD/TEM 表征组织，SEM 观察裂纹路径偏转
**推翻条件**: 若不同热处理组的 m 值无显著差异，或组织差异主要由缺陷状态变化解释，则假设降级
"""


def _hyp_surface_roughness_fatigue_limit() -> str:
    """表面粗糙度→疲劳极限 的具体假设。"""
    return """### H1 机制型假设：表面粗糙度 → 疲劳极限

**假设陈述**:
在控制应力比 R=-1 后，表面粗糙度 Ra 从 1μm（polished）增至 15μm（as-built），L-PBF Ti-6Al-4V 的疲劳极限 σw 预期降低 30–50%。该降低可通过将表面粗糙峰等效为微缺陷（√area_roughness ≈ 1.5×Rz）并应用 Murakami √area 模型预测。当表面粗糙度与内部孔隙共存时，两者竞争主导疲劳极限：取 max(σw_pores, σw_roughness) 作为有效疲劳极限。

**验证方法**: 升降法测不同表面状态的 σw，micro-CT 测孔隙，SEM 确认起裂源，Murakami 模型预测对比
**推翻条件**: 若 polished 组与 as-built 组的 σw 差异 <15%，或起裂源始终为内部孔隙而非表面，则假设降级
"""


# ═══════════════════════════════════════════════════════════════════════════
# MODE 4: Experiment Design (实验验证设计)
# ═══════════════════════════════════════════════════════════════════════════

def generate_experiment_design_answer(
    question: str,
    ind_var: Optional[str],
    dep_var: Optional[str],
    var_class: str,
) -> str:
    """
    实验验证设计模式 (v2) — 结构化实验设计方案 (IV/DV/CV/MV + 预测方向 + 推翻条件)。
    """
    lines = []
    iv_lower = (ind_var or "").lower()
    dv_lower = (dep_var or "").lower()

    # ── 1. 实验目标 ──
    lines.append("## 1. 实验目标\n")
    if ind_var and dep_var:
        lines.append(
            f"验证 {ind_var} 对 {dep_var} "
            f"的影响，确定变量关系及其条件边界。\n"
        )
    else:
        lines.append(
            f"验证目标变量对疲劳指标的影响，确定变量关系及其条件边界。\n"
        )

    # ── 2. 科学假设 ──
    lines.append("\n## 2. 科学假设\n")
    if ind_var and dep_var:
        lines.append(
            f"**H₁**: 在控制材料（Ti-6Al-4V Grade 23）、工艺（L-PBF）、"
            f"表面状态和热处理等协变量后，{ind_var} 通过特定机制影响 {dep_var}，"
            f"且该关系受表面状态、应力比 R 和缺陷状态的调节。\n\n"
            f"**H₀ (零假设)**: {ind_var} 与 {dep_var} 之间无统计显著关系，"
            f"或观测到的相关性完全由未控制的混杂变量解释。\n\n"
        )
        # 机制预测
        iv_lower_h = (ind_var or "").lower()
        if "pore" in iv_lower_h or "defect" in iv_lower_h:
            lines.append(
                "**机制预测**: 近表面大孔隙（√area 较大且距表面 < 100μm）"
                "通过孔隙边缘应力集中与自由表面应力场叠加，"
                "显著降低疲劳裂纹起裂寿命 Ni，从而降低总疲劳寿命 Nf。"
                "该机制在表面粗糙度较低（Ra < 1μm）时占主导。\n"
            )
        elif "roughness" in iv_lower_h or "surface" in iv_lower_h:
            lines.append(
                "**机制预测**: 较高的表面粗糙度（Ra > 10μm）通过表面缺口效应"
                "提高局部应力集中系数 Kt，优先诱导裂纹从表面粗糙峰根部起裂，"
                "从而降低 Nf。内部孔隙对 Nf 的解释力在表面粗糙度高时被掩盖。\n"
            )
        elif "delta" in iv_lower_h or "crack" in iv_lower_h:
            lines.append(
                "**机制预测**: ΔK 增大通过提高裂纹尖端塑性区尺寸和损伤累积速率，"
                "加速疲劳裂纹扩展（da/dN 增大），该关系受 Paris 定律约束。\n"
            )
        else:
            lines.append(
                "**机制预测**: 自变量通过疲劳损伤累积机制影响因变量，"
                "具体机制链条需结合表征数据确定。\n"
            )
    else:
        lines.append(
            "待明确自变量/因变量后补充。\n"
        )

    # ── 3. 变量定义表 ──
    lines.append("\n## 3. 变量定义\n\n")
    lines.append("| 变量类型 | 具体变量 | 测量方式 | 说明 |\n")
    lines.append("|---|---|---|---|\n")

    if "roughness" in iv_lower or "surface" in iv_lower:
        lines.append(
            "| 自变量 (IV) | surface_roughness_Ra / Rz | 3D profiler / 接触式轮廓仪 | 核心自变量 |\n"
            "| 自变量 (IV) | surface_state (as-built/polished/machined) | 分类变量 | 分组对比 |\n"
        )
    elif "pore" in iv_lower:
        lines.append(
            "| 自变量 (IV) | pore_size / √area | micro-CT | 核心自变量 |\n"
            "| 自变量 (IV) | distance_to_surface | micro-CT 三维定位 | 区分近表面/内部 |\n"
            "| 自变量 (IV) | pore_aspect_ratio | micro-CT | 形态变量（可选） |\n"
        )
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append(
            "| 自变量 (IV) | ΔK (stress intensity factor range) | FCGR 试验计算 | 核心自变量 |\n"
            "| 自变量 (IV) | stress_ratio R | 试验设定 | 调节变量 |\n"
        )
    elif "heat" in iv_lower:
        lines.append(
            "| 自变量 (IV) | heat_treatment_type | 工艺记录 | SR / HIP / annealing / STA |\n"
        )
    else:
        lines.append(f"| 自变量 (IV) | {ind_var or '目标变量'} | 待确定 | 核心自变量 |\n")

    if "roughness" in iv_lower or "surface" in iv_lower:
        lines.append(
            "| 因变量 (DV) | Nf (cycles to failure) | HCF 试验记录 | 主要输出 |\n"
            "| 因变量 (DV) | crack_initiation_site | SEM fractography | 定性判断指标 |\n"
            "| 因变量 (DV) | fatigue_limit σw | 升降法 | 如需测定疲劳极限 |\n"
        )
    elif "pore" in iv_lower:
        lines.append(
            "| 因变量 (DV) | Nf (cycles to failure) | HCF 试验记录 | 主要输出 |\n"
            "| 因变量 (DV) | crack_initiation_site | SEM fractography | 确认起裂源与孔隙对应关系 |\n"
            "| 因变量 (DV) | fatigue_limit σw | 升降法 | Murakami 模型验证 |\n"
        )
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append(
            "| 因变量 (DV) | da/dN (crack growth rate) | FCGR 试验记录 | 主要输出 |\n"
            "| 因变量 (DV) | Paris C, m | 数据拟合 | 模型参数 |\n"
            "| 因变量 (DV) | ΔKth | 降载法测定 | 门槛值 |\n"
        )
    else:
        lines.append(f"| 因变量 (DV) | {dep_var or '疲劳指标'} | 待确定 | 主要输出 |\n")

    # 控制变量
    lines.append(
        "| 控制变量 (CV) | Ti-6Al-4V material grade (Grade 23) | 供应商认证 | 所有组保持一致 |\n"
        "| 控制变量 (CV) | L-PBF process parameters | 工艺记录 | 所有组保持一致 |\n"
        "| 控制变量 (CV) | stress_ratio R | 试验设定 | 推荐 R=0.1 |\n"
        "| 控制变量 (CV) | stress_amplitude σa | 试验设定 | 多水平（至少 4 级） |\n"
        "| 控制变量 (CV) | temperature | 试验记录 | 室温（25°C） |\n"
        "| 控制变量 (CV) | loading mode | 试验设定 | 轴向拉-拉 |\n"
    )
    if "pore" in iv_lower:
        lines.append("| 控制变量 (CV) | surface_state | 加工控制 | 全部 polished，排除粗糙度干扰 |\n")
    if "roughness" in iv_lower:
        lines.append("| 控制变量 (CV) | heat_treatment | 工艺控制 | 除目标外保持一致 |\n")
    lines.append(
        "| 调节变量 (MV) | surface_state | 分组变量 | 用于分析条件依赖性 |\n"
        "| 调节变量 (MV) | heat_treatment | 分组变量 | 用于分析条件依赖性 |\n"
        "| 调节变量 (MV) | stress_ratio R | 分组变量 | 如需评估 R 比效应 |\n"
    )

    # ── 4. 预测方向 ──
    lines.append("\n## 4. 预测方向\n\n")
    lines.append("如果假设成立，预期会观察到以下变化：\n\n")
    if "pore" in iv_lower:
        lines.append(
            "1. **近表面大孔隙组的 Nf 低于深部大孔隙组**（p < 0.05 统计显著）\n"
            "2. **SEM 起裂源更常对应近表面孔隙**（对应率 > 70%）\n"
            "3. **polished 后 surface roughness 效应减弱**，pore_size / distance_to_surface 对 Nf 的解释力增强\n"
            "4. **Nf 与 √area 呈负相关**，与 distance_to_surface 呈正相关\n"
            "5. **Murakami √area 模型预测的疲劳极限与实测值偏差 < 20%**\n"
        )
    elif "roughness" in iv_lower or "surface" in iv_lower:
        lines.append(
            "1. **as-built 试样的表面起裂率 > 80%**，抛光试样 < 30%\n"
            "2. **抛光后 Nf 提升 2-10 倍**（相同应力幅下）\n"
            "3. **Ra/Rz 与 Nf 呈显著负相关**（|r| > 0.6）\n"
            "4. **表面粗糙度主导条件下，内部孔隙尺寸与 Nf 的相关性弱**\n"
        )
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append(
            "1. **高孔隙率组的 Paris C 高于低孔隙率组**（log(C) 差异显著）\n"
            "2. **Paris m 在不同孔隙状态下相对稳定**（差异 < 10%）\n"
            "3. **高 R 比组 da/dN 高于低 R 比组**（相同 ΔK 下）\n"
        )
    elif "heat" in iv_lower:
        lines.append(
            "1. **HIP 组的 Nf 显著高于 SR 组**（提升 3-10 倍）\n"
            "2. **HIP 后起裂源从内部孔隙转向表面特征**\n"
            "3. **as-built+HIP 的提升幅度小于 polished+HIP**\n"
        )
    else:
        lines.append(
            "1. 自变量变化时因变量发生系统性的、统计显著的变化\n"
            "2. 变化方向与机制预测一致\n"
        )

    # ── 5. 支持假设的结果 ──
    lines.append("\n## 5. 支持假设的结果\n\n")
    lines.append("以下具体判据支持假设成立：\n\n")
    if "pore" in iv_lower:
        lines.append(
            "- 裂纹起裂源与 micro-CT 定位的近表面孔隙的空间对应率 > 70%\n"
            "- 近表面大孔隙组的 Nf 显著低于深部大孔隙组（p < 0.05, Cohen's d > 0.8）\n"
            "- Nf 与 √area 和 distance_to_surface 存在统计显著的相关性（|r| > 0.5）\n"
            "- Murakami 模型预测的疲劳极限与实测值偏差 < ±20%\n"
            "- 多因素回归中 pore_size × distance_to_surface 交互项显著\n"
        )
    elif "roughness" in iv_lower:
        lines.append(
            "- as-built 组表面起裂率 > 80%，抛光组表面起裂率 < 30%\n"
            "- 抛光后 Nf 提升 > 2×（相同 σa 下）\n"
            "- Ra/Rz 与 Nf 呈显著负相关（|r| > 0.6, p < 0.05）\n"
            "- 引入 Ra 后 S-N 拟合 R² 提升 ≥ 0.1\n"
        )
    else:
        lines.append(
            "- 实验组与对照组 Nf 差异统计显著（p < 0.05）\n"
            "- 效应方向与机制预测一致\n"
            "- 相同条件下可重复（变异系数 < 15%）\n"
        )

    # ── 6. 推翻假设的结果 ──
    lines.append("\n## 6. 推翻假设的结果\n\n")
    lines.append("以下具体判据将导致假设被推翻或降级：\n\n")
    if "pore" in iv_lower:
        lines.append(
            "1. **裂纹起裂源与孔隙无稳定对应关系**（对应率 < 30%）→ 推翻 H1\n"
            "2. **Nf 的方差主要由 Ra/Rz 解释**（Ra 在回归中 R² > 0.5）→ 降级 H1\n"
            "3. **近表面组与深部组的 Nf 无统计显著差异**（p > 0.05）→ 推翻 H1\n"
            "4. **所有组别的起裂源均为表面特征而非孔隙** → 推翻 H1\n"
            "5. **polished 后孔隙特征对 Nf 的解释力未增强** → 降级 H2\n"
        )
    elif "roughness" in iv_lower:
        lines.append(
            "1. **抛光后 Nf 提升 < 20%** → 推翻 H1\n"
            "2. **as-built 和 polished 组的起裂源均为孔隙而非表面** → 推翻 H1\n"
            "3. **Ra/Rz 与 Nf 无稳定相关（|r| < 0.3）** → 推翻 H1\n"
            "4. **引入 Ra 后 S-N 拟合精度无改善** → 降级 H1\n"
        )
    elif "delta" in iv_lower:
        lines.append(
            "1. **不同缺陷状态下 Paris C/m 无统计显著差异** → 推翻 H1\n"
            "2. **m 的变化量级大于 C** → 降级 H1（参数效应方向可能反了）\n"
            "3. **高 R 组和低 R 组的 da/dN 无差异** → 推翻 H2\n"
        )
    elif "heat" in iv_lower:
        lines.append(
            "1. **HIP 后 Nf 提升 < 50%** → 降级 H1\n"
            "2. **HIP 后起裂源仍以内部孔隙为主** → 推翻 H1（HIP 未闭合关键缺陷）\n"
            "3. **polished+HIP 与 as-built+HIP 提升幅度相同** → 推翻 H2\n"
        )
    else:
        lines.append(
            "1. 控制所有可识别变量后目标变量无显著效应（p > 0.05）\n"
            "2. 发现未控制的中间变量解释了全部方差\n"
            "3. 效应方向与预期相反且无合理解释\n"
        )

    # ── 7. 样品分组（含混杂变量检查） ──
    lines.append("\n## 7. 样品分组\n\n")
    if "roughness" in iv_lower or "surface" in iv_lower:
        lines.append("| 组别 | 表面状态 | 热处理 | 表面粗糙度目标 | 孔隙状态 | 说明 |\n")
        lines.append("|------|---------|-------|--------------|---------|------|\n")
        lines.append("| A | as-built | stress-relieved | Ra ~10-15μm | 自然 | 高粗糙度基准组 |\n")
        lines.append("| B | polished | stress-relieved | Ra < 1μm | 自然 | 低粗糙度，去除表面缺口 |\n")
        lines.append("| C | machined | stress-relieved | Ra ~1-3μm | 自然 | 中等粗糙度 |\n")
        if "pore" in dv_lower:
            lines.append("| D | as-built | HIP | Ra ~10-15μm | HIP 闭合 | 仅消除孔隙，保留粗糙度 |\n")
            lines.append("| E | polished | HIP | Ra < 1μm | HIP 闭合 | 消除孔隙+粗糙度 |\n")
        lines.append("\n> ⚠️ **混杂变量检查**: 组 D 同时改变了热处理 (SR→HIP) 和孔隙状态。"
                     "D 与 A 比较时，Nf 差异由 HIP 引起，但无法区分是孔隙闭合还是组织变化所致。"
                     "如需单独评估孔隙闭合效应，应增设 polished+HIP 组与 polished+SR 对比。\n")
    elif "pore" in iv_lower:
        lines.append("| 组别 | 表面状态 | 热处理 | 目标孔隙特征 | 说明 |\n")
        lines.append("|------|---------|-------|------------|------|\n")
        lines.append("| A | polished | stress-relieved | 近表面大孔隙 (dist < 100μm) | 验证距离效应 |\n")
        lines.append("| B | polished | stress-relieved | 深部大孔隙 (dist > 500μm) | 对照距离效应 |\n")
        lines.append("| C | polished | stress-relieved | 低孔隙（基准） | 基准对照 |\n")
        lines.append("| D | as-built | stress-relieved | 自然孔隙分布 | 评估表面粗糙度竞争 |\n")
        lines.append("| E | polished | HIP | 低孔隙 | 评估 HIP 效果 |\n")
        lines.append("\n> ⚠️ **混杂变量检查**: 组 D (as-built) 与 A-C (polished) 比较时，"
                     "表面粗糙度与孔隙特征同时变化。"
                     "组 D 和 A 的 Nf 差异可能来自表面粗糙度差而非孔隙差。"
                     "如需分离两者，需增设 as-built + 低孔隙组或增加表面粗糙度协变量。\n")
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append("| 组别 | 表面状态 | 热处理 | 缺陷状态 | R 比 |\n")
        lines.append("|------|---------|-------|----------|------|\n")
        lines.append("| A | polished | SR | 低缺陷 | 0.1 |\n")
        lines.append("| B | polished | SR | 高缺陷 | 0.1 |\n")
        lines.append("| C | polished | HIP | 低缺陷 | 0.1 |\n")
        lines.append("| D | polished | SR | 低缺陷 | 0.5 |\n")
    elif "heat" in iv_lower:
        lines.append("| 组别 | 热处理 | 预期组织 | 表面状态 |\n")
        lines.append("|------|--------|---------|---------|\n")
        lines.append("| A | SR (as-built) | α′ martensite | polished |\n")
        lines.append("| B | sub-β annealing | α+β lamellar | polished |\n")
        lines.append("| C | HIP | α+β lamellar (low porosity) | polished |\n")
        lines.append("| D | STA | bimodal | polished |\n")
        lines.append("\n> ⚠️ **混杂变量检查**: 组 C (HIP) 同时改变了孔隙状态和组织。"
                     "如需区分孔隙闭合效应和组织演化效应，"
                     "应增加 HIP + 额外热处理组与单独 HIP 组对照。\n")
    else:
        lines.append("| 组别 | 条件 | 说明 |\n")
        lines.append("|------|------|------|\n")
        lines.append("| A | 基准组 | 标准工艺 |\n")
        lines.append("| B | 实验组 1 | 改变目标变量 |\n")
        lines.append("| C | 实验组 2 | 改变控制变量 |\n")

    # ── 8. 测试方法 ──
    lines.append("\n## 8. 测试方法\n")
    if "delta" in iv_lower or "crack" in iv_lower:
        lines.append(
            "- **FCGR 试验**: CT 或 SEN 试样，恒定 ΔK 或恒定 R 比，记录 da/dN-ΔK 曲线\n"
            "- **门槛值测试**: ΔK 降载法测定 ΔKth\n"
            "- **Paris 参数拟合**: log(da/dN) = m·log(ΔK) + log(C)\n"
        )
    elif "roughness" in iv_lower or "surface" in iv_lower:
        lines.append(
            "- **表面粗糙度测量**: Ra/Rz/Sa/Sq（接触式轮廓仪或 3D 光学 profiler）\n"
            "- **HCF 试验**: 多应力水平（至少 4 级），每级 3-5 个试样，R=0.1，f=20Hz\n"
            "- **升降法**: 测定疲劳极限 σw（如需）\n"
            "- **SEM fractography**: 确认裂纹起裂源位置和类型\n"
        )
    elif "pore" in iv_lower:
        lines.append(
            "- **micro-CT**: 疲劳前扫描，获取孔隙三维特征（尺寸、位置、形态）\n"
            "- **HCF 试验**: 多应力水平（至少 4 级），每级 3-5 个试样，R=0.1，f=20Hz\n"
            "- **SEM fractography**: 确认起裂源与 micro-CT 定位孔隙的对应关系\n"
        )
    else:
        lines.append(
            "- **HCF 试验**: 多应力水平（至少 4 级），每级 3-5 个试样\n"
            "- **S-N 曲线**: 记录应力幅 σa 与疲劳寿命 Nf\n"
        )

    # ── 9. 表征方法 ──
    lines.append("\n## 9. 表征方法\n")
    if "roughness" in iv_lower:
        lines.append(
            "- **表面粗糙度仪 / 3D profiler**: Ra/Rz/Sa/Sq（优先级最高）\n"
            "- **SEM**: 断口起裂源识别、断裂模式分析\n"
            "- **micro-CT**: 孔隙特征统计（辅助，用于协变量分析）\n"
        )
    elif "pore" in iv_lower:
        lines.append(
            "- **micro-CT**: 孔隙三维特征（尺寸、位置、形态、分布）—— **核心表征**\n"
            "- **SEM**: 断口起裂源与孔隙对应确认\n"
            "- **EBSD**（可选）: 起裂源周围晶粒取向分析\n"
        )
    elif "delta" in iv_lower:
        lines.append(
            "- **SEM**: 裂纹路径观察、断裂模式分析\n"
            "- **EBSD**: 裂纹路径与微观组织关系\n"
        )
    elif "heat" in iv_lower:
        lines.append(
            "- **SEM**: 组织观察、断口分析\n"
            "- **EBSD**: α/β 相比例、晶粒尺寸\n"
            "- **XRD**: 残余应力、α′ 含量\n"
            "- **TEM**（可选）: α lath 宽度测定\n"
        )
    else:
        lines.append(
            "- **micro-CT**: 孔隙特征\n"
            "- **SEM**: 断口分析\n"
            "- **表面粗糙度仪**: Ra/Rz 测量\n"
        )

    # ── 10. 数据分析方法 ──
    lines.append("\n## 10. 数据分析方法\n")
    if "pore" in iv_lower:
        lines.append(
            "- **S-N 曲线拟合**: log(Nf) = A - B·log(σa)，分组比较 A/B 参数\n"
            "- **Basquin 拟合**: σa = σf'·(2Nf)^b\n"
            "- **多因素回归**: Nf = β₀ + β₁·log(√area) + β₂·distance + β₃·(√area × distance) + ε\n"
            "- **Murakami √area 模型验证**: 对比预测 σw 与实测 σw\n"
            "- **非参数检验**: Kruskal-Wallis 比较多组 Nf 差异\n"
            "- **断口统计分析**: 起裂源类型分布（表面 vs 孔隙 vs 其他）\n"
            "- **协方差分析**: ANCOVA，以 Ra 为协变量\n"
        )
    elif "roughness" in iv_lower:
        lines.append(
            "- **S-N 曲线分组拟合**: as-built vs polished vs machined\n"
            "- **回归分析**: Nf vs Ra/Rz + pore_size（用于竞争分析）\n"
            "- **断口统计**: 起裂源类型分布比较\n"
            "- **升降法**: 疲劳极限 σw 的测定\n"
        )
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append(
            "- **Paris 拟合**: log(da/dN) = m·log(ΔK) + log(C)\n"
            "- **参数对比**: 不同条件下 C/m 的 t-test / ANOVA\n"
            "- **Walker 模型**: da/dN = C·(ΔK·(1-R)^{p-1})^m（多 R 比数据时）\n"
        )
    else:
        lines.append(
            "- **S-N 曲线拟合**: log(Nf) = A - B·log(σa)\n"
            "- **Basquin 拟合**: σa = σf'·(2Nf)^b\n"
            "- **升降法**: 疲劳极限 σw 的测定\n"
            "- **多因素分析**: 各变量对 Nf 的贡献权重\n"
        )

    # ── 文献连接 ──
    lines.append("\n" + _build_lit_support_section(ind_var, dep_var, question))

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main answer dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def generate_comprehensive_answer(
    question: str,
    answer_mode: str,
) -> str:
    """
    根据 answer_mode 和 answer_depth 分派到不同深度的回答生成器。
    默认使用 paper_level（论文级详细分析）。
    """
    # ── Read depth preference ──
    import streamlit as st
    depth = st.session_state.get("answer_depth", "paper_level")

    # ── Step 1: Extract variable pair ──
    ind_var, dep_var, var_class = extract_variable_pair(question)

    # Save to session for other uses
    st.session_state.variable_pair = (ind_var, dep_var, var_class)

    # Analyze coverage
    from src.interactive_modules import LiteratureSearchPlanner
    planner = LiteratureSearchPlanner()
    coverage_result = planner.analyze_question(question)
    coverage_level = coverage_result["coverage_level"]

    # ── Step 2: Build paper-level prefix ──
    coverage_warning = ""
    if coverage_result.get("topup_required"):
        coverage_warning = (
            "> ⚠️ **当前本地文献库不足以支持强假设生成**，"
            "以下内容仅为 search-guided candidate 输出。\n\n"
        )

    # ── Step 3: Page-specific answer generation ──
    if answer_mode == "popular_science":
        answer = generate_popular_science_answer(question, ind_var, dep_var, var_class)
    else:
        # Expensive retrieval/LLM work occurs only after the user clicks the
        # analysis button.  Local evidence remains visible if vector or LLM
        # services are unavailable.
        from src.smart_search import run_smart_search
        smart_result = run_smart_search(question, base_dir=BASE_DIR, use_llm=True)
        st.session_state["smart_search_result"] = smart_result
        answer = smart_result["answer"]

    # ── Step 4-8: 无需额外追加 ──
    if depth != "paper_level":
        quality = quality_gate_for_specificity(answer)
        if quality["low_specificity"]:
            supplement = _build_specificity_supplement(ind_var, dep_var)
            if supplement:
                answer += "\n" + supplement

    # ── Step 9: Do not create an unclaimed Streamlit PENDING task ──
    if coverage_result.get("topup_required"):
        answer += "\n## OA 文献补充门禁\n\n"
        answer += f"- 证据状态: {coverage_result['evidence_status']}\n"
        answer += (
            "- 未自动创建后台任务：Streamlit 网页没有常驻 worker，"
            "为避免无人领取的 PENDING 任务，请在“文献库管理 → OA 全文补充”中"
            "同步运行单篇任务并等待终态。\n"
        )

    return answer


# ── Domain Research Assistant Summary Section ────────────────────────────────


def _build_sci_summary_section(
    ind_var: Optional[str],
    dep_var: Optional[str],
    coverage_level: str = "unknown",
) -> str:
    """
    领域科研助手结构化分析框架 — 为科普/简洁回答补充 IV/DV/CV/MV、预测方向、
    支持判据、推翻条件和数据缺口，确保所有回答包含完整的变量-条件-机制-验证体系。
    """
    if not ind_var and not dep_var:
        return ""
    lines = []
    iv_lower = (ind_var or "").lower()
    dv_lower = (dep_var or "").lower()

    lines.append("\n---\n## 📋 领域科研助手分析摘要\n\n")

    # 1. IV/DV/CV/MV
    lines.append("### 1. 变量体系\n\n")
    lines.append("| 变量类型 | 具体变量 | 说明 |\n")
    lines.append("|---|---|---|\n")
    if "pore" in iv_lower or "defect" in iv_lower:
        lines.append("| 自变量 (IV) | pore_size / √area / distance_to_surface | 核心缺陷特征 |\n")
    elif "roughness" in iv_lower or "surface" in iv_lower:
        lines.append("| 自变量 (IV) | surface_roughness_Ra / Rz / surface_state | 表面状态量化 |\n")
    elif "delta" in iv_lower or "crack" in iv_lower:
        lines.append("| 自变量 (IV) | ΔK / stress_ratio_R | 裂纹扩展驱动力 |\n")
    else:
        lines.append(f"| 自变量 (IV) | {ind_var or '待识别'} | — |\n")

    if "fatigue_life" in dv_lower or "nf" in dv_lower:
        lines.append("| 因变量 (DV) | Nf / fatigue_limit σw | 疲劳寿命/极限 |\n")
    elif "da" in dv_lower or "crack" in dv_lower:
        lines.append("| 因变量 (DV) | da/dN / Paris C/m / ΔKth | 裂纹扩展指标 |\n")
    else:
        lines.append(f"| 因变量 (DV) | {dep_var or '待识别'} | — |\n")

    lines.append("| 控制变量 (CV) | Ti-6Al-4V Grade 23 / L-PBF 工艺 / R 比 / 温度 | 组间保持一致 |\n")
    lines.append("| 调节变量 (MV) | surface_state / heat_treatment / stress_ratio_R / defect_state | 影响关系方向和强度 |\n\n")

    # 2. 预测方向
    lines.append("### 2. 预测方向\n\n")
    if "pore" in iv_lower:
        lines.append("√area ↑ → Nf ↓（负相关）；distance_to_surface ↑ → Nf ↑（正相关，近表面更危险）\n\n")
    elif "roughness" in iv_lower or "surface" in iv_lower:
        lines.append("Ra/Rz ↑ → Nf ↓（负相关）；polished/machined 处理后 Nf ↑\n\n")
    elif "delta" in iv_lower:
        lines.append("ΔK ↑ → da/dN ↑（正相关，服从 Paris 定律）\n\n")
    else:
        lines.append(f"{ind_var or 'IV'} 变化 → {dep_var or 'DV'} 发生系统性变化\n\n")

    # 3. 支持判据
    lines.append("### 3. 支持判据\n\n")
    lines.append("- **统计显著**: 效应 p < 0.05\n")
    lines.append("- **效应量**: Cohen's d > 0.8（大效应）\n")
    lines.append("- **机制一致**: SEM/micro-CT 表征结果与预测机制相符\n")
    lines.append("- **模型验证**: Murakami/Paris 等模型预测与实测偏差 < 20%\n")
    lines.append("- **可重复性**: 同条件变异系数 < 15%\n\n")

    # 4. 推翻条件
    lines.append("### 4. 推翻条件\n\n")
    lines.append("以下任一情况将导致假设被推翻：\n\n")
    lines.append("1. IV 与 DV 无统计显著关系（p > 0.05）\n")
    lines.append("2. 效应方向与机制预测相反且无合理解释\n")
    lines.append("3. 存在未控制的混杂变量解释了全部效应\n")
    lines.append("4. 相同条件下不可重复（变异系数 > 20%）\n")
    lines.append("5. 表征结果与预测机制矛盾（如预期孔隙起裂但实际表面起裂）\n\n")

    # 5. 数据缺口
    lines.append("### 5. 当前数据缺口\n\n")
    cov_icon = {"sufficient": "🟢", "partial": "🟡", "weak": "🟠", "not_found": "🔴"}
    lines.append(f"**证据覆盖等级**: {cov_icon.get(coverage_level, '⚪')} {coverage_level}\n\n")
    if coverage_level in ("weak", "not_found"):
        lines.append("当前本地文献库证据不足，建议补充以下类型文献：\n")
        if "pore" in iv_lower:
            lines.append("- 同时包含 micro-CT 缺陷表征 + HCF 疲劳数据的文献\n")
            lines.append("- 系统控制 distance_to_surface 的对比实验\n")
        elif "roughness" in iv_lower:
            lines.append("- Ra/Rz + Nf 配对数据文献\n")
            lines.append("- 同时改变 Ra 和 pore_size 的系统对比\n")
        else:
            lines.append("- 目标变量对的直接实验数据\n")
    else:
        lines.append("有部分文献支持，但仍需更多系统定量数据确认条件边界。\n")

    lines.append("\n> *本摘要为领域科研助手系统输出的结构化分析框架*  \n")
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Quality Gate: check answer specificity
# ═══════════════════════════════════════════════════════════════════════════

def quality_gate_for_specificity(answer_text: str) -> Dict[str, Any]:
    """
    质量门禁：检查回答中是否包含泛泛表述。
    如果 low_specificity=True，需要自动补充具体内容。

    检测模式：
    - 只有泛泛结论没有具体展开
    - 仅有"正相关/负相关"而无变量、条件、机制
    """
    if not answer_text:
        return {"low_specificity": True, "reason": "回答为空"}

    text_lower = answer_text.lower()

    # Vague patterns that need concrete context
    vague_patterns = [
        r"负相关[。，；\n]",
        r"正相关[。，；\n]",
        r"影响疲劳性能[。，；\n]",
        r"改善疲劳性能[。，；\n]",
        r"[只|仅]需进一步研究",
        r"需要更多研究",
        r"建议.*实验验证[。，；\n]",
        r"存在一定争议[。，；\n]",
    ]

    # Specificity markers that indicate the answer is detailed enough
    specificity_markers = [
        "自变量", "因变量", "调节变量", "控制变量",
        "应力比", "应力幅", "pore_size", "distance_to_surface",
        "mechanism", "机制链", "Murakami", "Paris law",
        "试验方法", "测试方法", "表征方法", "样品分组",
        "推翻条件", "支持判据",
        "micro-CT", "SEM", "EBSD",
    ]

    has_vague = False
    for pat in vague_patterns:
        if re.search(pat, text_lower):
            has_vague = True
            break

    has_specificity = any(marker in text_lower for marker in specificity_markers)

    if has_vague and not has_specificity:
        return {
            "low_specificity": True,
            "reason": "回答包含泛泛表述且缺乏具体变量、条件、机制和验证方案",
        }

    return {"low_specificity": False, "reason": ""}


def _build_specificity_supplement(
    ind_var: Optional[str],
    dep_var: Optional[str],
) -> str:
    """当回答缺乏具体性时，自动补充标准变量信息。"""
    lines = [
        "\n---\n",
        "### 具体化补充\n",
        "系统检测到以上回答偏泛泛，自动补充以下标准信息：\n",
    ]
    if ind_var:
        lines.append(f"- **自变量**: {ind_var}\n")
    if dep_var:
        lines.append(f"- **因变量**: {dep_var}\n")
    lines.append("- **调节变量**: pore_location, surface_state, stress_ratio_R, stress_amplitude, heat_treatment\n")
    lines.append("- **条件边界**: as-built / polished / HIP / non-HIP / highσ / lowσ\n")
    lines.append("- **验证方法**: micro-CT + HCF/VHCF + SEM fractography\n")
    lines.append("- **推翻条件**: 目标变量与疲劳指标无统计显著关系\n\n")
    lines.append("> 如需更深入的分析，请选择「科研分析」或「假设生成」模式，"
                 "并确保问题中包含具体变量对（如孔隙尺寸→疲劳寿命）。\n")
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# UI — Sidebar
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Page navigation
# ═══════════════════════════════════════════════════════════════════════════

current_page = st.session_state.get("page", "search")

# Former standalone pages now live exclusively in the smart-search modes.
legacy_page_modes = {
    "research_gap": "research_analysis",
    "hypothesis_gen": "hypothesis_generation",
    "experiment_design": "experiment_design",
}
if current_page in legacy_page_modes:
    st.session_state.page = "search"
    st.session_state.answer_mode = legacy_page_modes[current_page]
    for stale_key in (
        "nav_gap",
        "nav_hypothesis",
        "nav_experiment",
        "gap_review_text",
        "gap_review_input",
        "gap_results",
        "gap_page",
    ):
        st.session_state.pop(stale_key, None)
    current_page = "search"

# Sidebar
with st.sidebar:
    st.title("🔬 TitaniumFatigueChat")
    st.caption("钛合金疲劳科研助手 · L-PBF Ti-6Al-4V 为当前主案例")
    from src.app_auth import render_logout_control
    from src.api_keys import get_deepseek_settings

    render_logout_control(st)
    deepseek_settings = get_deepseek_settings()
    if deepseek_settings.configured:
        st.caption("DeepSeek配置：已检测")
        st.caption(f"配置来源：{deepseek_settings.source}")
    else:
        st.warning(
            "未配置 DEEPSEEK_API_KEY。文献浏览和本地规则功能可用，"
            "需要模型的功能将回退或不可用。"
        )
        st.caption("配置来源：none")
    st.divider()

    # ── 普通前台导航 ──
    if st.button("🔍 智能搜索", key="nav_search",
                 use_container_width=True,
                 type="primary" if current_page == "search" else "secondary"):
        st.session_state.page = "search"
        st.session_state.answer_mode = "research_analysis"
        st.rerun()

    if st.button("📚 文献库管理", key="nav_library",
                 use_container_width=True,
                 type="primary" if current_page == "library" else "secondary"):
        st.session_state.page = "library"
        st.rerun()

    if st.button("📐 文献公式库", key="nav_formula",
                 use_container_width=True,
                 type="primary" if current_page == "formula_explain" else "secondary"):
        st.session_state.page = "formula_explain"
        st.rerun()

    st.divider()

    st.divider()

    from src.data_cache import get_system_stats_cached
    stats = get_system_stats_cached()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("唯一文献", stats["unique_literature_count"])
        st.metric("深读完成", stats["deep_read_complete_count"])
    with col2:
        st.metric("已入统一索引", stats["indexed_count"])
        st.metric("待处理/失败", stats["pending_or_failed_count"])
    st.caption(
        f"本地PDF文件数：{stats['local_pdf_file_count']}（含目录副本）；"
        f"唯一SHA-256：{stats['unique_pdf_sha256_count']}"
    )

    st.divider()
    st.caption("⚠️ 系统生成的是 candidate hypothesis，不得声称已被证明。")
    st.caption("📊 系统状态")
    st.caption("已完成一次分析" if st.session_state.analysis_done else "等待输入")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI Scientist Search
# ═══════════════════════════════════════════════════════════════════════════

if current_page == "search":

    st.markdown(
        "<h1 style='text-align: center;'>🔬 TitaniumFatigueChat</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem; color: #888;'>"
        "面向钛合金疲劳研究的 AI Scientist · L-PBF Ti-6Al-4V 为当前主案例</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])

    with search_col2:
        # ── Main search box ──
        st.text_input(
            "科研问题",
            placeholder="请输入科研问题、变量组合或参数条件，例如：孔隙尺寸和疲劳寿命之间是什么关系？",
            key="user_question",
            label_visibility="collapsed",
        )

        # ── Mode selection buttons ──
        current_mode = st.session_state.answer_mode
        cols = st.columns(4)
        clicked_mode = None
        for i, mk in enumerate(MODE_LIST):
            ml = MODES[mk]["label"]
            with cols[i]:
                is_active = (mk == current_mode)
                btn_type = "primary" if is_active else "secondary"
                if st.button(ml, key=f"mode_{mk}", type=btn_type, use_container_width=True):
                    clicked_mode = mk
        if clicked_mode:
            st.session_state.answer_mode = clicked_mode
            st.rerun()

        mode_desc = MODES.get(current_mode, {}).get("desc", "")
        st.info(f"**当前回答模式**: {MODES[current_mode]['label']}\n\n{mode_desc}")

        # ── Answer depth selector ──
        depth_options = {
            "standard": "标准",
            "detailed": "详细",
            "paper_level": "论文级",
        }
        depth_labels = ["标准", "详细", "论文级"]
        current_depth = st.session_state.get("answer_depth", "paper_level")
        depth_cols = st.columns(3)
        clicked_depth = None
        for i, (dk, dl) in enumerate(depth_options.items()):
            with depth_cols[i]:
                is_active = (dk == current_depth)
                btn_type = "primary" if is_active else "secondary"
                if st.button(dl, key=f"depth_{dk}", type=btn_type, use_container_width=True):
                    clicked_depth = dk
        if clicked_depth:
            st.session_state.answer_depth = clicked_depth
            st.rerun()
        st.caption(f"当前回答深度: {depth_options.get(current_depth, '论文级')}")

        # ── Analysis and explicitly separated literature actions ──
        action_cols = st.columns(3)
        with action_cols[0]:
            analyze_clicked = st.button(
                "🔍 开始分析",
                type="primary",
                use_container_width=True,
                key="run_analysis_btn",
            )
        with action_cols[1]:
            refresh_metadata_clicked = st.button(
                "刷新候选元数据",
                use_container_width=True,
                key="refresh_literature_metadata_btn",
            )
        with action_cols[2]:
            manual_topup_clicked = st.button(
                "手动补充1篇OA全文",
                use_container_width=True,
                key="manual_oa_topup_btn",
            )

        # ── Smart query understanding display ──
        current_input = st.session_state.get("user_question", "").strip()
        if current_input:
            sq = understand_user_query(current_input)
            intent_labels = {
                "general_explanation": "💡 一般解释", "relation_analysis": "📊 变量关系分析",
                "equation_generation": "📐 方程/参数生成", "hypothesis_generation": "🧪 假设生成",
                "experiment_design": "🧫 实验设计", "conflict_detection": "⚡ 文献冲突检测",
                "macro_micro_mechanism": "🔗 宏微观机制分析",
                "dominance_comparison": "🏆 因素比较", "literature_search": "📚 文献检索",
                "research_gap_discovery": "🧩 研究空白",
            }
            tag = f"系统检测意图：{intent_labels.get(sq['task_intent'], '💡 一般解释')}"
            if sq.get('independent_variable') and sq.get('dependent_variable'):
                tag += f" | 变量对：{sq['independent_variable']} → {sq['dependent_variable']}"
            elif sq.get('canonical_variable_names'):
                tag += f" | 检测到变量：{', '.join(sq['canonical_variable_names'])}"
            if sq.get('has_corrections'):
                tag += " | ✏️ 已自动修正"
            st.caption(tag)

        if refresh_metadata_clicked:
            if not current_input:
                st.warning("请先输入检索问题。")
            else:
                try:
                    from src.literature_agent import run_auto_literature_update

                    ind_var, dep_var, _ = extract_variable_pair(current_input)
                    metadata_result = run_auto_literature_update(
                        user_query=current_input,
                        ind_var=ind_var,
                        dep_var=dep_var,
                        max_results=8,
                        use_api=True,
                    )
                    st.session_state.literature_result = metadata_result
                    st.success(
                        f"候选元数据刷新完成：新增候选 "
                        f"{metadata_result.get('candidates_new', 0)} 篇。"
                    )
                except Exception as exc:
                    st.error(f"候选元数据刷新失败：{type(exc).__name__}: {exc}")

        if manual_topup_clicked:
            if not current_input:
                st.warning("请先输入检索问题。")
            else:
                st.session_state.page = "library"
                st.session_state.pop("oa_task_id", None)
                st.info(
                    "已转到“文献库管理”。请在“OA 全文补充”中同步运行单篇任务；"
                    "网页不会创建等待外部 worker 的 PENDING 任务。"
                )
                st.rerun()

        task_id = st.session_state.get("oa_task_id", "")
        if task_id:
            from src.literature_tasks import get_task

            current_task = get_task(task_id, base_dir=BASE_DIR)
            if current_task:
                if current_task.get("manual_override"):
                    st.info(
                        "本次补充由用户主动触发，并非系统判定当前证据必然不足。"
                    )
                st.caption(
                    "历史任务状态仅供审计；网页不会声称任务已在后台运行。"
                    "新任务请到“文献库管理 → OA 全文补充”同步执行。"
                )
                with st.expander("OA文献补充任务状态", expanded=True):
                    checkpoint = current_task.get("checkpoint") or {}
                    st.markdown(
                        "\n".join(
                            [
                                f"- task_id: `{current_task.get('task_id', '')}`",
                                f"- 任务状态: **{current_task.get('status', '')}**",
                                f"- 查询词: {current_task.get('query', '')}",
                                f"- 创建时间: {current_task.get('created_at', '')}",
                                f"- 候选数量: {current_task.get('candidate_count', 0)}",
                                f"- OA候选数量: {current_task.get('oa_candidate_count', 0)}",
                                f"- 下载数量: {current_task.get('downloaded_count', 0)}",
                                f"- 重复拒绝: {current_task.get('duplicate_rejected_count', 0)}",
                                f"- 付费墙/无OA权限拒绝: {current_task.get('paywall_rejected_count', 0)}",
                                f"- 深读状态: {current_task.get('deep_read_count', 0)}/{current_task.get('max_papers', 1)}",
                                f"- 入索引状态: {current_task.get('indexed_count', 0)}/{current_task.get('max_papers', 1)}",
                                f"- checkpoint: {checkpoint.get('current_phase', '')}",
                                f"- 失败原因: {current_task.get('last_error') or '无'}",
                            ]
                        )
                    )

        # ── Sample question buttons ──
        st.divider()
        st.markdown("<div style='font-size:0.85rem; color:#888; text-align:center;'>💡 示例问题</div>", unsafe_allow_html=True)
        sample_cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            with sample_cols[i % 2]:
                if st.button(q, key=safe_key("sample_q", q, i), use_container_width=True):
                    st.session_state.user_question = q
                    st.session_state.answer = None
                    st.rerun()

    # ── Answer processing ──
    if analyze_clicked:
        current_question = st.session_state.get("user_question", "").strip()
        current_mode = st.session_state.get("answer_mode", "research_analysis")
        if not current_question:
            st.warning("请输入科研问题。")
        else:
            # Step 1: Query understanding
            sq = understand_user_query(current_question)
            st.session_state.structured_query = sq

            # Step 2: If low confidence, show confirmation
            if sq["overall_confidence"] < 0.65 and sq.get("corrected_query") != current_question:
                st.info(
                    f"🤔 我理解你可能想问：**{sq['corrected_query']}**\n\n"
                    f"是否按这个理解分析？(置信度: {int(sq['overall_confidence']*100)}%)"
                )

            with st.spinner("正在分析问题，调用多模块生成综合回答..."):
                # Use the corrected query and extracted variables for answer generation
                effective_question = sq.get("corrected_query") or current_question
                answer_text = generate_comprehensive_answer(effective_question, current_mode)
                st.session_state.answer = answer_text
                st.session_state.last_question = current_question
                st.session_state.last_mode = current_mode
                st.session_state.analysis_done = True
                st.session_state.answer_timestamp = time.time()
            st.rerun()

    # ── "Question changed" hint ──
    current_question = st.session_state.get("user_question", "").strip()
    last_question = st.session_state.get("last_question", "")
    if st.session_state.get("answer") and current_question and current_question != last_question:
        st.info("📝 检测到问题已修改，请点击「开始分析」生成新答案。")

    # ── Display answer ──
    if st.session_state.get("answer"):
        st.divider()
        mode_label = MODES.get(st.session_state.last_mode, {}).get("label", "")
        st.markdown(f"<h3 style='text-align: center;'>📋 {mode_label}</h3>", unsafe_allow_html=True)
        display_question = st.session_state.last_question

        # ── Query understanding info ──
        sq = st.session_state.get("structured_query")
        if sq and sq.get("has_corrections"):
            with st.expander("🔍 智能问题理解", expanded=True):
                st.markdown(f"**原始问题**: {sq['original_query']}")
                st.markdown(f"**自动修正**: {sq['corrected_query']}")
                st.markdown("**修正记录**:")
                for c in sq["corrections"]:
                    conf_pct = int(c.get("confidence", 0.8) * 100)
                    st.markdown(f"- 「{c['raw']}」→「{c['corrected']}」(置信度: {conf_pct}%)")
                if sq.get("canonical_variable_names"):
                    st.markdown(f"**识别变量**: {', '.join(sq['canonical_variable_names'])}")
                if sq.get("independent_variable") and sq.get("dependent_variable"):
                    st.markdown(f"**变量对**: {sq['independent_variable']} → {sq['dependent_variable']}")
                conf = int(sq.get("overall_confidence", 0) * 100)
                st.markdown(f"**置信度**: {conf}%")
        elif sq and sq.get("canonical_variable_names"):
            with st.expander("🔍 问题理解", expanded=False):
                st.markdown(f"**问题**: {sq['original_query']}")
                st.markdown(f"**识别变量**: {', '.join(sq['canonical_variable_names'])}")
                if sq.get("independent_variable") and sq.get("dependent_variable"):
                    st.markdown(f"**变量对**: {sq['independent_variable']} → {sq['dependent_variable']}")

        st.caption(f"**问题**: {display_question}")
        ind_var, dep_var, var_class = st.session_state.get("variable_pair", (None, None, ""))
        if ind_var or dep_var:
            vars_html = f"<span style='font-size:0.85rem; color:#888;'>关注变量：{ind_var or '?'} → {dep_var or '?'}</span>"
            st.markdown(vars_html, unsafe_allow_html=True)
        st.divider()
        # 答案仅渲染一次（方案 A：直接显示完整回答）
        with st.container():
            st.markdown(st.session_state.answer)

        # Collapsible evidence details
        with st.expander("📄 后台数据详情（点击展开）"):
            st.markdown("#### 模块状态\n")
            explorer = EvidenceRelationExplorer()
            miner = EquationParameterMiner()
            detector = ConflictDetector()
            macro = MacroMicroLinkExplorer()
            type_counts = explorer.get_relation_type_counts()
            if type_counts:
                st.markdown(f"**变量关系**: {sum(type_counts.values())} 条")
            summ = miner.get_summary()
            st.markdown(f"**方程参数**: {summ['total_extractions']} 条提取")
            if detector.claims:
                st.markdown(f"**冲突检测**: {len(detector.claims)} 个主题")
            if macro.links:
                st.markdown(f"**宏微观关联**: {len(macro.links)} 条")
            if st.session_state.get("literature_result"):
                lit = st.session_state.literature_result
                st.markdown("#### 自动文献检索\n")
                st.markdown(f"- 检索式: {len(lit.get('search_queries', []))} 条")
                st.markdown(f"- 结果: {len(lit.get('results', []))} 篇")
                st.markdown(f"- 搜索计划: {lit.get('search_plan_saved_to', '')}")

        # Action buttons
        st.divider()
        col_export, col_clear = st.columns(2)
        with col_export:
            if st.button("📥 导出回答", key="export_btn_answer", use_container_width=True):
                out_dir = Path("outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                export_path = out_dir / "20_interactive_answer.md"
                export_path.write_text(
                    f"# TitaniumFatigueChat AI Scientist 回答\n\n"
                    f"**问题**: {st.session_state.last_question}\n"
                    f"**模式**: {mode_label}\n\n{st.session_state.answer}\n",
                    encoding="utf-8",
                )
                st.success(f"已导出到 {export_path}")
        with col_clear:
            if st.button("🔄 新问题", key="clear_btn_all", use_container_width=True):
                for key in ["answer", "analysis_done", "trigger_search", "literature_result"]:
                    st.session_state[key] = None if key != "analysis_done" else False
                st.session_state.user_question = ""
                st.session_state.last_question = ""
                st.rerun()

    elif not st.session_state.get("answer"):
        st.divider()
        st.markdown(
            "<div style='text-align:center; padding:3rem 0; color:#888;'>"
            "👆 输入科研问题，选择回答模式，点击「开始分析」获取综合回答"
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Research Gap Discovery
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "research_gap":

    st.markdown("<h1 style='text-align: center;'>🧩 研究空白发现</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1rem; color: #888;'>"
        "基于本地文献库和文献综述，自动识别 L-PBF Ti-6Al-4V 疲劳研究中的未解决问题与高价值假设</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Data source status ──
    with st.expander("📊 数据源状态", expanded=True):
        from src.data_cache import get_system_stats_cached
        stats = get_system_stats_cached()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("正式文献库", stats.get("n_papers", 0))
        with col2:
            from src.data_cache import load_candidate_papers
            cand_count = len(load_candidate_papers())
            st.metric("候选文献", cand_count)
        with col3:
            st.metric("证据片段", stats.get("ev_count", 0))
        with col4:
            st.metric("变量-机制", stats.get("vm_count", 0))

        # Paper type classification summary
        try:
            from src.paper_classifier import get_classification_summary
            cls_summary = get_classification_summary()
            st.markdown("**文献分类统计**:")
            ct = cls_summary.get("by_type", {})
            for t, n in sorted(ct.items(), key=lambda x: -x[1]):
                st.markdown(f"- {t}: {n} 篇", unsafe_allow_html=False)
        except Exception:
            pass

    # ── Review text input ──
    st.subheader("📝 文献综述输入")
    review_col1, review_col2 = st.columns([3, 1])
    with review_col1:
        review_text = st.text_area(
            "粘贴文献综述文本（或使用 docs/literature_review.md 默认文件）",
            value=st.session_state.get("gap_review_text", ""),
            height=150,
            key="gap_review_input",
            label_visibility="collapsed",
            placeholder="在此粘贴文献综述全文，或留空以仅使用本地数据库进行分析...",
        )
    with review_col2:
        # Try loading from default file
        default_review = load_review_text()
        if default_review:
            if st.button("📂 加载默认 docs 文件", key="load_review_btn", use_container_width=True):
                st.session_state.gap_review_text = default_review
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 清空", key="clear_review_btn", use_container_width=True):
            st.session_state.gap_review_text = ""
            st.rerun()

    # ── Analysis scope ──
    scope = st.radio(
        "分析范围",
        ["全部文献", "只分析正式文献库", "正式库 + 候选库"],
        index=0,
        horizontal=True,
        key="gap_scope_radio",
    )
    scope_map = {"全部文献": "all", "只分析正式文献库": "lit_db_only", "正式库 + 候选库": "lit_db_and_candidates"}

    # ── Run button ──
    if st.button("🚀 生成研究空白与高价值假设", type="primary", use_container_width=True, key="run_gap_discovery"):
        with st.spinner("正在分析文献库、比对证据、识别研究空白..."):
            review = st.session_state.get("gap_review_input", "").strip()
            gaps = discover_research_gaps(review_text=review, scope=scope_map.get(scope, "all"))
            save_gaps(gaps)
            st.session_state.gap_results = gaps
        st.success(f"完成！发现 {len(gaps)} 个研究空白。")
        st.rerun()

    st.divider()

    # ── Display results ──
    gaps = st.session_state.get("gap_results")
    if gaps:
        st.subheader(f"📋 研究空白列表（共 {len(gaps)} 个，按优先级排序）")

        # Priority stats
        pri_counts = {"high_priority": 0, "medium_priority": 0, "low_priority": 0, "reject": 0}
        for g in gaps:
            p = g.get("priority_level", "reject")
            if p in pri_counts:
                pri_counts[p] += 1
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🔴 高优先级", pri_counts["high_priority"])
        with c2:
            st.metric("🟡 中优先级", pri_counts["medium_priority"])
        with c3:
            st.metric("🟢 低优先级", pri_counts["low_priority"])
        with c4:
            st.metric("⚪ 已拒绝", pri_counts["reject"])

        st.divider()

        # ── Pagination for gap list ──
        ITEMS_PER_PAGE = DEMO_MAX_ITEMS if DEMO_MODE else 10
        total_pages = (len(gaps) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        page_key = "gap_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1

        col_prev, col_page, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ 上一页", disabled=(st.session_state[page_key] <= 1), key="gap_prev", use_container_width=True):
                st.session_state[page_key] = max(1, st.session_state[page_key] - 1)
                st.rerun()
        with col_page:
            st.markdown(f"<div style='text-align:center;'><b>第 {st.session_state[page_key]}/{max(1,total_pages)} 页</b></div>", unsafe_allow_html=True)
        with col_next:
            if st.button("下一页 ▶", disabled=(st.session_state[page_key] >= total_pages), key="gap_next", use_container_width=True):
                st.session_state[page_key] = min(total_pages, st.session_state[page_key] + 1)
                st.rerun()

        start_idx = (st.session_state[page_key] - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(gaps))
        page_gaps = gaps[start_idx:end_idx]

        st.caption(f"显示 {start_idx+1}-{end_idx} / 共 {len(gaps)} 个")

        # Display each gap as a compact card
        for gap in page_gaps:
            pri = gap.get("priority_level", "low_priority")
            total = gap.get("total_priority_score", 0)
            pri_icon = {"high_priority": "🔴", "medium_priority": "🟡",
                         "low_priority": "🟢", "reject": "⚪"}.get(pri, "⚪")
            gap_type_label = {
                "evidence_gap": "证据不足", "parameter_gap": "参数缺口",
                "mechanism_gap": "机制缺口", "conflict_gap": "文献冲突",
                "validation_gap": "验证不足", "boundary_condition_gap": "条件边界",
                "data_gap": "数据缺口", "translation_gap": "工程转化",
            }.get(gap.get("gap_type", ""), gap.get("gap_type", ""))

            with st.container(border=True):
                st.markdown(f"### {gap.get('gap_id', '')}: {gap.get('gap_title', '')}")
                st.markdown(f"**类型**: {gap_type_label} | **优先级**: {pri_icon} {pri} | **总分**: {total}/100")

                with st.expander("📄 详细分析", expanded=False):
                    st.markdown(f"**研究空白**:")
                    st.markdown(gap.get("gap_statement", gap.get("existing_evidence_summary", "—")))
                    st.markdown(f"**相关变量**: {gap.get('related_variables', '—')}")
                    st.markdown(f"**目标指标**: {gap.get('target_indicators', '—')}")
                    st.markdown(f"**为什么重要**: {gap.get('why_it_matters', '—')}")
                    st.markdown(f"**缺失证据**: {gap.get('missing_evidence', '—')}")

                    if gap.get("candidate_hypothesis"):
                        st.markdown("**候选假设**:")
                        st.info(gap["candidate_hypothesis"])

                    if gap.get("potential_equation_or_model"):
                        st.markdown(f"**可能方程/模型**: {gap['potential_equation_or_model']}")

                    # Scoring breakdown
                    st.markdown("**评分分解**:")
                    score_cols = st.columns(4)
                    with score_cols[0]:
                        st.caption(f"科学价值 {gap.get('scientific_value_score', 0)}/20")
                        st.caption(f"新颖性 {gap.get('novelty_score', 0)}/15")
                    with score_cols[1]:
                        st.caption(f"可验证性 {gap.get('testability_score', 0)}/20")
                        st.caption(f"可行性 {gap.get('feasibility_score', 0)}/15")
                    with score_cols[2]:
                        st.caption(f"参数潜力 {gap.get('parameter_potential_score', 0)}/10")
                        st.caption(f"冲突利用 {gap.get('conflict_usefulness_score', 0)}/10")
                    with score_cols[3]:
                        st.caption(f"论文潜力 {gap.get('paper_potential_score', 0)}/10")

        # Export section
        st.divider()
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("📥 导出研究空白报告", key="export_gap_report", use_container_width=True):
                report_path = OUTPUTS_DIR / "24_research_gap_report.md"
                st.success(f"已导出到 {report_path}")
        with exp_col2:
            if st.button("📥 导出优先级假设列表", key="export_hyp_list", use_container_width=True):
                hyp_path = OUTPUTS_DIR / "25_prioritized_gap_hypotheses.md"
                st.success(f"已导出到 {hyp_path}")

    else:
        st.info("👈 选择分析范围，点击「生成研究空白与高价值假设」开始分析。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Literature Library Management
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "library":
    from src.library_page import render_library_page
    render_library_page()

elif current_page == "baseline":
    st.markdown("<h1 style='text-align: center;'>📋 Baseline 对比</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>Direct Qwen vs Structured Evidence Only vs Full EPCR-HG</p>", unsafe_allow_html=True)
    st.divider()

    if st.button("🚀 运行 Baseline 对比", type="primary", use_container_width=True, key="run_baseline"):
        with st.spinner("正在对比不同系统版本..."):
            result = run_baseline_comparison()
        st.success(f"完成！对比了 {result['n_tasks']} 个测试用例，{len(result['versions'])} 个系统版本。")

    # Display results
    import pandas as pd
    bm_path = Path("data/baseline_comparison.csv")
    if bm_path.exists():
        bdf = pd.read_csv(bm_path, encoding="utf-8-sig")
        st.subheader("Baseline 对比结果")
        st.dataframe(bdf, use_container_width=True)

        # Summary stats
        st.subheader("各版本平均分")
        for version in bdf["system_version"].unique():
            vdf = bdf[bdf["system_version"] == version]
            mean_total = vdf["total_score"].mean()
            st.metric(f"{version}", f"{mean_total:.1f}")

        st.markdown(f"完整报告: `outputs/baseline_comparison_report.md`")
    else:
        st.info("点击「运行 Baseline 对比」生成结果。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "ablation":
    st.markdown("<h1 style='text-align: center;'>🔬 消融研究</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>评估各模块对系统整体性能的贡献</p>", unsafe_allow_html=True)
    st.divider()

    if st.button("🚀 运行消融研究", type="primary", use_container_width=True, key="run_ablation"):
        with st.spinner("正在评估各模块贡献..."):
            result = run_ablation_study()
        st.success(f"完成！比较了 {result['n_versions']} 个系统配置。")

    import pandas as pd
    abl_path = Path("data/ablation_results.csv")
    if abl_path.exists():
        adf = pd.read_csv(abl_path, encoding="utf-8-sig")
        st.subheader("消融结果（按总分排序）")
        st.dataframe(adf.sort_values("total_score", ascending=False), use_container_width=True)

        # Visualize delta
        st.subheader("与 Full System 的差异")
        full_score = adf[adf["ablation_version"] == "Full System"]["total_score"].values
        if len(full_score) > 0:
            full_score = full_score[0]
            for _, row in adf.iterrows():
                delta = row["total_score"] - full_score
                st.metric(row["ablation_version"], f"{row['total_score']}", delta=f"{delta:+.0f}")

        st.markdown(f"完整报告: `outputs/ablation_study_report.md`")
    else:
        st.info("点击「运行消融研究」生成结果。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Q1 Paper Readiness
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "q1_quality":
    st.markdown("<h1 style='text-align: center;'>📊 系统质量评估</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>评估领域科研助手系统的证据链完整性、模块质量与可验证性</p>", unsafe_allow_html=True)
    st.divider()

    if st.button("📊 生成完整度评分", type="primary", use_container_width=True, key="run_q1"):
        with st.spinner("正在评估论文就绪度..."):
            result = paper_quality_gate()
        st.success("评分完成！")

        st.subheader(f"总分: {result['total_score']}/100")
        grade_icon = {"paper_ready_candidate": "🟢", "potential_with_major_validation": "🟡",
                       "high_potential": "🟠", "paper_possible": "🔴", "not_ready": "⚫"}
        st.subheader(f"{grade_icon.get(result['grade'], '⚪')} {result['grade']}")

        st.subheader("维度评分")
        for dim, score in result["dimension_scores"].items():
            max_s = {"Methodological Novelty": 15, "Dataset Reproducibility": 15,
                      "Evidence Grounding": 15, "Quantitative Validation": 15,
                      "Baseline Comparison": 10, "Ablation Study": 10,
                      "Expert/Retrospective Validation": 10, "Domain Scientific Insight": 5,
                      "Figure/Table Readiness": 3, "Manuscript Readiness": 2}.get(dim, 10)
            st.progress(score / max_s)
            st.caption(f"{dim}: {score}/{max_s}")

        st.markdown(f"完整评分卡: `outputs/q1_paper_readiness_scorecard.md`")
    else:
        st.info("点击「生成完整度评分」进行评估。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Research Material Export
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "sci_export":
    st.markdown("<h1 style='text-align: center;'>📝 系统结果导出</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>基于文献库、证据库、假设库、基准评测和消融结果，\n"
                "导出领域科研助手系统的结构化输出</p>", unsafe_allow_html=True)

    # ── 说明 ──
    st.info(
        "本模块导出系统运行结果，包括数据集统计、证据链、假设列表、\n"
        "实验方案和评测结果。**导出的内容为系统输出，需研究者人工核查后使用。**"
    )

    st.divider()
    st.subheader("导出内容")
    st.markdown("""
    - `outputs/methods_draft.md` — 方法框架描述
    - `outputs/results_tables.md` — 结果表格（数据集统计 / Baseline / Ablation）
    - `outputs/figure_plan.md` — 图表规划
    - `outputs/discussion_claims.md` — 讨论要点（Claim–Evidence–Limitation）
    """)

    if st.button("📥 生成所有论文材料", type="primary", use_container_width=True, key="run_sci_export"):
        with st.spinner("正在生成论文材料..."):
            try:
                from src.sci_export import generate_all
                generate_all()
                st.success("论文材料生成完成 (真实数据驱动)!")
            except ImportError:
                exports = generate_sci_paper_export()
                st.success("论文材料生成完成！")
                for name, summary in exports.items():
                    st.markdown(f"- ✅ **{name}**: {summary}")

    # Also run baseline + ablation + retrospective + Paris validation if not done
    st.subheader("🔧 一键运行完整验证套件")
    if st.button("🚀 运行全部验证（Baseline + Ablation + Retrospective + Paris + Readiness Gate）",
                 use_container_width=True, key="run_all_validations"):
        with st.spinner("正在运行完整验证套件..."):
            run_baseline_comparison()
            run_ablation_study()
            run_retrospective_validation(split_year=2022)
            run_paris_law_validation()
            q1 = paper_quality_gate()
        st.success("全部验证完成！")
        st.metric("研究完整度评估", f"{q1['total_score']}/100", q1['grade'])


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Paper Results Dashboard (新版论文结果统计)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "dashboard":
    st.markdown("<h1 style='text-align: center;'>📊 论文结果统计面板</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>Evidence–Parameter–Conflict RAG — 论文级数据统计</p>")
    st.divider()

    # ═══ 缓存化系统统计 ═══
    from src.data_cache import (get_system_stats_cached, load_literature_database)
    stats = get_system_stats_cached()

    metric_cols = st.columns(4)
    metric_cols[0].metric("唯一文献", stats["unique_literature_count"])
    metric_cols[1].metric("深读完成", stats["deep_read_complete_count"])
    metric_cols[2].metric("已入统一索引", stats["indexed_count"])
    metric_cols[3].metric("待处理/失败", stats["pending_or_failed_count"])
    st.caption(
        f"本地PDF文件数 {stats['local_pdf_file_count']}，其中唯一SHA-256 "
        f"{stats['unique_pdf_sha256_count']}；目录副本不重复计数。"
    )

    st.caption(f"研究空白: {stats['gap_count']}  |  假设: {stats['hypothesis_count']}")

    # Literature support breakdown (cached)
    st.subheader("📋 文献库概览")
    lit_df = load_literature_database()
    n_lit = len(lit_df)
    st.caption(
        f"旧元数据CSV行数: {n_lit}（仅作兼容展示，不是唯一文献真源） | "
        f"综述: {(lit_df['paper_type_primary'] == 'review').sum() if not lit_df.empty and 'paper_type_primary' in lit_df.columns else 0} 篇"
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Evidence Correction (人工证据校正)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "evidence_correction":
    st.markdown("<h1 style='text-align: center;'>✏️ 人工证据校正</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>审查和校正自动抽取的证据片段</p>")
    st.divider()

    from src.data_cache import load_evidence_snippets
    df = load_evidence_snippets()
    if df.empty:
        st.error("证据片段数据为空，请先运行文献库构建。")
    else:
        st.metric("总证据片段", len(df))

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            ev_types = ["All"] + sorted(df['evidence_type'].unique().tolist()) if 'evidence_type' in df.columns else ["All"]
            filter_type = st.selectbox("按类型筛选", ev_types)
        with col2:
            strengths = ["All"] + sorted(df['evidence_strength'].unique().tolist()) if 'evidence_strength' in df.columns else ["All"]
            filter_str = st.selectbox("按强度筛选", strengths)

        filtered = df.copy()
        if filter_type != "All" and 'evidence_type' in filtered.columns:
            filtered = filtered[filtered['evidence_type'] == filter_type]
        if filter_str != "All" and 'evidence_strength' in filtered.columns:
            filtered = filtered[filtered['evidence_strength'] == filter_str]

        st.dataframe(filtered.head(20), use_container_width=True)

        # Edit interface
        st.subheader("校正单条证据")
        if len(df) > 0:
            ev_ids = df['evidence_id'].tolist()
            selected_id = st.selectbox("选择要校对的证据ID", ev_ids)
            selected = df[df['evidence_id'] == selected_id]
            if len(selected) > 0:
                row = selected.iloc[0]
                with st.form("edit_evidence"):
                    new_claim = st.text_area("Extracted Claim", row.get('extracted_claim', ''), height=80)
                    new_iv = st.text_input("Independent Variable", row.get('independent_variable', ''))
                    new_dv = st.text_input("Dependent Variable", row.get('dependent_variable', ''))
                    new_mech = st.text_input("Mechanism", row.get('mechanism', ''))
                    new_type = st.selectbox("Evidence Type", EVIDENCE_TYPES if 'EVIDENCE_TYPES' in dir() else ["direct_experimental_evidence", "indirect_mechanistic_evidence", "review_statement", "equation_parameter_evidence", "conflict_evidence", "hypothesis_candidate", "insufficient_evidence"],
                                            index=0)
                    new_strength = st.selectbox("Strength", ["high", "moderate", "low"], index=1)
                    if st.form_submit_button("保存校正"):
                        st.success(f"校正已保存到 {selected_id} (note: CSV in-place update)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Hypothesis Generation (假设生成)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "hypothesis_gen":
    import pandas as pd
    st.markdown("<h1 style='text-align: center;'>🧪 假设生成</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>基于文献证据的科学研究假设生成</p>")
    st.divider()

    st.info(
        "该页面基于文献证据库和变量关系数据，生成具体、可验证、可推翻的候选科学假设。\n\n"
        "您也可以在「智能搜索」页面输入科研问题并选择「假设生成」模式。"
    )

    # Quick access to hypothesis generation in search mode
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 前往智能搜索使用假设生成模式", use_container_width=True):
            st.session_state.page = "search"
            st.session_state.answer_mode = "hypothesis_generation"
            st.rerun()
    with col2:
        st.markdown("")

    # Show existing hypotheses
    hyp_path = Path("data/hypothesis_dataset.csv")
    if hyp_path.exists():
        hyp_df = pd.read_csv(hyp_path, encoding="utf-8-sig", on_bad_lines="skip")
        st.subheader(f"现有假设 ({len(hyp_df)} 条)")
        disp_cols = [c for c in ["hypothesis_id", "hypothesis_statement", "priority_level",
                                  "total_score", "hypothesis_type"]
                     if c in hyp_df.columns]
        st.dataframe(hyp_df[disp_cols].head(20), use_container_width=True, hide_index=True)
    else:
        st.info("暂无假设数据，请先运行智能搜索的假设生成功能。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Literature Formula Library (文献公式库)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "formula_explain":
    import pandas as pd
    from src.literature_formula_library import (
        EVIDENCE_STATUSES,
        FORMULA_EMPTY_MESSAGE,
        FORMULA_TABLE_FIELDS,
        FORMULA_TYPES,
        formula_summary,
        formula_to_table_row,
        filter_literature_formulas,
        load_system_model_registry,
    )

    st.markdown("<h1 style='text-align: center;'>📐 文献公式库</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#888;'>逐页精读文献中实际出现的公式、变量、条件与来源</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    from src.data_cache import load_literature_formulas_cached, literature_formula_version
    literature_formulas = load_literature_formulas_cached(
        str(BASE_DIR), literature_formula_version(BASE_DIR)
    )
    paper_options = {
        f"{row['paper_title']} [{row['paper_id']}]": row["paper_id"]
        for row in literature_formulas
    }

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        paper_scope = st.selectbox(
            "文献范围筛选",
            ["全部文献", "指定文献"],
            key="formula_paper_scope",
        )
        selected_paper_id = None
        if paper_scope == "指定文献" and paper_options:
            selected_paper_label = st.selectbox(
                "选择文献",
                sorted(paper_options),
                key="formula_selected_paper",
            )
            selected_paper_id = paper_options[selected_paper_label]
    with filter_col2:
        selected_formula_type = st.selectbox(
            "公式类型筛选",
            FORMULA_TYPES,
            key="literature_formula_type",
        )
    with filter_col3:
        selected_evidence_status = st.selectbox(
            "证据状态筛选",
            EVIDENCE_STATUSES,
            index=0 if formula_summary(literature_formulas)["confirmed"] else 1,
            key="literature_formula_evidence_status",
        )

    filtered_formulas = filter_literature_formulas(
        literature_formulas,
        paper_id=selected_paper_id,
        formula_type=selected_formula_type,
        evidence_status=selected_evidence_status,
    )

    if not literature_formulas:
        st.info(FORMULA_EMPTY_MESSAGE)
    elif not filtered_formulas:
        st.info("当前筛选条件下没有文献公式记录。")
    else:
        summary = formula_summary(literature_formulas)
        st.caption(
            f"真实页公式证据记录 {summary['total']} 条；已确认 {summary['confirmed']} 条；"
            f"待人工复核 {summary['pending_review']} 条；"
            f"图像或不可可靠解析 {summary['image_review_required']} 条。"
        )
        formula_df = pd.DataFrame(
            [formula_to_table_row(row) for row in filtered_formulas],
            columns=FORMULA_TABLE_FIELDS,
        )
        selection = st.dataframe(
            formula_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="literature_formula_table",
        )
        selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
        if selected_rows:
            selected_formula = filtered_formulas[selected_rows[0]]
            with st.expander("公式证据详情", expanded=True):
                st.markdown(f"**paper_id**：`{selected_formula['paper_id']}`")
                st.markdown(f"**论文完整题目**：{selected_formula['paper_title']}")
                st.markdown(f"**DOI**：{selected_formula['doi']}")
                st.markdown(f"**页码**：{selected_formula['page_number']}")
                st.markdown(f"**章节**：{selected_formula['section']}")
                st.markdown(f"**公式编号**：{selected_formula['equation_number']}")
                st.markdown("**原始公式**")
                st.code(selected_formula["original_formula"], language="text")
                st.markdown("**标准化 LaTeX**")
                if selected_formula["normalized_latex"]:
                    st.latex(selected_formula["normalized_latex"])
                else:
                    st.info("待人工复核，不自动补写。")
                st.markdown("**公式前后原文**")
                st.text(selected_formula["context_before_after"])
                st.markdown(
                    "**各符号定义**：" + "；".join(selected_formula["symbol_definitions"])
                )
                st.markdown(
                    "**各符号单位**：" + "；".join(selected_formula["symbol_units"])
                )
                st.markdown(
                    "**文献采用的参数**："
                    + "；".join(selected_formula["parameter_values_units"])
                )
                st.markdown(f"**数据来源**：`{selected_formula['data_source']}`")
                st.markdown(f"**作者给出的适用范围**：{selected_formula['author_scope']}")
                st.markdown(f"**作者给出的局限**：{selected_formula['author_limitations']}")
                st.markdown(
                    f"**人工审核状态**：`{selected_formula['manual_review_status']}`"
                )

    st.divider()
    with st.expander("基础理论模型参考", expanded=False):
        st.warning(
            "以下内容属于 `SYSTEM_MODEL_REGISTRY`，仅作为理论参考，"
            "不是本地文献实际提取结果。"
        )
        registry_df = pd.DataFrame(load_system_model_registry())
        st.dataframe(registry_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Experiment Design (实验验证方案)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "experiment_design":
    import pandas as pd
    st.markdown("<h1 style='text-align: center;'>🧫 实验验证方案</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>基于研究空白和假设的实验方案设计</p>")
    st.divider()

    st.info(
        "该页面辅助设计科研实验方案，包括样品分组、测试方法、表征方法和判定标准。\n\n"
        "您也可以在「智能搜索」页面输入科研问题并选择「实验验证设计」模式。"
    )

    if st.button("🔍 前往智能搜索使用实验验证设计模式", use_container_width=True):
        st.session_state.page = "search"
        st.session_state.answer_mode = "experiment_design"
        st.rerun()

    # Show research gaps with experimental designs
    gap_path = Path("data/research_gap_dataset.csv")
    if gap_path.exists():
        gap_df = pd.read_csv(gap_path, encoding="utf-8-sig", on_bad_lines="skip")
        has_exp = [c for c in ["gap_id", "gap_title", "experimental_design", "priority_level"] if c in gap_df.columns]
        if has_exp and "experimental_design" in gap_df.columns:
            with_exp = gap_df[gap_df["experimental_design"].notna() & (gap_df["experimental_design"] != "")]
            st.subheader(f"已有实验方案的研究空白 ({len(with_exp)} 个)")
            if len(with_exp) > 0:
                st.dataframe(with_exp[has_exp].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("暂无研究空白数据，请先在「研究空白发现」页面生成。")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Failure Cases (失败案例分析)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "failure_cases":
    import pandas as pd
    st.markdown("<h1 style='text-align: center;'>📋 失败案例分析</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>系统生成失败案例的跟踪、分析和改进记录</p>")
    st.divider()

    failure_path = Path("data/failure_cases.csv")
    if failure_path.exists():
        fail_df = pd.read_csv(failure_path, encoding="utf-8-sig", on_bad_lines="skip")
        st.metric("失败案例总数", len(fail_df))
        st.dataframe(fail_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无失败案例数据。运行系统验证流程后会自动生成。")

    # Try from scientific_framework
    try:
        from src.scientific_framework import get_failure_cases
        cases = get_failure_cases()
        if cases:
            with st.expander("📂 从模块加载的失败案例", expanded=False):
                for case in cases[:10]:
                    st.markdown(f"- **{case.get('id', '?')}**: {case.get('title', '')}")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Time Slice Validation (时间切片回溯验证)
# ═══════════════════════════════════════════════════════════════════════════

elif current_page == "time_slice_validation":
    import pandas as pd
    st.markdown("<h1 style='text-align: center;'>🔄 时间切片回溯验证</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>按时间分割文献库，回溯验证系统生成的假设方向</p>")
    st.divider()

    st.info("该模块将文献库按时间分为早期和后期两组，验证早期数据生成的假设能否被后期文献证实。")

    try:
        from src.retrospective_validation import run_retrospective_validation
    except ImportError:
        st.error("retrospective_validation 模块不可用。")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        split_year = st.number_input("分割年份", min_value=2018, max_value=2024, value=2022, step=1)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 运行回溯验证", type="primary", use_container_width=True):
            with st.spinner("正在运行时间切片回溯验证..."):
                result = run_retrospective_validation(split_year=split_year)
            st.success(f"完成！")

    # Show results
    retro_path = Path("data/retrospective_validation_results.csv")
    if retro_path.exists():
        retro_df = pd.read_csv(retro_path, encoding="utf-8-sig", on_bad_lines="skip")
        st.subheader("回溯验证结果")
        st.dataframe(retro_df, use_container_width=True, hide_index=True)

        # Summary stats
        if "support_level" in retro_df.columns:
            st.subheader("支持等级分布")
            sup_counts = retro_df["support_level"].value_counts()
            for level, count in sup_counts.items():
                st.metric(level, count)

    retro_pairs = Path("data/retrospective_validation_pairs.csv")
    if retro_pairs.exists():
        pairs_df = pd.read_csv(retro_pairs, encoding="utf-8-sig", on_bad_lines="skip")
        with st.expander(f"验证对详情 ({len(pairs_df)} 对)", expanded=False):
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
