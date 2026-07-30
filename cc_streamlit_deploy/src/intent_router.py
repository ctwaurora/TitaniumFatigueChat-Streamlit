"""
intent_router.py — LLM-based Intent Classification Router

使用 DeepSeek 识别用户意图，返回匹配的模块列表。
支持复杂自然语言，不是关键词匹配。
"""

import json
import re
from typing import List, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── 所有可识别的意图 ──────────────────────────────────────────────────

INTENT_CATEGORIES = {
    "literature_review": {
        "name": "文献综述",
        "description": "用户想了解某主题的文献概况、研究现状",
        "keywords": ["综述", "回顾", "概括", "总结", "overview", "review", "现状"],
    },
    "evidence_extraction": {
        "name": "文献证据提取",
        "description": "用户想提取某变量关系的实验证据和条件",
        "keywords": ["证据", "实验条件", "提取", "条件", "字段"],
    },
    "mechanism_analysis": {
        "name": "机制分析",
        "description": "用户想了解某现象的损伤机制、机制链",
        "keywords": ["机制", "机理", "裂纹萌生", "扩展", "起裂", "mechanism", "initiation"],
    },
    "mechanism_comparison": {
        "name": "多机制比较",
        "description": "用户想比较多种机制在不同条件下的主导关系",
        "keywords": ["竞争", "主导", "比较", "哪个", "vs", "versus", "还是"],
    },
    "condition_mechanism_map": {
        "name": "条件-机制相图",
        "description": "用户想看实验条件到机制的映射关系",
        "keywords": ["相图", "条件-机制", "map", "映射", "条件图"],
    },
    "research_gap": {
        "name": "研究空白",
        "description": "用户想了解当前领域缺少哪些证据或研究",
        "keywords": ["空白", "gap", "未解决", "缺乏", "不足", "下一步"],
    },
    "counter_evidence": {
        "name": "反向证据检索",
        "description": "用户想找反驳某假设的证据",
        "keywords": ["反证", "反驳", "反向证据", "counter", "质疑", "争议"],
    },
    "experiment_design": {
        "name": "实验设计",
        "description": "用户想设计实验验证某假设",
        "keywords": ["实验", "验证", "方案", "怎么做", "如何验证", "设计实验", "分组"],
    },
    "data_analysis": {
        "name": "数据分析方法",
        "description": "用户想知道如何分析实验数据",
        "keywords": ["数据分析", "统计", "拟合", "回归", "怎么分析", "如何处理"],
    },
    "formula_explanation": {
        "name": "公式/模型解释",
        "description": "用户想了解某公式或模型的含义",
        "keywords": ["公式", "模型", "方程", "Paris", "Murakami", "Kitagawa", "Basquin"],
    },
    "formula_comparison": {
        "name": "多公式比较",
        "description": "用户想比较多个模型或公式的适用性",
        "keywords": ["哪个模型", "哪个公式", "比较模型", "选择模型", "对比"],
    },
    "full_analysis": {
        "name": "全面分析",
        "description": "用户明确要求输出全部模块的完整分析",
        "keywords": ["全面分析", "完整分析", "系统分析", "全面", "完整报告", "full analysis"],
    },
}

INTENT_KEYS = list(INTENT_CATEGORIES.keys())
INTENT_NAMES = {k: v["name"] for k, v in INTENT_CATEGORIES.items()}


# ═══════════════════════════════════════════════════════════════════════
# LLM-based Intent Classification
# ═══════════════════════════════════════════════════════════════════════

def classify_intent_llm(question: str) -> List[str]:
    """
    使用 DeepSeek 对用户问题进行意图分类。
    返回匹配的 intent key 列表。

    Args:
        question: 用户原始问题

    Returns:
        intent_keys: 如 ["mechanism_analysis", "experiment_design"]
    """
    # 首先检查是否明确要求全面分析
    q_lower = question.lower()
    full_triggers = ["全面分析", "完整分析", "系统分析", "全面报告",
                     "full analysis", "complete analysis", "all modules"]
    if any(t in q_lower for t in full_triggers):
        return ["full_analysis"]

    # 尝试调用 DeepSeek API 进行分类
    try:
        return _call_llm_classify(question)
    except Exception:
        # LLM 失败时回退到关键词辅助分类
        return _classify_fallback(question)


def _call_llm_classify(question: str) -> List[str]:
    """调用 DeepSeek API 进行意图分类。"""
    from src.api_keys import get_deepseek_settings

    settings = get_deepseek_settings()
    if not settings.configured:
        return _classify_fallback(question)

    prompt = f"""你是一个科研助手系统的意图分类器。
你的任务是将用户的科研问题分类到以下类别中（可多选）。

类别列表：
{json.dumps(INTENT_NAMES, ensure_ascii=False, indent=2)}

分类规则：
1. 仔细阅读用户问题，理解他真正想做什么
2. 如果问题涉及多个方面，返回多个类别
3. 如果用户问"怎么做实验"→ 实验设计
4. 如果用户问"为什么这样"→ 机制分析
5. 如果用户问"什么关系"→ 文献证据提取 + 机制分析
6. 如果用户问"哪个更重要"→ 多机制比较
7. 如果用户问还缺少什么 → 研究空白
8. 如果用户问有没有反例 → 反向证据检索
9. 如果用户问公式/模型 → 公式/模型解释
10. 如果用户想比较模型 → 多公式比较

用户问题：{question}

请只输出一个 JSON 数组，如 ["mechanism_analysis", "experiment_design"]。
不要输出其他文字。如果无法确定，默认返回 ["literature_review", "evidence_extraction"]。"""

    try:
        from src.deepseek_client import DeepSeekClient
        from src.deepseek_usage import log_call

        client = DeepSeekClient(settings)
        text = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            timeout=15,
        )
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            intents = json.loads(match.group(0))
            if isinstance(intents, list) and intents:
                valid = [intent for intent in intents if intent in INTENT_KEYS]
                if valid:
                    log_call("intent_classification", settings.model, "科研问题意图分类", True)
                    return valid
    except Exception:
        pass

    return _classify_fallback(question)


def _classify_fallback(question: str) -> List[str]:
    """
    回退分类器：基于规则+关键词的意图识别。
    不使用纯关键词匹配，而是结合语义模式。
    """
    q = question.lower().strip()

    # 模式匹配：按优先级从高到低
    patterns = [
        # 实验设计模式
        ("experiment_design", [
            r"怎么.*验证", r"如何.*验证", r"设计.*实验", r"实验.*方案",
            r"验证.*假设", r"怎么.*做", r"试验.*设计", r"如何.*设计",
            r"样品.*分组", r"测试.*方法", r"表征.*方法",
            r"if i want to verify", r"experiment", r"test plan",
        ]),
        # 机制分析模式
        ("mechanism_analysis", [
            r"机制", r"机理", r"为什么会", r"裂纹.*萌生", r"裂纹.*扩展",
            r"起裂", r"损伤.*机理", r"原因", r"mechanism", r"why",
            r"initiation", r"propagation",
        ]),
        # 多机制比较模式
        ("mechanism_comparison", [
            r"哪个.*更容易", r"哪个.*更.*主导", r"比较.*机制",
            r"竞争", r"vs", r"versus", r"还是.*主导",
            r"which.*dominat", r"competition",
        ]),
        # 条件-机制相图
        ("condition_mechanism_map", [
            r"条件.*机制", r"相图", r"条件.*图", r"机制.*条件",
            r"condition.*map", r"map.*mechanism",
        ]),
        # 反向证据检索
        ("counter_evidence", [
            r"反证", r"反驳", r"质疑", r"争议", r"反例",
            r"有没有.*不同", r"有.*相反", r"counter.*evidence",
            r"反对", r"不同.*结论",
        ]),
        # 研究空白
        ("research_gap", [
            r"研究空白", r"gap", r"未解决", r"还缺.*什么",
            r"不足.*之处", r"research gap", r"missing",
            r"下一步.*研究", r"future work",
        ]),
        # 公式/模型解释
        ("formula_explanation", [
            r"公式.*解释", r"模型.*解释", r"paris.*公式", r"murakami.*模型",
            r"kitagawa.*takahashi", r"basquin", r"walker.*模型",
            r"公式.*意思", r"模型.*含义", r"equation", r"formula",
        ]),
        # 多公式比较
        ("formula_comparison", [
            r"哪个.*模型", r"哪个.*公式", r"比较.*模型", r"选择.*模型",
            r"model.*comparison", r"which model",
        ]),
        # 数据分析
        ("data_analysis", [
            r"数据.*分析", r"统计.*方法", r"怎么.*拟合", r"怎么.*回归",
            r"data analysis", r"how.*analyze", r"curve fitting",
        ]),
        # 文献证据提取
        ("evidence_extraction", [
            r"证据", r"实验条件", r"提取.*数据", r"文献.*证据",
            r"evidence", r"condition", r"parameter",
        ]),
        # 文献综述（最低优先级）
        ("literature_review", [
            r"综述", r"总结.*文献", r"研究.*现状", r"overview", r"review",
        ]),
    ]

    scores = {}
    for intent, regex_list in patterns:
        score = 0
        for regex in regex_list:
            if re.search(regex, q):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        # 默认：文献证据 + 机制分析
        return ["literature_review", "evidence_extraction"]

    # 按分数排序
    sorted_intents = sorted(scores.keys(), key=lambda k: -scores[k])

    # 如果实验设计 + 机制分析同时出现，说明是"验证机制"类型
    if "experiment_design" in sorted_intents and "mechanism_analysis" in sorted_intents:
        return ["experiment_design", "mechanism_analysis"]

    # 返回最高分的前 1-2 个
    return sorted_intents[:2]


# ── 模块调度映射 ──────────────────────────────────────────────────────

# 每个 intent 对应的 unified_answer 模块函数
INTENT_TO_MODULE = {
    "literature_review": ["section_question_understanding", "section_direct_conclusion"],
    "evidence_extraction": ["summarize_condition_evidence_for_question"],
    "mechanism_analysis": ["section_mechanism_map", "section_direct_conclusion"],
    "mechanism_comparison": ["section_mechanism_map", "section_counter_evidence"],
    "condition_mechanism_map": ["section_mechanism_map"],
    "research_gap": ["section_data_gaps", "section_counter_evidence"],
    "counter_evidence": ["section_counter_evidence"],
    "experiment_design": ["section_experiment_design"],
    "data_analysis": ["section_data_gaps"],
    "formula_explanation": ["section_model_comparison"],
    "formula_comparison": ["section_model_comparison"],
    "full_analysis": [
        "section_question_understanding", "section_direct_conclusion",
        "summarize_condition_evidence_for_question", "section_mechanism_map",
        "section_counter_evidence", "section_hypotheses",
        "section_experiment_design", "section_model_comparison",
        "section_data_gaps", "section_hypothesis_scoring",
    ],
}

# Intent 中文名映射（用于回答顶部显示）
INTENT_DISPLAY = {
    "literature_review": "📚 文献综述",
    "evidence_extraction": "🔬 文献证据提取",
    "mechanism_analysis": "⚙️ 机制分析",
    "mechanism_comparison": "⚖️ 多机制比较",
    "condition_mechanism_map": "🗺️ 条件-机制相图",
    "research_gap": "🔍 研究空白发现",
    "counter_evidence": "🔄 反向证据检索",
    "experiment_design": "🧪 实验设计",
    "data_analysis": "📊 数据分析方法",
    "formula_explanation": "📐 公式/模型解释",
    "formula_comparison": "📏 多公式比较",
    "full_analysis": "📋 全面分析",
}


def get_intent_display(intents: List[str]) -> str:
    """生成 intent 显示字符串。"""
    names = [INTENT_DISPLAY.get(i, i) for i in intents]
    return " | ".join(names)


def get_modules_for_intents(intents: List[str]) -> List[str]:
    """根据 intent 列表返回需要调用的模块函数名列表。"""
    modules = []
    seen = set()
    for intent in intents:
        for mod in INTENT_TO_MODULE.get(intent, []):
            if mod not in seen:
                modules.append(mod)
                seen.add(mod)
    return modules
