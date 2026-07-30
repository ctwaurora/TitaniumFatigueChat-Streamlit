from typing import Any, Dict, List

from .deepseek_skill import call_deepseek
from .library_skill import load_cards


def _get_deepseek_content(response: Any) -> str:
    """
    从 DeepSeek 返回结果中提取文本内容。
    兼容 call_deepseek 返回 dict 或 str 的情况。
    """
    if isinstance(response, str):
        return response

    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(response)


def _get_card_from_item(item: Any) -> Dict[str, Any]:
    """
    兼容两种文献卡片存储格式：
    1. {"card": {...}}
    2. {...}
    """
    if isinstance(item, dict) and isinstance(item.get("card"), dict):
        return item["card"]

    if isinstance(item, dict):
        return item

    return {}


def generate_case_demo() -> str:
    """
    生成案例演示报告。
    用于展示 TitaniumFatigueChat 如何基于已有文献库生成科研假设与验证方案。
    """

    cards = load_cards()
    sample_size = min(3, len(cards))

    if sample_size == 0:
        return "没有文献数据，无法生成演示案例。请先上传或搜索文献。"

    # 准备样本文献数据
    samples: List[str] = []

    for i in range(sample_size):
        card = _get_card_from_item(cards[i])

        title = card.get("title", "无标题")
        alloy_type = (
            card.get("alloy_type")
            or card.get("material")
            or card.get("keywords")
            or "未知材料体系"
        )

        if isinstance(alloy_type, list):
            alloy_type = "、".join(str(x) for x in alloy_type[:3])

        samples.append(f"- {alloy_type}: {title}")

    samples_text = "\n".join(samples)

    prompt = f"""
请生成以下案例演示报告。

案例名称：
“钛合金疲劳寿命预测文献库驱动的 AI Scientist 科研假设生成案例”

背景：
本案例展示 TitaniumFatigueChat 系统如何帮助研究人员围绕钛合金疲劳研究构建个人文献库，并基于文献检索、RAG 问答和智能体分析生成可验证的科研假设。

案例数据：
当前系统已读取 {sample_size} 篇精选文献：

{samples_text}

报告要求：
1. 系统功能概述，控制在 200 字以内。
2. 案例操作流程，包括：文献上传 → 文献解析 → 文献检索 → RAG 问答 → 科研假设生成。
3. 生成 1 个具体的研究假设示例。
4. 给出该假设的验证方案亮点。
5. 应用价值总结，控制在 200 字以内。

格式要求：
- 使用中文。
- 包含清晰小标题。
- 重点突出技术价值。
- 保持专业科研风格。
- 不要写成宣传广告，要像题目求解中的案例验证。

输出格式：

### 案例演示报告

[正文内容]
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = call_deepseek(messages)
    return _get_deepseek_content(response)
