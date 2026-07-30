import json
from pathlib import Path

from .deepseek_skill import call_deepseek_text
from .rag_skill import get_evidence_text


CARDS_PATH = Path("data/literature_cards.jsonl")


def _read_cards_text() -> str:
    if not CARDS_PATH.exists():
        return ""

    cards = []
    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(line)

    return "\n".join(cards[:20])


def generate_research_gaps() -> str:
    cards_text = _read_cards_text()

    if not cards_text:
        return "当前文献库为空，请先上传并抽取文献卡片。"

    evidence = get_evidence_text("疲劳 寿命 裂纹 扩展 热处理 组织 模型 研究不足", top_k=6)

    prompt = f"""
你是材料疲劳断裂方向的科研助手。
请根据文献卡片和 RAG 证据，生成研究空白报告。

输出结构：
1. 已有研究基础
2. 研究热点
3. 研究不足
4. 低覆盖组合
5. 可切入科学问题
6. 推荐研究方向
7. 对应证据片段

文献卡片：
{cards_text}

RAG证据：
{evidence}
"""

    return call_deepseek_text(prompt, max_tokens=4000, temperature=0.2)
