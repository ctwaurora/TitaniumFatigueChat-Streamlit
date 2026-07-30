from .deepseek_skill import call_deepseek_text
from .rag_skill import get_evidence_text


def review_evidence_chain(hypothesis: str) -> str:
    evidence = get_evidence_text("疲劳 寿命 证据 机制 模型 验证", top_k=6)

    prompt = f"""
你是科研方案评审专家。
请对下面的科学假设进行证据链审查和评分。

输出结构：
1. 假设是否清晰
2. 证据是否充分
3. 是否存在空泛表述
4. 需要补充哪些文献
5. 需要补充哪些实验或仿真
6. 可验证性评分，满分100
7. 修改建议
8. 改进后的假设版本

科学假设：
{hypothesis}

可用证据：
{evidence}
"""

    return call_deepseek_text(prompt, max_tokens=4000, temperature=0.2)
