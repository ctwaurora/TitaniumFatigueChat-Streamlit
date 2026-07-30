from .deepseek_skill import call_deepseek_text
from .rag_skill import get_evidence_text


def generate_hypothesis(gap_report: str) -> str:
    evidence = get_evidence_text("疲劳 寿命预测 裂纹扩展 热处理 显微组织 机器学习", top_k=6)

    prompt = f"""
你是材料科学 AI Scientist。
请根据研究空白报告和文献证据，生成可验证科学假设与研究计划。

输出结构：
1. 研究问题
2. 科学假设
3. 理论依据
4. 技术路线
5. 所需数据
6. 实验或仿真验证方案
7. 可用算法
8. 评价指标
9. 预期结果
10. 论文题目和摘要草案

研究空白报告：
{gap_report}

文献证据：
{evidence}
"""

    return call_deepseek_text(prompt, max_tokens=5000, temperature=0.25)
