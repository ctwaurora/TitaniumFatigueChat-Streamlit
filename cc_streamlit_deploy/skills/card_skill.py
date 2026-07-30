"""
card_skill.py — 文献卡片抽取（增强版）

调用 DeepSeek 从 PDF 文本中抽取结构化文献卡片，包含全部领域字段。
"""

import json
import re
from typing import Dict, Any, List

from .deepseek_skill import call_deepseek


def _get_deepseek_content(response: Any) -> str:
    """从 DeepSeek 返回结果中提取文本内容。"""
    if isinstance(response, str):
        return response
    # OpenAI-compatible format: choices[0].message.content
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    raise ValueError("DeepSeek 返回格式异常")


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON。"""
    text = text.strip()

    # 去掉 markdown 代码块
    code_block_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 从文本中截取第一个 { 到最后一个 }
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("模型响应中未找到有效 JSON")
    json_text = text[json_start:json_end]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{str(e)}\n原始内容：{text}")


def _normalize_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """统一文献卡片字段，确保所有字段存在。"""
    default_card = {
        "title": "",
        "authors": [],
        "publication_year": "",
        "abstract": "",
        "key_findings": [],
        "methods": "",
        "conclusion": "",
        "keywords": [],
        "doi": "",
        "journal": "",

        # 领域扩展字段
        "material_system": "",
        "processing_method": "",
        "heat_treatment": "",
        "microstructure": "",
        "loading_condition": "",
        "stress_ratio_R": "",
        "temperature_environment": "",
        "experimental_methods": "",
        "characterization_methods": "",
        "mechanical_indicators": "",
        "crack_initiation": "",
        "crack_growth_mechanism": "",
        "model_or_method": "",
        "limitations": "",
        "possible_innovation": "",
        "evidence_text": "",
    }

    if not isinstance(card, dict):
        raise ValueError("文献卡片结果不是字典格式")

    for key, default_value in default_card.items():
        if key not in card or card[key] is None:
            card[key] = default_value

    # 字段类型修正
    if isinstance(card.get("authors"), str):
        card["authors"] = [card["authors"]]
    if isinstance(card.get("key_findings"), str):
        card["key_findings"] = [card["key_findings"]]
    if isinstance(card.get("keywords"), str):
        card["keywords"] = [card["keywords"]]

    return card


def extract_literature_card(pdf_text: str) -> Dict[str, Any]:
    """调用 DeepSeek 从 PDF 文本中抽取结构化文献卡片（增强版）。

    包含全部领域字段：材料体系、制备工艺、热处理、微观组织、
    载荷条件、表征方法、力学指标、裂纹起裂与扩展机制等。
    """
    if not pdf_text or not pdf_text.strip():
        raise ValueError("PDF 文本为空，无法提取文献卡片")

    text_for_analysis = pdf_text[:12000]

    prompt = f"""你是一名钛合金疲劳断裂领域的高级文献分析专家。
请从下面的论文文本中提取关键信息，生成结构化文献卡片。

请严格按照以下 JSON 格式输出（只输出 JSON，不要 Markdown 代码块，不要解释）：

字段说明（关键要求）：
- title: 论文完整标题（字符串）
- authors: 作者列表（数组）
- publication_year: 发表年份（字符串，如"2025"）
- doi: DOI 编号（字符串，无则填空字符串）
- journal: 期刊名称（字符串，无则填空字符串）
- abstract: 摘要（300字以内）
- keywords: 关键词（数组，5个左右）
- material_system: 材料体系，如 "Ti-6Al-4V (TC4)"、"TC17"、"Ti60"、"7085铝合金" 等
- processing_method: 制备/加工工艺，如 "锻造"、"L-PBF增材制造"、"SLM"、"EBM"、"真空自耗电弧熔炼" 等
- heat_treatment: 热处理制度，如 "920°C固溶+550°C时效" 等
- microstructure: 微观组织，如 "α+β双态组织"、"片层组织"、"等轴组织"、"α'马氏体" 等
- loading_condition: 载荷条件，如 "低周疲劳(LCF)"、"高周疲劳(HCF)"、"超高周疲劳(VHCF)"、"疲劳裂纹扩展(FCGR)"、"变幅载荷" 等
- stress_ratio_R: 应力比R值，如 "0.1"、"-1" 等
- temperature_environment: 温度/环境，如 "室温"、"400°C"、"腐蚀环境" 等
- experimental_methods: 实验方法，如 "拉伸试验"、"疲劳试验"、"疲劳裂纹扩展试验" 等
- characterization_methods: 表征方法，如 "SEM"、"EBSD"、"TEM"、"XRD"、"DIC"、"X-ray CT"、"断口分析" 等
- mechanical_indicators: 力学/疲劳指标，如 "疲劳寿命Nf"、"S-N曲线"、"da/dN"、"ΔK"、"ΔKth"、"疲劳强度" 等
- crack_initiation: 裂纹起裂机制，如 "表面滑移带起裂"、"夹杂物/缺陷起裂"、"晶界起裂" 等
- crack_growth_mechanism: 裂纹扩展机制，如 "Paris区稳态扩展"、"裂纹闭合效应"、"氧化辅助扩展" 等
- model_or_method: 模型/方法，如 "Paris公式"、"Walker模型"、"NASGRO"、"GRNN"、"SVR"、"有限元" 等
- key_findings: 主要发现（数组，3-5条）
- conclusion: 结论
- methods: 研究方法简述
- limitations: 研究局限性/作者指出的不足，如 "未覆盖高温工况"、"微观机制验证不足" 等
- possible_innovation: 可能的创新点或拓展方向

注意：
1. 如果字段无法从文本判断，填写空字符串或空数组
2. authors、key_findings、keywords 必须是数组
3. material_system 等字段只需简单提取，不需要分析推理
4. 必须输出合法 JSON

论文内容如下：

{text_for_analysis}"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = call_deepseek(messages)
        content = _get_deepseek_content(response)
        card = _extract_json_from_text(content)
        card = _normalize_card(card)
        return card

    except json.JSONDecodeError:
        # 尝试 fallback: 从原文中提取基本信息
        return _fallback_extraction(pdf_text)
    except Exception as e:
        raise Exception(f"文献卡片抽取失败：{str(e)}")


def _fallback_extraction(pdf_text: str) -> Dict[str, Any]:
    """当 LLM 抽取失败时的规则回退。"""
    lines = pdf_text.split("\n")
    title_lines = [l.strip() for l in lines[:20] if l.strip() and len(l.strip()) > 20]
    title = title_lines[0][:200] if title_lines else ""

    return {
        "title": title,
        "authors": [],
        "publication_year": "",
        "doi": "",
        "journal": "",
        "abstract": "",
        "keywords": [],
        "material_system": "",
        "processing_method": "",
        "heat_treatment": "",
        "microstructure": "",
        "loading_condition": "",
        "stress_ratio_R": "",
        "temperature_environment": "",
        "experimental_methods": "",
        "characterization_methods": "",
        "mechanical_indicators": "",
        "crack_initiation": "",
        "crack_growth_mechanism": "",
        "model_or_method": "",
        "key_findings": [pdf_text[:500]],
        "conclusion": "",
        "methods": "",
        "limitations": "",
        "possible_innovation": "",
        "evidence_text": pdf_text[:1000],
    }
