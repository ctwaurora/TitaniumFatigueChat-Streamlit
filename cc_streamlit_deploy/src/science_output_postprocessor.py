"""Chinese scientific-output cleanup without adding scientific content."""

from __future__ import annotations

import re
from typing import Any


FIELD_LABELS = {
    "temperature": "温度",
    "environment": "环境",
    "build_orientation": "构建方向",
    "loading_mode": "加载方式",
    "surface_condition": "表面状态",
    "surface_state": "表面状态",
    "surface_treatment": "表面处理",
    "heat_treatment": "热处理",
    "stress_ratio": "应力比",
    "stress_ratio_R": "应力比",
    "crack_length": "裂纹长度",
    "material_batch": "材料批次",
    "manufacturing_window": "制造工艺窗口",
    "alloy_grade": "合金牌号",
    "manufacturing_process": "制造工艺",
    "frequency": "频率",
    "hip": "HIP状态",
    "initial_crack_length": "初始裂纹长度",
}

_DUPLICATES = {
    "表面表面表面": "表面",
    "表面表面": "表面",
    "粗糙度度": "粗糙度",
    "裂纹裂纹": "裂纹",
    "速率速率": "速率",
    "疲劳疲劳": "疲劳",
    "组织组织": "组织",
    "采用采用": "采用",
}
_JSON_OBJECT = re.compile(r"(?m)^\s*[\[{]\s*[\"']?[A-Za-z_][\s\S]{0,1200}?[\]}]\s*$")
_CODE_FENCE = re.compile(r"```(?:json|python)?\s*([\s\S]*?)```", re.I)


def postprocess_science_output(text: Any) -> str:
    output = str(text or "").replace("\x00", "")
    for source, target in _DUPLICATES.items():
        output = output.replace(source, target)
    for source, target in sorted(FIELD_LABELS.items(), key=lambda item: -len(item[0])):
        output = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])", target, output)
    output = output.replace("系统推断（当前引文未直接验证）：", "【系统候选推断】")
    output = re.sub(r"(?:【系统候选推断】\s*){2,}", "【系统候选推断】", output)
    output = re.sub(r"[。．]{2,}", "。", output)
    output = re.sub(r"[。．](?=[，,；;])", "", output)
    output = re.sub(r"[；;]{2,}", "；", output)
    output = re.sub(r"。[；;]", "；", output)
    output = re.sub(r"(?m)^\s*[，,；;]\s*", "", output)
    output = re.sub(r"[ \t]+([，。；：！？])", r"\1", output)
    output = re.sub(r"([，。；：！？])[ \t]+", r"\1", output)
    output = re.sub(r"\n{3,}", "\n\n", output)
    output = _CODE_FENCE.sub(lambda match: match.group(1).strip(), output)
    return output.strip()


def science_text_quality_audit(text: Any) -> dict[str, Any]:
    value = str(text or "")
    audit_value = "\n".join(
        line
        for line in value.splitlines()
        if not line.strip().startswith("原文摘录：")
    )
    duplicate_hits = {
        token: audit_value.count(token) for token in _DUPLICATES if token in audit_value
    }
    raw_fields = sorted(
        field for field in FIELD_LABELS
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", audit_value)
    )
    markdown_leaks = []
    if "```json" in value.casefold() or "```python" in value.casefold():
        markdown_leaks.append("code_fence")
    if _JSON_OBJECT.search(value):
        markdown_leaks.append("serialized_object")
    incomplete = [
        line for line in value.splitlines()
        if line.strip().startswith(("，", ",", "；", ";"))
    ]
    repeated_punctuation = re.findall(r"[。．；;][。．，,；;]", audit_value)
    return {
        "passed": not duplicate_hits and not raw_fields and not markdown_leaks and not incomplete and not repeated_punctuation,
        "duplicate_word_hits": duplicate_hits,
        "repeated_punctuation_hits": repeated_punctuation,
        "raw_field_names": raw_fields,
        "raw_markdown_or_object_hits": markdown_leaks,
        "incomplete_sentence_lines": incomplete,
        "error_count": sum(duplicate_hits.values()) + len(repeated_punctuation) + len(raw_fields) + len(markdown_leaks) + len(incomplete),
    }
