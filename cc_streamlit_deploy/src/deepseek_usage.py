"""DeepSeek call auditing without recording prompts, responses, or secrets."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
_CALL_LOG: List[Dict[str, str]] = []


def log_call(
    stage: str,
    model_name: str,
    purpose: str,
    success: bool,
) -> str:
    call_id = f"D{len(_CALL_LOG) + 1:04d}"
    _CALL_LOG.append(
        {
            "call_id": call_id,
            "stage": stage,
            "model_name": model_name,
            "purpose": purpose,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success_or_fail": "success" if success else "fail",
        }
    )
    return call_id


def run_deepseek_usage_report() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "deepseek_call_log.csv"
    fields = [
        "call_id",
        "stage",
        "model_name",
        "purpose",
        "timestamp",
        "success_or_fail",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_CALL_LOG)
    success = sum(row["success_or_fail"] == "success" for row in _CALL_LOG)
    report = [
        "# DeepSeek Usage Report",
        "",
        f"- Calls: {len(_CALL_LOG)}",
        f"- Success: {success}",
        f"- Fail: {len(_CALL_LOG) - success}",
        "",
        "本报告不记录提示词、模型响应、API Key 或访问密码。",
    ]
    (OUTPUTS_DIR / "11_deepseek_usage_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return {
        "total_calls": len(_CALL_LOG),
        "success": success,
        "fail": len(_CALL_LOG) - success,
        "call_log": str(csv_path),
    }
