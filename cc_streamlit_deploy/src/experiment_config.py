"""Versioned configuration contract for frozen paper experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_RELATIVE = Path("config/paper_experiment_config.json")


def load_paper_experiment_config(base_dir: Path) -> dict[str, Any]:
    path = Path(base_dir).resolve() / CONFIG_RELATIVE
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / CONFIG_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "reranking", "gate", "model"}
    missing = required - set(payload)
    if missing:
        raise ValueError("PAPER_EXPERIMENT_CONFIG_MISSING:" + ",".join(sorted(missing)))
    if payload["reranking"].get("stable_tie_break") != [
        "final_evidence_score_desc", "document_id_asc", "evidence_id_asc"
    ]:
        raise ValueError("PAPER_EXPERIMENT_CONFIG_UNSTABLE_TIE_BREAK")
    return payload


def paper_experiment_config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
