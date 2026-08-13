"""Version-aware dataset selection without mutating frozen manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTIVE_POINTER = "data/system/active_dataset_manifest.json"
V1_FALLBACK = "data/system/verified_dataset_v1_candidate_manifest.json"


def active_dataset_manifest_path(base_dir: Path) -> Path:
    root = Path(base_dir)
    pointer = root / ACTIVE_POINTER
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        relative = str(payload.get("manifest_path") or "").strip()
        if not relative or Path(relative).is_absolute():
            raise RuntimeError("ACTIVE_DATASET_POINTER_INVALID")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError("ACTIVE_DATASET_POINTER_ESCAPES_PROJECT") from exc
        if not candidate.is_file():
            raise RuntimeError("ACTIVE_DATASET_MANIFEST_MISSING")
        return candidate
    return root / V1_FALLBACK


def load_active_dataset_manifest(base_dir: Path) -> dict[str, Any]:
    path = active_dataset_manifest_path(base_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = payload.get("papers") or payload.get("documents") or []
    declared = payload.get("verified_paper_count", payload.get("formal_document_count"))
    if not isinstance(papers, list) or declared is None or len(papers) != int(declared):
        raise RuntimeError("FORMAL_PROVENANCE_VIOLATION:ACTIVE_DATASET_MANIFEST_INVALID")
    return payload


def active_dataset_ids(base_dir: Path) -> set[str]:
    payload = load_active_dataset_manifest(base_dir)
    papers = payload.get("papers") or payload.get("documents") or []
    ids = {
        str(row.get("document_id") or row.get("paper_id") or "")
        for row in papers
        if row.get("document_id") or row.get("paper_id")
    }
    if len(ids) != len(papers):
        raise RuntimeError("FORMAL_PROVENANCE_VIOLATION:ACTIVE_DATASET_IDS_INVALID")
    return ids
