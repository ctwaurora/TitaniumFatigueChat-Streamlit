"""Stable, traceable names for PDFs accepted into the formal RAG library."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from src.stage1_store import load_paper_manifest, load_pdf_file_records, stage1_paths


FORMAL_LIBRARY_STATUS = "FORMAL"
FORMAL_RAG_STATUS = "INDEXED_STAGE3_UNIFIED"
INVALID_WINDOWS_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
SEQUENCE_NAME = re.compile(r"^(\d{3,})_")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_title_filename(title: str, *, max_length: int = 170) -> str:
    value = INVALID_WINDOWS_CHARS.sub(" ", str(title or "").strip())
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip(" ._")
    if not value:
        value = "Untitled_formal_paper"
    value = value[:max_length].rstrip(" ._")
    return value or "Untitled_formal_paper"


def _formal_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_paper_manifest(base_dir)
        if row.get("library_status") == FORMAL_LIBRARY_STATUS
        and row.get("rag_status") == FORMAL_RAG_STATUS
        and row.get("pdf_valid") is True
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("title") or "").casefold(),
            str(row.get("paper_id") or ""),
        ),
    )


def build_rename_plan(base_dir: Path, *, append_only: bool = False) -> list[dict[str, Any]]:
    pdf_dir = (base_dir / "paper" / "pdfs").resolve()
    rows = _formal_rows(base_dir)
    if not rows:
        return []
    existing_numbers = []
    for row in rows:
        match = SEQUENCE_NAME.match(Path(str(row.get("canonical_pdf_path") or "")).name)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    plan: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for ordinal, row in enumerate(rows, start=1):
        old_path = Path(str(row.get("canonical_pdf_path") or "")).resolve()
        old_match = SEQUENCE_NAME.match(old_path.name)
        if append_only and old_match:
            number = int(old_match.group(1))
        elif append_only:
            number = next_number
            next_number += 1
        else:
            number = ordinal
        stem = safe_title_filename(str(row.get("title") or ""))
        candidate = f"{number:03d}_{stem}.pdf"
        suffix = 2
        while candidate.casefold() in used_names:
            candidate = f"{number:03d}_{stem}_{suffix}.pdf"
            suffix += 1
        used_names.add(candidate.casefold())
        new_path = pdf_dir / candidate
        plan.append(
            {
                "sequence": number,
                "old_filename": old_path.name,
                "new_filename": candidate,
                "paper_id": str(row.get("paper_id") or ""),
                "title": str(row.get("title") or ""),
                "doi": str(row.get("doi") or ""),
                "sha256": str(row.get("file_hash_sha256") or "").lower(),
                "old_path": str(old_path),
                "new_path": str(new_path),
                "rename_status": "UNCHANGED" if old_path == new_path else "PLANNED",
                "error_reason": "",
            }
        )
    return plan


def validate_rename_plan(plan: list[dict[str, Any]]) -> dict[str, int]:
    old_paths: set[Path] = set()
    new_paths: set[Path] = set()
    hashes: set[str] = set()
    for item in plan:
        old_path = Path(item["old_path"])
        new_path = Path(item["new_path"])
        expected = str(item["sha256"] or "").lower()
        if not old_path.is_file():
            raise RuntimeError(f"MISSING_SOURCE_PDF:{old_path}")
        actual = file_sha256(old_path)
        if not expected or actual != expected:
            raise RuntimeError(f"SOURCE_SHA_MISMATCH:{item['paper_id']}:{old_path}")
        old_key = old_path.resolve()
        new_key = new_path.resolve()
        if old_key in old_paths or new_key in new_paths:
            raise RuntimeError(f"DUPLICATE_RENAME_PATH:{new_path}")
        if expected in hashes:
            raise RuntimeError(f"DUPLICATE_FORMAL_SHA:{expected}")
        if new_path.exists() and new_key not in {Path(row["old_path"]).resolve() for row in plan}:
            raise RuntimeError(f"TARGET_ALREADY_EXISTS:{new_path}")
        old_paths.add(old_key)
        new_paths.add(new_key)
        hashes.add(expected)
    return {"verified_count": len(plan), "planned_count": sum(item["rename_status"] == "PLANNED" for item in plan)}


def _replace_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(item, path_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, path_map) for item in value]
    if isinstance(value, str):
        return path_map.get(value.casefold(), value)
    return value


def _update_active_references(base_dir: Path, plan: list[dict[str, Any]]) -> dict[str, int]:
    path_map = {
        str(Path(item["old_path"]).resolve()).casefold(): str(Path(item["new_path"]).resolve())
        for item in plan
    }
    paths = stage1_paths(base_dir)
    manifest = load_paper_manifest(base_dir)
    manifest_updates = 0
    for row in manifest:
        replaced = _replace_paths(row, path_map)
        if replaced != row:
            row.clear()
            row.update(replaced)
            manifest_updates += 1
    _write_jsonl(paths["paper_manifest"], manifest)

    pdf_rows = load_pdf_file_records(base_dir)
    pdf_updates = 0
    for row in pdf_rows:
        replaced = _replace_paths(row, path_map)
        if replaced != row:
            row.clear()
            row.update(replaced)
            row["current_filename"] = Path(str(row.get("canonical_pdf_path") or "")).name
            pdf_updates += 1
    _write_jsonl(paths["pdf_files"], pdf_rows)

    structured_updates = 0
    # Stage-2 artifacts retain the source PDF path for provenance.  Those
    # references must move in the same transaction as the manifest and asset
    # rows, otherwise the next incremental RAG build rejects every renamed
    # formal paper as untrusted.
    candidates = [
        base_dir / "data" / "deep_read",
        base_dir / "data" / "tasks",
        base_dir / "data" / "system",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for path in [*root.rglob("*.json"), *root.rglob("*.jsonl")]:
            try:
                if path.suffix == ".jsonl":
                    value = _read_jsonl(path)
                    replaced = _replace_paths(value, path_map)
                    if replaced != value:
                        _write_jsonl(path, replaced)
                        structured_updates += 1
                else:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    replaced = _replace_paths(value, path_map)
                    if replaced != value:
                        temp = path.with_suffix(path.suffix + ".tmp")
                        temp.write_text(
                            json.dumps(replaced, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        temp.replace(path)
                        structured_updates += 1
            except (OSError, json.JSONDecodeError):
                continue
    return {
        "manifest_updates": manifest_updates,
        "pdf_record_updates": pdf_updates,
        "structured_reference_updates": structured_updates,
    }


def execute_rename_plan(base_dir: Path, plan: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rename_plan(plan)
    moved: list[tuple[Path, Path, Path]] = []
    try:
        for item in plan:
            source = Path(item["old_path"])
            target = Path(item["new_path"])
            if source == target:
                continue
            temporary = source.with_name(f".{source.name}.{uuid.uuid4().hex}.rename_tmp")
            source.replace(temporary)
            moved.append((source, temporary, target))
        for source, temporary, target in moved:
            temporary.replace(target)
        updates = _update_active_references(base_dir, plan)
    except Exception:
        for source, temporary, target in reversed(moved):
            current = target if target.exists() else temporary
            if current.exists() and not source.exists():
                current.replace(source)
        raise

    failures = []
    renamed = 0
    for item in plan:
        target = Path(item["new_path"])
        actual = file_sha256(target) if target.is_file() else ""
        if actual != item["sha256"]:
            item["rename_status"] = "FAILED"
            item["error_reason"] = "POST_RENAME_SHA_MISMATCH_OR_MISSING"
            failures.append(item["paper_id"])
        elif item["old_path"] == item["new_path"]:
            item["rename_status"] = "UNCHANGED"
        else:
            item["rename_status"] = "RENAMED"
            renamed += 1
    if failures:
        raise RuntimeError("POST_RENAME_VALIDATION_FAILED:" + ",".join(failures))
    return {"renamed_count": renamed, "verified_count": len(plan), **updates}


def rename_new_formal_pdf(base_dir: Path, paper_id: str) -> dict[str, Any]:
    plan = [
        item
        for item in build_rename_plan(base_dir, append_only=True)
        if item["paper_id"] == paper_id
    ]
    if not plan:
        return {"renamed_count": 0, "verified_count": 0}
    return execute_rename_plan(base_dir, plan)
