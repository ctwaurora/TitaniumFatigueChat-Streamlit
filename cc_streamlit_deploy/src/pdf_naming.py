"""Stable, traceable names for PDFs accepted into the formal RAG library."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

import joblib

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
    manifest_rows = [
        row
        for row in load_paper_manifest(base_dir)
        if row.get("library_status") == FORMAL_LIBRARY_STATUS
        and row.get("rag_status") == FORMAL_RAG_STATUS
        and row.get("pdf_valid") is True
    ]
    by_id = {str(row.get("paper_id") or ""): dict(row) for row in manifest_rows}
    # The active registry is append-only and therefore provides the formal
    # acceptance order.  Keep that order stable; only fall back to document_id
    # for records absent from the registry.
    registry = _read_jsonl(base_dir / "data" / "system" / "active_formal_registry.jsonl")
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in registry:
        paper_id = str(record.get("document_id") or record.get("paper_id") or "")
        row = by_id.get(paper_id)
        if row is not None and paper_id not in seen:
            # The Active Formal Registry is authoritative for the currently
            # accepted binary.  A small number of historical Stage-1 rows keep
            # a canonical_historical_sha while the active lock intentionally
            # uses a verified alternate binary.
            row.update(
                {
                    "paper_id": paper_id,
                    "title": record.get("title") or row.get("title") or "",
                    "doi": record.get("doi") or row.get("doi") or "",
                    "canonical_pdf_path": record.get("local_file_path")
                    or row.get("canonical_pdf_path")
                    or "",
                    "file_hash_sha256": record.get("sha256")
                    or row.get("file_hash_sha256")
                    or "",
                }
            )
            ordered.append(row)
            seen.add(paper_id)
    ordered.extend(
        sorted(
            (row for paper_id, row in by_id.items() if paper_id not in seen),
            key=lambda row: str(row.get("paper_id") or ""),
        )
    )
    return ordered


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
        return {
            _replace_paths(key, path_map) if isinstance(key, str) else key: _replace_paths(item, path_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_paths(item, path_map) for item in value]
    if isinstance(value, str):
        exact = path_map.get(value.casefold())
        if exact is not None:
            return exact
        # Some promoted papers were deep-read in an isolated gate workspace,
        # so their runtime provenance retained a different directory with the
        # same old PDF basename.  Once accepted into the single Active Corpus,
        # source-path metadata must resolve to the canonical active PDF.
        if "/" in value or "\\" in value:
            canonical = path_map.get("@basename:" + Path(value).name.casefold())
            if canonical is not None:
                return canonical
        return value
    return value


def _path_replacements(base_dir: Path, plan: list[dict[str, Any]]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    pdf_root = (base_dir / "paper" / "pdfs").resolve()
    for item in plan:
        old_path = Path(item["old_path"]).resolve()
        new_path = Path(item["new_path"]).resolve()
        variants = {
            str(old_path): str(new_path),
            old_path.name: new_path.name,
            (Path("paper") / "pdfs" / old_path.name).as_posix(): (
                Path("paper") / "pdfs" / new_path.name
            ).as_posix(),
            str(Path("paper") / "pdfs" / old_path.name): str(
                Path("paper") / "pdfs" / new_path.name
            ),
        }
        try:
            variants[str(old_path.relative_to(pdf_root))] = str(new_path.relative_to(pdf_root))
        except ValueError:
            pass
        replacements.update({old.casefold(): new for old, new in variants.items()})
        replacements["@basename:" + old_path.name.casefold()] = str(new_path)
    return replacements


def _structured_reference_files(base_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in (
        "data/deep_read",
        "data/tasks",
        "data/rag",
        "data/evidence",
        "data/stage3_5",
        "data/oa",
    ):
        root = base_dir / relative
        if root.exists():
            files.update(root.rglob("*.json"))
            files.update(root.rglob("*.jsonl"))
    # Only active system contracts are mutable.  Historical v1.1 manifests
    # intentionally keep their original filenames for audit reproducibility.
    for name in (
        "active_formal_dataset_manifest.json",
        "active_formal_pdf_lock_manifest.json",
        "active_formal_registry.jsonl",
        "formal_rag_whitelist.json",
        "pdf_watch_queue.json",
        "corpus_statistics.json",
    ):
        path = base_dir / "data" / "system" / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def _write_replaced_jsonl(path: Path, path_map: dict[str, str]) -> bool:
    temporary = path.with_suffix(path.suffix + ".rename_tmp")
    changed = False
    with path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            replaced = _replace_paths(value, path_map)
            changed = changed or replaced != value
            target.write(json.dumps(replaced, ensure_ascii=False, sort_keys=True) + "\n")
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink(missing_ok=True)
    return changed


def _refresh_binary_rag_metadata(base_dir: Path, path_map: dict[str, str]) -> int:
    updates = 0
    cache = base_dir / "data" / "rag" / "search_documents.joblib"
    if cache.is_file():
        documents = joblib.load(cache)
        replaced = _replace_paths(documents, path_map)
        if replaced != documents:
            temporary = cache.with_suffix(cache.suffix + ".rename_tmp")
            joblib.dump(replaced, temporary, compress=0)
            temporary.replace(cache)
            updates += 1

    lookup = base_dir / "data" / "rag" / "search_documents.sqlite3"
    if lookup.is_file():
        temporary = lookup.with_suffix(lookup.suffix + ".rename_tmp")
        temporary.unlink(missing_ok=True)
        shutil.copy2(lookup, temporary)
        changed = 0
        connection = sqlite3.connect(temporary)
        try:
            for doc_id, payload in connection.execute("SELECT doc_id, payload FROM documents"):
                value = json.loads(payload)
                replaced = _replace_paths(value, path_map)
                if replaced != value:
                    connection.execute(
                        "UPDATE documents SET payload = ? WHERE doc_id = ?",
                        (json.dumps(replaced, ensure_ascii=False), doc_id),
                    )
                    changed += 1
            connection.commit()
        finally:
            connection.close()
        if changed:
            temporary.replace(lookup)
            updates += 1
        else:
            temporary.unlink(missing_ok=True)
    return updates


def _update_active_references(base_dir: Path, plan: list[dict[str, Any]]) -> dict[str, int]:
    path_map = _path_replacements(base_dir, plan)
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
    for path in _structured_reference_files(base_dir):
        try:
            if path.suffix == ".jsonl":
                structured_updates += int(_write_replaced_jsonl(path, path_map))
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                replaced = _replace_paths(value, path_map)
                if replaced != value:
                    temp = path.with_suffix(path.suffix + ".rename_tmp")
                    temp.write_text(
                        json.dumps(replaced, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    temp.replace(path)
                    structured_updates += 1
        except (OSError, json.JSONDecodeError):
            continue
    binary_updates = _refresh_binary_rag_metadata(base_dir, path_map)
    return {
        "manifest_updates": manifest_updates,
        "pdf_record_updates": pdf_updates,
        "structured_reference_updates": structured_updates,
        "binary_rag_reference_updates": binary_updates,
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
