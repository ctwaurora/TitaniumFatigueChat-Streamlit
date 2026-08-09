"""Safe local PDF consolidation and bounded canonical-library import."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.auto_oa_pipeline import index_auto_validated_paper
from src.deep_read_pipeline import deep_read_pdf
from src.domain_scope import OUT_OF_SCOPE, classify_literature_scope
from src.metadata_service import fetch_metadata, is_valid_title
from src.stage1_store import (
    BASE_DIR,
    extract_basic_pdf_metadata,
    load_paper_manifest,
    load_pdf_file_records,
    normalize_doi,
    normalize_title,
    register_pdf_bytes,
    sha256_bytes,
    stage1_paths,
    update_paper_domain_scope,
    update_paper_metadata_fields,
    update_paper_library_status,
    update_paper_rag_status,
    validate_pdf_bytes,
)
from src.unified_rag import build_unified_rag, rag_paths
from src.formal_pdf_protection import validate_formal_pdf_locks


ProgressCallback = Callable[[str, Dict[str, Any]], None]
LEGACY_ROOTS = ("early_papers", "followup_papers", "papers")
TARGET_RELATIVE = Path("paper") / "pdfs"
INVALID_VALUES = {"", "nan", "none", "null", "unknown", "untitled"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit(
    callback: Optional[ProgressCallback], phase: str, **payload: Any
) -> None:
    if callback:
        callback(phase, {"phase": phase, **payload})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temp.replace(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _year(value: Any) -> str:
    match = re.search(r"(?:D:)?((?:19|20)\d{2})", str(value or ""))
    return match.group(1) if match else ""


def _metadata_valid(metadata: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    authors = " ".join(str(metadata.get("authors") or "").split())
    if not is_valid_title(metadata.get("title")):
        reasons.append("TITLE_REQUIRED")
    if not authors or authors.lower() in INVALID_VALUES:
        reasons.append("AUTHORS_REQUIRED")
    year = str(
        metadata.get("publication_date") or metadata.get("year") or ""
    ).strip()
    if not _year(year) and year.upper() != "UNKNOWN":
        reasons.append("YEAR_REQUIRED")
    source_url = str(metadata.get("source_url") or "").strip()
    if not normalize_doi(metadata.get("doi") or "") and not source_url.startswith(
        "https://"
    ):
        reasons.append("DOI_OR_TRUSTED_SOURCE_URL_REQUIRED")
    return not reasons, reasons


def _visible_first_page_metadata(content: bytes) -> Dict[str, str]:
    """Conservatively read bibliographic text printed on the first page."""
    try:
        import fitz

        document = fitz.open(stream=content, filetype="pdf")
        text = document[0].get_text("text") if document.page_count else ""
        document.close()
    except Exception:
        return {}
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        return {}

    title_start = 0
    for index, line in enumerate(lines[:20]):
        lower = line.lower()
        boilerplate = (
            lower.startswith(("contents lists", "journal homepage", "guest editorial"))
            or lower in {"international journal of fatigue"}
            or lower.startswith("journal of ")
            or lower.startswith("available online ")
            or "published by elsevier" in lower
            or lower.startswith("vol.")
            or line.startswith("第")
            or bool(re.match(r"^(?:19|20)\d{2}\s*年", line))
            or bool(re.match(r"^[A-Za-z]+\s+(?:19|20)\d{2}$", line))
            or "www." in lower
            or "http://" in lower
            or "https://" in lower
            or "scientific reports" in lower
            or bool(re.match(r"^\d+\s+vol", lower))
            or (line.isascii() and line.isupper() and len(line) < 45)
        )
        if not boilerplate and len(line) >= 8:
            title_start = index
            break
    title_lines: List[str] = []
    author_start: Optional[int] = None
    for index, line in enumerate(lines[title_start : title_start + 8], start=title_start):
        lower = line.lower()
        looks_like_author = bool(
            re.search(
                r"(?:^|,\s*)[A-Z]\.\s*[A-ZÀ-Ž][A-Za-zÀ-ž'’-]+",
                line,
            )
            or "⇑" in line
            or re.search(r"\b(?:et al\.?)\b", lower)
            or line.count(",") >= 2
            or line.count("，") >= 2
        )
        if index > title_start and looks_like_author:
            author_start = index
            break
        if lower.startswith(("abstract", "article history", "keywords")):
            break
        title_lines.append(line)
    title = " ".join(title_lines).strip()[:500]
    if "/4.0/)." in title:
        title = title.split("/4.0/).", 1)[1].strip()
    if author_start is None:
        author_start = title_start + len(title_lines)

    author_lines: List[str] = []
    for line in lines[author_start : author_start + 4]:
        lower = line.lower()
        if lower.startswith(
            (
                "abstract",
                "article history",
                "keywords",
                "received",
                "available online",
            )
        ):
            break
        if re.match(r"^[\uFF08(]\d+\.", line):
            break
        if re.match(r"^[a-z]\s+(?:university|school|department|institute|faculty|direct)", lower):
            break
        if re.search(
            r"\b(?:university|institute|department|school|laboratory|center|centre)\b",
            lower,
        ):
            break
        author_lines.append(line)
    authors = " ".join(author_lines)
    authors = re.sub(r"\s*[a-z](?:,[a-z])?\s*(?:,|⇑|$)", "; ", authors)
    authors = authors.replace("⇑", "")
    authors = " ".join(authors.split()).strip(" ,;")

    year = ""
    for pattern in (
        r"Available online[^\n]*(19|20)\d{2}",
        r"(?:©|\x01|\x02|\x03)\s*((?:19|20)\d{2})",
        r"((?:19|20)\d{2})\s*\u5e74",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year_match = re.search(r"(?:19|20)\d{2}", match.group(0))
            year = year_match.group(0) if year_match else ""
            break
    abstract = ""
    abstract_match = re.search(
        r"(?:a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|abstract)\s*(.+?)"
        r"(?:\n\s*1\.?\s*introduction|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract = " ".join(abstract_match.group(1).split())[:4000]
    return {
        "title": title,
        "authors": authors,
        "publication_date": year,
        "abstract": abstract,
    }


def _metadata_quality(row: Dict[str, Any]) -> tuple[int, int, int, int]:
    title = str(row.get("title") or "").strip()
    lower = title.lower()
    suspicious = int(
        "http" in lower
        or "www." in lower
        or "scientific reports |" in lower
        or lower.startswith("contents lists")
        or "journal homepage" in lower
        or lower.startswith("international journal of fatigue")
        or "/4.0/)." in lower
        or " department of " in lower
        or len(title) > 350
        or title.startswith("\u7b2c")
        or "\u6458 \u8981" in title
    )
    return (
        int(is_valid_title(title)) - suspicious,
        int(bool(str(row.get("authors") or "").strip())),
        int(bool(_year(row.get("publication_date") or row.get("year")))),
        min(len(title), 500),
    )


def _candidate_paths(base_dir: Path) -> List[tuple[str, Path]]:
    rows = [(name, base_dir / name) for name in LEGACY_ROOTS]
    rows.append((TARGET_RELATIVE.as_posix(), base_dir / TARGET_RELATIVE))
    paper_root = base_dir / "paper"
    target = (base_dir / TARGET_RELATIVE).resolve()
    if paper_root.exists():
        for path in paper_root.rglob("*.pdf"):
            resolved = path.resolve()
            if not resolved.is_relative_to(target):
                rows.append(("paper_legacy_outside_pdfs", resolved.parent))
    unique: List[tuple[str, Path]] = []
    seen: set[str] = set()
    for label, path in rows:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append((label, path))
    return unique


def inventory_local_pdfs(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    """Read every PDF without deleting, moving, or modifying a source file."""
    directory_reports: List[Dict[str, Any]] = []
    all_files: Dict[str, Path] = {}
    for label, root in _candidate_paths(base_dir):
        files = sorted(
            (path for path in root.rglob("*.pdf") if path.is_file()),
            key=lambda value: str(value).lower(),
        ) if root.exists() else []
        for path in files:
            all_files.setdefault(str(path.resolve()).lower(), path.resolve())
        directory_reports.append(
            {
                "directory": label,
                "path": str(root.resolve()),
                "pdf_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "files": [
                    {
                        "path": str(path.resolve()),
                        "relative_path": str(path.resolve().relative_to(base_dir.resolve())),
                        "bytes": path.stat().st_size,
                    }
                    for path in files
                ],
            }
        )

    manifest_rows = load_paper_manifest(base_dir)
    identity_rows = [
        *manifest_rows,
        *_read_csv(base_dir / "data" / "literature_database.csv"),
        *_read_csv(base_dir / "data" / "candidate_papers.csv"),
    ]
    metadata_by_doi: Dict[str, Dict[str, Any]] = {}
    for row in identity_rows:
        doi = normalize_doi(row.get("doi") or "")
        if not doi:
            continue
        if (
            doi not in metadata_by_doi
            or _metadata_quality(row) > _metadata_quality(metadata_by_doi[doi])
        ):
            metadata_by_doi[doi] = row

    manifest_by_hash: Dict[str, Dict[str, Any]] = {}
    for row in manifest_rows:
        file_hash = str(row.get("file_hash_sha256") or "")
        if file_hash:
            current = manifest_by_hash.get(file_hash)
            if current is None or (
                bool(row.get("deep_read_complete")),
                bool(row.get("pdf_valid")),
            ) > (
                bool(current.get("deep_read_complete")),
                bool(current.get("pdf_valid")),
            ):
                manifest_by_hash[file_hash] = row
    file_rows: List[Dict[str, Any]] = []
    for path in sorted(all_files.values(), key=lambda value: str(value).lower()):
        try:
            content = path.read_bytes()
            file_hash = sha256_bytes(content)
            validation = validate_pdf_bytes(content)
            extracted = extract_basic_pdf_metadata(content, path.name)
            visible = _visible_first_page_metadata(content)
            existing = manifest_by_hash.get(file_hash) or {}
            extracted_doi = normalize_doi(
                existing.get("doi") or extracted.get("doi") or ""
            )
            authoritative = metadata_by_doi.get(extracted_doi) or {}
            extracted_title = str(extracted.get("title") or "").strip()
            visible_title = str(visible.get("title") or "").strip()
            best_extracted_title = (
                visible_title
                if is_valid_title(visible_title)
                and len(visible_title) > len(extracted_title)
                else extracted_title
            )
            existing_title = str(
                authoritative.get("title") or existing.get("title") or ""
            ).strip()
            visible_better = _metadata_quality(
                {"title": visible_title}
            ) > _metadata_quality({"title": existing_title})
            best_title = (
                visible_title
                if is_valid_title(visible_title)
                and (
                    visible_better
                    or (
                        len(visible_title) > len(existing_title)
                        and normalize_title(visible_title).startswith(
                            normalize_title(existing_title)
                        )
                    )
                    or not is_valid_title(existing_title)
                )
                else existing_title or best_extracted_title
            )
            existing_authors = str(
                authoritative.get("authors")
                or existing.get("authors")
                or ""
            ).strip()
            authors_suspicious = (
                existing_authors.lower()
                in {"admin", "unknown", "untitled", "none", "nan"}
                or len(existing_authors) > 350
                or (
                    is_valid_title(best_title)
                    and normalize_title(existing_authors).startswith(
                        normalize_title(best_title)[:50]
                    )
                )
            )
            metadata = {
                "title": best_title,
                "authors": str(
                    (
                        ""
                        if authors_suspicious
                        else existing_authors
                    )
                    or visible.get("authors")
                    or extracted.get("authors")
                    or ""
                ).strip(),
                "publication_date": str(
                    authoritative.get("publication_date")
                    or authoritative.get("year")
                    or existing.get("publication_date")
                    or visible.get("publication_date")
                    or extracted.get("publication_date")
                    or ""
                ),
                "doi": extracted_doi,
                "source_url": str(
                    authoritative.get("source_url")
                    or authoritative.get("url")
                    or existing.get("source_url")
                    or ""
                ).strip(),
                "metadata_source": str(
                    authoritative.get("metadata_source")
                    or existing.get("metadata_source")
                    or ""
                ).strip(),
                "abstract": str(visible.get("abstract") or "").strip(),
            }
            scope = classify_literature_scope(
                {
                    **metadata,
                }
            )
            stored_scope = str(existing.get("domain_scope") or "")
            metadata_valid, metadata_errors = _metadata_valid(metadata)
            file_rows.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.resolve().relative_to(base_dir.resolve())),
                    "bytes": len(content),
                    "sha256": file_hash,
                    "pdf_valid": bool(validation.get("pdf_valid")),
                    "real_page_count": int(validation.get("real_page_count") or 0),
                    "pdf_error": str(validation.get("error") or ""),
                    **metadata,
                    **scope,
                    "stored_domain_scope": stored_scope,
                    "metadata_valid": metadata_valid,
                    "metadata_errors": metadata_errors,
                    "existing_paper_id": str(existing.get("paper_id") or ""),
                }
            )
        except Exception as exc:
            file_rows.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.resolve().relative_to(base_dir.resolve())),
                    "bytes": path.stat().st_size,
                    "sha256": "",
                    "pdf_valid": False,
                    "real_page_count": 0,
                    "pdf_error": f"{type(exc).__name__}:{exc}",
                    "title": "",
                    "authors": "",
                    "publication_date": "",
                    "doi": "",
                    "domain_scope": OUT_OF_SCOPE,
                    "scope_reason": "PDF_INVENTORY_FAILED",
                    "metadata_valid": False,
                    "metadata_errors": ["PDF_INVENTORY_FAILED"],
                    "existing_paper_id": "",
                }
            )

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        if row["sha256"]:
            by_hash[row["sha256"]].append(row)
    unique_rows = [
        sorted(
            rows,
            key=lambda row: (
                not row["relative_path"].replace("\\", "/").startswith(
                    f"{TARGET_RELATIVE.as_posix()}/"
                ),
                row["relative_path"].lower(),
            ),
        )[0]
        for rows in by_hash.values()
    ]
    doi_groups = Counter(
        normalize_doi(row.get("doi") or "")
        for row in unique_rows
        if normalize_doi(row.get("doi") or "")
    )
    title_groups = Counter(
        normalize_title(row.get("title") or "")
        for row in unique_rows
        if is_valid_title(row.get("title"))
    )
    return {
        "generated_at": _now(),
        "base_dir": str(base_dir.resolve()),
        "target_dir": str((base_dir / TARGET_RELATIVE).resolve()),
        "directories": directory_reports,
        "files": file_rows,
        "original_pdf_count": len(file_rows),
        "unique_pdf_count": len(by_hash),
        "duplicate_pdf_count": sum(
            max(0, len(rows) - 1) for rows in by_hash.values()
        ),
        "exact_duplicate_groups": {
            key: [row["path"] for row in rows]
            for key, rows in by_hash.items()
            if len(rows) > 1
        },
        "duplicate_doi_count": sum(value - 1 for value in doi_groups.values() if value > 1),
        "duplicate_title_count": sum(
            value - 1 for value in title_groups.values() if value > 1
        ),
        "failures": [
            row for row in file_rows if not row.get("pdf_valid")
        ],
    }


def local_pdf_directory_summary(
    base_dir: Path = BASE_DIR,
) -> List[Dict[str, Any]]:
    """Fast pre-scan counts for the web page; no hashing or PDF parsing."""
    rows: List[Dict[str, Any]] = []
    for label, root in _candidate_paths(base_dir):
        files = list(root.rglob("*.pdf")) if root.exists() else []
        rows.append(
            {
                "directory": label,
                "pdf_count": len(files),
                "total_bytes": sum(
                    path.stat().st_size for path in files if path.is_file()
                ),
            }
        )
    return rows


def _safe_target_path(target_dir: Path, file_hash: str) -> Path:
    preferred = target_dir / f"PDF_{file_hash[:20].upper()}.pdf"
    if not preferred.exists():
        return preferred
    try:
        if sha256_bytes(preferred.read_bytes()) == file_hash:
            return preferred
    except OSError:
        pass
    for index in range(1, 1000):
        candidate = target_dir / (
            f"PDF_{file_hash[:20].upper()}_{index:03d}.pdf"
        )
        if not candidate.exists():
            return candidate
        try:
            if sha256_bytes(candidate.read_bytes()) == file_hash:
                return candidate
        except OSError:
            continue
    raise RuntimeError(f"NO_SAFE_TARGET_NAME:{file_hash}")


def _replace_strings(value: Any, path_map: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, path_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_strings(item, path_map) for item in value]
    if isinstance(value, str):
        return path_map.get(value.lower(), value)
    return value


def _reconcile_references(
    base_dir: Path,
    hash_targets: Dict[str, Path],
    inventory: Dict[str, Any],
) -> Dict[str, int]:
    paths = stage1_paths(base_dir)
    path_hash = {
        str(Path(row["path"]).resolve()).lower(): row["sha256"]
        for row in inventory["files"]
        if row.get("sha256")
    }
    path_map = {
        old_path: str(hash_targets[file_hash].resolve())
        for old_path, file_hash in path_hash.items()
        if file_hash in hash_targets
    }
    manifest = load_paper_manifest(base_dir)
    manifest_updates = 0
    for row in manifest:
        file_hash = str(row.get("file_hash_sha256") or "")
        target = hash_targets.get(file_hash)
        if target:
            target_text = str(target.resolve())
            if row.get("canonical_pdf_path") != target_text:
                row["canonical_pdf_path"] = target_text
                manifest_updates += 1
            linked = [
                path_map.get(str(Path(value).resolve()).lower(), value)
                for value in (row.get("linked_versions") or [])
                if value
            ]
            row["linked_versions"] = list(
                dict.fromkeys([target_text, *linked])
            )
    _write_jsonl(paths["paper_manifest"], manifest)

    pdf_rows = load_pdf_file_records(base_dir)
    pdf_updates = 0
    deduplicated_pdf_rows: Dict[str, Dict[str, Any]] = {}
    for row in pdf_rows:
        file_hash = str(row.get("file_hash_sha256") or "")
        target = hash_targets.get(file_hash)
        if target and row.get("canonical_pdf_path") != str(target.resolve()):
            row["canonical_pdf_path"] = str(target.resolve())
            pdf_updates += 1
        deduplicated_pdf_rows.setdefault(file_hash or str(row), row)
    _write_jsonl(paths["pdf_files"], list(deduplicated_pdf_rows.values()))

    deep_updates = 0
    deep_root = base_dir / "data" / "deep_read"
    for path in (
        list(deep_root.rglob("*.json"))
        + list(deep_root.rglob("*.jsonl"))
        if deep_root.exists()
        else []
    ):
        try:
            if path.suffix == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                replaced = _replace_strings(rows, path_map)
                if replaced != rows:
                    _write_jsonl(path, replaced)
                    deep_updates += 1
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                replaced = _replace_strings(value, path_map)
                if replaced != value:
                    _write_json(path, replaced)
                    deep_updates += 1
        except (OSError, json.JSONDecodeError):
            continue

    csv_updates = 0
    for path in (base_dir / "data").glob("*.csv"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        replaced = text
        for old_path, target in path_map.items():
            # CSV path references are not JSON escaped.
            for row in inventory["files"]:
                if str(Path(row["path"]).resolve()).lower() == old_path:
                    replaced = replaced.replace(row["path"], target)
        if replaced != text:
            path.write_text(replaced, encoding="utf-8-sig")
            csv_updates += 1
    return {
        "manifest_path_updates": manifest_updates,
        "pdf_record_path_updates": pdf_updates,
        "deep_read_file_updates": deep_updates,
        "csv_reference_updates": csv_updates,
        "path_updates": (
            manifest_updates + pdf_updates + deep_updates + csv_updates
        ),
    }


def validate_local_pdf_links(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    manifest = load_paper_manifest(base_dir)
    paper_ids = {
        str(row.get("paper_id") or "") for row in manifest if row.get("paper_id")
    }
    inaccessible = [
        {
            "paper_id": row.get("paper_id"),
            "path": row.get("canonical_pdf_path"),
        }
        for row in manifest
        if row.get("pdf_valid")
        and (
            not row.get("canonical_pdf_path")
            or not Path(str(row["canonical_pdf_path"])).is_file()
        )
    ]
    deep_orphans: List[str] = []
    deep_root = base_dir / "data" / "deep_read"
    if deep_root.exists():
        deep_orphans = [
            path.name
            for path in deep_root.iterdir()
            if path.is_dir() and path.name not in paper_ids
        ]
    evidence_orphans: List[str] = []
    for row in _read_csv(
        base_dir / "data" / "evidence" / "trusted_evidence.csv"
    ):
        paper_id = str(row.get("paper_id") or "")
        if paper_id and paper_id not in paper_ids:
            evidence_orphans.append(paper_id)
    rag_manifest_path = base_dir / "data" / "rag" / "manifest.json"
    rag_manifest = {}
    try:
        rag_manifest = json.loads(rag_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    rag_orphans = [
        str(value)
        for value in (rag_manifest.get("paper_ids") or [])
        if str(value) not in paper_ids
    ]
    passed = not (
        inaccessible or deep_orphans or evidence_orphans or rag_orphans
    )
    return {
        "passed": passed,
        "canonical_record_count": len(manifest),
        "inaccessible_pdf_records": inaccessible,
        "deep_read_orphans": sorted(set(deep_orphans)),
        "evidence_orphans": sorted(set(evidence_orphans)),
        "rag_orphans": sorted(set(rag_orphans)),
        "rag_indexed_count": len(rag_manifest.get("paper_ids") or []),
    }


def scan_and_import_local_pdfs(
    *,
    base_dir: Path = BASE_DIR,
    max_files: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    report_name: str = "pdf_storage_unification_report.json",
) -> Dict[str, Any]:
    """Copy unique bytes, preserve sources, and run the existing Stage-2/3 path."""
    validate_formal_pdf_locks(base_dir)
    inventory = inventory_local_pdfs(base_dir)
    target_dir = base_dir / TARGET_RELATIVE
    target_dir.mkdir(parents=True, exist_ok=True)
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in inventory["files"]:
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row)
    ordered_hashes = sorted(by_hash)
    if max_files is not None:
        ordered_hashes = ordered_hashes[: max(0, int(max_files))]

    hash_targets: Dict[str, Path] = {}
    copied = 0
    results: List[Dict[str, Any]] = []
    before_manifest_ids = {
        str(row.get("paper_id") or "")
        for row in load_paper_manifest(base_dir)
    }
    for index, file_hash in enumerate(ordered_hashes, start=1):
        rows = by_hash[file_hash]
        source = next(
            (
                Path(row["path"])
                for row in rows
                if row["relative_path"].replace("\\", "/").startswith(
                    f"{TARGET_RELATIVE.as_posix()}/"
                )
            ),
            Path(rows[0]["path"]),
        )
        target = _safe_target_path(target_dir, file_hash)
        hash_targets[file_hash] = target
        _emit(
            progress_callback,
            "SCANNING",
            index=index,
            total=len(ordered_hashes),
            path=source.name,
        )
        if not target.exists():
            shutil.copy2(source, target)
            copied += 1
        representative = next(
            (row for row in rows if row.get("existing_paper_id")),
            rows[0],
        )
        representative = dict(representative)
        pre_metadata = {
            "title": representative.get("title") or "",
            "authors": representative.get("authors") or "",
            "publication_date": representative.get("publication_date") or "",
            "year": _year(representative.get("publication_date")) or "UNKNOWN",
            "doi": representative.get("doi") or "",
            "source_url": representative.get("source_url") or "",
        }
        pre_valid, _ = _metadata_valid(pre_metadata)
        if not pre_valid and normalize_doi(pre_metadata["doi"]):
            try:
                enriched = fetch_metadata(pre_metadata["doi"], timeout=15)
            except Exception as exc:
                enriched = {
                    "validation_status": "INVALID_METADATA",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            if enriched.get("validation_status") == "VALID":
                representative.update(
                    {
                        "title": enriched.get("title")
                        or representative.get("title")
                        or "",
                        "authors": enriched.get("authors")
                        or representative.get("authors")
                        or "",
                        "publication_date": enriched.get("year")
                        or representative.get("publication_date")
                        or "UNKNOWN",
                        "doi": normalize_doi(
                            enriched.get("doi")
                            or representative.get("doi")
                            or ""
                        ),
                        "source_url": enriched.get("source_url") or "",
                        "metadata_source": enriched.get("metadata_source")
                        or "",
                    }
                )
        detail: Dict[str, Any] = {
            "sha256": file_hash,
            "source_files": [row["path"] for row in rows],
            "target_file": str(target.resolve()),
            "copied": str(source.resolve()) != str(target.resolve())
            and target.exists(),
            "title": representative.get("title") or "",
            "authors": representative.get("authors") or "",
            "year": _year(representative.get("publication_date")),
            "doi": representative.get("doi") or "",
            "metadata_source": representative.get("metadata_source") or "",
            "real_page_count": representative.get("real_page_count") or 0,
            "domain_scope": representative.get("domain_scope"),
            "status": "PENDING",
            "failure_reason": "",
        }
        if not representative.get("pdf_valid"):
            detail["status"] = "QUARANTINED"
            detail["failure_reason"] = representative.get("pdf_error") or "INVALID_PDF"
            results.append(detail)
            continue
        content = target.read_bytes()
        registration = register_pdf_bytes(
            content,
            target.name,
            source_path=str(source.resolve()),
            source_type="LOCAL_PDF_SCAN",
            metadata_override={
                "title": str(representative.get("title") or ""),
                "authors": str(representative.get("authors") or ""),
                "publication_date": str(
                    representative.get("publication_date") or ""
                ),
                "doi": str(representative.get("doi") or ""),
            },
            base_dir=base_dir,
        )
        paper_id = str(registration.get("paper_id") or "")
        detail["paper_id"] = paper_id
        if paper_id:
            update_paper_metadata_fields(
                paper_id,
                title=str(representative.get("title") or ""),
                authors=str(representative.get("authors") or ""),
                publication_date=(
                    _year(representative.get("publication_date"))
                    or str(representative.get("publication_date") or "")
                ),
                doi=str(representative.get("doi") or ""),
                metadata_source=str(
                    representative.get("metadata_source")
                    or "LOCAL_PDF_VISIBLE_METADATA"
                ),
                source_url=str(representative.get("source_url") or ""),
                base_dir=base_dir,
            )
        scope = classify_literature_scope(
            {
                "title": registration.get("title") or detail["title"],
                "authors": registration.get("authors") or detail["authors"],
                "abstract": representative.get("abstract") or "",
            }
        )
        detail.update(scope)
        if paper_id:
            update_paper_domain_scope(
                paper_id,
                scope["domain_scope"],
                scope_reason=scope["scope_reason"],
                base_dir=base_dir,
            )
        current = next(
            (
                row
                for row in load_paper_manifest(base_dir)
                if str(row.get("paper_id") or "") == paper_id
            ),
            {},
        )
        if (
            scope["domain_scope"] != OUT_OF_SCOPE
            and current.get("deep_read_complete")
            and current.get("evidence_status")
            in {"AUTO_VALIDATED", "HUMAN_CONFIRMED"}
            and current.get("rag_status") == "INDEXED_STAGE3_UNIFIED"
        ):
            update_paper_library_status(
                paper_id, "FORMAL", base_dir=base_dir
            )
            detail.update(
                {
                    "status": "EXISTING_FORMAL",
                    "deep_read_complete": True,
                    "evidence_status": current.get("evidence_status"),
                    "rag_status": current.get("rag_status"),
                }
            )
            results.append(detail)
            continue
        metadata = {
            "title": registration.get("title") or detail["title"],
            "authors": registration.get("authors") or detail["authors"],
            "publication_date": (
                registration.get("publication_date") or detail["year"]
            ),
            "doi": registration.get("doi") or detail["doi"],
            "source_url": representative.get("source_url") or "",
        }
        metadata_valid, metadata_errors = _metadata_valid(metadata)
        if not metadata_valid or scope["domain_scope"] == OUT_OF_SCOPE:
            reasons = [
                *metadata_errors,
                *(
                    ["OUT_OF_TITANIUM_FATIGUE_SCOPE"]
                    if scope["domain_scope"] == OUT_OF_SCOPE
                    else []
                ),
            ]
            detail["status"] = "QUARANTINED"
            detail["failure_reason"] = ";".join(reasons)
            if paper_id:
                update_paper_library_status(
                    paper_id,
                    "QUARANTINED",
                    quarantine_reason=detail["failure_reason"],
                    base_dir=base_dir,
                )
                update_paper_rag_status(
                    paper_id, "NOT_INDEXED", base_dir
                )
            results.append(detail)
            continue

        _emit(progress_callback, "DEEP_READING", paper_id=paper_id)
        deep = deep_read_pdf(
            target,
            paper_id=paper_id,
            title=str(metadata["title"]),
            base_dir=base_dir,
            force=False,
        )
        detail["deep_read_complete"] = bool(deep.get("deep_read_complete"))
        detail["processed_page_count"] = int(
            deep.get("processed_page_count")
            or deep.get("page_record_count")
            or 0
        )
        detail["evidence_count"] = int(deep.get("evidence_count") or 0)
        if not deep.get("deep_read_complete"):
            detail["status"] = "QUARANTINED"
            detail["failure_reason"] = str(
                deep.get("error") or "DEEP_READ_INCOMPLETE"
            )
            update_paper_library_status(
                paper_id,
                "QUARANTINED",
                quarantine_reason=detail["failure_reason"],
                base_dir=base_dir,
            )
            results.append(detail)
            continue
        _emit(progress_callback, "QUALITY_GATING", paper_id=paper_id)
        indexed = index_auto_validated_paper(paper_id, base_dir=base_dir)
        gate = indexed.get("gate") or {}
        detail["quality_gate"] = "PASSED" if gate.get("passed") else "FAILED"
        detail["evidence_status"] = gate.get("evidence_status")
        detail["rag_status"] = indexed.get("status")
        if indexed.get("status") == "INDEXED_STAGE3_UNIFIED":
            detail["status"] = "FORMAL"
            update_paper_library_status(
                paper_id, "FORMAL", base_dir=base_dir
            )
        else:
            detail["status"] = "QUARANTINED"
            detail["failure_reason"] = ";".join(gate.get("reasons") or [])
            update_paper_library_status(
                paper_id,
                "QUARANTINED",
                quarantine_reason=detail["failure_reason"],
                base_dir=base_dir,
            )
        results.append(detail)

    # If a previously indexed record is now verified as OUT_OF_SCOPE, rebuild
    # the existing corpus without it.  This uses the normal Stage-3 builder.
    rag_manifest_path = rag_paths(base_dir)["manifest"]
    try:
        rag_manifest = json.loads(rag_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        rag_manifest = {}
    manifest_scope = {
        str(row.get("paper_id") or ""): str(row.get("domain_scope") or "")
        for row in load_paper_manifest(base_dir)
    }
    current_rag_ids = [
        str(value)
        for value in (rag_manifest.get("paper_ids") or [])
        if value
    ]
    manifest_status = {
        str(row.get("paper_id") or ""): str(row.get("library_status") or "")
        for row in load_paper_manifest(base_dir)
    }
    if any(
        manifest_scope.get(paper_id) == OUT_OF_SCOPE
        or manifest_status.get(paper_id) != "FORMAL"
        for paper_id in current_rag_ids
    ):
        build_unified_rag(
            [
                paper_id
                for paper_id in current_rag_ids
                if manifest_scope.get(paper_id) != OUT_OF_SCOPE
                and manifest_status.get(paper_id) == "FORMAL"
            ],
            base_dir=base_dir,
        )

    # Include already-present target hashes outside a bounded smoke batch when
    # reconciling only if the full run was requested.
    if max_files is None:
        for file_hash, rows in by_hash.items():
            if file_hash in hash_targets:
                continue
            target = _safe_target_path(target_dir, file_hash)
            if target.exists():
                hash_targets[file_hash] = target

    reference_updates = _reconcile_references(
        base_dir, hash_targets, inventory
    )
    validation = validate_local_pdf_links(base_dir)
    after_manifest = load_paper_manifest(base_dir)
    after_manifest_ids = {
        str(row.get("paper_id") or "") for row in after_manifest
    }
    report = {
        **inventory,
        "mode": "SMOKE_3" if max_files == 3 else "FULL",
        "processed_unique_pdf_count": len(ordered_hashes),
        "successful_copy_count": copied,
        "new_canonical_record_count": len(
            after_manifest_ids - before_manifest_ids
        ),
        "new_candidate_count": sum(
            row.get("status") == "CANDIDATE" for row in results
        ),
        "new_formal_count": sum(
            row.get("status") == "FORMAL" for row in results
        ),
        "deep_read_success_count": sum(
            bool(row.get("deep_read_complete")) for row in results
        ),
        "indexed_count": sum(
            row.get("rag_status") == "INDEXED_STAGE3_UNIFIED"
            for row in results
        ),
        "failure_count": sum(
            row.get("status") == "QUARANTINED" for row in results
        ),
        "processing_results": results,
        **reference_updates,
        "validation": validation,
        "legacy_sources_retained": [
            str((base_dir / name).resolve()) for name in LEGACY_ROOTS
        ],
        "legacy_marked": False,
        "completed_at": _now(),
    }
    if validation["passed"] and max_files is None:
        for name in LEGACY_ROOTS:
            root = base_dir / name
            if root.exists():
                marker = root / "LEGACY_PDF_SOURCE_README.txt"
                marker.write_text(
                    "Legacy PDF source retained after SHA-256 consolidation. "
                    "Do not delete until the user reviews the migration report.\n",
                    encoding="utf-8",
                )
        report["legacy_marked"] = True
    report_path = base_dir / "data" / "migrations" / report_name
    _write_json(report_path, report)
    report["report_path"] = str(report_path.resolve())
    return report
