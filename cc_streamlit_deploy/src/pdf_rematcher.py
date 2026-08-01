"""Deterministic local-PDF rematching for canonical no-PDF records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz

from src.stage1_store import normalize_doi, normalize_title


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf(path: Path) -> dict[str, Any]:
    result = {
        "header_valid": False,
        "pymupdf_open_valid": False,
        "real_page_count": 0,
        "error": "",
    }
    try:
        result["header_valid"] = path.read_bytes()[:5] == b"%PDF-"
        if not result["header_valid"]:
            result["error"] = "INVALID_PDF_HEADER"
            return result
        with fitz.open(path) as document:
            result["real_page_count"] = len(document)
            result["pymupdf_open_valid"] = len(document) > 0
    except Exception as exc:
        result["error"] = f"PYMUPDF_OPEN_FAILED:{type(exc).__name__}:{exc}"
    return result


def _first_page_features(path: Path) -> dict[str, Any]:
    try:
        with fitz.open(path) as document:
            text = document[0].get_text("text") if len(document) else ""
    except Exception:
        return {"text": "", "title_candidates": [], "doi_candidates": []}
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    candidates: list[str] = []
    for index in range(min(24, len(lines))):
        for width in (1, 2, 3):
            candidate = " ".join(lines[index : index + width]).strip()
            if 15 <= len(candidate) <= 350:
                candidates.append(candidate)
    dois = [
        normalize_doi(value)
        for value in re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    ]
    return {
        "text": text,
        "title_candidates": list(dict.fromkeys(candidates)),
        "doi_candidates": list(dict.fromkeys(value for value in dois if value)),
    }


def build_local_asset_index(base_dir: Path) -> list[dict[str, Any]]:
    inventory = _read_json(
        base_dir / "data/tasks/full_library_deep_read/inventory.json", {}
    )
    pdf_rows = _read_jsonl(base_dir / "data/pdf_files.jsonl")
    manifest = _read_jsonl(base_dir / "data/paper_manifest.jsonl")
    pdf_by_hash = {
        str(row.get("file_hash_sha256") or ""): row
        for row in pdf_rows
        if row.get("file_hash_sha256")
    }
    manifest_by_hash = defaultdict(list)
    for row in manifest:
        if row.get("file_hash_sha256"):
            manifest_by_hash[str(row["file_hash_sha256"])].append(row)

    assets: list[dict[str, Any]] = []
    for row in inventory.get("files") or []:
        path = Path(str(row.get("local_path") or ""))
        if not path.is_file():
            continue
        file_hash = str(row.get("sha256") or "") or _sha256(path)
        pdf_record = pdf_by_hash.get(file_hash) or {}
        linked = manifest_by_hash.get(file_hash) or []
        features = _first_page_features(path)
        assets.append(
            {
                **row,
                "path": str(path.resolve()),
                "sha256": file_hash,
                "pdf_asset_id": str(pdf_record.get("pdf_file_id") or ""),
                "linked_paper_ids": [
                    str(value.get("paper_id") or "") for value in linked
                ],
                "first_page_text": features["text"],
                "first_page_title_candidates": features["title_candidates"],
                "first_page_normalized_title_candidates": [
                    normalize_title(value) for value in features["title_candidates"]
                ],
                "first_page_token_set": set(
                    normalize_title(features["text"]).split()
                ),
                "first_page_doi_candidates": features["doi_candidates"],
            }
        )
    return assets


def _author_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_title(value).split()
        if len(token) >= 3 and token not in {"and", "the"}
    }


def _year(value: Any) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def match_record_to_local_asset(
    record: dict[str, Any], assets: list[dict[str, Any]]
) -> dict[str, Any]:
    saved_asset_id = str(record.get("pdf_asset_id") or "")
    saved_hash = str(record.get("file_hash_sha256") or "").casefold()
    doi = normalize_doi(record.get("doi") or "")
    title = normalize_title(record.get("title") or "")
    authors = _author_tokens(str(record.get("authors") or ""))
    year = _year(record.get("publication_date") or record.get("year"))

    def result(asset: dict[str, Any], basis: str, confidence: float) -> dict[str, Any]:
        validation = validate_pdf(Path(asset["path"]))
        return {
            "found": bool(
                validation["header_valid"] and validation["pymupdf_open_valid"]
            ),
            "match_basis": basis,
            "match_confidence": confidence,
            "asset": asset,
            "validation": validation,
        }

    if saved_asset_id:
        candidates = [a for a in assets if a.get("pdf_asset_id") == saved_asset_id]
        if len(candidates) == 1:
            return result(candidates[0], "SAVED_PDF_ASSET_ID", 1.0)
    if saved_hash:
        candidates = [a for a in assets if str(a.get("sha256") or "").casefold() == saved_hash]
        if len(candidates) == 1:
            return result(candidates[0], "SHA256_EXACT", 1.0)
    if doi:
        candidates = [
            a
            for a in assets
            if normalize_doi(a.get("doi") or "") == doi
            or doi in set(a.get("first_page_doi_candidates") or [])
        ]
        unique = {str(a["sha256"]): a for a in candidates}
        if len(unique) == 1:
            return result(next(iter(unique.values())), "DOI_EXACT", 0.99)
    if title:
        candidates = [
            a for a in assets if normalize_title(a.get("title") or "") == title
        ]
        unique = {str(a["sha256"]): a for a in candidates}
        if len(unique) == 1:
            return result(next(iter(unique.values())), "NORMALIZED_TITLE_EXACT", 0.97)

    scored: list[tuple[float, dict[str, Any], str]] = []
    if title and authors and year:
        for asset in assets:
            asset_title = normalize_title(asset.get("title") or "")
            similarity = SequenceMatcher(None, title, asset_title).ratio()
            asset_authors = _author_tokens(str(asset.get("authors") or ""))
            asset_year = _year(asset.get("year"))
            if (
                similarity >= 0.88
                and authors & asset_authors
                and (not asset_year or asset_year == year)
            ):
                scored.append((similarity, asset, "AUTHOR_YEAR_TITLE_SIMILARITY"))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) == 1 or scored[0][0] > scored[1][0] + 0.03:
            return result(scored[0][1], scored[0][2], round(scored[0][0], 4))

    first_page_scored: list[tuple[float, dict[str, Any]]] = []
    if title:
        title_tokens = set(title.split())
        for asset in assets:
            page_tokens = set(asset.get("first_page_token_set") or set())
            if not title_tokens or (
                len(title_tokens & page_tokens) / len(title_tokens) < 0.60
            ):
                continue
            best = max(
                (
                    SequenceMatcher(None, title, candidate).ratio()
                    for candidate in asset.get(
                        "first_page_normalized_title_candidates"
                    )
                    or []
                    if len(title_tokens & set(candidate.split())) / len(title_tokens)
                    >= 0.60
                ),
                default=0.0,
            )
            if best >= 0.90:
                first_page_scored.append((best, asset))
    first_page_scored.sort(key=lambda item: item[0], reverse=True)
    if first_page_scored and (
        len(first_page_scored) == 1
        or first_page_scored[0][0] > first_page_scored[1][0] + 0.03
    ):
        return result(
            first_page_scored[0][1],
            "PDF_FIRST_PAGE_TITLE_AUTHOR",
            round(first_page_scored[0][0], 4),
        )
    return {
        "found": False,
        "match_basis": "NO_RELIABLE_LOCAL_MATCH",
        "match_confidence": 0.0,
        "asset": None,
        "validation": {},
        "reason": (
            "No saved asset id, SHA-256, DOI, normalized title, "
            "author/year/title or first-page match met the reliability gate"
        ),
    }


def audit_pdf_not_acquired(base_dir: Path) -> list[dict[str, Any]]:
    manifest = _read_jsonl(base_dir / "data/paper_manifest.jsonl")
    assets = build_local_asset_index(base_dir)
    output: list[dict[str, Any]] = []
    for record in manifest:
        if record.get("pdf_valid"):
            continue
        match = match_record_to_local_asset(record, assets)
        asset = match.get("asset") or {}
        linked_ids = [value for value in asset.get("linked_paper_ids") or [] if value]
        can_repair = bool(match.get("found")) and (
            not linked_ids or str(record.get("paper_id") or "") in linked_ids
        )
        output.append(
            {
                "canonical_paper_id": str(record.get("paper_id") or ""),
                "title": str(record.get("title") or ""),
                "doi": normalize_doi(record.get("doi") or ""),
                "local_pdf_found": bool(match.get("found")),
                "match_basis": match.get("match_basis"),
                "match_confidence": float(match.get("match_confidence") or 0),
                "matched_pdf_path": str(asset.get("path") or ""),
                "matched_pdf_sha256": str(asset.get("sha256") or ""),
                "matched_pdf_asset_id": str(asset.get("pdf_asset_id") or ""),
                "matched_existing_paper_ids": linked_ids,
                "pdf_header_valid": bool(
                    (match.get("validation") or {}).get("header_valid")
                ),
                "pymupdf_open_valid": bool(
                    (match.get("validation") or {}).get("pymupdf_open_valid")
                ),
                "real_page_count": int(
                    (match.get("validation") or {}).get("real_page_count") or 0
                ),
                "repair_completed": False,
                "repairable_without_canonical_merge": can_repair,
                "reason": str(
                    match.get("reason")
                    or (
                        "MATCHED_ASSET_ALREADY_LINKED_TO_ANOTHER_CANONICAL"
                        if match.get("found") and not can_repair
                        else "MATCH_FOUND_AWAITING_APPLY"
                    )
                ),
                "next_action": (
                    "LEGAL_OA_LOOKUP_OR_MANUAL_UPLOAD"
                    if not match.get("found")
                    else "SAFE_LOCAL_LINK"
                    if can_repair
                    else "CANONICAL_MERGE_REQUIRED"
                ),
            }
        )
    return output
