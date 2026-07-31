"""Canonical Stage-1 storage, PDF validation, and deduplication services.

Canonical roots:
  paper/pdfs/                 immutable PDF copies for all new imports
  data/paper_manifest.jsonl   canonical paper records
  data/pdf_files.jsonl        physical PDF/version records
  data/duplicate_records.jsonl
  data/rag/manifest.json      Stage-3 scientific RAG source-of-truth manifest
  data/rag/paper_index.json   LEGACY compatibility availability index
  data/rag/chunks/*.json      LEGACY compatibility text chunks
  data/evidence/trusted_evidence.csv
  data/evidence/quarantined_evidence.csv

Legacy stores remain read-only migration sources.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from dataclasses import fields
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data_contracts import (
    DATA_SCHEMA_VERSION,
    DIRECTNESS_VALUES,
    DuplicateRecord,
    EvidenceRecord,
    PaperRecord,
    PdfFileRecord,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CANONICAL_PDF_DIR = BASE_DIR / "paper" / "pdfs"
PAPER_MANIFEST_PATH = DATA_DIR / "paper_manifest.jsonl"
PDF_FILES_PATH = DATA_DIR / "pdf_files.jsonl"
DUPLICATE_RECORDS_PATH = DATA_DIR / "duplicate_records.jsonl"
RAG_ROOT = DATA_DIR / "rag"
# Stage-3 scientific retrieval must use RAG_MANIFEST_PATH through
# src.unified_rag.  The two legacy constants remain only for old upload/library
# code and are explicitly excluded by the Stage-3 manifest.
RAG_MANIFEST_PATH = RAG_ROOT / "manifest.json"
RAG_CHUNKS_DIR = RAG_ROOT / "chunks"
RAG_INDEX_PATH = RAG_ROOT / "paper_index.json"
EVIDENCE_ROOT = DATA_DIR / "evidence"
TRUSTED_EVIDENCE_PATH = EVIDENCE_ROOT / "trusted_evidence.csv"
QUARANTINED_EVIDENCE_PATH = EVIDENCE_ROOT / "quarantined_evidence.csv"
MANUAL_REVIEW_DIR = DATA_DIR / "manual_review"

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_stage1_dirs(base_dir: Path = BASE_DIR) -> Dict[str, Path]:
    paths = stage1_paths(base_dir)
    for key in (
        "canonical_pdf_dir",
        "rag_root",
        "rag_chunks_dir",
        "evidence_root",
        "manual_review_dir",
        "deep_read_root",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def stage1_paths(base_dir: Path = BASE_DIR) -> Dict[str, Path]:
    data_dir = base_dir / "data"
    rag_root = data_dir / "rag"
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "canonical_pdf_dir": base_dir / "paper" / "pdfs",
        "paper_manifest": data_dir / "paper_manifest.jsonl",
        "pdf_files": data_dir / "pdf_files.jsonl",
        "duplicate_records": data_dir / "duplicate_records.jsonl",
        "rag_root": rag_root,
        "rag_manifest": rag_root / "manifest.json",
        "rag_chunks_dir": rag_root / "chunks",
        "rag_index": rag_root / "paper_index.json",
        "evidence_root": data_dir / "evidence",
        "trusted_evidence": data_dir / "evidence" / "trusted_evidence.csv",
        "quarantined_evidence": data_dir / "evidence" / "quarantined_evidence.csv",
        "manual_review_dir": data_dir / "manual_review",
        "deep_read_root": data_dir / "deep_read",
    }


def normalize_doi(value: str) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(".,; ")


def normalize_title(value: str) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def title_similarity(a: str, b: str) -> float:
    left, right = normalize_title(a), normalize_title(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf_bytes(content: bytes) -> Dict[str, Any]:
    result = {
        "pdf_valid": False,
        "real_page_count": 0,
        "file_size": len(content),
        "error": "",
        "metadata": {},
    }
    if not content.startswith(b"%PDF"):
        result["error"] = "INVALID_PDF_HEADER"
        return result
    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            if doc.page_count < 1:
                result["error"] = "PDF_HAS_NO_PAGES"
                return result
            result["pdf_valid"] = True
            result["real_page_count"] = int(doc.page_count)
            result["metadata"] = dict(doc.metadata or {})
    except Exception as exc:
        result["error"] = f"PDF_PARSE_FAILED: {exc}"
    return result


def validate_pdf_path(path: Path) -> Dict[str, Any]:
    try:
        return validate_pdf_bytes(path.read_bytes())
    except OSError as exc:
        return {
            "pdf_valid": False,
            "real_page_count": 0,
            "file_size": 0,
            "error": f"PDF_READ_FAILED: {exc}",
            "metadata": {},
        }


def _looks_like_scholarly_title(value: object) -> bool:
    title = " ".join(str(value or "").split()).strip()
    lower = title.lower()
    if lower in {"", "nan", "none", "null", "unknown", "untitled"}:
        return False
    if re.fullmatch(r"[a-z]{0,5}\d{5,}[a-z0-9._-]*", lower):
        return False
    if lower.startswith((
        "university of ", "materials research, vol.", "accepted manuscript",
        "microsoft word - ", "type of the paper ",
        "arxiv:", "proceedings of ",
    )) or lower.endswith((".doc", ".docx", ".pdf")):
        return False
    if re.match(r"^(?:metals|materials|fatigue|journal)\s+20\d{2}\s*,\s*\d+", lower):
        return False
    if len(re.findall(r"[^\W\d_]", title, flags=re.UNICODE)) < 4:
        return False
    if re.search(r"[a-z]", lower) and not re.search(r"[\u3400-\u9fff]", title):
        return len(re.findall(r"[a-z0-9]+", lower)) >= 4
    return len(title) >= 4


def clean_pdf_title(value: object) -> str:
    """Remove recognizable publisher furniture without inventing a title."""
    title = " ".join(str(value or "").split()).strip()
    numbered = re.search(
        r"(?:文章编号|Article\s+ID)\s*[:：]\s*\S+\s+(.+)$", title, re.I
    )
    if numbered and _looks_like_scholarly_title(numbered.group(1)):
        title = numbered.group(1).strip()
    return title


def _title_from_first_page(page: Any) -> str:
    """Use prominent real first-page text, never the filesystem name."""
    try:
        lines = [" ".join(line.split()) for line in page.get_text("text").splitlines()]
        lines = [line for line in lines if line]
        for index, line in enumerate(lines[:30]):
            cleaned = clean_pdf_title(line)
            if not _looks_like_scholarly_title(cleaned):
                continue
            if cleaned.lower().startswith((
                "available from ", "copyright ", "materials research, vol.",
                "the chinese journal of ", "received:", "published:",
            )):
                continue
            title_lines = [cleaned]
            if not re.search(r"[.!?。]\s*$", cleaned):
                for follow in lines[index + 1:index + 5]:
                    low = follow.lower()
                    if (
                        low.startswith(("abstract", "keywords", "available from", "received", "published"))
                        or "@" in follow or "http" in low
                        or re.search(r"\b(?:university|institute|laboratory|correspondence)\b", low)
                        or (follow.count(",") >= 2 and re.search(r"\b\d\b", follow))
                    ):
                        break
                    if len(follow) < 8:
                        break
                    title_lines.append(follow)
                    if re.search(r"[.!?。]\s*$", follow):
                        break
            sequential = clean_pdf_title(" ".join(title_lines)).rstrip(".")
            if _looks_like_scholarly_title(sequential) and len(sequential) <= 600:
                return sequential
        candidates: List[Tuple[float, float, str]] = []
        for block in (page.get_text("dict") or {}).get("blocks") or []:
            for line in block.get("lines") or []:
                spans = line.get("spans") or []
                text = " ".join(str(span.get("text") or "") for span in spans).strip()
                if not text:
                    continue
                size = max(float(span.get("size") or 0.0) for span in spans)
                y = min(float((span.get("bbox") or [0, 0])[1]) for span in spans)
                if y < 18 or re.match(r"^(?:article|research article|open access)$", text, re.I):
                    continue
                candidates.append((size, y, text))
        plausible = [row for row in candidates if _looks_like_scholarly_title(row[2])]
        if plausible:
            max_size = max(row[0] for row in plausible)
            prominent = [row for row in candidates if row[0] >= max_size - 0.6]
            prominent.sort(key=lambda row: row[1])
            joined = " ".join(row[2] for row in prominent)
            if _looks_like_scholarly_title(joined) and len(joined) <= 600:
                return clean_pdf_title(joined)
    except Exception:
        pass
    return ""


def extract_basic_pdf_metadata(content: bytes, filename: str = "") -> Dict[str, str]:
    """Extract only Stage-1 identity metadata and real parser metadata."""
    result = {
        "title": "",
        "authors": "",
        "publication_date": "",
        "doi": "",
    }
    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            metadata = doc.metadata or {}
            result["title"] = clean_pdf_title(metadata.get("title") or "")
            result["authors"] = str(metadata.get("author") or "").strip()
            result["publication_date"] = str(
                metadata.get("creationDate") or metadata.get("modDate") or ""
            ).strip()
            first_text = "\n".join(
                doc.load_page(i).get_text("text")
                for i in range(min(3, doc.page_count))
            )
            doi_match = DOI_RE.search(first_text)
            if doi_match:
                result["doi"] = normalize_doi(doi_match.group(0))
            if not _looks_like_scholarly_title(result["title"]):
                result["title"] = _title_from_first_page(doc.load_page(0))
            if not _looks_like_scholarly_title(result["title"]):
                result["title"] = ""
                for line in first_text.splitlines()[:40]:
                    clean = " ".join(line.split())
                    if _looks_like_scholarly_title(clean) and not clean.lower().startswith(
                        ("abstract", "doi", "http")
                    ):
                        result["title"] = clean_pdf_title(clean[:500])
                        break
    except Exception:
        pass
    return result


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for attempt in range(5):
        try:
            temp_path.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            # Windows virus scanners/indexers can briefly retain a handle
            # after the writer closes.  Retry the same atomic replacement.
            time.sleep(0.05 * (attempt + 1))


def load_paper_manifest(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    return _read_jsonl(stage1_paths(base_dir)["paper_manifest"])


def load_pdf_file_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    return _read_jsonl(stage1_paths(base_dir)["pdf_files"])


def load_duplicate_records(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    return _read_jsonl(stage1_paths(base_dir)["duplicate_records"])


def _stable_paper_id(doi: str, title: str, file_hash: str) -> str:
    identity = normalize_doi(doi) or normalize_title(title) or file_hash
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16].upper()
    return f"PAPER_{digest}"


def _find_duplicate(
    records: Sequence[Dict[str, Any]],
    doi: str,
    title: str,
    file_hash: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    normalized_doi = normalize_doi(doi)
    normalized_title = normalize_title(title)
    # Exact file identity is stronger than metadata identity.  Checking it
    # first prevents a local PDF that already has Stage-2/3 state from being
    # attached to an older metadata-only DOI record.
    if file_hash:
        for row in records:
            if row.get("file_hash_sha256") == file_hash:
                return row, "HASH_EXACT", file_hash
    if normalized_doi:
        for row in records:
            if normalize_doi(row.get("doi", "")) == normalized_doi:
                return row, "DOI_EXACT", normalized_doi
    if normalized_title:
        for row in records:
            if normalize_title(row.get("title", "")) == normalized_title:
                return row, "TITLE_EXACT", normalized_title
    return None, "", ""


def semantic_duplicate_candidates(
    title: str,
    records: Sequence[Dict[str, Any]],
    threshold: float = 0.88,
) -> List[Dict[str, Any]]:
    candidates = []
    for row in records:
        score = title_similarity(title, row.get("title", ""))
        if score >= threshold and score < 1.0:
            candidates.append(
                {
                    "paper_id": row.get("paper_id", ""),
                    "title": row.get("title", ""),
                    "similarity": round(score, 4),
                    "status": "NEEDS_MANUAL_REVIEW",
                }
            )
    return sorted(candidates, key=lambda item: item["similarity"], reverse=True)


def register_pdf_bytes(
    content: bytes,
    original_filename: str,
    *,
    source_path: str = "",
    source_type: str = "USER_UPLOAD",
    metadata_override: Optional[Dict[str, str]] = None,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Validate, copy, deduplicate, and register one PDF without deep reading."""
    paths = ensure_stage1_dirs(base_dir)
    validation = validate_pdf_bytes(content)
    if not validation["pdf_valid"]:
        return {
            "status": "REJECTED_INVALID_PDF",
            "pdf_valid": False,
            "real_page_count": 0,
            "file_hash_sha256": sha256_bytes(content),
            "error": validation["error"],
        }

    file_hash = sha256_bytes(content)
    metadata = extract_basic_pdf_metadata(content, original_filename)
    if metadata_override:
        for key, value in metadata_override.items():
            if value not in (None, ""):
                metadata[key] = str(value)

    title = metadata.get("title", "")
    doi = normalize_doi(metadata.get("doi", ""))
    authors = metadata.get("authors", "")
    publication_date = metadata.get("publication_date", "")
    canonical_path = paths["canonical_pdf_dir"] / f"PDF_{file_hash[:20].upper()}.pdf"
    if not canonical_path.exists():
        canonical_path.write_bytes(content)

    records = load_paper_manifest(base_dir)
    duplicate, match_level, match_value = _find_duplicate(
        records, doi, title, file_hash
    )
    now = utc_now()
    semantic_candidates = semantic_duplicate_candidates(title, records)
    canonical_upgrade = False

    if duplicate:
        paper_id = duplicate["paper_id"]
        metadata_only = not duplicate.get("pdf_valid") or not duplicate.get(
            "canonical_pdf_path"
        )
        if metadata_only:
            canonical_upgrade = True
            duplicate_status = "UNIQUE"
            duplicate_of = ""
            existing_title = str(duplicate.get("title") or "").strip()
            existing_title_usable = (
                bool(existing_title)
                and normalize_title(existing_title)
                not in {"nan", "none", "null", "unknown", "untitled"}
            )
            duplicate.update(
                {
                    "title": existing_title if existing_title_usable else title,
                    "authors": duplicate.get("authors") or authors or "",
                    "publication_date": (
                        duplicate.get("publication_date")
                        or publication_date
                        or ""
                    ),
                    "doi": duplicate.get("doi") or doi or "",
                    "normalized_title": normalize_title(
                        title or duplicate.get("title") or ""
                    ),
                    "source_type": source_type,
                    "source_path": source_path or original_filename,
                    "canonical_pdf_path": str(canonical_path.resolve()),
                    "file_hash_sha256": file_hash,
                    "real_page_count": int(validation["real_page_count"]),
                    "pdf_valid": True,
                    "library_status": "PDF_DOWNLOADED",
                    "duplicate_status": "UNIQUE",
                    "duplicate_of": "",
                    "updated_at": now,
                }
            )
            _write_jsonl(paths["paper_manifest"], records)
        else:
            duplicate_status = "DUPLICATE"
            duplicate_of = paper_id
            linked = list(duplicate.get("linked_versions") or [])
            canonical_str = str(canonical_path.resolve())
            if canonical_str not in linked and canonical_str != duplicate.get(
                "canonical_pdf_path"
            ):
                linked.append(canonical_str)
                duplicate["linked_versions"] = linked
                duplicate["updated_at"] = now
                _write_jsonl(paths["paper_manifest"], records)
            duplicate_record = DuplicateRecord(
                duplicate_id=f"DUP_{hashlib.sha256((paper_id + file_hash).encode()).hexdigest()[:16].upper()}",
                source_record_id=file_hash,
                duplicate_of=paper_id,
                match_level=match_level,
                match_value=match_value,
                status="LINKED_VERSION" if match_level == "DOI_EXACT" else "DUPLICATE",
                reason=f"Matched existing canonical paper by {match_level}",
                created_at=now,
            ).to_dict()
            duplicate_rows = load_duplicate_records(base_dir)
            if not any(
                row.get("duplicate_id") == duplicate_record["duplicate_id"]
                for row in duplicate_rows
            ):
                duplicate_rows.append(duplicate_record)
                _write_jsonl(paths["duplicate_records"], duplicate_rows)
    else:
        paper_id = _stable_paper_id(doi, title, file_hash)
        duplicate_status = (
            "POSSIBLE_DUPLICATE" if semantic_candidates else "UNIQUE"
        )
        duplicate_of = ""
        record = PaperRecord(
            paper_id=paper_id,
            title=title,
            authors=authors,
            publication_date=publication_date,
            doi=doi,
            normalized_title=normalize_title(title),
            source_type=source_type,
            source_path=source_path or original_filename,
            canonical_pdf_path=str(canonical_path.resolve()),
            file_hash_sha256=file_hash,
            real_page_count=int(validation["real_page_count"]),
            pdf_valid=True,
            library_status="PDF_DOWNLOADED",
            rag_status="NOT_INDEXED",
            evidence_status="NOT_EXTRACTED",
            duplicate_status=duplicate_status,
            duplicate_of="",
            linked_versions=[],
            created_at=now,
            updated_at=now,
        ).to_dict()
        records.append(record)
        _write_jsonl(paths["paper_manifest"], records)

    pdf_record = PdfFileRecord(
        pdf_file_id=f"PDF_{file_hash[:20].upper()}",
        paper_id=paper_id,
        source_path=source_path or original_filename,
        canonical_pdf_path=str(canonical_path.resolve()),
        file_hash_sha256=file_hash,
        pdf_valid=True,
        real_page_count=int(validation["real_page_count"]),
        file_size=len(content),
        ingest_status="REGISTERED",
        duplicate_status=duplicate_status,
        created_at=now,
        updated_at=now,
    ).to_dict()
    pdf_rows = load_pdf_file_records(base_dir)
    if not any(row.get("file_hash_sha256") == file_hash for row in pdf_rows):
        pdf_rows.append(pdf_record)
        _write_jsonl(paths["pdf_files"], pdf_rows)

    if semantic_candidates:
        review_path = paths["manual_review_dir"] / "semantic_duplicates.jsonl"
        review_rows = _read_jsonl(review_path)
        for candidate in semantic_candidates:
            item = {
                "paper_id": paper_id,
                "title": title,
                **candidate,
                "created_at": now,
                "data_version": DATA_SCHEMA_VERSION,
            }
            if item not in review_rows:
                review_rows.append(item)
        _write_jsonl(review_path, review_rows)

    return {
        "status": (
            "UPDATED_CANONICAL"
            if canonical_upgrade
            else "DUPLICATE"
            if duplicate
            else "REGISTERED"
        ),
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "publication_date": publication_date,
        "doi": doi,
        "normalized_title": normalize_title(title),
        "source_path": source_path or original_filename,
        "canonical_pdf_path": str(canonical_path.resolve()),
        "local_path": str(canonical_path.resolve()),
        "file_hash_sha256": file_hash,
        "pdf_valid": True,
        "real_page_count": int(validation["real_page_count"]),
        "file_size": len(content),
        "ingest_status": "REGISTERED",
        "duplicate_status": duplicate_status,
        "duplicate_of": duplicate_of,
        "semantic_duplicate_candidates": semantic_candidates,
        "data_version": DATA_SCHEMA_VERSION,
    }


def register_metadata_record(
    metadata: Dict[str, Any],
    *,
    source_record_id: str,
    source_type: str,
    library_status: str = "CANDIDATE",
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Register a legacy metadata row without claiming that a PDF exists."""
    paths = ensure_stage1_dirs(base_dir)
    records = load_paper_manifest(base_dir)
    doi = normalize_doi(metadata.get("doi", ""))
    title = str(metadata.get("title") or "").strip()
    file_hash = str(
        metadata.get("file_hash_sha256") or metadata.get("file_hash") or ""
    )
    existing, match_level, match_value = _find_duplicate(
        records, doi, title, file_hash
    )
    now = utc_now()
    if existing:
        for target, sources in {
            "title": ("title",),
            "authors": ("authors",),
            "publication_date": ("publication_date", "year"),
            "doi": ("doi",),
        }.items():
            if existing.get(target):
                continue
            for source in sources:
                value = metadata.get(source)
                if value:
                    existing[target] = str(value)
                    break
        existing["normalized_title"] = normalize_title(existing.get("title", ""))
        existing["metadata_source"] = str(
            existing.get("metadata_source")
            or metadata.get("metadata_source")
            or source_type
        )
        existing["oa_status"] = str(
            metadata.get("oa_status") or existing.get("oa_status") or "UNKNOWN"
        )
        existing["source_url"] = str(
            metadata.get("source_url")
            or metadata.get("landing_page")
            or existing.get("source_url")
            or existing.get("source_path")
            or ""
        )
        existing["domain_scope"] = str(
            metadata.get("domain_scope")
            or existing.get("domain_scope")
            or ""
        )
        existing["scope_reason"] = str(
            metadata.get("scope_reason")
            or existing.get("scope_reason")
            or ""
        )
        existing["updated_at"] = now
        _write_jsonl(paths["paper_manifest"], records)
        duplicate = DuplicateRecord(
            duplicate_id=f"DUP_{hashlib.sha256((source_record_id + existing['paper_id']).encode()).hexdigest()[:16].upper()}",
            source_record_id=source_record_id,
            duplicate_of=existing["paper_id"],
            match_level=match_level,
            match_value=match_value,
            status="LINKED_VERSION" if match_level == "DOI_EXACT" else "DUPLICATE",
            reason=f"Legacy metadata matched canonical paper by {match_level}",
            created_at=now,
        ).to_dict()
        duplicate_rows = load_duplicate_records(base_dir)
        if not any(
            row.get("duplicate_id") == duplicate["duplicate_id"]
            for row in duplicate_rows
        ):
            duplicate_rows.append(duplicate)
            _write_jsonl(paths["duplicate_records"], duplicate_rows)
        return {
            "status": "DUPLICATE",
            "paper_id": existing["paper_id"],
            "duplicate_of": existing["paper_id"],
            "match_level": match_level,
        }

    paper_id = _stable_paper_id(doi, title, file_hash or source_record_id)
    semantic_candidates = semantic_duplicate_candidates(title, records)
    record = PaperRecord(
        paper_id=paper_id,
        title=title,
        authors=str(metadata.get("authors") or ""),
        publication_date=str(
            metadata.get("publication_date") or metadata.get("year") or ""
        ),
        doi=doi,
        normalized_title=normalize_title(title),
        source_type=source_type,
        source_path=str(metadata.get("source_path") or ""),
        canonical_pdf_path=str(metadata.get("canonical_pdf_path") or ""),
        file_hash_sha256=file_hash,
        real_page_count=int(metadata.get("real_page_count") or 0),
        pdf_valid=bool(metadata.get("pdf_valid") or False),
        library_status=library_status,
        rag_status="NOT_INDEXED",
        evidence_status="NOT_EXTRACTED",
        duplicate_status=(
            "POSSIBLE_DUPLICATE" if semantic_candidates else "UNIQUE"
        ),
        created_at=now,
        updated_at=now,
    ).to_dict()
    record.update(
        {
            "metadata_source": str(metadata.get("metadata_source") or source_type),
            "oa_status": str(metadata.get("oa_status") or "UNKNOWN"),
            "source_url": str(
                metadata.get("source_url")
                or metadata.get("landing_page")
                or metadata.get("source_path")
                or ""
            ),
            "domain_scope": str(metadata.get("domain_scope") or ""),
            "scope_reason": str(metadata.get("scope_reason") or ""),
        }
    )
    records.append(record)
    _write_jsonl(paths["paper_manifest"], records)
    return {
        "status": "REGISTERED",
        "paper_id": paper_id,
        "duplicate_of": "",
        "match_level": "",
    }


def register_pdf_path(
    path: Path,
    *,
    source_type: str = "LEGACY_IMPORT",
    metadata_override: Optional[Dict[str, str]] = None,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    return register_pdf_bytes(
        path.read_bytes(),
        path.name,
        source_path=str(path.resolve()),
        source_type=source_type,
        metadata_override=metadata_override,
        base_dir=base_dir,
    )


def update_paper_rag_status(
    paper_id: str, rag_status: str, base_dir: Path = BASE_DIR
) -> None:
    paths = stage1_paths(base_dir)
    rows = load_paper_manifest(base_dir)
    changed = False
    for row in rows:
        if row.get("paper_id") == paper_id:
            row["rag_status"] = rag_status
            row["updated_at"] = utc_now()
            changed = True
            break
    if changed:
        _write_jsonl(paths["paper_manifest"], rows)


def update_paper_library_status(
    paper_id: str,
    library_status: str,
    *,
    quarantine_reason: str = "",
    base_dir: Path = BASE_DIR,
) -> None:
    """Update the lifecycle state of one canonical paper record in place."""
    paths = stage1_paths(base_dir)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        row["library_status"] = library_status
        row["quarantine_reason"] = quarantine_reason
        row["updated_at"] = utc_now()
        _write_jsonl(paths["paper_manifest"], rows)
        return


def update_paper_verified_metadata(
    paper_id: str,
    metadata: Dict[str, Any],
    *,
    base_dir: Path = BASE_DIR,
) -> None:
    """Apply metadata returned by a real scholarly API to one canonical row."""
    paths = stage1_paths(base_dir)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        for target, source in (
            ("title", "title"), ("authors", "authors"),
            ("publication_date", "year"), ("doi", "doi"),
        ):
            value = str(metadata.get(source) or "").strip()
            if value:
                row[target] = value
        row["normalized_title"] = normalize_title(row.get("title") or "")
        row["metadata_source"] = str(metadata.get("metadata_source") or "SCHOLARLY_API")
        row["metadata_confidence"] = "VERIFIED_API"
        row["updated_at"] = utc_now()
        _write_jsonl(paths["paper_manifest"], rows)
        return


def set_primary_pdf_version(
    paper_id: str,
    pdf_path: Path | str,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Select a validated higher-quality linked PDF as the canonical read version."""
    path = Path(pdf_path).resolve()
    validation = validate_pdf_path(path)
    if not validation.get("pdf_valid"):
        raise ValueError(str(validation.get("error") or "INVALID_PDF"))
    file_hash = sha256_file(path)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        row["canonical_pdf_path"] = str(path)
        row["file_hash_sha256"] = file_hash
        row["real_page_count"] = int(validation["real_page_count"])
        row["pdf_valid"] = True
        row["updated_at"] = utc_now()
        _write_jsonl(stage1_paths(base_dir)["paper_manifest"], rows)
        return {
            "paper_id": paper_id, "canonical_pdf_path": str(path),
            "file_hash_sha256": file_hash,
            "real_page_count": int(validation["real_page_count"]),
        }
    raise KeyError(paper_id)


def update_paper_domain_scope(
    paper_id: str,
    domain_scope: str,
    *,
    scope_reason: str = "",
    base_dir: Path = BASE_DIR,
) -> None:
    """Persist titanium-fatigue scope without changing scientific artifacts."""
    paths = stage1_paths(base_dir)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        row["domain_scope"] = domain_scope
        row["scope_reason"] = scope_reason
        row["updated_at"] = utc_now()
        _write_jsonl(paths["paper_manifest"], rows)
        return


def update_paper_extraction_status(
    paper_id: str,
    *,
    extraction_status: str,
    deep_read_complete: bool,
    page_record_path: str = "",
    page_coverage_ratio: float = 0.0,
    evidence_status: str = "",
    base_dir: Path = BASE_DIR,
) -> None:
    """Update only Stage-2 status fields on the canonical paper manifest."""
    paths = stage1_paths(base_dir)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        row["extraction_status"] = extraction_status
        row["deep_read_complete"] = bool(deep_read_complete)
        row["page_record_path"] = page_record_path
        row["page_coverage_ratio"] = float(page_coverage_ratio)
        if evidence_status:
            row["evidence_status"] = evidence_status
        row["data_version"] = "stage2.0"
        row["updated_at"] = utc_now()
        _write_jsonl(paths["paper_manifest"], rows)
        return


def is_title_derived_evidence(row: Dict[str, Any]) -> bool:
    title = str(row.get("title") or "")
    original = str(
        row.get("original_text")
        or row.get("original_sentence")
        or row.get("evidence_text")
        or ""
    )
    source_method = str(
        row.get("source_method") or row.get("source_field") or ""
    ).lower()
    notes = str(row.get("notes") or "").lower()
    if "title" in source_method or "title_classification" in notes:
        return True
    return bool(title and original and title_similarity(title, original) >= 0.9)


EVIDENCE_COLUMNS = [
    field.name for field in fields(EvidenceRecord)
]


def evidence_record_from_legacy(row: Dict[str, Any], index: int = 0) -> EvidenceRecord:
    quarantined = is_title_derived_evidence(row)
    directness_raw = str(
        row.get("directness") or row.get("direct_or_indirect") or ""
    ).upper()
    directness_map = {
        "DIRECT": "DIRECT",
        "INDIRECT": "INDIRECT",
        "INFERRED": "INFERRED",
        "MENTION_ONLY": "MENTION_ONLY",
    }
    directness = "INVALID" if quarantined else directness_map.get(
        directness_raw, "INVALID"
    )
    review_status = (
        "QUARANTINED_TITLE_DERIVED" if quarantined else "UNREVIEWED"
    )
    page_raw = row.get("page_number", row.get("page", 0))
    try:
        page_number = int(float(page_raw or 0))
    except (TypeError, ValueError):
        page_number = 0
    try:
        confidence = float(row.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    now = utc_now()
    return EvidenceRecord(
        evidence_id=str(row.get("evidence_id") or f"LEGACY_EV_{index:06d}"),
        paper_id=str(row.get("paper_id") or ""),
        claim=str(row.get("claim") or row.get("extracted_claim") or ""),
        original_text=str(
            row.get("original_text")
            or row.get("original_sentence")
            or row.get("evidence_text")
            or ""
        ),
        page_number=page_number,
        section=str(row.get("section") or row.get("source_section") or ""),
        directness=directness,
        confidence=confidence,
        review_status=review_status,
        source_method=str(
            row.get("source_method")
            or row.get("source_field")
            or "LEGACY_MIGRATION"
        ),
        created_at=str(row.get("creation_time") or now),
        updated_at=now,
    )


def write_evidence_partitions(
    legacy_rows: Iterable[Dict[str, Any]], base_dir: Path = BASE_DIR
) -> Dict[str, int]:
    paths = ensure_stage1_dirs(base_dir)
    trusted: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    for index, row in enumerate(legacy_rows):
        record = evidence_record_from_legacy(row, index).to_dict()
        if record["review_status"] == "QUARANTINED_TITLE_DERIVED":
            quarantined.append(record)
        elif record["directness"] != "INVALID":
            trusted.append(record)
        else:
            quarantined.append(record)
    _write_csv(paths["trusted_evidence"], trusted, EVIDENCE_COLUMNS)
    _write_csv(paths["quarantined_evidence"], quarantined, EVIDENCE_COLUMNS)
    return {"trusted": len(trusted), "quarantined": len(quarantined)}


def _write_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def load_trusted_evidence_rows(base_dir: Path = BASE_DIR) -> List[Dict[str, str]]:
    path = stage1_paths(base_dir)["trusted_evidence"]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("directness") != "INVALID"
        and row.get("review_status") != "QUARANTINED_TITLE_DERIVED"
    ]


def upsert_trusted_evidence(
    rows: Sequence[Dict[str, Any]],
    *,
    paper_id: str,
    total_pages: int,
    title: str = "",
    base_dir: Path = BASE_DIR,
) -> Dict[str, int]:
    """Idempotently replace one paper's trusted evidence after provenance checks."""
    paths = ensure_stage1_dirs(base_dir)
    existing = load_trusted_evidence_rows(base_dir)
    retained = [row for row in existing if row.get("paper_id") != paper_id]
    valid: List[Dict[str, Any]] = []
    normalized_title = normalize_title(title)
    for row in rows:
        try:
            page_number = int(row.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        original_text = str(row.get("original_text") or "").strip()
        directness = str(row.get("directness") or "INVALID").upper()
        section = str(row.get("section") or "").strip()
        review_status = str(row.get("review_status") or "")
        if directness not in DIRECTNESS_VALUES or directness == "INVALID":
            continue
        if not original_text or not 1 <= page_number <= total_pages:
            continue
        if normalized_title and normalize_title(original_text) == normalized_title:
            continue
        if directness == "DIRECT" and (not section or section in {"title", "unclassified"}):
            continue
        if directness == "INFERRED" and "MANUAL" not in review_status:
            continue
        normalized = {column: row.get(column, "") for column in EVIDENCE_COLUMNS}
        conditions = normalized.get("experimental_conditions")
        if isinstance(conditions, (dict, list)):
            normalized["experimental_conditions"] = json.dumps(
                conditions, ensure_ascii=False, sort_keys=True
            )
        valid.append(normalized)
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in valid:
        deduped[str(row["evidence_id"])] = row
    combined = retained + list(deduped.values())
    _write_csv(paths["trusted_evidence"], combined, EVIDENCE_COLUMNS)
    return {
        "paper_evidence_count": len(deduped),
        "trusted_total": len(combined),
    }


def discover_pdf_files(
    roots: Optional[Sequence[Path]] = None,
    *,
    base_dir: Path = BASE_DIR,
) -> List[Path]:
    if roots is None:
        roots = (
            base_dir / "paper" / "pdfs",
            base_dir / "papers",
            base_dir / "early_papers",
            base_dir / "followup_papers",
            base_dir / "data" / "uploaded_papers",
        )
    found: Dict[str, Path] = {}
    seen_hashes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pdf"), key=lambda item: str(item).lower()):
            if path.is_file():
                resolved = path.resolve()
                resolved_key = str(resolved).lower()
                if resolved_key in found:
                    continue
                try:
                    file_hash = sha256_file(resolved)
                except OSError:
                    file_hash = ""
                if file_hash and file_hash in seen_hashes:
                    continue
                found[resolved_key] = resolved
                if file_hash:
                    seen_hashes.add(file_hash)
    return list(found.values())


def update_paper_metadata_fields(
    paper_id: str,
    *,
    title: str = "",
    authors: str = "",
    publication_date: str = "",
    doi: str = "",
    metadata_source: str = "",
    source_url: str = "",
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """Fill verified metadata on one canonical record without replacing identity."""
    paths = ensure_stage1_dirs(base_dir)
    rows = load_paper_manifest(base_dir)
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        current_title = str(row.get("title") or "").strip()
        incoming_title = str(title or "").strip()
        current_title_suspicious = (
            "http" in current_title.lower()
            or "www." in current_title.lower()
            or current_title.lower().startswith("contents lists")
            or "journal homepage" in current_title.lower()
            or current_title.lower().startswith(
                "international journal of fatigue"
            )
            or "/4.0/)." in current_title.lower()
            or current_title.lower().startswith("journal of ")
            or "published by elsevier" in current_title.lower()
            or " department of " in current_title.lower()
            or len(current_title) > 350
            or current_title.startswith("\u7b2c")
            or "\u6458 \u8981" in current_title
        )
        if incoming_title and (
            not current_title
            or current_title_suspicious
            or (
                normalize_title(incoming_title).startswith(
                    normalize_title(current_title)
                )
                and len(incoming_title) > len(current_title)
            )
        ):
            row["title"] = incoming_title
            row["normalized_title"] = normalize_title(incoming_title)
        current_authors = str(row.get("authors") or "").strip()
        authors_suspicious = (
            "http" in current_authors.lower()
            or "scientific reports" in current_authors.lower()
            or current_authors.lower() in {"admin", "unknown", "untitled"}
            or len(current_authors) > 350
            or (
                current_title
                and normalize_title(current_authors).startswith(
                    normalize_title(current_title)[:50]
                )
            )
        )
        if authors and (not current_authors or authors_suspicious):
            row["authors"] = str(authors).strip()
        if publication_date and (
            not str(row.get("publication_date") or "").strip()
            or str(row.get("publication_date") or "").startswith("D:")
        ):
            row["publication_date"] = str(publication_date).strip()
        if doi and not normalize_doi(row.get("doi") or ""):
            row["doi"] = normalize_doi(doi)
        if metadata_source and not str(row.get("metadata_source") or "").strip():
            row["metadata_source"] = str(metadata_source).strip()
        if source_url and not str(row.get("source_url") or "").strip():
            row["source_url"] = str(source_url).strip()
        row["updated_at"] = utc_now()
        _write_jsonl(paths["paper_manifest"], rows)
        return row
    return {}


def load_rag_index(base_dir: Path = BASE_DIR) -> Dict[str, Any]:
    path = stage1_paths(base_dir)["rag_index"]
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_rag_index(index: Dict[str, Any], base_dir: Path = BASE_DIR) -> None:
    path = ensure_stage1_dirs(base_dir)["rag_index"]
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)
