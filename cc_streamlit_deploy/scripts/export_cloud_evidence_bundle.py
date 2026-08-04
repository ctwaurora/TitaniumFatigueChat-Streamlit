"""Export a read-only, provenance-preserving evidence bundle for Streamlit Cloud."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "TitaniumFatigueChat-Streamlit-GitHub"
    / "cc_streamlit_deploy"
    / "data"
    / "cloud_bundle"
)
BUNDLE_VERSION = "1.0.0"
SCHEMA_VERSION = "cloud-evidence-bundle-1.0"
MAX_EVIDENCE_EXCERPT = 700
MAX_RAG_EXCERPT = 700
INDEX_VOCABULARY_SIZE = 16_000
VECTOR_DIMENSIONS = 96

CONDITION_FIELDS = (
    "alloy_grade", "manufacturing_process", "heat_treatment", "hip",
    "surface_treatment", "build_orientation", "loading_mode", "stress_ratio_R",
    "frequency", "temperature", "environment", "fatigue_regime",
    "specimen_geometry", "defect_type", "defect_size", "defect_location",
    "defect_distance_to_surface", "surface_roughness", "stress_amplitude",
    "maximum_stress", "fatigue_life", "fatigue_limit", "da_dN", "delta_K",
    "crack_initiation_location", "fracture_mechanism", "mechanism",
    "result_direction", "mechanism_dominance_direction",
)


class CloudBundleExportError(RuntimeError):
    """Raised when the local formal corpus cannot produce a safe cloud bundle."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudBundleExportError(f"Cannot read required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CloudBundleExportError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise CloudBundleExportError(f"Missing required JSONL: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CloudBundleExportError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if isinstance(value, dict):
                yield value


def _clean_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\b[A-Za-z]:[\\/]\S+", "[local path omitted]", text)
    text = re.sub(r"\b[^\s<>\"']+\.pdf\b", "[file name omitted]", text, flags=re.I)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _safe_url(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.match(r"^https?://", text, re.I) else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tokenize(value: str) -> list[str]:
    text = str(value or "").lower()
    latin = re.findall(r"[a-z0-9]+(?:[/.-][a-z0-9]+)*", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return latin + chinese


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise CloudBundleExportError(f"Refusing to export empty dataset: {path.name}")
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            frame[column] = frame[column].map(_safe_scalar)
    frame.to_parquet(path, index=False, compression="zstd")


def _build_indexes(rag_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    texts = [str(row.get("text") or "") for row in rag_rows]
    document_ids = [str(row["doc_id"]) for row in rag_rows]
    vectorizer = CountVectorizer(
        tokenizer=_tokenize,
        token_pattern=None,
        lowercase=False,
        max_features=INDEX_VOCABULARY_SIZE,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(texts).tocsr()
    if counts.shape[0] != len(document_ids) or counts.shape[1] == 0:
        raise CloudBundleExportError("Cloud BM25 vocabulary is empty or misaligned")

    vocabulary = [""] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        vocabulary[index] = term
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int32)

    sparse.save_npz(output_dir / "bm25_index.npz", counts, compressed=True)
    np.save(output_dir / "bm25_document_lengths.npy", lengths, allow_pickle=False)
    np.save(output_dir / "bm25_document_frequency.npy", document_frequency, allow_pickle=False)
    (output_dir / "index_document_ids.json").write_text(
        json.dumps(document_ids, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (output_dir / "index_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    transformer = TfidfTransformer(norm="l2")
    tfidf = transformer.fit_transform(counts)
    dimensions = min(VECTOR_DIMENSIONS, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=dimensions, random_state=42, n_iter=5)
    embeddings = normalize(svd.fit_transform(tfidf)).astype(np.float16)
    components = svd.components_.astype(np.float16)
    np.save(output_dir / "vector_embeddings.npy", embeddings, allow_pickle=False)
    np.save(output_dir / "vector_components.npy", components, allow_pickle=False)
    np.save(
        output_dir / "vector_idf.npy",
        np.asarray(transformer.idf_, dtype=np.float32),
        allow_pickle=False,
    )
    index_metadata = {
        "schema_version": SCHEMA_VERSION,
        "document_count": len(document_ids),
        "vocabulary_size": len(vocabulary),
        "vector_dimensions": dimensions,
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "average_document_length": float(lengths.mean()),
        },
        "vector": {
            "method": "tfidf_truncated_svd_cosine",
            "dtype": "float16",
        },
    }
    (output_dir / "index_metadata.json").write_text(
        json.dumps(index_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [
        "bm25_index.npz",
        "bm25_document_lengths.npy",
        "bm25_document_frequency.npy",
        "index_document_ids.json",
        "index_vocabulary.json",
        "vector_embeddings.npy",
        "vector_components.npy",
        "vector_idf.npy",
        "index_metadata.json",
    ]


def export_cloud_evidence_bundle(
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    project = project_root.resolve()
    destination = output_dir.resolve()
    if destination == project or project in destination.parents:
        raise CloudBundleExportError("Cloud bundle output must not be inside the source project")
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.iterdir():
        if old.is_file():
            old.unlink()
        else:
            raise CloudBundleExportError(f"Unexpected directory in cloud bundle: {old}")

    whitelist_payload = _read_json(project / "data/system/formal_rag_whitelist.json")
    whitelist = {str(value) for value in whitelist_payload.get("paper_ids") or []}
    manifest_rows = {
        str(row.get("paper_id") or ""): row
        for row in _read_jsonl(project / "data/paper_manifest.jsonl")
    }
    formal_ids = {
        paper_id
        for paper_id in whitelist
        if paper_id in manifest_rows
        and manifest_rows[paper_id].get("formal_status") == "FORMAL_INDEXED"
        and manifest_rows[paper_id].get("rag_status") == "INDEXED_STAGE3_UNIFIED"
    }
    if not formal_ids or formal_ids != whitelist:
        raise CloudBundleExportError(
            "Formal whitelist and current FORMAL_INDEXED/INDEXED_STAGE3_UNIFIED records differ"
        )

    evidence_rows: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    evidence_counts: Counter[str] = Counter()
    with (project / "data/evidence/trusted_evidence.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "")
            evidence_id = str(row.get("evidence_id") or "")
            if paper_id not in formal_ids or not evidence_id:
                continue
            evidence_ids.add(evidence_id)
            evidence_counts[paper_id] += 1
            evidence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "paper_id": paper_id,
                    "claim": _clean_text(row.get("claim"), MAX_EVIDENCE_EXCERPT),
                    "short_excerpt": _clean_text(
                        row.get("original_text"), MAX_EVIDENCE_EXCERPT
                    ),
                    "page_number": int(float(row.get("page_number") or 0)),
                    "section": _clean_text(row.get("section"), 160),
                    "directness": str(row.get("directness") or ""),
                    "confidence": float(row.get("confidence") or 0),
                    "review_status": str(row.get("review_status") or ""),
                    "support_or_counter": str(row.get("support_or_counter") or ""),
                    "variables": _safe_scalar(row.get("variables")),
                    "conditions": _safe_scalar(row.get("conditions")),
                    "data_version": str(row.get("data_version") or ""),
                }
            )

    condition_rows: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    for row in _read_jsonl(project / "data/evidence/condition_evidence_records.jsonl"):
        paper_id = str(row.get("canonical_paper_id") or "")
        evidence_id = str(row.get("evidence_id") or "")
        if paper_id not in formal_ids or evidence_id not in evidence_ids:
            continue
        condition_counts[paper_id] += 1
        item = {
            "condition_evidence_id": str(row.get("condition_evidence_id") or ""),
            "evidence_id": evidence_id,
            "paper_id": paper_id,
            "claim": _clean_text(row.get("claim"), MAX_EVIDENCE_EXCERPT),
            "page_number": int(float(row.get("page_number") or 0)),
            "section": _clean_text(row.get("section"), 160),
            "directness": str(row.get("directness") or ""),
            "evidence_role": str(row.get("evidence_role") or ""),
            "independent_variables": _safe_scalar(row.get("independent_variables")),
            "dependent_variables": _safe_scalar(row.get("dependent_variables")),
        }
        item.update({key: _safe_scalar(row.get(key)) for key in CONDITION_FIELDS})
        condition_rows.append(item)

    formula_rows: list[dict[str, Any]] = []
    formula_counts: Counter[str] = Counter()
    formula_source = project / "data/rag/formula_documents.jsonl"
    for row in _read_jsonl(formula_source):
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in formal_ids:
            continue
        formula_counts[paper_id] += 1
        formula_rows.append(
            {
                "formula_id": str(row.get("doc_id") or ""),
                "paper_id": paper_id,
                "equation": _clean_text(row.get("equation"), 800),
                "parameters": _safe_scalar(row.get("parameters")),
                "units": _safe_scalar(row.get("units")),
                "applicable_conditions": _safe_scalar(row.get("applicable_conditions")),
                "claim": _clean_text(row.get("claim"), MAX_EVIDENCE_EXCERPT),
                "short_excerpt": _clean_text(
                    row.get("original_text"), MAX_EVIDENCE_EXCERPT
                ),
                "page_number": int(float(row.get("page_number") or 0)),
                "section": _clean_text(row.get("section"), 160),
                "confidence": float(row.get("confidence") or 0),
            }
        )

    rag_rows: list[dict[str, Any]] = []
    rag_counts: Counter[str] = Counter()
    topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rag_root = project / "data/rag"
    for index_type, filename in (
        ("page", "page_documents.jsonl"),
        ("section", "section_documents.jsonl"),
        ("evidence", "evidence_documents.jsonl"),
        ("condition", "condition_documents.jsonl"),
        ("formula", "formula_documents.jsonl"),
    ):
        for row in _read_jsonl(rag_root / filename):
            paper_id = str(row.get("paper_id") or "")
            doc_id = str(row.get("doc_id") or "")
            if paper_id not in formal_ids or not doc_id:
                continue
            text = _clean_text(
                row.get("claim") or row.get("text") or row.get("original_text"),
                MAX_RAG_EXCERPT,
            )
            conditions = row.get("experimental_conditions") or {}
            rag_counts[paper_id] += 1
            for token in _tokenize(text):
                if len(token) > 2 and token not in {"fatigue", "titanium", "alloy"}:
                    topic_counts[paper_id][token] += 1
            rag_rows.append(
                {
                    "doc_id": doc_id,
                    "paper_id": paper_id,
                    "index_type": index_type,
                    "title": _clean_text(row.get("title"), 400),
                    "text": text,
                    "claim": text,
                    "page_number": int(float(row.get("page_number") or 0)),
                    "section": _clean_text(row.get("section"), 160),
                    "experimental_conditions": _safe_scalar(conditions),
                    "directness": str(row.get("directness") or ""),
                    "confidence": float(row.get("confidence") or 0),
                    "review_status": str(row.get("review_status") or ""),
                    "source_method": str(row.get("source_method") or ""),
                    "data_version": str(row.get("data_version") or ""),
                }
            )

    formal_rows: list[dict[str, Any]] = []
    for paper_id in sorted(formal_ids):
        row = manifest_rows[paper_id]
        formal_rows.append(
            {
                "paper_id": paper_id,
                "title": _clean_text(row.get("title"), 500),
                "authors": _clean_text(row.get("authors"), 500),
                "year": str(row.get("year") or row.get("publication_date") or ""),
                "journal": "",
                "doi": str(row.get("doi") or ""),
                "oa_url": _safe_url(row.get("source_url")),
                "document_type": str(row.get("document_type") or ""),
                "domain_scope": str(row.get("domain_scope") or ""),
                "formal_status": "FORMAL_INDEXED",
                "rag_status": "INDEXED_STAGE3_UNIFIED",
                "evidence_record_count": evidence_counts[paper_id],
                "condition_evidence_record_count": condition_counts[paper_id],
                "formula_record_count": formula_counts[paper_id],
                "rag_chunk_count": rag_counts[paper_id],
                "topics": json.dumps(
                    [name for name, _ in topic_counts[paper_id].most_common(8)],
                    ensure_ascii=False,
                ),
            }
        )

    rag_paper_ids = {row["paper_id"] for row in rag_rows}
    if rag_paper_ids != formal_ids:
        raise CloudBundleExportError("Every formal paper must have current Stage-3 RAG chunks")
    if not evidence_rows or not condition_rows or not formula_rows or not rag_rows:
        raise CloudBundleExportError("Cloud bundle evidence layers must all be non-empty")
    if any(row["paper_id"] not in formal_ids for row in evidence_rows + condition_rows + formula_rows):
        raise CloudBundleExportError("Non-formal paper leaked into cloud bundle")

    parquet_files = {
        "formal_literature.parquet": formal_rows,
        "evidence_records.parquet": evidence_rows,
        "condition_evidence_records.parquet": condition_rows,
        "formula_records.parquet": formula_rows,
        "rag_chunks.parquet": rag_rows,
    }
    for filename, rows in parquet_files.items():
        _write_parquet(rows, destination / filename)
    index_files = _build_indexes(rag_rows, destination)

    rag_manifest = _read_json(project / "data/rag/manifest.json")
    corpus_snapshot = _read_json(project / "data/system/corpus_statistics.json")
    dataset_seed = json.dumps(
        {
            "rag_built_at": rag_manifest.get("built_at"),
            "whitelist": sorted(formal_ids),
            "corpus_signature": corpus_snapshot.get("signature"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    dataset_version = hashlib.sha256(dataset_seed.encode("utf-8")).hexdigest()
    exported_files = [*parquet_files, *index_files]
    file_hashes = {
        name: _sha256(destination / name)
        for name in sorted(exported_files)
    }
    traceable_papers = {
        row["paper_id"]
        for row in evidence_rows
        if row["page_number"] > 0 and row["section"]
    }
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "export_time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_commit": source_commit or _source_commit(project),
        "formal_literature_count": len(formal_rows),
        "formal_rag_count": len(rag_paper_ids),
        "traceable_literature_count": len(traceable_papers),
        "evidence_record_count": len(evidence_rows),
        "condition_evidence_record_count": len(condition_rows),
        "formula_record_count": len(formula_rows),
        "rag_chunk_count": len(rag_rows),
        "files": file_hashes,
        "text_policy": {
            "full_text_included": False,
            "maximum_evidence_excerpt_characters": MAX_EVIDENCE_EXCERPT,
            "maximum_rag_excerpt_characters": MAX_RAG_EXCERPT,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = export_cloud_evidence_bundle(PROJECT_ROOT, args.output)
    except CloudBundleExportError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
