"""Validated read-only access to the Streamlit Cloud evidence bundle."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


SCHEMA_VERSION = "cloud-evidence-bundle-1.0"
FAIL_CLOSED_MESSAGE = "云端正式知识库未成功加载，当前禁止生成科研结论。"


class CloudBundleError(RuntimeError):
    """Raised when a required cloud bundle is absent or invalid."""


@dataclass(frozen=True)
class CloudEvidenceBundle:
    root: Path
    manifest: dict[str, Any]
    formal_literature: pd.DataFrame
    evidence_records: pd.DataFrame
    condition_evidence_records: pd.DataFrame
    formula_records: pd.DataFrame
    rag_chunks: pd.DataFrame
    document_ids: tuple[str, ...]
    document_lookup: Mapping[str, dict[str, Any]]
    vocabulary: tuple[str, ...]
    vocabulary_lookup: Mapping[str, int]
    bm25_index: sparse.csr_matrix
    bm25_document_lengths: np.ndarray
    bm25_document_frequency: np.ndarray
    vector_embeddings: np.ndarray
    vector_components: np.ndarray
    vector_idf: np.ndarray

    @property
    def dataset_version(self) -> str:
        return str(self.manifest["dataset_version"])


def cloud_bundle_root(base_dir: Path) -> Path:
    from src.private_rag_loader import resolve_private_rag_root

    return resolve_private_rag_root(Path(base_dir))


def cloud_bundle_required(
    base_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    explicit = str(env.get("TFC_DATA_MODE") or "").strip().upper()
    if explicit == "LOCAL":
        return False
    if explicit == "CLOUD":
        return True
    root = Path(base_dir)
    if (root / "DEPLOY_VERSION.json").is_file():
        return True
    try:
        return cloud_bundle_root(root).is_dir()
    except Exception:
        return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_signature(root: Path) -> str:
    path = root / "manifest.json"
    try:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}:{_sha256(path)}"
    except OSError:
        return "missing"


def _frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise CloudBundleError(f"Cannot load {path.name}: {exc}") from exc


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            key: ("" if pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _validate_counts(manifest: dict[str, Any], frames: Mapping[str, pd.DataFrame]) -> None:
    expected = {
        "formal_literature_count": len(frames["formal_literature"]),
        "evidence_record_count": len(frames["evidence_records"]),
        "condition_evidence_record_count": len(frames["condition_evidence_records"]),
        "formula_record_count": len(frames["formula_records"]),
        "rag_chunk_count": len(frames["rag_chunks"]),
    }
    for key, actual in expected.items():
        if int(manifest.get(key) or 0) != actual or actual <= 0:
            raise CloudBundleError(f"Manifest count mismatch or empty dataset: {key}")
    formal_ids = set(frames["formal_literature"]["paper_id"].astype(str))
    rag_ids = set(frames["rag_chunks"]["paper_id"].astype(str))
    if int(manifest.get("formal_rag_count") or 0) != len(rag_ids) or rag_ids != formal_ids:
        raise CloudBundleError("Formal literature and formal RAG paper IDs differ")
    evidence = frames["evidence_records"]
    evidence_ids = set(evidence["evidence_id"].astype(str))
    if not set(evidence["paper_id"].astype(str)) <= formal_ids:
        raise CloudBundleError("EvidenceRecord references a non-formal paper")
    conditions = frames["condition_evidence_records"]
    if not set(conditions["evidence_id"].astype(str)) <= evidence_ids:
        raise CloudBundleError("ConditionEvidenceRecord references an unknown EvidenceRecord")
    if not set(conditions["paper_id"].astype(str)) <= formal_ids:
        raise CloudBundleError("ConditionEvidenceRecord references a non-formal paper")
    if not set(frames["formula_records"]["paper_id"].astype(str)) <= formal_ids:
        raise CloudBundleError("FormulaRecord references a non-formal paper")


@functools.lru_cache(maxsize=4)
def _load_cached(
    root_text: str, signature: str, base_dir_text: str
) -> CloudEvidenceBundle:
    del signature
    root = Path(root_text)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudBundleError("Cloud bundle manifest is missing or invalid") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CloudBundleError("Cloud bundle schema is incompatible")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CloudBundleError("Cloud bundle checksums are missing")
    for filename, expected in files.items():
        path = root / str(filename)
        if not path.is_file() or _sha256(path) != str(expected):
            raise CloudBundleError(f"Cloud bundle checksum failed: {filename}")

    frames = {
        "formal_literature": _frame(root / "formal_literature.parquet"),
        "evidence_records": _frame(root / "evidence_records.parquet"),
        "condition_evidence_records": _frame(
            root / "condition_evidence_records.parquet"
        ),
        "formula_records": _frame(root / "formula_records.parquet"),
        "rag_chunks": _frame(root / "rag_chunks.parquet"),
    }
    _validate_counts(manifest, frames)
    # Product code and the mounted private bundle must describe the same
    # active dataset.  A mismatch is a scientific provenance failure, not a
    # recoverable display warning.
    try:
        from src.dataset_versioning import get_active_dataset_manifest

        get_active_dataset_manifest(
            Path(base_dir_text), mounted_manifest=manifest
        )
    except RuntimeError as exc:
        if "ACTIVE_DATASET_CONTRACT_MISSING" not in str(exc):
            raise CloudBundleError(str(exc)) from exc
    document_ids = tuple(
        json.loads((root / "index_document_ids.json").read_text(encoding="utf-8"))
    )
    vocabulary = tuple(
        json.loads((root / "index_vocabulary.json").read_text(encoding="utf-8"))
    )
    bm25_index = sparse.load_npz(root / "bm25_index.npz").tocsr()
    lengths = np.load(root / "bm25_document_lengths.npy", allow_pickle=False)
    document_frequency = np.load(
        root / "bm25_document_frequency.npy", allow_pickle=False
    )
    embeddings = np.load(root / "vector_embeddings.npy", allow_pickle=False)
    components = np.load(root / "vector_components.npy", allow_pickle=False)
    vector_idf = np.load(root / "vector_idf.npy", allow_pickle=False)
    rag_records = _records(frames["rag_chunks"])
    if (
        len(document_ids) != len(rag_records)
        or bm25_index.shape != (len(document_ids), len(vocabulary))
        or lengths.shape[0] != len(document_ids)
        or document_frequency.shape[0] != len(vocabulary)
        or embeddings.shape[0] != len(document_ids)
        or components.shape[1] != len(vocabulary)
        or vector_idf.shape[0] != len(vocabulary)
    ):
        raise CloudBundleError("Cloud retrieval indexes are not aligned")
    if list(document_ids) != [str(row["doc_id"]) for row in rag_records]:
        raise CloudBundleError("Cloud document IDs are not aligned with RAG chunks")
    for row in rag_records:
        row["experimental_conditions"] = _parse_json_object(
            row.get("experimental_conditions")
        )
        row["original_text"] = str(row.get("text") or "")
    lookup = {str(row["doc_id"]): row for row in rag_records}
    return CloudEvidenceBundle(
        root=root,
        manifest=manifest,
        formal_literature=frames["formal_literature"],
        evidence_records=frames["evidence_records"],
        condition_evidence_records=frames["condition_evidence_records"],
        formula_records=frames["formula_records"],
        rag_chunks=frames["rag_chunks"],
        document_ids=document_ids,
        document_lookup=lookup,
        vocabulary=vocabulary,
        vocabulary_lookup={term: index for index, term in enumerate(vocabulary)},
        bm25_index=bm25_index,
        bm25_document_lengths=lengths,
        bm25_document_frequency=document_frequency,
        vector_embeddings=embeddings,
        vector_components=components,
        vector_idf=vector_idf,
    )


def load_cloud_bundle(base_dir: Path) -> CloudEvidenceBundle:
    try:
        root = cloud_bundle_root(Path(base_dir)).resolve()
    except Exception as exc:
        raise CloudBundleError("Private RAG bundle source is unavailable or invalid") from exc
    if not root.is_dir():
        raise CloudBundleError("Cloud bundle directory is missing")
    return _load_cached(
        str(root), _manifest_signature(root), str(Path(base_dir).resolve())
    )


def cloud_bundle_status(base_dir: Path) -> dict[str, Any]:
    required = cloud_bundle_required(base_dir)
    if not required:
        return {"required": False, "ready": False, "mode": "LOCAL", "error": ""}
    try:
        bundle = load_cloud_bundle(base_dir)
    except CloudBundleError as exc:
        return {
            "required": True,
            "ready": False,
            "mode": "CLOUD",
            "error": str(exc),
            "message": FAIL_CLOSED_MESSAGE,
        }
    return {
        "required": True,
        "ready": True,
        "mode": "CLOUD",
        "error": "",
        "manifest": dict(bundle.manifest),
    }


def require_cloud_bundle(base_dir: Path) -> CloudEvidenceBundle | None:
    if not cloud_bundle_required(base_dir):
        return None
    try:
        return load_cloud_bundle(base_dir)
    except CloudBundleError as exc:
        raise CloudBundleError(FAIL_CLOSED_MESSAGE) from exc


def _tokens(text: str) -> list[str]:
    lowered = str(text or "").lower()
    return (
        re.findall(r"[a-z0-9]+(?:[/.-][a-z0-9]+)*", lowered)
        + re.findall(r"[\u4e00-\u9fff]", lowered)
    )


def cloud_bm25_scores(query: str, bundle: CloudEvidenceBundle) -> dict[str, float]:
    term_counts = Counter(
        bundle.vocabulary_lookup[token]
        for token in _tokens(query)
        if token in bundle.vocabulary_lookup
    )
    if not term_counts:
        return {}
    scores = np.zeros(len(bundle.document_ids), dtype=np.float32)
    n_docs = len(bundle.document_ids)
    avgdl = max(float(bundle.bm25_document_lengths.mean()), 1.0)
    k1, b = 1.5, 0.75
    for column, query_frequency in term_counts.items():
        values = bundle.bm25_index[:, column].toarray().ravel()
        df = float(bundle.bm25_document_frequency[column])
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denominator = values + k1 * (
            1.0 - b + b * bundle.bm25_document_lengths / avgdl
        )
        scores += query_frequency * idf * (
            values * (k1 + 1.0) / np.maximum(denominator, 1e-8)
        )
    top = np.flatnonzero(scores > 0)
    return {bundle.document_ids[index]: float(scores[index]) for index in top}


def cloud_vector_scores(query: str, bundle: CloudEvidenceBundle) -> dict[str, float]:
    counts: dict[int, float] = Counter(
        bundle.vocabulary_lookup[token]
        for token in _tokens(query)
        if token in bundle.vocabulary_lookup
    )
    if not counts:
        return {}
    row = np.zeros(len(bundle.vocabulary), dtype=np.float32)
    for index, value in counts.items():
        row[index] = value
    row *= bundle.vector_idf
    norm = float(np.linalg.norm(row))
    if norm <= 0:
        return {}
    dense = (row / norm) @ bundle.vector_components.astype(np.float32).T
    dense_norm = float(np.linalg.norm(dense))
    if dense_norm <= 0:
        return {}
    values = bundle.vector_embeddings.astype(np.float32) @ (dense / dense_norm)
    top = np.flatnonzero(values > 0)
    return {bundle.document_ids[index]: float(values[index]) for index in top}


def cloud_documents_by_ids(
    document_ids: Sequence[str],
    bundle: CloudEvidenceBundle,
) -> list[dict[str, Any]]:
    return [
        dict(bundle.document_lookup[doc_id])
        for doc_id in dict.fromkeys(str(value) for value in document_ids if value)
        if doc_id in bundle.document_lookup
    ]


def cloud_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _records(frame)
