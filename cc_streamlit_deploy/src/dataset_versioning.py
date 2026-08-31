"""Version-aware dataset selection without mutating frozen manifests."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping


ACTIVE_POINTER = "data/system/active_dataset_manifest.json"
PUBLIC_ACTIVE_CONTRACT = "data/active_dataset.json"
V1_FALLBACK = "data/system/verified_dataset_v1_candidate_manifest.json"


class DatasetVersionMismatchError(RuntimeError):
    """Raised when the declared active dataset and mounted RAG disagree."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ACTIVE_DATASET_MANIFEST_INVALID:{path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ACTIVE_DATASET_MANIFEST_INVALID:{path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_dataset_contract_path(base_dir: Path) -> Path:
    """Return the single active-product contract for this runtime.

    The full local project uses ``data/system/active_dataset_manifest.json``.
    The public deployment contains only the sanitized equivalent
    ``data/active_dataset.json``.
    """
    root = Path(base_dir)
    local = root / ACTIVE_POINTER
    if local.is_file():
        return local
    public = root / PUBLIC_ACTIVE_CONTRACT
    if public.is_file():
        return public
    return local


def active_dataset_manifest_path(base_dir: Path) -> Path:
    root = Path(base_dir)
    pointer = active_dataset_contract_path(root)
    if pointer.is_file():
        payload = _read_json(pointer)
        relative = str(payload.get("manifest_path") or "").strip()
        # The sanitized public contract intentionally omits the full manifest.
        if not relative and pointer.name == Path(PUBLIC_ACTIVE_CONTRACT).name:
            return pointer
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
    payload = _read_json(path)
    papers = payload.get("papers") or payload.get("documents") or []
    declared = payload.get("verified_paper_count", payload.get("formal_document_count"))
    if path.name == Path(PUBLIC_ACTIVE_CONTRACT).name:
        declared = payload.get("paper_count")
        papers = [None] * int(declared or 0)
    if not isinstance(papers, list) or declared is None or len(papers) != int(declared):
        raise RuntimeError("FORMAL_PROVENANCE_VIOLATION:ACTIVE_DATASET_MANIFEST_INVALID")
    return payload


def _local_rag_counts(base_dir: Path) -> tuple[int, int] | None:
    path = Path(base_dir) / "data/rag/manifest.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    paper_ids = payload.get("paper_ids") or []
    document_counts = payload.get("document_counts") or {}
    if not isinstance(paper_ids, list) or not isinstance(document_counts, dict):
        raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:LOCAL_RAG_MANIFEST_INVALID")
    return len(set(map(str, paper_ids))), sum(int(value or 0) for value in document_counts.values())


def _mounted_rag_counts(manifest: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(manifest.get("formal_rag_count") or 0),
        int(manifest.get("rag_chunk_count") or 0),
        str(
            manifest.get("declared_dataset_version")
            or manifest.get("active_dataset_version")
            or ""
        ),
    )


def get_active_dataset_manifest(
    base_dir: Path,
    *,
    mounted_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return and validate the normalized active v1.x product contract.

    Historical manifests are never rewritten or inferred here.  When a local
    RAG manifest or a mounted cloud bundle is available, its version-bearing
    counts must match the declared active contract or the call fails closed.
    """
    root = Path(base_dir)
    contract_path = active_dataset_contract_path(root)
    if not contract_path.is_file():
        raise RuntimeError("ACTIVE_DATASET_CONTRACT_MISSING")
    contract = _read_json(contract_path)
    required = {
        "product_version",
        "dataset_version",
        "dataset_hash",
        "paper_count",
        "rag_count",
        "chunk_count",
        "evidence_record_count",
        "condition_evidence_record_count",
        "formula_candidate_count",
        "formula_confirmed_count",
        "private_rag_version",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError("ACTIVE_DATASET_CONTRACT_INCOMPLETE:" + ",".join(missing))

    manifest_path_text = str(contract.get("manifest_path") or "").strip()
    source_payload: dict[str, Any] = {}
    if manifest_path_text:
        manifest_path = (root / manifest_path_text).resolve()
        try:
            manifest_path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError("ACTIVE_DATASET_POINTER_ESCAPES_PROJECT") from exc
        if not manifest_path.is_file():
            raise RuntimeError("ACTIVE_DATASET_MANIFEST_MISSING")
        source_payload = _read_json(manifest_path)
        source_rows = source_payload.get("papers") or source_payload.get("documents") or []
        if str(source_payload.get("dataset_version") or "") != str(contract["dataset_version"]):
            raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:DECLARED_VERSION")
        if len(source_rows) != int(contract["paper_count"]):
            raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:PAPER_COUNT")
        if _sha256(manifest_path) != str(contract["dataset_hash"]):
            raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:DATASET_HASH")

    if mounted_manifest is not None:
        rag_count, chunk_count, mounted_version = _mounted_rag_counts(mounted_manifest)
        if mounted_version and mounted_version != str(contract["dataset_version"]):
            raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:MOUNTED_VERSION")
    else:
        local_counts = _local_rag_counts(root)
        rag_count, chunk_count = local_counts or (
            int(contract["rag_count"]),
            int(contract["chunk_count"]),
        )
    if rag_count != int(contract["rag_count"]):
        raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:RAG_COUNT")
    if chunk_count != int(contract["chunk_count"]):
        raise DatasetVersionMismatchError("DATASET_VERSION_MISMATCH:CHUNK_COUNT")

    statistics_path = root / "data/system/corpus_statistics.json"
    if statistics_path.is_file():
        counts = _read_json(statistics_path).get("counts") or {}
        comparisons = {
            "evidence_record_count": "EVIDENCE_COUNT",
            "condition_evidence_record_count": "CONDITION_EVIDENCE_COUNT",
            "formula_candidate_count": "FORMULA_CANDIDATE_COUNT",
        }
        source_keys = {
            "evidence_record_count": "evidence_record_count",
            "condition_evidence_record_count": "condition_evidence_record_count",
            "formula_candidate_count": "formula_record_count",
        }
        for contract_key, label in comparisons.items():
            actual = counts.get(source_keys[contract_key])
            if actual is not None and int(actual) != int(contract[contract_key]):
                raise DatasetVersionMismatchError(f"DATASET_VERSION_MISMATCH:{label}")

    normalized = dict(source_payload)
    normalized.update(contract)
    normalized["verified_paper_count"] = int(contract["paper_count"])
    normalized["formal_document_count"] = int(contract["paper_count"])
    normalized["formal_rag_count"] = int(contract["rag_count"])
    normalized["rag_chunk_count"] = int(contract["chunk_count"])
    normalized["status"] = "ACTIVE"
    return normalized


def active_dataset_ids(base_dir: Path) -> set[str]:
    payload = load_active_dataset_manifest(base_dir)
    papers = payload.get("papers") or payload.get("documents") or []
    if not papers and (Path(base_dir) / PUBLIC_ACTIVE_CONTRACT).is_file():
        raise RuntimeError("ACTIVE_DATASET_IDS_NOT_EXPORTED")
    ids = {
        str(row.get("document_id") or row.get("paper_id") or "")
        for row in papers
        if row.get("document_id") or row.get("paper_id")
    }
    if len(ids) != len(papers):
        raise RuntimeError("FORMAL_PROVENANCE_VIOLATION:ACTIVE_DATASET_IDS_INVALID")
    return ids
