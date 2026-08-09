"""Immutable run snapshots for reproducible paper experiments."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tfc-run-snapshot-1.0"


class RunSnapshotError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_commit(base_dir: Path) -> str:
    git = base_dir / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"
    if head.startswith("ref: "):
        reference = head[5:]
        ref_path = git / reference
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and line.endswith(" " + reference):
                    return line.split(" ", 1)[0]
    return head if len(head) == 40 else "UNKNOWN"


def _config_contract(base_dir: Path) -> tuple[str, str, dict[str, Any]]:
    path = base_dir / "config/evidence_weight_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        from src.evidence_weighting import DEFAULT_CONFIG

        payload = {"schema_version": "evidence-weight-default-v1", **DEFAULT_CONFIG}
    return (
        str(payload.get("schema_version") or "UNVERSIONED"),
        _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        payload,
    )


def _dataset_contract(base_dir: Path, fallback_fingerprint: str) -> tuple[str, str, str]:
    """Return the declared dataset version, contract hash, and content fingerprint.

    ``smart_search_dataset_version`` intentionally returns a content-sensitive
    fingerprint for cache invalidation.  A paper run also needs the human-readable
    frozen dataset version, so preserve both values instead of overloading one
    field with two unrelated meanings.
    """
    path = base_dir / "data/system/verified_dataset_v1_candidate_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback_fingerprint, "UNAVAILABLE", fallback_fingerprint
    return (
        str(payload.get("dataset_version") or fallback_fingerprint),
        str(payload.get("dataset_contract_sha256") or "UNAVAILABLE"),
        fallback_fingerprint,
    )


def _new_run_id(skill: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"RUN_{stamp}_{skill}_{uuid.uuid4().hex[:8]}"


def create_run_snapshot(
    *,
    base_dir: Path,
    question: str,
    skill: str,
    prompt_version: str,
    prompt_template: str,
    prompt: str,
    dataset_version: str,
    model: str,
    temperature: float,
    top_p: float | None,
    seed: int | None,
    top_k: int,
    research_result: dict[str, Any],
    raw_model_output: str,
    cleaned_output: str,
    response_time: float,
    token_usage: dict[str, Any] | None,
    generation_status: str,
    completion_class: str,
) -> dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    run_id = _new_run_id(skill)
    config_version, config_hash, retrieval_config = _config_contract(base_dir)
    from src.experiment_config import (
        load_paper_experiment_config,
        paper_experiment_config_hash,
    )

    experiment_config = load_paper_experiment_config(base_dir)
    declared_dataset_version, dataset_contract_hash, dataset_fingerprint = _dataset_contract(
        base_dir, dataset_version
    )
    selected = list(
        research_result.get("selected_evidence_bundle")
        or research_result.get("retrieved_evidence_pool")
        or []
    )
    retrieved_documents = list(dict.fromkeys(
        str(row.get("paper_id") or row.get("document_id") or "")
        for row in selected
        if row.get("paper_id") or row.get("document_id")
    ))
    evidence = [
        {
            "evidence_id": str(row.get("doc_id") or row.get("evidence_id") or ""),
            "document_id": str(row.get("paper_id") or row.get("document_id") or ""),
            "role": str(row.get("verified_evidence_role") or row.get("evidence_role") or ""),
            "condition_match": row.get("condition_match_score"),
            "page_number": row.get("page_number"),
        }
        for row in selected
    ]
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "question": question,
        "skill": skill,
        "timestamp": _now(),
        "git_commit": _git_commit(base_dir),
        "dataset_version": declared_dataset_version,
        "dataset_contract_sha256": dataset_contract_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "prompt_version": prompt_version,
        "prompt_template_hash": _sha256_text(prompt_template),
        "prompt_hash": _sha256_text(prompt),
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "retrieval_config": retrieval_config,
        "retrieval_config_version": config_version,
        "retrieval_config_hash": config_hash,
        "experiment_config_version": experiment_config["schema_version"],
        "experiment_config_hash": paper_experiment_config_hash(experiment_config),
        "gate_config": experiment_config["gate"],
        "model_config": experiment_config["model"],
        "top_k": top_k,
        "retrieved_document_ids": retrieved_documents,
        "evidence": evidence,
        "evidence_ids": [row["evidence_id"] for row in evidence],
        "evidence_roles": [row["role"] for row in evidence],
        "condition_match": [row["condition_match"] for row in evidence],
        "raw_model_output": raw_model_output,
        "cleaned_output": cleaned_output,
        "response_time_seconds": response_time,
        "token_usage": token_usage or {},
        "generation_status": generation_status,
        "completion_class": completion_class,
    }
    run_dir = base_dir / "runs" / datetime.now(timezone.utc).strftime("%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{run_id}.json"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise RunSnapshotError(f"RUN_SNAPSHOT_WRITE_FAILED:{exc}") from exc
    return {"run_id": run_id, "snapshot_path": str(path.resolve()), "snapshot": snapshot}


def load_run_snapshot(run_id: str, *, base_dir: Path) -> dict[str, Any]:
    matches = list((Path(base_dir).resolve() / "runs").glob(f"*/{run_id}.json"))
    if len(matches) != 1:
        raise RunSnapshotError(f"RUN_SNAPSHOT_NOT_FOUND_OR_AMBIGUOUS:{run_id}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def generate_report_from_snapshot(
    run_id: str,
    *,
    base_dir: Path,
    output_path: Path | None = None,
) -> str:
    snapshot = load_run_snapshot(run_id, base_dir=base_dir)
    report = "\n".join([
        f"# TitaniumFatigueChat Run {run_id}",
        "",
        f"- Question: {snapshot['question']}",
        f"- Skill: `{snapshot['skill']}`",
        f"- Dataset: `{snapshot['dataset_version']}`",
        f"- Git commit: `{snapshot['git_commit']}`",
        f"- Prompt: `{snapshot['prompt_version']}` / `{snapshot['prompt_hash']}`",
        f"- Retrieval config: `{snapshot['retrieval_config_version']}` / `{snapshot['retrieval_config_hash']}`",
        f"- Model: `{snapshot['model']}`",
        f"- Evidence IDs: {', '.join(snapshot['evidence_ids'])}",
        "",
        "## Frozen output",
        "",
        snapshot["cleaned_output"],
        "",
    ])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report
