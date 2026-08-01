"""EvidenceRecord deduplication, provenance audit, and non-destructive quarantine."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_ALREADY_AUDITED = "PAPER_1EF9D65943440F0C"
REQUIRED_COLUMNS = (
    "evidence_role", "independent_variables", "dependent_variables",
    "control_variables", "result_direction", "mechanism", "uncertainty",
    "source_hash",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().replace("‐", "-").replace("–", "-")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return re.sub(r"[^\w\u3400-\u9fff]+", " ", text).strip()


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _reference_pages(base_dir: Path, paper_ids: set[str]) -> set[tuple[str, int]]:
    output = set()
    for paper_id in paper_ids:
        pages = read_jsonl(base_dir / "data/deep_read" / paper_id / "page_records.jsonl")
        reference_started = False
        for page in pages:
            number = int(page.get("page_number") or 0)
            section = str(page.get("section_title") or "").casefold()
            text = str(page.get("cleaned_text") or page.get("page_text") or "")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            numbered = sum(bool(re.match(r"^\[?\d{1,3}[.)\]]\s+", line)) for line in lines)
            heading = bool(re.search(r"(^|\n)\s*(references|bibliography|参考文献)\s*($|\n)", text, re.I))
            if "reference" in section or "bibliograph" in section or heading or (len(lines) >= 8 and numbered >= 4):
                reference_started = True
            # References normally occupy a terminal run; an appendix ends it.
            if reference_started and re.search(r"(^|\n)\s*appendix\b", text, re.I):
                reference_started = False
            if reference_started:
                output.add((paper_id, number))
    return output


def _source_hash(row: dict[str, Any]) -> str:
    raw = "|".join((
        str(row.get("canonical_paper_id") or row.get("paper_id") or ""),
        str(row.get("page_number") or ""),
        normalize_text(row.get("original_text")),
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _reclassify_directness(row: dict[str, Any], document_type: str) -> tuple[str, str]:
    current = str(row.get("directness") or "MENTION_ONLY").upper()
    text = str(row.get("original_text") or "")
    if document_type == "REVIEW_ARTICLE" and current == "DIRECT":
        return "INDIRECT", "REVIEW_SECONDARY_EVIDENCE"
    cited = bool(re.search(r"\b(?:previous studies|studies have|reported by|et al\.|according to|literature)\b|\[[0-9,\- ]+\]", text, re.I))
    if current == "DIRECT" and cited:
        return "INDIRECT", "SECONDARY_CITATION_LANGUAGE"
    if len(normalize_text(text)) < 35 and not re.search(r"\d", text):
        return "MENTION_ONLY", "TOO_SHORT_FOR_DIRECT_CLAIM"
    return current if current in {"DIRECT", "INDIRECT", "MENTION_ONLY"} else "MENTION_ONLY", ""


def audit_evidence(base_dir: Path) -> dict[str, Any]:
    source = base_dir / "data/evidence/trusted_evidence.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    manifest = {row["paper_id"]: row for row in read_jsonl(base_dir / "data/paper_manifest.jsonl")}
    paper_ids = {str(row.get("canonical_paper_id") or row.get("paper_id") or "") for row in rows}
    reference_pages = _reference_pages(base_dir, paper_ids)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = base_dir / "data/backups" / f"evidence_quality_{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, backup / source.name)

    repeated_short = defaultdict(set)
    for row in rows:
        key = normalize_text(row.get("original_text"))
        if 10 <= len(key) <= 180:
            repeated_short[(str(row.get("canonical_paper_id") or row.get("paper_id") or ""), key)].add(str(row.get("page_number") or ""))
    headers = {key for key, pages in repeated_short.items() if len(pages) >= 3}

    kept, rejected, mapping = [], [], []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    directness_changes = Counter()
    reject_reasons = Counter()
    for row in rows:
        paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "")
        row["canonical_paper_id"] = paper_id
        normalized = normalize_text(row.get("original_text"))
        page = int(float(row.get("page_number") or 0))
        reason = ""
        if paper_id != TARGET_ALREADY_AUDITED:
            section = str(row.get("section") or "").casefold()
            if (paper_id, page) in reference_pages or "reference" in section or "bibliograph" in section:
                reason = "REFERENCE_MISIDENTIFICATION"
            elif (paper_id, normalized) in headers:
                reason = "REPEATED_HEADER_OR_FOOTER"
            elif re.search(r"\btable of contents\b|^contents\s*$|\.\s*\.\s*\.\s*\d+$", str(row.get("original_text") or ""), re.I):
                reason = "TABLE_OF_CONTENTS_MISIDENTIFICATION"
            elif len(normalized) < 12:
                reason = "NON_EVIDENTIARY_FRAGMENT"
            elif (paper_id, normalized) in seen:
                reason = "EXACT_DUPLICATE"
        if reason:
            row["audit_decision"] = "QUARANTINED"
            row["audit_timestamp"] = now()
            row["quarantine_reason"] = reason
            rejected.append(row)
            reject_reasons[reason] += 1
            mapping.append({"source_evidence_id": row.get("evidence_id"), "retained_evidence_id": (seen.get((paper_id, normalized)) or {}).get("evidence_id", ""), "decision": "QUARANTINED", "reason": reason})
            continue
        seen[(paper_id, normalized)] = row
        document_type = str((manifest.get(paper_id) or {}).get("document_type") or "")
        new_directness, why = _reclassify_directness(row, document_type)
        if new_directness != str(row.get("directness") or "").upper():
            directness_changes[f"{row.get('directness')}->{new_directness}:{why}"] += 1
        row["directness"] = new_directness
        row["evidence_role"] = str(row.get("support_or_counter") or "SUPPORT").upper()
        row["source_hash"] = _source_hash(row)
        variables = parse_mapping(row.get("variables"))
        conditions = parse_mapping(row.get("experimental_conditions") or row.get("conditions"))
        row["independent_variables"] = json.dumps(variables.get("independent_variables") or [], ensure_ascii=False)
        row["dependent_variables"] = json.dumps(variables.get("dependent_variables") or [], ensure_ascii=False)
        row["control_variables"] = json.dumps(variables.get("control_variables") or [], ensure_ascii=False)
        row["result_direction"] = str(variables.get("result_direction") or conditions.get("result_direction") or "NOT_REPORTED")
        row["mechanism"] = str(variables.get("deepseek_mechanism_relation") or variables.get("mechanism") or "NOT_REPORTED")
        row["uncertainty"] = str(conditions.get("deepseek_uncertainty") or row.get("confidence") or "NOT_REPORTED")
        row["audit_decision"] = "TRUSTED_AFTER_QUALITY_AUDIT"
        row["audit_timestamp"] = now()
        kept.append(row)

    # Merge only short, truly adjacent fragments after invalid-content removal.
    groups = defaultdict(list)
    for row in kept:
        if str(row.get("canonical_paper_id")) == TARGET_ALREADY_AUDITED:
            continue
        try:
            paragraph = int(float(row.get("paragraph_index") or 0))
        except ValueError:
            paragraph = 0
        if len(normalize_text(row.get("original_text"))) <= 180:
            key = (row["canonical_paper_id"], row.get("page_number"), row.get("section"), row.get("directness"), row.get("evidence_type"), row.get("experimental_conditions"))
            groups[key].append((paragraph, row))
    merged_away = set()
    merged_rows = []
    for values in groups.values():
        values.sort(key=lambda item: item[0])
        run = []
        for paragraph, row in values:
            if run and paragraph not in {run[-1][0], run[-1][0] + 1}:
                if len(run) >= 2:
                    merged_rows.append(run)
                run = []
            run.append((paragraph, row))
        if len(run) >= 2:
            merged_rows.append(run)
    for run in merged_rows:
        originals = [row for _, row in run if row["evidence_id"] not in merged_away]
        if len(originals) < 2:
            continue
        merged = dict(originals[0])
        merged["original_text"] = " ".join(str(row.get("original_text") or "").strip() for row in originals)
        merged["claim"] = " ".join(dict.fromkeys(str(row.get("claim") or "").strip() for row in originals if row.get("claim")))
        digest = hashlib.sha256("|".join(str(row["evidence_id"]) for row in originals).encode()).hexdigest()[:20].upper()
        merged["evidence_id"] = f"EV_MERGED_{digest}"
        merged["source_hash"] = _source_hash(merged)
        merged["audit_decision"] = "MERGED_ADJACENT_FRAGMENTS"
        for row in originals:
            merged_away.add(row["evidence_id"])
            mapping.append({"source_evidence_id": row["evidence_id"], "retained_evidence_id": merged["evidence_id"], "decision": "MERGED", "reason": "ADJACENT_SHORT_FRAGMENT"})
        kept.append(merged)
    kept = [row for row in kept if row.get("evidence_id") not in merged_away]

    all_fields = list(dict.fromkeys(fieldnames + list(REQUIRED_COLUMNS) + ["quarantine_reason"]))
    temporary = source.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(kept)
    temporary.replace(source)
    audit_dir = base_dir / "data/evidence/audit/full_library"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("quarantined_records.jsonl", rejected), ("repair_mapping.jsonl", mapping)):
        with (audit_dir / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    before_counts = Counter(str(row.get("canonical_paper_id") or row.get("paper_id") or "") for row in rows)
    after_counts = Counter(str(row.get("canonical_paper_id") or row.get("paper_id") or "") for row in kept)
    report = {
        "generated_at": now(), "backup_path": str(backup),
        "before_count": len(rows), "after_trusted_count": len(kept),
        "quarantined_count": len(rejected), "merged_source_count": len(merged_away),
        "merged_record_count": len(merged_rows), "reject_reason_counts": dict(reject_reasons),
        "directness_change_counts": dict(directness_changes),
        "papers_over_300_before": {key: value for key, value in before_counts.items() if value > 300},
        "papers_over_300_after": {key: value for key, value in after_counts.items() if value > 300},
        "target_preserved_without_reaudit": TARGET_ALREADY_AUDITED,
        "per_paper": [{"canonical_paper_id": key, "before": before_counts[key], "after": after_counts[key], "removed_or_merged": before_counts[key] - after_counts[key]} for key in sorted(before_counts)],
    }
    return report
