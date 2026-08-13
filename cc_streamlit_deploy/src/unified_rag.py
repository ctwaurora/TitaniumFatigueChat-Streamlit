"""Stage-3 unified, provenance-gated hybrid RAG.

The only scientific retrieval source is ``data/rag/manifest.json`` and the
five JSONL document layers referenced by that manifest.  Legacy Stage-1 chunks
are retained, but are never loaded by this module.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from src.research_topics import (
    document_topics,
    expand_topic_query,
    identify_topics,
    pore_is_dominant,
    query_mentions_pores,
)
from src.query_frame import parse_query_frame
from src.formal_pdf_protection import validate_formal_pdf_locks
from src.evidence_weighting import load_weight_config, score_evidence, select_evidence_budget
from src.claim_evidence_verifier import (
    ALTERNATIVE_MECHANISM,
    CONDITION_DEPENDENT,
    DIRECT_COUNTER,
    DIRECT_SUPPORT,
    LIMITATION_EVIDENCE,
    REVIEW_BACKGROUND,
    SUPPORTING_CONTEXT,
    classify_evidence_for_claim,
)
from src.stage1_store import BASE_DIR, load_paper_manifest, update_paper_rag_status
from src.dataset_versioning import active_dataset_ids


RAG_SCHEMA_VERSION = "stage3.0"
ALLOWED_DIRECTNESS = {"DIRECT", "INDIRECT", "MENTION_ONLY", "INFERRED"}
FORBIDDEN_REVIEW = {"QUARANTINED_TITLE_DERIVED"}
REFERENCE_SECTION_NAMES = {"reference", "references", "bibliography"}
INDEX_FILES = {
    "page": "page_documents.jsonl",
    "section": "section_documents.jsonl",
    "evidence": "evidence_documents.jsonl",
    "condition": "condition_documents.jsonl",
    "formula": "formula_documents.jsonl",
}

CONDITION_FIELDS = (
    "alloy_grade", "manufacturing_process", "build_orientation", "layer_thickness",
    "laser_power", "scan_speed", "energy_density", "relative_density", "heat_treatment",
    "hip", "surface_treatment", "defect_type", "defect_size", "defect_morphology",
    "defect_distance_to_surface", "porosity", "defect_location", "defect_distribution",
    "stress_amplitude", "maximum_stress", "stress_ratio_R", "frequency", "cycle_count",
    "fatigue_regime", "loading_mode", "environment", "temperature", "specimen_geometry",
    "surface_roughness", "ct_resolution", "sem", "ebsd", "crack_detection_method",
    "fatigue_life", "fatigue_limit", "crack_initiation_location", "da_dN", "delta_K",
    "fracture_mechanism", "mechanism_dominance_direction",
)

# Coverage is measured over ten scientific condition families rather than all
# optional schema fields. A record can satisfy a family through a legacy alias
# or its canonical ConditionEvidenceRecord name.
SUFFICIENCY_CONDITION_GROUPS = (
    ("alloy_grade", "material"),
    ("manufacturing_process", "process"),
    ("surface_treatment", "surface_state", "surface_roughness"),
    ("heat_treatment", "hip"),
    ("stress_ratio_R",),
    ("fatigue_regime",),
    ("temperature",),
    ("specimen_geometry", "sample_geometry"),
    ("loading_mode", "testing_method"),
    ("sem", "ebsd", "ct_resolution", "characterization_method"),
)

QUERY_EXPANSIONS = {
    "孔隙": "pore porosity defect lack of fusion gas pore",
    "尺寸": "size diameter area sqrt area",
    "疲劳寿命": "fatigue life Nf cycles S-N",
    "距表面距离": "distance from surface subsurface near-surface internal depth",
    "裂纹起裂": "crack initiation origin nucleation",
    "表面": "surface roughness as-built machined polished",
    "热处理": "heat treatment anneal HIP hot isostatic pressing",
    "应力比": "stress ratio load ratio R Pmin Pmax",
    "裂纹扩展": "fatigue crack growth propagation da/dN Delta K",
    "巴黎": "Paris law C m da/dN Delta K",
    "公式": "equation formula Paris law C m",
    "组织": "microstructure grain lath texture phase EBSD SEM TEM XRD",
    "as-built": "as-built rough surface",
    "machined": "machined polished surface",
    "c和m": "Paris C m coefficient exponent",
}

VARIABLE_TERMS = {
    "pore_size": ("pore size", "diameter", "sqrt area", "defect size", "孔隙尺寸"),
    "surface_distance": (
        "distance from surface",
        "near-surface",
        "subsurface",
        "internal",
        "depth",
        "距表面",
    ),
    "fatigue_life": ("fatigue life", "nf", "cycles", "疲劳寿命"),
    "surface_state": ("as-built", "machined", "polished", "surface roughness"),
    "crack_initiation": ("crack initiation", "crack origin", "nucleation"),
    "crack_growth": ("crack growth", "da/dn", "delta k", "paris"),
    "paris_c": ("paris c", "coefficient c", "parameter c"),
    "paris_m": ("paris m", "exponent m", "parameter m"),
    "stress_ratio_R": ("stress ratio", "load ratio", "r ="),
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rag_paths(base_dir: Path = BASE_DIR) -> Dict[str, Path]:
    root = base_dir / "data" / "rag"
    result = {
        "root": root,
        "manifest": root / "manifest.json",
        "status": root / "index_status.json",
        "bm25_dir": root / "bm25",
        "bm25": root / "bm25" / "index.json",
        "bm25_cache": root / "bm25" / "index.joblib",
        "document_cache": root / "search_documents.joblib",
        "document_lookup": root / "search_documents.sqlite3",
        "vector_dir": root / "vector",
        "vector_model": root / "vector" / "model.joblib",
        "vector_embeddings": root / "vector" / "embeddings.npy",
        "vector_ids": root / "vector" / "document_ids.json",
    }
    for index_type, filename in INDEX_FILES.items():
        result[index_type] = root / filename
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _replace_with_windows_retry(temp, path)


def _atomic_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _replace_with_windows_retry(temp, path)


def _replace_with_windows_retry(temp: Path, target: Path) -> None:
    """Tolerate short-lived antivirus/indexer handles on Windows."""
    attempts = 20
    for attempt in range(attempts):
        try:
            temp.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.15 * (attempt + 1), 1.0))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_reference_page(page: Dict[str, Any]) -> bool:
    """Return True only for pages classified wholly as a reference list."""
    section = str(page.get("section_title") or "").strip().casefold()
    page_type = str(page.get("page_type") or "").strip().casefold()
    return section in REFERENCE_SECTION_NAMES or page_type in REFERENCE_SECTION_NAMES


def _trim_reference_tail(text: str) -> str:
    """Keep body text before a standalone References heading on a mixed page."""
    clean = str(text or "")
    match = re.search(r"(?im)^\s*references\s*$", clean)
    return clean[: match.start()].rstrip() if match else clean


def _norm_key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", _norm(text).lower(), flags=re.UNICODE).strip()


def _tokens(text: str) -> List[str]:
    latin = re.findall(r"[a-zA-Z0-9]+(?:[/.-][a-zA-Z0-9]+)*", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return latin + chinese


def expand_question(question: str) -> str:
    return expand_topic_query(question)


def identify_variables(question: str) -> List[str]:
    lower = expand_question(question).lower()
    found = []
    for variable, terms in VARIABLE_TERMS.items():
        if any(term.lower() in lower for term in terms):
            found.append(variable)
    return found


def _counter_signal(text: str) -> bool:
    return bool(re.search(
        r"\b(?:no significant|not significant|however|whereas|contrary|exception|"
        r"depend(?:s|ed|ent)? on|limited to|did not|failed to|dominat(?:e|ed|es|ing)?|controlled by|governed by)\b|"
        r"无显著|并不显著|然而|相反|例外|取决于|仅适用于|未提高|未降低",
        str(text or ""),
        re.I,
    ))


def _conditions_from_text(text: str, supplied: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    lower = text.lower()
    conditions: Dict[str, Any] = {key: "" for key in CONDITION_FIELDS}
    supplied = supplied or {}
    for key in CONDITION_FIELDS:
        value = supplied.get(key)
        if value:
            conditions[key] = value
    if re.search(r"\bti[- ]?6al[- ]?4v\b|\bti64\b", lower):
        conditions["material"] = "Ti-6Al-4V"
    processes = []
    for name, pattern in (
        ("L-PBF", r"\bl-?pbf\b|laser powder bed fusion"),
        ("SLM", r"\bslm\b|selective laser melting"),
        ("AM", r"additive manufactur"),
    ):
        if re.search(pattern, lower):
            processes.append(name)
    if processes:
        conditions["process"] = sorted(set(processes))
    surfaces = [
        value
        for value in ("as-built", "machined", "polished", "rough")
        if value in lower
    ]
    if surfaces:
        conditions["surface_state"] = surfaces
    treatments = []
    if re.search(r"\bhip\b|hot isostatic press", lower):
        treatments.append("HIP")
    if re.search(r"\banneal", lower):
        treatments.append("annealed")
    if re.search(r"heat treat", lower):
        treatments.append("heat-treated")
    if treatments:
        conditions["heat_treatment"] = sorted(set(treatments))
    ratio = re.search(
        r"\b(?:stress|load)?\s*ratio(?:\s+of)?\s*R?\s*[=:]?\s*(-?\d+(?:\.\d+)?)"
        r"|\bR\s*=\s*(?:P\s*min\s*/\s*P\s*max\s*=\s*)?(-?\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    if ratio:
        conditions["stress_ratio_R"] = next(
            (value for value in ratio.groups() if value is not None), ""
        )
    regime = []
    for name, pattern in (
        ("LCF", r"\blcf\b|low.?cycle"),
        ("HCF", r"\bhcf\b|high.?cycle"),
        ("VHCF", r"\bvhcf\b|very.?high.?cycle"),
        ("FCG", r"fatigue crack (?:growth|propagation)|\bfcgr?\b"),
    ):
        if re.search(pattern, lower):
            regime.append(name)
    if regime:
        conditions["fatigue_regime"] = regime
    temp = re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:°\s*C|º\s*C|K)\b", text, re.I)
    if temp:
        conditions["temperature"] = temp.group(0)
    geometry = re.search(
        r"\b(?:C\(T\)|compact tension|dog[- ]?bone|cylindrical|rectangular|"
        r"four[- ]point bend|three[- ]point bend)\b",
        text,
        re.I,
    )
    if geometry:
        conditions["sample_geometry"] = geometry.group(0)
    tests = []
    for name, pattern in (
        ("fatigue", r"fatigue test"),
        ("FCG", r"crack growth test|ASTM\s*E647"),
        ("four-point bending", r"four[- ]point bend"),
    ):
        if re.search(pattern, lower):
            tests.append(name)
    if tests:
        conditions["testing_method"] = tests
    methods = [
        method
        for method in ("EBSD", "SEM", "TEM", "XRD", "optical microscopy", "micro-CT")
        if method.lower() in lower
    ]
    if methods:
        conditions["characterization_method"] = methods
    return conditions


def _parse_conditions(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _doc(
    *,
    doc_id: str,
    index_type: str,
    paper: Dict[str, Any],
    text: str,
    page_number: int,
    section: str,
    claim: str = "",
    directness: str = "",
    confidence: float = 0.0,
    review_status: str = "",
    conditions: Optional[Dict[str, Any]] = None,
    source_method: str = "",
) -> Dict[str, Any]:
    clean = _norm(text)
    return {
        "doc_id": doc_id,
        "index_type": index_type,
        "paper_id": str(paper.get("paper_id") or ""),
        "title": str(paper.get("title") or ""),
        "claim": _norm(claim or clean),
        "original_text": clean,
        "text": clean,
        "page_number": int(page_number or 0),
        "section": str(section or ""),
        "experimental_conditions": _conditions_from_text(clean, conditions),
        "directness": directness,
        "confidence": float(confidence or 0.0),
        "review_status": review_status,
        "source_method": source_method,
        "source_pdf_path": str(paper.get("source_pdf_path") or ""),
        "file_hash_sha256": str(paper.get("file_hash_sha256") or ""),
        "evidence_tier": str(paper.get("evidence_tier") or "TIER1_CORE_DIRECT"),
        "tier_reason": str(paper.get("tier_reason") or "V1_FROZEN_FORMAL_CORPUS"),
        "data_version": RAG_SCHEMA_VERSION,
    }


def _paper_stage2_record(base_dir: Path, paper_id: str) -> Optional[Dict[str, Any]]:
    root = base_dir / "data" / "deep_read" / paper_id
    status = _read_json(root / "extraction_status.json", {})
    pages = _read_jsonl(root / "page_records.jsonl")
    if (
        not status.get("deep_read_complete")
        or status.get("pipeline_version") not in {"stage2.0", "stage2.1"}
        or not pages
        or len(pages) != int(status.get("real_page_count") or 0)
        or [int(row.get("page_number") or 0) for row in pages]
        != list(range(1, len(pages) + 1))
    ):
        return None
    source_pdf = Path(str(status.get("source_pdf_path") or ""))
    if not source_pdf.exists():
        return None
    return {"status": status, "pages": pages, "root": root}


def _legacy_inventory(base_dir: Path) -> List[Dict[str, Any]]:
    candidates = (
        base_dir / "data" / "chunks.jsonl",
        base_dir / "data" / "chunks",
        base_dir / "data" / "paper_index.json",
        base_dir / "data" / "rag" / "paper_index.json",
        base_dir / "data" / "rag" / "chunks",
    )
    return [
        {
            "path": str(path.resolve()),
            "status": "LEGACY",
            "participates_in_scientific_retrieval": False,
            "exists": path.exists(),
        }
        for path in candidates
    ]


def _load_trusted_evidence(base_dir: Path) -> List[Dict[str, Any]]:
    path = base_dir / "data" / "evidence" / "trusted_evidence.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_documents(
    base_dir: Path, paper_ids: Sequence[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    manifest_rows = {
        row.get("paper_id"): row for row in load_paper_manifest(base_dir)
    }
    trusted = _load_trusted_evidence(base_dir)
    output = {key: [] for key in INDEX_FILES}
    excluded = Counter()
    accepted_papers = []
    for paper_id in paper_ids:
        stage2 = _paper_stage2_record(base_dir, paper_id)
        if not stage2:
            excluded["paper_not_stage2_trusted"] += 1
            continue
        status, pages, root = stage2["status"], stage2["pages"], stage2["root"]
        paper = dict(manifest_rows.get(paper_id) or {})
        paper.update(
            {
                "paper_id": paper_id,
                "title": status.get("title") or paper.get("title") or "",
                "source_pdf_path": status.get("source_pdf_path") or "",
                "file_hash_sha256": status.get("file_hash_sha256") or "",
            }
        )
        page_by_number = {int(row["page_number"]): row for row in pages}
        accepted_papers.append(paper_id)
        retrievable_pages: List[Dict[str, Any]] = []
        for page in pages:
            if _is_reference_page(page):
                excluded["reference_page_not_retrievable"] += 1
                continue
            retrievable_text = _trim_reference_tail(page.get("cleaned_text") or "")
            if not _norm(retrievable_text):
                excluded["empty_page_after_reference_trim"] += 1
                continue
            retrievable_page = dict(page)
            retrievable_page["cleaned_text"] = retrievable_text
            retrievable_pages.append(retrievable_page)
            page_number = int(page["page_number"])
            output["page"].append(
                _doc(
                    doc_id=f"PAGE_{paper_id}_{page_number:04d}",
                    index_type="page",
                    paper=paper,
                    text=retrievable_text,
                    page_number=page_number,
                    section=page.get("section_title") or "",
                    directness="",
                    confidence=float(page.get("classification_confidence") or 0),
                    review_status=page.get("parse_status") or "",
                    source_method="STAGE2_PAGE_RECORD",
                )
            )
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for page in retrievable_pages:
            grouped[str(page.get("section_title") or "unclassified")].append(page)
        for section, section_pages in grouped.items():
            text = "\n\n".join(str(row.get("cleaned_text") or "") for row in section_pages)
            output["section"].append(
                _doc(
                    doc_id=f"SECTION_{paper_id}_{hashlib.sha1(section.encode()).hexdigest()[:10]}",
                    index_type="section",
                    paper=paper,
                    text=text,
                    page_number=int(section_pages[0]["page_number"]),
                    section=section,
                    confidence=sum(
                        float(row.get("classification_confidence") or 0)
                        for row in section_pages
                    )
                    / len(section_pages),
                    review_status="STAGE2_SECTION_AGGREGATE",
                    source_method="STAGE2_SECTION_RECORD",
                )
            )
        for row in trusted:
            if row.get("paper_id") != paper_id:
                continue
            directness = str(row.get("directness") or "").upper()
            review_status = str(row.get("review_status") or "")
            try:
                page_number = int(float(row.get("page_number") or 0))
                confidence = float(row.get("confidence") or 0)
            except (TypeError, ValueError):
                excluded["invalid_numeric_provenance"] += 1
                continue
            original = _norm(row.get("original_text") or "")
            page = page_by_number.get(page_number)
            if directness not in ALLOWED_DIRECTNESS:
                excluded["invalid_directness"] += 1
                continue
            if review_status in FORBIDDEN_REVIEW:
                excluded["quarantined_title_derived"] += 1
                continue
            if directness == "INFERRED" and "MANUAL" not in review_status.upper():
                excluded["inferred_without_manual_review"] += 1
                continue
            if not page or not original:
                excluded["missing_page_provenance"] += 1
                continue
            page_text = _norm(page.get("cleaned_text") or "")
            if _norm_key(original) not in _norm_key(page_text):
                excluded["text_not_found_on_page"] += 1
                continue
            conditions = _parse_conditions(row.get("experimental_conditions"))
            evidence_doc = _doc(
                doc_id=str(row.get("evidence_id") or ""),
                index_type="evidence",
                paper=paper,
                text=original,
                page_number=page_number,
                section=str(row.get("section") or ""),
                claim=str(row.get("claim") or original),
                directness=directness,
                confidence=confidence,
                review_status=review_status,
                conditions=conditions,
                source_method=str(row.get("source_method") or ""),
            )
            output["evidence"].append(evidence_doc)
            if any(
                value not in (None, "", [], {}, "NOT_REPORTED")
                for value in evidence_doc["experimental_conditions"].values()
            ):
                condition_doc = dict(evidence_doc)
                condition_doc["doc_id"] = f"COND_{evidence_doc['doc_id']}"
                condition_doc["index_type"] = "condition"
                output["condition"].append(condition_doc)
        for index, equation in enumerate(_read_jsonl(root / "equations.jsonl"), 1):
            page_number = int(equation.get("page_number") or 0)
            if page_number not in page_by_number:
                excluded["formula_bad_page"] += 1
                continue
            if _is_reference_page(page_by_number[page_number]):
                excluded["formula_on_reference_page"] += 1
                continue
            text = _norm(equation.get("original_text") or "")
            if not text:
                continue
            conditions = _conditions_from_text(text)
            output["formula"].append(
                {
                    **_doc(
                        doc_id=f"FORMULA_{paper_id}_{page_number:04d}_{index:03d}",
                        index_type="formula",
                        paper=paper,
                        text=text,
                        page_number=page_number,
                        section=str(page_by_number[page_number].get("section_title") or ""),
                        claim=text,
                        directness="DIRECT",
                        confidence=0.85,
                        review_status=str(equation.get("review_status") or "TEXT_EXTRACTED"),
                        conditions=conditions,
                        source_method="STAGE2_FORMULA_RECORD",
                    ),
                    "equation": str(equation.get("latex_candidate") or text),
                    "parameters": re.findall(
                        r"\b(?:C|m|R|Kmax|Kmin|Delta\s*K|da/dN)\b", text, re.I
                    ),
                    "units": re.findall(
                        r"\b(?:MPa(?:\s*sqrt\(m\))?|m/cycle|mm|Hz)\b", text, re.I
                    ),
                    "applicable_conditions": conditions,
                    "source_paper": paper_id,
                }
            )
    return output, {
        "accepted_paper_ids": accepted_papers,
        "excluded": dict(excluded),
    }


def _build_bm25(documents: Sequence[Dict[str, Any]], path: Path) -> None:
    tokenized = [_tokens(doc["text"]) for doc in documents]
    document_frequency: Counter[str] = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))
    payload = {
        "schema_version": RAG_SCHEMA_VERSION,
        "document_ids": [doc["doc_id"] for doc in documents],
        "term_frequencies": [dict(Counter(terms)) for terms in tokenized],
        "document_lengths": [len(terms) for terms in tokenized],
        "document_frequency": dict(document_frequency),
        "average_document_length": (
            sum(map(len, tokenized)) / len(tokenized) if tokenized else 0.0
        ),
        "k1": 1.5,
        "b": 0.75,
    }
    _atomic_json(path, payload)
    joblib.dump(payload, path.with_suffix(".joblib"), compress=0)


def _build_vector(documents: Sequence[Dict[str, Any]], paths: Dict[str, Path]) -> None:
    texts = [doc["text"] for doc in documents]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w/.-]+\b",
    )
    sparse = vectorizer.fit_transform(texts)
    max_components = min(128, max(1, sparse.shape[0] - 1), max(1, sparse.shape[1] - 1))
    if max_components >= 2:
        svd: Optional[TruncatedSVD] = TruncatedSVD(
            n_components=max_components, random_state=20260727
        )
        dense = svd.fit_transform(sparse)
    else:
        svd = None
        dense = sparse.toarray()
    normalizer = Normalizer(copy=False)
    dense = normalizer.fit_transform(dense)
    paths["vector_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "svd": svd, "normalizer": normalizer},
        paths["vector_model"],
    )
    np.save(paths["vector_embeddings"], np.asarray(dense, dtype=np.float32))
    _atomic_json(paths["vector_ids"], [doc["doc_id"] for doc in documents])


def _build_document_lookup(documents: Sequence[Dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO documents(doc_id, payload) VALUES (?, ?)",
            ((str(row["doc_id"]), json.dumps(row, ensure_ascii=False)) for row in documents),
        )
        connection.commit()
    finally:
        connection.close()
    _replace_with_windows_retry(temporary, path)


def build_unified_rag(
    paper_ids: Sequence[str],
    *,
    base_dir: Path = BASE_DIR,
    required_paper_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Rebuild the Stage-3 source only from completed Stage-2 artifacts."""
    requested_ids = list(dict.fromkeys(paper_ids))
    validate_formal_pdf_locks(base_dir, requested_document_ids=requested_ids)
    paths = rag_paths(base_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    documents, build_info = _build_documents(base_dir, requested_ids)
    accepted_ids = set(build_info["accepted_paper_ids"])
    required_ids = set(
        requested_ids if required_paper_ids is None else required_paper_ids
    )
    missing_required = sorted(required_ids - accepted_ids)
    if missing_required:
        # This check deliberately happens before the first index write.  A
        # failed incremental intake must leave the complete previous RAG on
        # disk and must not downgrade any existing canonical record.
        raise RuntimeError(
            "UNIFIED_RAG_REQUIRED_PAPERS_REJECTED:"
            + ",".join(missing_required)
        )
    for index_type, rows in documents.items():
        _atomic_jsonl(paths[index_type], rows)
    all_documents = [
        row for index_type in INDEX_FILES for row in documents[index_type]
    ]
    joblib.dump(all_documents, paths["document_cache"], compress=0)
    _build_document_lookup(all_documents, paths["document_lookup"])
    _build_bm25(all_documents, paths["bm25"])
    _build_vector(all_documents, paths)
    counts = {key: len(value) for key, value in documents.items()}
    manifest = {
        "schema_version": RAG_SCHEMA_VERSION,
        "source_of_truth": str(paths["manifest"].resolve()),
        "scientific_retrieval_entrypoint": "src.unified_rag.retrieve_research_evidence",
        "document_layers": {
            key: str(paths[key].resolve()) for key in INDEX_FILES
        },
        "paper_ids": build_info["accepted_paper_ids"],
        "document_counts": counts,
        "corpus_statistics_source": str(
            (base_dir / "data" / "system" / "corpus_statistics.json").resolve()
        ),
        "legacy_sources": _legacy_inventory(base_dir),
        "legacy_direct_access_allowed": False,
        "built_at": _now(),
    }
    _atomic_json(paths["manifest"], manifest)
    status = {
        "status": "READY",
        "schema_version": RAG_SCHEMA_VERSION,
        "paper_count": len(build_info["accepted_paper_ids"]),
        "document_count": len(all_documents),
        "document_counts": counts,
        "bm25_ready": paths["bm25"].exists(),
        "vector_ready": (
            paths["vector_model"].exists() and paths["vector_embeddings"].exists()
        ),
        "excluded": build_info["excluded"],
        "built_at": manifest["built_at"],
    }
    _atomic_json(paths["status"], status)
    for row in load_paper_manifest(base_dir):
        paper_id = str(row.get("paper_id") or "")
        if (
            paper_id
            and paper_id not in accepted_ids
            and row.get("rag_status") == "INDEXED_STAGE3_UNIFIED"
        ):
            update_paper_rag_status(
                paper_id, "NOT_INDEXED_CURRENT_WHITELIST", base_dir
            )
    for paper_id in accepted_ids:
        update_paper_rag_status(paper_id, "INDEXED_STAGE3_UNIFIED", base_dir)
    return status


def load_unified_documents(base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    from src.cloud_evidence_bundle import (
        cloud_bundle_required,
        require_cloud_bundle,
    )

    if cloud_bundle_required(base_dir):
        bundle = require_cloud_bundle(base_dir)
        assert bundle is not None
        return [
            dict(bundle.document_lookup[doc_id])
            for doc_id in bundle.document_ids
        ]
    paths = rag_paths(base_dir)
    manifest = _read_json(paths["manifest"], {})
    if (
        manifest.get("schema_version") != RAG_SCHEMA_VERSION
        or manifest.get("legacy_direct_access_allowed") is not False
    ):
        raise RuntimeError("Stage-3 unified RAG manifest is missing or invalid")
    version = str(manifest.get("built_at") or paths["manifest"].stat().st_mtime_ns)
    return _load_unified_documents_cached(str(Path(base_dir).resolve()), version)


@functools.lru_cache(maxsize=4)
def _load_unified_documents_cached(base_dir_text: str, version: str) -> List[Dict[str, Any]]:
    paths = rag_paths(Path(base_dir_text))
    if paths["document_cache"].exists():
        return joblib.load(paths["document_cache"])
    return [
        row
        for index_type in INDEX_FILES
        for row in _read_jsonl(paths[index_type])
    ]


@functools.lru_cache(maxsize=4)
def _load_bm25_cached(base_dir_text: str, version: str) -> Dict[str, Any]:
    paths = rag_paths(Path(base_dir_text))
    if paths["bm25_cache"].exists():
        return joblib.load(paths["bm25_cache"])
    return _read_json(paths["bm25"], {})


@functools.lru_cache(maxsize=4)
def _load_vector_cached(base_dir_text: str, version: str) -> Tuple[Any, np.ndarray, List[str]]:
    paths = rag_paths(Path(base_dir_text))
    return (
        joblib.load(paths["vector_model"]),
        np.load(paths["vector_embeddings"]),
        _read_json(paths["vector_ids"], []),
    )


def _rag_version(base_dir: Path) -> str:
    from src.cloud_evidence_bundle import (
        cloud_bundle_required,
        require_cloud_bundle,
    )

    if cloud_bundle_required(base_dir):
        bundle = require_cloud_bundle(base_dir)
        assert bundle is not None
        return bundle.dataset_version
    manifest = _read_json(rag_paths(base_dir)["manifest"], {})
    return str(manifest.get("built_at") or "missing")


def _load_documents_by_ids(base_dir: Path, document_ids: Sequence[str]) -> List[Dict[str, Any]]:
    from src.cloud_evidence_bundle import (
        cloud_bundle_required,
        cloud_documents_by_ids,
        require_cloud_bundle,
    )

    if cloud_bundle_required(base_dir):
        bundle = require_cloud_bundle(base_dir)
        assert bundle is not None
        return cloud_documents_by_ids(document_ids, bundle)
    wanted = list(dict.fromkeys(str(value) for value in document_ids if value))
    lookup = rag_paths(base_dir)["document_lookup"]
    if not wanted or not lookup.exists():
        return [row for row in load_unified_documents(base_dir) if row.get("doc_id") in set(wanted)]
    rows: List[Dict[str, Any]] = []
    with sqlite3.connect(f"file:{lookup}?mode=ro", uri=True) as connection:
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for payload, in connection.execute(
                f"SELECT payload FROM documents WHERE doc_id IN ({placeholders})", chunk
            ):
                try:
                    rows.append(json.loads(payload))
                except json.JSONDecodeError:
                    continue
    return rows


def _bm25_scores(query: str, base_dir: Path) -> Dict[str, float]:
    from src.cloud_evidence_bundle import (
        cloud_bm25_scores,
        cloud_bundle_required,
        require_cloud_bundle,
    )

    if cloud_bundle_required(base_dir):
        bundle = require_cloud_bundle(base_dir)
        assert bundle is not None
        return cloud_bm25_scores(expand_question(query), bundle)
    payload = _load_bm25_cached(str(Path(base_dir).resolve()), _rag_version(base_dir))
    terms = _tokens(expand_question(query))
    n_docs = len(payload.get("document_ids") or [])
    avgdl = float(payload.get("average_document_length") or 1.0)
    dfs = payload.get("document_frequency") or {}
    k1, b = float(payload.get("k1") or 1.5), float(payload.get("b") or 0.75)
    scores: Dict[str, float] = {}
    for index, (doc_id, frequencies, dl) in enumerate(
        zip(
            payload.get("document_ids") or [],
            payload.get("term_frequencies") or [],
            payload.get("document_lengths") or [],
        )
    ):
        if index and index % 500 == 0:
            time.sleep(0)
        score = 0.0
        for term in terms:
            tf = float(frequencies.get(term) or 0)
            if not tf:
                continue
            df = float(dfs.get(term) or 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * float(dl) / avgdl)
            )
        if score > 0:
            scores[doc_id] = score
    return scores


def _vector_scores(query: str, base_dir: Path) -> Dict[str, float]:
    from src.cloud_evidence_bundle import (
        cloud_bundle_required,
        cloud_vector_scores,
        require_cloud_bundle,
    )

    if cloud_bundle_required(base_dir):
        bundle = require_cloud_bundle(base_dir)
        assert bundle is not None
        return cloud_vector_scores(expand_question(query), bundle)
    model, embeddings, ids = _load_vector_cached(
        str(Path(base_dir).resolve()), _rag_version(base_dir)
    )
    sparse = model["vectorizer"].transform([expand_question(query)])
    dense = model["svd"].transform(sparse) if model["svd"] is not None else sparse.toarray()
    dense = model["normalizer"].transform(dense)
    values = embeddings @ np.asarray(dense[0], dtype=np.float32)
    return {
        doc_id: float(score)
        for doc_id, score in zip(ids, values)
        if float(score) > 0
    }


def _matches_filters(doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    conditions = doc.get("experimental_conditions") or {}
    for key, wanted in filters.items():
        if wanted in (None, "", []):
            continue
        actual = doc.get(key) if key in {"paper_id", "section", "index_type"} else conditions.get(key)
        wanted_values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        actual_text = json.dumps(actual, ensure_ascii=False).lower()
        if not any(str(value).lower() in actual_text for value in wanted_values):
            return False
    return True


def _verified_dataset_ids(base_dir: Path) -> set[str] | None:
    """Return active version IDs while preserving v1 as the fail-safe fallback."""
    fallback = Path(base_dir) / "data/system/verified_dataset_v1_candidate_manifest.json"
    pointer = Path(base_dir) / "data/system/active_dataset_manifest.json"
    if not fallback.is_file() and not pointer.is_file():
        return None
    return active_dataset_ids(Path(base_dir))


def _deduplicate(results: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: set[Tuple[str, str]] = set()
    output = []
    removed = 0
    for row in results:
        key = (
            str(row.get("paper_id") or ""),
            hashlib.sha1(_norm_key(row.get("original_text") or "").encode()).hexdigest(),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(row)
    return output, removed


def retrieve_research_evidence(
    question: str,
    task_type: str = "research",
    required_variables: Optional[Sequence[str]] = None,
    condition_filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    *,
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    started = time.perf_counter()
    query_frame = parse_query_frame(question)
    query_topics = query_frame.topic_labels
    normalized_question = expand_question(question)
    normalization_complete = time.perf_counter()
    errors = []
    try:
        bm25 = _bm25_scores(normalized_question, base_dir)
        bm25_executed = True
    except Exception as exc:
        bm25, bm25_executed = {}, False
        errors.append(f"BM25:{type(exc).__name__}:{exc}")
    bm25_complete = time.perf_counter()
    try:
        vector = _vector_scores(normalized_question, base_dir)
        vector_executed = True
    except Exception as exc:
        vector, vector_executed = {}, False
        errors.append(f"VECTOR:{type(exc).__name__}:{exc}")
    vector_complete = time.perf_counter()
    candidate_ids = [key for key, _ in sorted(bm25.items(), key=lambda item: item[1], reverse=True)[:1000]]
    candidate_ids.extend(key for key, _ in sorted(vector.items(), key=lambda item: item[1], reverse=True)[:1000])
    documents = _load_documents_by_ids(base_dir, candidate_ids)
    verified_ids = _verified_dataset_ids(base_dir)
    max_bm25 = max(bm25.values(), default=1.0)
    variables = list(required_variables or identify_variables(question))
    candidates = []
    for index, doc in enumerate(documents):
        if index and index % 100 == 0:
            time.sleep(0)
        if doc["doc_id"] not in bm25 and doc["doc_id"] not in vector:
            continue
        if verified_ids is not None and str(doc.get("paper_id") or "") not in verified_ids:
            continue
        if not _matches_filters(doc, condition_filters or {}):
            continue
        lexical = bm25.get(doc["doc_id"], 0.0) / max_bm25
        semantic = max(0.0, vector.get(doc["doc_id"], 0.0))
        retrieval_score = 0.50 * lexical + 0.50 * semantic
        lower = doc["text"].lower()
        coverage = sum(
            any(term in lower for term in VARIABLE_TERMS.get(variable, (variable,)))
            for variable in variables
        )
        provenance = 1.0 if doc.get("page_number", 0) > 0 else 0.0
        direct_bonus = {
            "DIRECT": 0.12,
            "INDIRECT": 0.05,
            "MENTION_ONLY": -0.04,
            "INFERRED": -0.08,
        }.get(doc.get("directness"), 0.0)
        layer_bonus = {
            "evidence": 0.10,
            "formula": 0.09,
            "condition": 0.06,
            "page": 0.02,
            "section": 0.0,
        }.get(doc.get("index_type"), 0.0)
        doc_topics = document_topics(doc.get("text") or "", doc.get("experimental_conditions") or {})
        topic_overlap = len(set(query_topics) & set(doc_topics))
        topic_diversity_score = topic_overlap / max(1, len(set(query_topics)))
        entity_match_score = coverage / max(1, len(variables)) if variables else 0.5
        conditions = doc.get("experimental_conditions") or {}
        frame_conditions = {
            "alloy_grade": query_frame.alloy_grade,
            "manufacturing_process": query_frame.manufacturing_process,
            "stress_ratio_R": query_frame.stress_ratio,
            "temperature": query_frame.temperature,
            "environment": query_frame.environment,
            "loading_mode": query_frame.loading_mode,
        }
        checked_conditions = [(key, value) for key, value in frame_conditions.items() if value]
        condition_matches = 0
        condition_conflicts = []
        for key, expected in checked_conditions:
            actual = conditions.get(key) or conditions.get("process" if key == "manufacturing_process" else key)
            if not actual:
                continue
            actual_text = " ".join(str(item) for item in actual) if isinstance(actual, list) else str(actual)
            if str(expected).casefold() in actual_text.casefold() or actual_text.casefold() in str(expected).casefold():
                condition_matches += 1
            else:
                condition_conflicts.append({"field": key, "query": expected, "evidence": actual})
        condition_match_score = condition_matches / max(1, len(checked_conditions)) if checked_conditions else 0.5
        stage_terms = " ".join((query_frame.fatigue_stage, query_frame.crack_stage)).casefold()
        fatigue_stage_match_score = 0.5 if not stage_terms.strip() else float(any(term.casefold() in lower or term in doc_topics for term in (query_frame.fatigue_stage, query_frame.crack_stage) if term))
        formula_match_score = 1.0 if query_frame.requested_formulas and doc.get("index_type") == "formula" else 0.4 if doc.get("index_type") == "formula" else 0.0
        source_quality_score = min(1.0, 0.55 * provenance + 0.30 * float(doc.get("directness") == "DIRECT") + 0.15 * float(doc.get("confidence") or 0.0))
        contradiction_relevance_score = float(task_type == "counter_target" and _counter_signal(doc.get("text") or ""))
        unsolicited_pore_penalty = (
            0.30
            if not query_mentions_pores(question)
            and "DEFECT" in doc_topics
            and pore_is_dominant(doc.get("text") or "")
            else 0.0
        )
        rerank_score = (
            0.20 * lexical + 0.20 * semantic + 0.14 * entity_match_score
            + 0.14 * condition_match_score + 0.08 * fatigue_stage_match_score
            + 0.06 * formula_match_score + 0.08 * source_quality_score
            + 0.06 * contradiction_relevance_score + 0.04 * topic_diversity_score
            + direct_bonus + layer_bonus - unsolicited_pore_penalty
        )
        item = {
            **doc,
            "retrieval_score": round(retrieval_score, 6),
            "rerank_score": round(rerank_score, 6),
            "source_index_type": doc["index_type"],
            "bm25_score": round(bm25.get(doc["doc_id"], 0.0), 6),
            "vector_score": round(vector.get(doc["doc_id"], 0.0), 6),
            "semantic_score": round(semantic, 6),
            "lexical_score": round(lexical, 6),
            "entity_match_score": round(entity_match_score, 6),
            "condition_match_score": round(condition_match_score, 6),
            "fatigue_stage_match_score": round(fatigue_stage_match_score, 6),
            "formula_match_score": round(formula_match_score, 6),
            "source_quality_score": round(source_quality_score, 6),
            "contradiction_relevance_score": round(contradiction_relevance_score, 6),
            "topic_diversity_score": round(topic_diversity_score, 6),
            "condition_conflicts": condition_conflicts,
            "matched_variables": [
                variable
                for variable in variables
                if any(
                    term in lower
                    for term in VARIABLE_TERMS.get(variable, (variable,))
                )
            ],
            "matched_topics": sorted(set(query_topics) & set(doc_topics)),
            "document_topics": doc_topics,
            "unsolicited_pore_penalty": unsolicited_pore_penalty,
        }
        weighted = score_evidence(item, question, config=None)
        # Keep the legacy rerank score for diagnostics. Candidate ordering now
        # uses the normalized, configurable evidence score.
        weighted["legacy_rerank_score"] = item["rerank_score"]
        weighted["rerank_score"] = weighted["evidence_weighted_score"]
        candidates.append(weighted)
    candidates.sort(
        key=lambda row: (
            float(row.get("evidence_weighted_score") or 0.0),
            float(row.get("rerank_score") or 0.0),
            str(row.get("doc_id") or ""),
        ),
        reverse=True,
    )
    # Duplicate diagnostics are measured on the retrieval window, not across
    # the entire corpus (where condition/evidence layers intentionally overlap).
    retrieval_window = candidates[: max(top_k * 5, 20)]
    deduped, removed = _deduplicate(retrieval_window)
    results = deduped[: max(0, int(top_k))]
    rerank_complete = time.perf_counter()
    return {
        "question": question,
        "task_type": task_type,
        "required_variables": variables,
        "identified_topics": query_topics,
        "query_frame": query_frame.as_dict(),
        "condition_filters": condition_filters or {},
        "bm25_executed": bm25_executed,
        "vector_executed": vector_executed,
        "metadata_filter_executed": True,
        "reranker_executed": True,
        "duplicate_removed": removed,
        "retrieved_paper_count": len({row.get("paper_id") for row in results if row.get("paper_id")}),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "phase_seconds": {
            "normalization_complete": round(normalization_complete - started, 6),
            "bm25_complete": round(bm25_complete - started, 6),
            "vector_complete": round(vector_complete - started, 6),
            "rerank_complete": round(rerank_complete - started, 6),
        },
        "errors": errors,
        "degraded_mode": "BM25_ONLY" if bm25_executed and not vector_executed else "VECTOR_ONLY" if vector_executed and not bm25_executed else "NONE",
        "results": results,
    }


def retrieve_supporting_evidence(
    question: str, *, top_k: int = 10, base_dir: Path = BASE_DIR
) -> List[Dict[str, Any]]:
    return retrieve_research_evidence(
        question, "supporting", top_k=top_k, base_dir=base_dir
    )["results"]


def generate_counter_targets(question: str) -> List[str]:
    """Generate alternate-mechanism targets without asserting they are true."""
    variables = set(identify_variables(question))
    topics = set(identify_topics(question))
    targets = []
    if {"pore_size", "surface_distance", "crack_initiation", "fatigue_life"} & variables:
        targets.extend(
            [
                "internal pore defect crack initiation fatigue",
                "surface roughness dominates crack initiation fatigue",
                "pore distance no significant effect fatigue",
                "HIP pore closure reduced defect influence fatigue",
                "VHCF fish-eye deep internal defect initiation",
                "stress amplitude controlled defect effect fatigue",
            ]
        )
    if {"paris_c", "paris_m", "crack_growth"} & variables:
        targets.extend(
            [
                "Paris constants depend on stress ratio microstructure environment",
                "Paris C m not directly comparable different units fitting interval",
                "crack closure threshold regime invalid Paris comparison",
            ]
        )
    topic_targets = {
        "RESIDUAL_STRESS": "residual stress relaxation redistribution no significant crack growth",
        "MICROSTRUCTURE": "microstructure lath grain texture no significant fatigue effect",
        "HEAT_TREATMENT": "heat treatment coarsening adverse fatigue no improvement",
        "HIP": "HIP microstructure coarsening fatigue no improvement condition dependent",
        "SURFACE_CONDITION": "surface treatment roughness machining condition dependent fatigue",
        "BUILD_ORIENTATION": "build orientation anisotropy no significant after machining heat treatment",
        "FATIGUE_LOADING": "stress ratio crack closure threshold condition dependent",
        "ENVIRONMENT": "environment temperature crack growth condition dependent",
        "SHORT_CRACK": "short crack long crack model invalid microstructure condition",
    }
    for topic in topics:
        if topic in topic_targets:
            targets.append(topic_targets[topic])
    if not targets:
        targets = [
            f"alternative mechanism conditions for {expand_question(question)}",
            f"no significant effect exception {expand_question(question)}",
        ]
    return targets


def retrieve_counter_evidence(
    question: str, *, top_k: int = 10, base_dir: Path = BASE_DIR
) -> List[Dict[str, Any]]:
    targets = generate_counter_targets(question)
    combined_target = " ; ".join(targets)
    result = retrieve_research_evidence(
        combined_target,
        "counter_target",
        required_variables=identify_variables(question),
        top_k=max(20, top_k * 3),
        base_dir=base_dir,
    )
    pooled = []
    for row in result["results"]:
        text = str(row.get("text") or row.get("original_text") or row.get("claim") or "")
        if not _counter_signal(text):
            continue
        item = dict(row)
        item["counter_target"] = targets[0] if targets else combined_target
        item["counter_targets"] = targets
        item["counter_evidence_validated"] = True
        item["counter_relation"] = "EXPLICIT_COUNTER_SIGNAL"
        pooled.append(item)
    pooled.sort(key=lambda row: float(row.get("rerank_score") or 0.0), reverse=True)
    deduped, _ = _deduplicate(pooled)
    return deduped[:top_k]


def retrieve_condition_dependent_evidence(
    question: str,
    *,
    condition_filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    base_dir: Path = BASE_DIR,
) -> List[Dict[str, Any]]:
    result = retrieve_research_evidence(
        question,
        "condition_dependent",
        condition_filters=condition_filters,
        top_k=top_k * 3,
        base_dir=base_dir,
    )
    return [
        row
        for row in result["results"]
        if any((row.get("experimental_conditions") or {}).values())
    ][:top_k]


def build_evidence_retrieval_subtasks(question: str) -> List[Dict[str, str]]:
    """Create entity-driven retrieval facets without hard-coding one question."""
    frame = parse_query_frame(question)
    variables = set(frame.independent_variables) | set(frame.dependent_variables)
    microstructure_variables = {
        "microstructure", "alpha_lath_width", "prior_beta_grain", "crystallographic_texture"
    }
    if "residual_stress" in variables and variables & microstructure_variables:
        subject = " ".join(
            value for value in (frame.manufacturing_process, frame.alloy_grade) if value
        ) or "titanium alloy"
        return [
            {
                "name": "residual_stress_evidence_query",
                "search_query": f"{subject} residual stress fatigue crack growth da/dN Delta Keff crack closure cyclic relaxation",
                "claim_query": "残余应力影响裂纹扩展速率da/dN",
            },
            {
                "name": "microstructure_evidence_query",
                "search_query": f"{subject} alpha alpha-prime microstructure lath prior beta grain texture EBSD short crack path growth transition",
                "claim_query": "微观组织影响短裂纹路径与扩展速率da/dN",
            },
            {
                "name": "residual_stress_microstructure_interaction_query",
                "search_query": f"{subject} residual stress microstructure simultaneous measurement same specimen short crack growth interaction",
                "claim_query": "残余应力与α或α′微观组织在同一试样共同影响短裂纹扩展速率da/dN",
            },
        ]
    return [{"name": "primary_evidence_query", "search_query": question, "claim_query": question}]


@dataclass
class SufficiencyThresholds:
    min_direct_papers: int = 3
    min_condition_coverage: float = 0.60
    min_variable_coverage: float = 0.80
    min_counter_evidence: int = 1
    min_page_traceability: float = 1.0
    min_direct_evidence_rate: float = 0.50
    max_duplicate_rate: float = 0.40


def evaluate_evidence_sufficiency(
    supporting: Sequence[Dict[str, Any]],
    counter: Sequence[Dict[str, Any]],
    required_variables: Sequence[str],
    *,
    duplicate_removed: int = 0,
    thresholds: Optional[SufficiencyThresholds] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or SufficiencyThresholds()
    all_rows = list(supporting) + list(counter)
    paper_ids = {row.get("paper_id") for row in all_rows if row.get("paper_id")}
    direct_papers = {
        row.get("paper_id")
        for row in all_rows
        if row.get("directness") == "DIRECT"
    }
    traced = sum(
        int(row.get("page_number") or 0) > 0
        and bool(row.get("source_pdf_path"))
        for row in all_rows
    )
    direct = sum(row.get("directness") == "DIRECT" for row in all_rows)
    covered_vars = {
        variable
        for row in all_rows
        for variable in row.get("matched_variables") or []
        if variable in required_variables
    }
    condition_fields = {
        key
        for row in all_rows
        for key, value in (row.get("experimental_conditions") or {}).items()
        if value not in (None, "", [], {}, "NOT_REPORTED")
    }
    variable_coverage = (
        len(covered_vars) / len(set(required_variables)) if required_variables else 1.0
    )
    covered_condition_groups = sum(
        any(alias in condition_fields for alias in aliases)
        for aliases in SUFFICIENCY_CONDITION_GROUPS
    )
    condition_coverage = covered_condition_groups / len(SUFFICIENCY_CONDITION_GROUPS)
    page_traceability = traced / len(all_rows) if all_rows else 0.0
    direct_rate = direct / len(all_rows) if all_rows else 0.0
    duplicate_rate = duplicate_removed / (len(all_rows) + duplicate_removed) if (all_rows or duplicate_removed) else 0.0
    failures = []
    critical_failure = False
    if not supporting:
        failures.append("no_supporting_evidence")
        critical_failure = True
    if len(direct_papers) < thresholds.min_direct_papers:
        failures.append("direct_independent_papers_below_3")
    if variable_coverage < thresholds.min_variable_coverage:
        failures.append("variable_coverage_incomplete")
    if condition_coverage < thresholds.min_condition_coverage:
        failures.append("condition_coverage_below_60_percent")
    if len(counter) < thresholds.min_counter_evidence:
        failures.append("no_counter_evidence")
    if page_traceability < thresholds.min_page_traceability:
        failures.append("page_traceability_incomplete")
    if direct_rate < thresholds.min_direct_evidence_rate:
        failures.append("direct_evidence_rate_low")
    if duplicate_rate > thresholds.max_duplicate_rate:
        failures.append("retrieval_highly_duplicate")
    only_weak = bool(all_rows) and all(
        row.get("directness") in {"MENTION_ONLY", "INDIRECT", ""}
        for row in all_rows
    )
    if only_weak:
        failures.append("only_abstract_or_mention_level_evidence")
    # Two independent gate failures indicate that the corpus is not merely
    # incomplete in one dimension; it is insufficient for a research answer.
    if not all_rows or critical_failure or len(failures) >= 2:
        state = "INSUFFICIENT"
    elif failures:
        state = "PARTIALLY_SUFFICIENT"
    else:
        state = "SUFFICIENT"
    return {
        "status": state,
        "direct_paper_count": len(direct_papers),
        "independent_paper_count": len(paper_ids),
        "condition_coverage": round(condition_coverage, 4),
        "variable_coverage": round(variable_coverage, 4),
        "counter_evidence_count": len(counter),
        "page_traceability_rate": round(page_traceability, 4),
        "direct_evidence_rate": round(direct_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "failure_reasons": failures,
    }


def answer_research_question(
    question: str,
    *,
    top_k: int = 10,
    base_dir: Path = BASE_DIR,
    include_counter: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    subtasks = build_evidence_retrieval_subtasks(question)
    required_variables = identify_variables(question)

    # Warm shared immutable indexes before starting facet workers. Without
    # this, concurrent cold-cache calls can deserialize the same index three
    # times and contend on Python BM25 initialization.
    from src.cloud_evidence_bundle import cloud_bundle_required, require_cloud_bundle

    if cloud_bundle_required(base_dir):
        require_cloud_bundle(base_dir)
    else:
        version = _rag_version(base_dir)
        root_text = str(Path(base_dir).resolve())
        paths = rag_paths(base_dir)
        if paths["bm25_cache"].exists() or paths["bm25"].exists():
            _load_bm25_cached(root_text, version)
        if all(
            paths[key].exists()
            for key in ("vector_model", "vector_embeddings", "vector_ids")
        ):
            _load_vector_cached(root_text, version)

    def execute_subtask(subtask: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Any]]:
        search_query = subtask["search_query"]
        return subtask, retrieve_research_evidence(
            search_query,
            required_variables=identify_variables(subtask["claim_query"]),
            top_k=max(20, top_k * 2),
            base_dir=base_dir,
        )

    with ThreadPoolExecutor(max_workers=min(3, len(subtasks))) as executor:
        subtask_results = list(executor.map(execute_subtask, subtasks))

    pooled_by_id: Dict[str, Dict[str, Any]] = {}
    role_audits: list[dict[str, Any]] = []
    verifier_started = time.perf_counter()
    for subtask, result in subtask_results:
        for source in result["results"]:
            item = dict(source)
            item["retrieval_subtask"] = subtask["name"]
            assessment = classify_evidence_for_claim(
                subtask["claim_query"], item, query_frame=parse_query_frame(question).as_dict()
            )
            item["verified_evidence_role"] = assessment.role
            item["evidence_role_reason"] = assessment.reason
            item["claim_concept_coverage"] = assessment.concept_coverage
            role_audits.append({
                "subtask": subtask["name"],
                "claim_query": subtask["claim_query"],
                **assessment.as_dict(),
            })
            item = score_evidence(item, question)
            doc_id = str(item.get("doc_id") or "")
            previous = pooled_by_id.get(doc_id)
            if previous is None or float(item.get("final_evidence_score") or 0) > float(previous.get("final_evidence_score") or 0):
                pooled_by_id[doc_id] = item
            elif previous:
                previous.setdefault("matched_retrieval_subtasks", []).append(subtask["name"])
    # Preserve stable retrieval order in the public diagnostics contract.
    # Budget selection independently ranks this pool by final evidence score.
    pool = list(pooled_by_id.values())
    verifier_elapsed = round(time.perf_counter() - verifier_started, 6)

    def selected(role: str, limit: int = top_k) -> List[Dict[str, Any]]:
        rows = [row for row in pool if row.get("verified_evidence_role") == role]
        # Keep at least one strong row from each retrieval facet before filling by rank.
        output: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for subtask in subtasks:
            candidate = next((row for row in rows if row.get("retrieval_subtask") == subtask["name"]), None)
            if candidate and str(candidate.get("doc_id") or "") not in seen_ids:
                output.append(candidate)
                seen_ids.add(str(candidate.get("doc_id") or ""))
        output.extend(row for row in rows if str(row.get("doc_id") or "") not in seen_ids)
        return output[:limit]

    weight_config = load_weight_config(base_dir)
    budgeted = select_evidence_budget(pool, question=question, config=weight_config)
    selected_pool = budgeted["_selected"]
    supporting = [row for row in selected_pool if row.get("verified_evidence_role") == DIRECT_SUPPORT]
    supporting_context = [row for row in selected_pool if row.get("verified_evidence_role") == SUPPORTING_CONTEXT]
    counter = [row for row in selected_pool if row.get("verified_evidence_role") == DIRECT_COUNTER] if include_counter else []
    conditional = [row for row in selected_pool if row.get("verified_evidence_role") == CONDITION_DEPENDENT]
    alternative = [row for row in selected_pool if row.get("verified_evidence_role") == ALTERNATIVE_MECHANISM]
    limitations = [row for row in selected_pool if row.get("verified_evidence_role") == LIMITATION_EVIDENCE]
    review_background = [row for row in selected_pool if row.get("verified_evidence_role") == REVIEW_BACKGROUND]
    base = max(
        (result for _, result in subtask_results),
        key=lambda result: len(result.get("results") or []),
    )
    sufficiency = evaluate_evidence_sufficiency(
        supporting,
        counter,
        base["required_variables"],
        duplicate_removed=base["duplicate_removed"],
    )
    return {
        "question": question,
        "retrieved_papers": sorted(
            {
                row["paper_id"]
                for row in supporting + counter + conditional + alternative + supporting_context
                if row.get("paper_id")
            }
        ),
        "supporting_evidence": supporting,
        "counter_evidence": counter,
        "condition_dependent_evidence": conditional,
        "alternative_mechanism_evidence": alternative,
        "limitation_evidence": limitations,
        "review_background_evidence": review_background,
        "supporting_context_evidence": supporting_context,
        "retrieved_evidence_pool": pool,
        "selected_evidence_bundle": selected_pool,
        "evidence_budget": {
            "requested_total": int(weight_config["budget"]["total"]),
            "selected_total": len(selected_pool),
            "single_paper_max_fraction": float(weight_config["budget"]["max_single_paper_fraction"]),
            "role_counts": dict(Counter(row.get("verified_evidence_role") for row in selected_pool)),
        },
        "identified_topics": identify_topics(question),
        "insufficient": sufficiency["status"] == "INSUFFICIENT",
        "duplicate_removed": base["duplicate_removed"],
        "page_traceability_rate": sufficiency["page_traceability_rate"],
        "evidence_sufficiency": sufficiency,
        "whether_oa_topup_triggered": False,
        "retrieval_diagnostics": {
            key: base[key]
            for key in (
                "bm25_executed",
                "vector_executed",
                "metadata_filter_executed",
                "reranker_executed",
                "elapsed_seconds",
                "errors",
                "degraded_mode",
                "retrieved_paper_count",
            )
        },
        "retrieval_subtasks": subtasks,
        "claim_evidence_role_audit": role_audits,
        "evidence_role_counts": dict(Counter(row.get("verified_evidence_role") for row in pool)),
        "claim_evidence_verifier_seconds": verifier_elapsed,
        "retrieval_subtask_metrics": [
            {
                "name": subtask["name"],
                "elapsed_seconds": result.get("elapsed_seconds"),
                "retrieved_evidence_count": len(result.get("results") or []),
                "retrieved_paper_count": result.get("retrieved_paper_count"),
                "cache_hit": result.get("cache_hit", False),
            }
            for subtask, result in subtask_results
        ],
        "retrieval_phase_seconds": dict(base.get("phase_seconds") or {}),
        "first_evidence_ready_seconds": round(time.perf_counter() - started, 6),
    }
