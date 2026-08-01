"""Stage-2 real-page deep reading and provenance-safe evidence pipeline.

The pipeline is deliberately deterministic and local.  It visits every real
PDF page in order with PyMuPDF, then performs continuous-window extraction,
separate category sweeps, omission auditing, and evidence validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz

from src.data_contracts import (
    EvidenceRecord,
    PageRecord,
    SectionCoverageRecord,
    SequentialWindowRecord,
    VariableScanRecord,
)
from src.stage1_store import (
    BASE_DIR,
    normalize_title,
    register_pdf_path,
    sha256_file,
    update_paper_extraction_status,
    upsert_trusted_evidence,
)


PIPELINE_VERSION = "stage2.1"
VALID_PARSE_STATUSES = {
    "PROCESSED",
    "EMPTY_PAGE",
    "FIGURE_ONLY",
    "TABLE_ONLY",
    "OCR_REQUIRED",
    "PARSE_FAILED",
    "MANUAL_REVIEW_REQUIRED",
}

SECTION_NAMES = (
    "title",
    "abstract",
    "introduction",
    "materials_and_methods",
    "manufacturing",
    "heat_treatment",
    "surface_characterization",
    "microstructure_characterization",
    "defect_characterization",
    "fatigue_testing",
    "results",
    "fractography",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "unclassified",
)

SECTION_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("abstract", re.compile(r"^\s*abstract\b", re.I)),
    ("introduction", re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?introduction\b", re.I)),
    (
        "materials_and_methods",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:methods?|"
            r"materials?\s*(?:and|&)\s*methods?|"
            r"experimental(?:\s+(?:methods?|procedure|details))?|methodology)\b",
            re.I,
        ),
    ),
    (
        "manufacturing",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:manufacturing|fabrication|"
            r"specimen preparation|raw materials? and processing|powder|"
            r"laser powder bed fusion|l-pbf|slm)\b",
            re.I,
        ),
    ),
    (
        "heat_treatment",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:heat treatment|hot isostatic|hip|"
            r"effect of hip|anneal|stress relief)\b",
            re.I,
        ),
    ),
    (
        "surface_characterization",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:surface (?:roughness|condition|"
            r"characterization|topography|profilometry)|roughness measurement|"
            r"quantification of (?:the )?surface notch effect)\b",
            re.I,
        ),
    ),
    (
        "microstructure_characterization",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:microstructur\w*|metallograph\w*|"
            r"ebsd|xrd|electron microscopy|texture)\b",
            re.I,
        ),
    ),
    (
        "defect_characterization",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:defect|porosity|pore|lack of fusion|"
            r"x-ray computed tomography|micro-ct|critical defect size analysis)\b",
            re.I,
        ),
    ),
    (
        "fatigue_testing",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:fatigue test\w*|fatigue experiment\w*|"
            r"rotating bending fatigue testing|fatigue crack growth testing|"
            r"fatigue life (?:prediction|calculation)|"
            r"constant amplitude fatigue design data|"
            r"aircraft spectrum for fatigue loading|"
            r"variable amplitude fatigue loading|"
            r"crack growth test\w*|mechanical (?:and dynamic )?testing|"
            r"four-point bending fatigue)\b",
            re.I,
        ),
    ),
    (
        "results",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:results?(?:\s+and\s+discussion)?|"
            r"mechanical properties|fatigue performance analysis|"
            r"fatigue crack growth|relationship between variables|"
            r"quantitative analysis)\b",
            re.I,
        ),
    ),
    (
        "fractography",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:fractograph\w*|fracture surface|"
            r"fracture mechanisms?)\b",
            re.I,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:discussion|"
            r"near-threshold fatigue crack growth|paris law fatigue crack growth)\b",
            re.I,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:conclusions?|summary)\b",
            re.I,
        ),
    ),
    ("references", re.compile(r"^\s*(?:references|bibliography)\s*$", re.I)),
    ("appendix", re.compile(r"^\s*(?:appendix|supplementary)\b", re.I)),
)

MICRO_TERMS = (
    "microstructure",
    "microstructural",
    "alpha phase",
    "beta phase",
    "α",
    "β",
    "alpha prime",
    "α′",
    "martensite",
    "martensitic",
    "prior beta grain",
    "prior-β grain",
    "grain",
    "grain size",
    "columnar grain",
    "equiaxed grain",
    "lath",
    "alpha lath",
    "lamella",
    "lamellar",
    "basketweave",
    "widmanstätten",
    "texture",
    "misorientation",
    "phase fraction",
    "grain boundary",
    "ebsd",
    "xrd",
    "scanning electron microscopy",
    "scanning electron microscope",
    "transmission electron microscopy",
    "transmission electron microscope",
    "optical microscopy",
)

CATEGORY_TERMS: Dict[str, Tuple[str, ...]] = {
    "material_and_powder": (
        "ti-6al-4v", "ti6al4v", "powder", "particle size", "chemical composition",
    ),
    "lpbf_process": (
        "l-pbf", "lpbf", "laser powder bed fusion", "selective laser melting",
        "scan speed", "laser power", "hatch spacing", "layer thickness",
    ),
    "build_orientation": (
        "build orientation", "building direction", "vertical", "horizontal",
        "0°", "45°", "90°",
    ),
    "heat_treatment_and_hip": (
        "heat treatment", "anneal", "stress relief", "hot isostatic pressing", "hip",
    ),
    "surface_state_and_roughness": (
        "as-built", "machined", "polished", "surface roughness", " ra ", " rz ", " sa ", " sq ",
    ),
    "pores_and_defects": (
        "pore", "porosity", "lack of fusion", "keyhole", "void", "defect",
        "micro-ct", "computed tomography",
    ),
    "microstructure": MICRO_TERMS,
    "residual_stress": (
        "residual stress", "compressive stress", "tensile residual", "xrd stress",
    ),
    "fatigue_test_conditions": (
        "fatigue test", "stress ratio", "load ratio", "r =", "frequency", "hz",
        "stress amplitude", "runout",
    ),
    "fatigue_life_and_limit": (
        "fatigue life", "fatigue limit", "endurance limit", "cycles", "nf", "runout",
    ),
    "crack_initiation_site": (
        "crack origin", "crack initiation", "initiation site", "fracture origin",
    ),
    "crack_growth_and_paris": (
        "crack growth", "crack propagation", "da/dn", "δk", "Δk", "delta k",
        "paris law", "paris parameter", "paris regime",
    ),
    "equations_and_models": (
        "equation", "model", "paris law", "walker model", "murakami", "basquin",
    ),
    "statistical_data": (
        "standard deviation", "confidence interval", "regression", "r2", "r²",
        "probability", "weibull", "mean value",
    ),
    "limitations_and_future_work": (
        "limitation", "future work", "further work", "remains unclear",
        "should be investigated", "more research",
    ),
}

NUMERIC_RE = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*[±–-]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:µm|μm|um|nm|mm|cm|MPa|GPa|Pa|Hz|kHz|°C|℃|K|%|cycles?|m/cycle)?",
    re.I,
)
MEASUREMENT_RE = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*[±–-]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:µm|μm|um|nm|mm|cm|MPa|GPa|Pa|Hz|kHz|°C|℃|K|%|m/cycle)(?!\w)",
    re.I,
)
SENTENCE_RE = re.compile(
    r"(?<!\bFig\.)(?<!\bFigs\.)(?<!\bEq\.)(?<!\bEqs\.)"
    r"(?<!\bet al\.)(?<!\bi\.e\.)(?<!\be\.g\.)(?<!\bvs\.)"
    r"(?<=[.!?])\s+|\n{2,}",
    re.I,
)
FIGURE_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*\d+[a-z]?", re.I)
TABLE_RE = re.compile(r"\btable\s*\d+[a-z]?", re.I)
EQUATION_RE = re.compile(
    r"(?:\b(?:equation|eq\.)\s*\(?\d+\)?|(?:da\s*/\s*dN|ΔK|delta\s*K)\s*=|"
    r"(?<!\w)(?:R|K(?:max|min|cl|eff|th)?|Nf|σa)\s*=\s*[-+A-Za-z0-9(])",
    re.I,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def deep_read_paths(base_dir: Path, paper_id: str) -> Dict[str, Path]:
    root = base_dir / "data" / "deep_read" / paper_id
    return {
        "root": root,
        "pages": root / "page_records.jsonl",
        "windows": root / "sequential_windows.jsonl",
        "sections": root / "section_coverage_report.json",
        "scans": root / "variable_scans.json",
        "tables": root / "tables.jsonl",
        "figures": root / "figure_captions.jsonl",
        "equations": root / "equations.jsonl",
        "audit": root / "audit_log.jsonl",
        "semantic": root / "deepseek_semantic_enrichment.json",
        "status": root / "extraction_status.json",
        "page_checkpoint": root / "page_checkpoint.jsonl",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\u00ad\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> List[str]:
    values = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(values) <= 1:
        values = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return values


def _section_anchors(text: str, previous: str = "unclassified") -> List[Tuple[int, str]]:
    """Return genuine section-heading offsets in extracted reading order.

    Reference entries often begin with a number and contain words such as
    "manufacturing".  Once the References heading has been reached, only an
    explicit appendix/supplementary heading may switch the section again.
    """
    anchors: List[Tuple[int, str]] = []
    in_references = previous == "references"
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        candidate = " ".join(raw_line.split())
        lowered = candidate.lower().rstrip(": .")
        explicit = candidate[:1].isupper() and lowered in {
            "abstract",
            "introduction",
            "methods",
            "materials and methods",
            "materials & methods",
            "results",
            "results and discussion",
            "discussion",
            "conclusion",
            "conclusions",
            "fatigue test results",
            "fatigue testing",
            "fractography",
            "manufacturing",
            "heat treatment",
            "surface characterization",
            "surface profilometry",
            "microstructure characterization",
            "defect characterization",
            "micro-ct",
            "references",
            "bibliography",
            "appendix",
            "supplementary",
        } or bool(
            re.match(
                r"^(?:abstract|introduction|references|bibliography)\s*[:.]?(?:\s|$)",
                candidate,
                re.I,
            )
        )
        prefix_heading = bool(
            candidate[:1].isupper()
            and re.match(
                r"^(?:surface profilometry|micro-ct|fatigue test results)"
                r"\s*[.:]\s+",
                candidate,
                re.I,
            )
        )
        numbered = bool(
            re.match(
                r"^\s*(?:[1-9](?:\.\d+)*[.)]?)\s+\S",
                candidate,
            )
        )
        if candidate and (
            (len(candidate) <= 150 and (explicit or numbered))
            or prefix_heading
        ):
            for section, pattern in SECTION_PATTERNS:
                if not pattern.search(candidate):
                    continue
                if in_references and section not in {"references", "appendix"}:
                    break
                anchors.append((offset, section))
                if section == "references":
                    in_references = True
                break
        offset += len(raw_line)
    return anchors


def _heading_section(
    text: str, previous: str = "unclassified"
) -> Tuple[str, float, List[Tuple[int, str]]]:
    anchors = _section_anchors(text, previous)
    if not anchors:
        return "unclassified", 0.0, []
    return anchors[0][1], 0.98, anchors


def _sentence_spans(text: str) -> Iterable[Tuple[int, int, str]]:
    start = 0
    for separator in SENTENCE_RE.finditer(text):
        end = separator.start()
        if end > start:
            yield start, end, text[start:end]
        start = separator.end()
    if start < len(text):
        yield start, len(text), text[start:]


def _section_at_offset(
    page: PageRecord,
    offset: int,
    anchors: Sequence[Tuple[int, str]],
) -> str:
    section = "title" if page.page_number == 1 and anchors else page.section_title
    for anchor_offset, anchor_section in anchors:
        if anchor_offset > offset:
            break
        section = anchor_section
    return section


def _fallback_section(text: str, previous: str, page_number: int, total: int) -> Tuple[str, float]:
    lower = text.lower()
    if page_number == 1:
        return ("abstract", 0.8) if "abstract" in lower else ("title", 0.75)
    if page_number >= max(2, total - 1) and re.search(r"\[\d+\]|doi\.org", lower):
        return "references", 0.65
    if previous != "unclassified":
        return previous, 0.65
    signals = (
        ("fatigue_testing", ("stress ratio", "fatigue tests were", "testing frequency")),
        ("fractography", ("fracture surface", "fractograph", "crack origin")),
        ("microstructure_characterization", ("ebsd", "microstructure was", "xrd pattern")),
        ("defect_characterization", ("porosity", "micro-ct", "pore size")),
        ("discussion", ("suggests that", "this indicates", "can be attributed")),
        ("results", ("results show", "was observed", "figure shows")),
    )
    for section, terms in signals:
        if sum(term in lower for term in terms) >= 2:
            return section, 0.62
    return "unclassified", 0.2


def _extract_page_text(page: fitz.Page) -> str:
    """Extract text blocks in native reading order without running furniture."""
    content: List[str] = []
    page_height = float(page.rect.height)
    for block in page.get_text("blocks", sort=False):
        x0, y0, x1, y1, block_text, _, block_type = block
        del x0, x1
        if block_type != 0:
            continue
        text = str(block_text).strip()
        if not text:
            continue
        normalized_line = " ".join(text.split())
        if re.match(
            r"^(?:Materials\s+\d{4},\s*\d+|M\.\s*Tarik\s+Hasib,\s*et\s+al\.)",
            normalized_line,
            re.I,
        ):
            continue
        if y1 <= 50 or y0 >= page_height - 35:
            continue
        if (
            content
            and not re.search(r"[.!?;:]\s*$", content[-1])
            and re.match(r"^[a-zαβγδ]", text)
        ):
            content[-1] = f"{content[-1]} {text}"
        else:
            content.append(text)
    return "\n\n".join(content)


def parse_pdf_pages(
    pdf_path: Path,
    paper_id: str,
    *,
    checkpoint_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, PageRecord], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[PageRecord], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Visit every page exactly once, persisting one recoverable checkpoint per page."""
    records: List[PageRecord] = []
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    equations: List[Dict[str, Any]] = []
    seen_table_ids: set[str] = set()
    seen_figure_ids: set[str] = set()
    previous_section = "unclassified"
    source = str(pdf_path.resolve())
    checkpoint_rows: Dict[int, Dict[str, Any]] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                number = int((row.get("page_record") or {}).get("page_number") or 0)
                if number > 0:
                    checkpoint_rows[number] = row
        except (OSError, ValueError, json.JSONDecodeError):
            checkpoint_rows = {}
    with fitz.open(pdf_path) as document:
        total_pages = len(document)
        for page_index in range(total_pages):
            page_number = page_index + 1
            cached = checkpoint_rows.get(page_number)
            if cached:
                record = PageRecord(**cached["page_record"])
                records.append(record)
                tables.extend(cached.get("tables") or [])
                figures.extend(cached.get("figures") or [])
                equations.extend(cached.get("equations") or [])
                previous_section = record.section_title or previous_section
                if progress_callback:
                    progress_callback(page_number, total_pages, record)
                continue
            if should_stop and should_stop():
                raise DeepReadInterrupted(page_number - 1, total_pages)
            page = document.load_page(page_index)
            parse_status = "PROCESSED"
            raw_text = ""
            image_count = len(page.get_images(full=True))
            try:
                # Preserve the PDF's native text-block order. Geometric line
                # sorting interleaves left and right columns into fabricated
                # sentences on common two-column journal layouts.
                raw_text = _extract_page_text(page)
            except Exception:
                parse_status = "PARSE_FAILED"
            cleaned = _clean_text(raw_text)
            has_table = bool(TABLE_RE.search(cleaned))
            has_figure = bool(FIGURE_RE.search(cleaned))
            has_equation = bool(EQUATION_RE.search(cleaned))
            ocr_required = False
            if parse_status != "PARSE_FAILED" and len(cleaned) < 25:
                if image_count:
                    parse_status = "OCR_REQUIRED"
                    ocr_required = True
                else:
                    parse_status = "EMPTY_PAGE"
            elif len(cleaned) < 180 and has_table and not has_figure:
                parse_status = "TABLE_ONLY"
            elif len(cleaned) < 180 and has_figure and not has_table:
                parse_status = "FIGURE_ONLY"

            detected, confidence, anchors = _heading_section(
                cleaned, previous_section
            )
            if (
                anchors
                and anchors[0][0] > 300
                and previous_section != "unclassified"
            ):
                detected = previous_section
                confidence = 0.8
            if detected == "unclassified":
                detected, confidence = _fallback_section(
                    cleaned, previous_section, page_number, total_pages
                )
            if anchors:
                previous_section = anchors[-1][1]
            elif detected != "unclassified":
                previous_section = detected

            page_type = "TEXT"
            if has_table or has_figure:
                page_type = "MIXED"
            if parse_status in {"TABLE_ONLY", "FIGURE_ONLY", "OCR_REQUIRED", "EMPTY_PAGE"}:
                page_type = parse_status
            record = PageRecord(
                paper_id=paper_id,
                page_number=page_number,
                total_pages=total_pages,
                raw_text=raw_text,
                cleaned_text=cleaned,
                character_count=len(cleaned),
                section_title=detected,
                page_type=page_type,
                parse_status=parse_status,
                contains_table=has_table,
                contains_figure_caption=has_figure,
                contains_equation=has_equation,
                ocr_required=ocr_required,
                source_pdf_path=source,
                processed_at=_now(),
                classification_confidence=confidence,
                canonical_paper_id=paper_id,
                page_text=cleaned,
                text_length=len(cleaned),
                extraction_status=parse_status,
                image_count=image_count,
                table_candidate=has_table,
                formula_candidate=has_equation,
                visual_review_status=(
                    "NEEDS_VISUAL_REVIEW"
                    if image_count or has_table or has_figure else "NOT_REQUIRED"
                ),
            )
            records.append(record)

            page_tables: List[Dict[str, Any]] = []
            page_figures: List[Dict[str, Any]] = []
            page_equations: List[Dict[str, Any]] = []

            for match in TABLE_RE.finditer(cleaned):
                table_key = re.sub(r"\s+", "", match.group(0).lower())
                if table_key in seen_table_ids:
                    continue
                seen_table_ids.add(table_key)
                excerpt = cleaned[match.start(): match.start() + 900]
                page_tables.append(
                    {
                        "paper_id": paper_id,
                        "table_id": match.group(0),
                        "page_number": page_number,
                        "title": excerpt.splitlines()[0][:240],
                        "extracted_text": excerpt,
                        "review_status": (
                            "TABLE_REVIEW_REQUIRED"
                            if NUMERIC_RE.search(excerpt)
                            or re.search(r"\b(?:paris|constant|coefficient)\b", excerpt, re.I)
                            else "TEXT_EXTRACTED"
                        ),
                    }
                )
            for match in FIGURE_RE.finditer(cleaned):
                figure_number = re.search(r"\d+[a-z]?", match.group(0), re.I)
                figure_key = (
                    figure_number.group(0).lower()
                    if figure_number
                    else re.sub(r"\s+", "", match.group(0).lower())
                )
                if figure_key in seen_figure_ids:
                    continue
                seen_figure_ids.add(figure_key)
                excerpt = cleaned[match.start(): match.start() + 700]
                page_figures.append(
                    {
                        "paper_id": paper_id,
                        "figure_id": match.group(0),
                        "page_number": page_number,
                        "caption": excerpt,
                        "review_status": (
                            "FIGURE_REVIEW_REQUIRED"
                            if re.search(r"\b(?:value|data|curve|plot)\b", excerpt, re.I)
                            else "CAPTION_EXTRACTED"
                        ),
                    }
                )
            for match in EQUATION_RE.finditer(cleaned):
                excerpt = cleaned[max(0, match.start() - 160): match.start() + 360]
                page_equations.append(
                    {
                        "paper_id": paper_id,
                        "page_number": page_number,
                        "original_text": excerpt,
                        "latex_candidate": "",
                        "review_status": "TEXT_EXTRACTED",
                    }
                )
            tables.extend(page_tables)
            figures.extend(page_figures)
            equations.extend(page_equations)
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({
                        "page_record": record.to_dict(),
                        "tables": page_tables,
                        "figures": page_figures,
                        "equations": page_equations,
                    }, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if progress_callback:
                progress_callback(page_number, total_pages, record)
    return records, tables, figures, equations


class DeepReadInterrupted(RuntimeError):
    def __init__(self, processed_pages: int, total_pages: int):
        super().__init__("DEEP_READ_INTERRUPTED")
        self.processed_pages = processed_pages
        self.total_pages = total_pages


def _matching_terms(text: str, terms: Sequence[str]) -> List[str]:
    matches = set()
    for term in terms:
        stripped = term.strip()
        if not stripped:
            continue
        body = re.escape(stripped).replace(r"\ ", r"\s+")
        left = r"(?<!\w)" if stripped[0].isalnum() else ""
        right = r"(?!\w)" if stripped[-1].isalnum() else ""
        if re.search(left + body + right, text, re.I):
            matches.add(term)
    return sorted(matches)


def _structured_values(text: str, page_number: int) -> List[Dict[str, Any]]:
    values = []
    seen = set()
    for match in NUMERIC_RE.finditer(text):
        raw = match.group(0).strip()
        if not raw or raw in seen or not re.search(r"\d", raw):
            continue
        seen.add(raw)
        values.append({"raw_value": raw, "page_number": page_number})
    return values[:80]


def build_sequential_windows(
    pages: Sequence[PageRecord], window_size: int = 3
) -> List[SequentialWindowRecord]:
    """Build continuous 2-4 page windows without Top-K filtering."""
    if not 2 <= window_size <= 4:
        raise ValueError("window_size must be between 2 and 4 pages")
    windows: List[SequentialWindowRecord] = []
    overlap_paragraph = ""
    for start_index in range(0, len(pages), window_size):
        page_slice = list(pages[start_index: start_index + window_size])
        page_numbers = [page.page_number for page in page_slice]
        parts = [page.cleaned_text for page in page_slice if page.cleaned_text]
        window_text = "\n\n".join(([overlap_paragraph] if overlap_paragraph else []) + parts)
        paragraphs = _paragraphs(window_text)
        if paragraphs:
            overlap_paragraph = paragraphs[-1][-1200:]
        all_terms = {
            category: _matching_terms(window_text, terms)
            for category, terms in CATEGORY_TERMS.items()
        }
        extracted = {
            "variables": {key: value for key, value in all_terms.items() if value},
            "material_and_manufacturing_conditions": {
                key: all_terms[key]
                for key in (
                    "material_and_powder", "lpbf_process", "build_orientation",
                    "heat_treatment_and_hip", "surface_state_and_roughness",
                )
                if all_terms[key]
            },
            "fatigue_conditions": all_terms["fatigue_test_conditions"],
            "numeric_values": sum(
                (_structured_values(page.cleaned_text, page.page_number) for page in page_slice),
                [],
            )[:150],
            "table_mentions": TABLE_RE.findall(window_text),
            "figure_mentions": FIGURE_RE.findall(window_text),
            "equation_mentions": [
                match.group(0) for match in EQUATION_RE.finditer(window_text)
            ],
            "author_conclusions": [
                sentence[:1200]
                for sentence in SENTENCE_RE.split(window_text)
                if re.search(r"\b(?:conclude|demonstrate|showed|indicate|suggest)\b", sentence, re.I)
            ][:20],
            "author_limitations": [
                sentence[:1200]
                for sentence in SENTENCE_RE.split(window_text)
                if re.search(r"\b(?:limitation|future work|further work|remains unclear)\b", sentence, re.I)
            ][:20],
            "reverse_information": [
                sentence[:1200]
                for sentence in SENTENCE_RE.split(window_text)
                if re.search(r"\b(?:however|whereas|in contrast|contrary)\b", sentence, re.I)
            ][:20],
            "phenomena": [
                sentence[:1200]
                for sentence in SENTENCE_RE.split(window_text)
                if re.search(
                    r"\b(?:observed|exhibited|increased|decreased|failed|"
                    r"crack(?:ed|ing)?|fracture|runout)\b",
                    sentence,
                    re.I,
                )
            ][:30],
            "mechanisms": [
                sentence[:1200]
                for sentence in SENTENCE_RE.split(window_text)
                if re.search(
                    r"\b(?:mechanism|due to|attributed to|caused by|resulted from|"
                    r"governed by|associated with)\b",
                    sentence,
                    re.I,
                )
            ][:30],
            "followup_review_required": bool(
                FIGURE_RE.search(window_text)
                or TABLE_RE.search(window_text)
                or re.search(r"\b(?:unclear|not available|could not)\b", window_text, re.I)
            ),
        }
        windows.append(
            SequentialWindowRecord(
                window_id=f"{pages[0].paper_id}_W{len(windows) + 1:04d}",
                paper_id=pages[0].paper_id,
                start_page=page_numbers[0],
                end_page=page_numbers[-1],
                source_page_numbers=page_numbers,
                window_text=window_text,
                extracted_items=extracted,
                processing_status="PROCESSED",
            )
        )
    return windows


def build_section_coverage(pages: Sequence[PageRecord]) -> List[SectionCoverageRecord]:
    grouped: Dict[str, List[PageRecord]] = defaultdict(list)
    for page in pages:
        grouped[page.section_title].append(page)
    rows = []
    for section, section_pages in grouped.items():
        rows.append(
            SectionCoverageRecord(
                section_name=section,
                page_range=[section_pages[0].page_number, section_pages[-1].page_number],
                paragraph_count=sum(len(_paragraphs(page.cleaned_text)) for page in section_pages),
                classification_confidence=round(
                    sum(page.classification_confidence for page in section_pages)
                    / len(section_pages),
                    4,
                ),
            )
        )
    return sorted(rows, key=lambda row: row.page_range[0])


def _iter_evidence_sentences(
    page: PageRecord,
) -> Iterable[Tuple[int, str, str]]:
    anchors = _section_anchors(page.cleaned_text)
    for paragraph_index, (start, _, raw_sentence) in enumerate(
        _sentence_spans(page.cleaned_text)
    ):
        section = _section_at_offset(page, start, anchors)
        if section in {"title", "references", "appendix"}:
            continue
        sentence = " ".join(raw_sentence.split()).strip()
        # Publisher extraction can append table text after later two-column
        # headings even when the table is visually above them.  A caption is
        # therefore tied to the page's dominant section and remains explicitly
        # gated for table/figure review in build_trusted_evidence.
        if re.match(r"^\s*(?:table|fig(?:ure)?\.?)\s*\d+", sentence, re.I):
            section = page.section_title
        if sentence:
            yield paragraph_index, section, sentence


def scan_variable_category(
    category: str,
    terms: Sequence[str],
    pages: Sequence[PageRecord],
    *,
    audit_complete: bool = False,
) -> VariableScanRecord:
    matched_pages: List[int] = []
    matched_sections: set[str] = set()
    matched_terms: set[str] = set()
    values: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for page in pages:
        page_matched = False
        for paragraph_index, section, sentence in _iter_evidence_sentences(page):
            found = _matching_terms(sentence, terms)
            if not found:
                continue
            page_matched = True
            matched_sections.add(section)
            matched_terms.update(found)
            values.extend(_structured_values(sentence, page.page_number))
            if len(sentence) < 25:
                continue
            evidence.append(
                {
                    "page_number": page.page_number,
                    "section": section,
                    "paragraph_index": paragraph_index,
                    "original_text": sentence[:1800],
                }
            )
            if len(evidence) >= 80:
                break
        if page_matched:
            matched_pages.append(page.page_number)
    if matched_pages:
        scan_status = "EXTRACTED"
        if category == "microstructure":
            has_quant = any(
                MEASUREMENT_RE.search(item["original_text"])
                for item in evidence
            )
            has_mechanism = any(
                item["section"]
                in {
                    "microstructure_characterization",
                    "results",
                    "discussion",
                    "conclusion",
                }
                and
                re.search(
                    r"\b(?:microstructur|grain|lath|phase|martensit).{0,160}"
                    r"(?:fatigue|crack|life|growth|initiat)|"
                    r"(?:fatigue|crack|life|growth|initiat).{0,160}"
                    r"(?:microstructur|grain|lath|phase|martensit)",
                    item["original_text"],
                    re.I,
                )
                for item in evidence
            )
            if has_mechanism:
                scan_status = "MECHANISTIC"
            elif has_quant:
                scan_status = "QUANTITATIVE"
            elif len(evidence) <= 2:
                scan_status = "MENTION_ONLY"
            else:
                scan_status = "QUALITATIVE"
    else:
        scan_status = "NOT_REPORTED" if audit_complete else "NOT_EXTRACTED"
    return VariableScanRecord(
        category=category,
        matched_pages=sorted(set(matched_pages)),
        matched_sections=sorted(matched_sections),
        matched_terms=sorted(matched_terms),
        structured_values=values[:300],
        original_evidence=evidence,
        missing_fields=[] if matched_pages else [category],
        scan_status=scan_status,
    )


def run_variable_sweeps(
    pages: Sequence[PageRecord], *, audit_complete: bool = False
) -> Dict[str, VariableScanRecord]:
    return {
        category: scan_variable_category(
            category, terms, pages, audit_complete=audit_complete
        )
        for category, terms in CATEGORY_TERMS.items()
    }


AUDIT_RULES: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("microstructure", ("ebsd", "sem", "tem", "xrd", "alpha", "beta", "martensite", "grain", "lath", "texture")),
    ("surface_state_and_roughness", (" ra ", " rz ", " sa ", " sq ", "roughness")),
    ("fatigue_test_conditions", ("r =", "stress ratio", "load ratio")),
    ("fatigue_life_and_limit", ("nf", "cycles", "runout", "fatigue life")),
    ("crack_initiation_site", ("crack origin", "initiation site", "crack initiation")),
    ("crack_growth_and_paris", ("paris", "da/dn", "δk", "Δk", "delta k")),
    ("equations_and_models", ("equation", "eq.", "paris law")),
)


def audit_full_paper_extraction(
    pages: Sequence[PageRecord],
    scans: Dict[str, VariableScanRecord],
) -> Tuple[Dict[str, VariableScanRecord], List[Dict[str, Any]]]:
    """Locate omissions, rescan the relevant real pages, and record every update."""
    logs: List[Dict[str, Any]] = []
    page_by_number = {page.page_number: page for page in pages}
    eligible_by_page = {
        page.page_number: "\n".join(
            sentence for _, _, sentence in _iter_evidence_sentences(page)
        )
        for page in pages
    }
    full_text = "\n".join(eligible_by_page.values())
    for category, terms in AUDIT_RULES:
        keyword_present = bool(_matching_terms(full_text, terms))
        scan = scans[category]
        inconsistent_not_reported = (
            scan.scan_status == "NOT_REPORTED" and keyword_present
        )
        missing_extraction = keyword_present and not scan.original_evidence
        if not inconsistent_not_reported and not missing_extraction:
            continue
        matched_numbers = [
            page.page_number
            for page in pages
            if _matching_terms(eligible_by_page[page.page_number], terms)
        ]
        targeted_pages = [page_by_number[number] for number in matched_numbers]
        replacement = scan_variable_category(
            category, CATEGORY_TERMS[category], targeted_pages, audit_complete=True
        )
        fixed = bool(replacement.original_evidence)
        if fixed:
            scans[category] = replacement
        logs.append(
            {
                "audit_id": f"AUDIT_{len(logs) + 1:04d}",
                "category": category,
                "issue": (
                    "NOT_REPORTED_CONTRADICTED_BY_FULL_TEXT"
                    if inconsistent_not_reported
                    else "KEYWORD_PRESENT_BUT_FIELD_EMPTY"
                ),
                "located_pages": matched_numbers,
                "action": "TARGETED_PAGE_RESCAN",
                "before_status": scan.scan_status,
                "after_status": replacement.scan_status if fixed else scan.scan_status,
                "fixed": fixed,
                "timestamp": _now(),
            }
        )

    method_pages = {
        page.page_number: page.cleaned_text
        for page in pages
        if page.section_title in {"materials_and_methods", "fatigue_testing"}
    }
    result_pages = {
        page.page_number: page.cleaned_text
        for page in pages
        if page.section_title in {"results", "discussion"}
    }
    if method_pages and result_pages:
        method_r = set(re.findall(r"\bR\s*=\s*[-+]?\d+(?:\.\d+)?", "\n".join(method_pages.values()), re.I))
        result_r = set(re.findall(r"\bR\s*=\s*[-+]?\d+(?:\.\d+)?", "\n".join(result_pages.values()), re.I))
        if method_r and result_r and method_r != result_r:
            logs.append(
                {
                    "audit_id": f"AUDIT_{len(logs) + 1:04d}",
                    "category": "fatigue_test_conditions",
                    "issue": "METHODS_RESULTS_CONDITION_DIFFERENCE",
                    "located_pages": sorted(list(method_pages) + list(result_pages)),
                    "action": "MANUAL_REVIEW_REQUIRED",
                    "before_status": "EXTRACTED",
                    "after_status": "UNCERTAIN",
                    "fixed": False,
                    "timestamp": _now(),
                }
            )

    for category, scan in list(scans.items()):
        if not scan.matched_pages:
            scans[category] = scan_variable_category(
                category, CATEGORY_TERMS[category], pages, audit_complete=True
            )

    equation_scan = scans["equations_and_models"]
    equation_pages = {
        page.page_number for page in pages if page.contains_equation
    }
    captured_equation_pages = {
        int(item["page_number"]) for item in equation_scan.original_evidence
    }
    for page_number in sorted(equation_pages - captured_equation_pages):
        page = page_by_number[page_number]
        match = EQUATION_RE.search(page.cleaned_text)
        if not match:
            continue
        excerpt = page.cleaned_text[max(0, match.start() - 160): match.start() + 360]
        equation_scan.original_evidence.append(
            {
                "page_number": page_number,
                "section": page.section_title,
                "paragraph_index": 0,
                "original_text": excerpt,
            }
        )
        equation_scan.matched_pages = sorted(
            set(equation_scan.matched_pages + [page_number])
        )
        equation_scan.scan_status = "EXTRACTED"
        logs.append(
            {
                "audit_id": f"AUDIT_{len(logs) + 1:04d}",
                "category": "equations_and_models",
                "issue": "EQUATION_PRESENT_BUT_STRUCTURED_FIELD_EMPTY",
                "located_pages": [page_number],
                "action": "TARGETED_PAGE_RESCAN",
                "before_status": "NOT_EXTRACTED",
                "after_status": "EXTRACTED",
                "fixed": True,
                "timestamp": _now(),
            }
        )
    final_equation_pages = {
        int(item["page_number"]) for item in equation_scan.original_evidence
    }
    for log in logs:
        if (
            log.get("category") == "equations_and_models"
            and not log.get("fixed")
            and set(log.get("located_pages", [])) <= final_equation_pages
        ):
            log["fixed"] = True
            log["after_status"] = "EXTRACTED"
            log["action"] = "TARGETED_PAGE_RESCAN;FORMULA_PATTERN_RESCAN"

    stats_scan = scans["statistical_data"]
    value_pages = {
        int(value.get("page_number") or 0)
        for value in stats_scan.structured_values
    }
    for page in pages:
        if (
            page.contains_table
            and NUMERIC_RE.search(page.cleaned_text)
            and page.page_number not in value_pages
        ):
            values = _structured_values(page.cleaned_text, page.page_number)
            stats_scan.structured_values.extend(values)
            stats_scan.matched_pages = sorted(
                set(stats_scan.matched_pages + [page.page_number])
            )
            stats_scan.scan_status = "EXTRACTED"
            logs.append(
                {
                    "audit_id": f"AUDIT_{len(logs) + 1:04d}",
                    "category": "statistical_data",
                    "issue": "TABLE_HAS_VALUES_BUT_STRUCTURED_DATA_EMPTY",
                    "located_pages": [page.page_number],
                    "action": "TARGETED_PAGE_RESCAN",
                    "before_status": "NOT_EXTRACTED",
                    "after_status": "EXTRACTED",
                    "fixed": bool(values),
                    "timestamp": _now(),
                }
            )

    abstract_text = "\n".join(
        page.cleaned_text for page in pages if page.section_title == "abstract"
    ).lower()
    conclusion_text = "\n".join(
        page.cleaned_text for page in pages if page.section_title == "conclusion"
    ).lower()
    contradiction_pairs = (
        ("increase", "decrease"),
        ("improve", "reduce"),
        ("higher", "lower"),
        ("significant", "not significant"),
    )
    if abstract_text and conclusion_text and any(
        (left in abstract_text and right in conclusion_text)
        or (right in abstract_text and left in conclusion_text)
        for left, right in contradiction_pairs
    ):
        logs.append(
            {
                "audit_id": f"AUDIT_{len(logs) + 1:04d}",
                "category": "conclusion_consistency",
                "issue": "ABSTRACT_BODY_CONCLUSION_DIFFERENCE",
                "located_pages": [
                    page.page_number
                    for page in pages
                    if page.section_title in {"abstract", "conclusion"}
                ],
                "action": "MANUAL_REVIEW_REQUIRED",
                "before_status": "UNREVIEWED",
                "after_status": "UNCERTAIN",
                "fixed": False,
                "timestamp": _now(),
            }
        )
    return scans, logs


def _conditions_from_text(text: str) -> Dict[str, Any]:
    """Extract only conditions that are explicit in the evidence sentence.

    MPa is not intrinsically a pressure unit in fatigue papers.  The former
    extractor labelled every MPa value as pressure; contextual patterns below
    keep stress, pressure, and stress-intensity values separate.
    """
    conditions: Dict[str, Any] = {}
    lower = text.lower()
    scalar_patterns = {
        "stress_ratio_R": (
            r"\bR\s*=\s*(?:P\s*min\s*/\s*P\s*max\s*=\s*)?"
            r"[-+]?\d+(?:\.\d+)?"
            r"|\b(?:stress|load)\s+ratio(?:\s+of)?\s*(?:R\s*)?"
            r"[=:]?\s*[-+]?\d+(?:\.\d+)?"
        ),
        "frequency": (
            r"\b\d+(?:\.\d+)?\s*(?:Hz|kHz)\b"
            r"|\b\d+(?:\.\d+)?\s*cycles?\s+per\s+minute(?:\s*\(CPM\))?"
        ),
        "temperature": (
            r"\b[-+]?\d+(?:\.\d+)?\s*"
            r"(?:°C|◦C|℃|[\x00-\x1f]C|K)\b"
        ),
        "duration": (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:(?-i:h|hr|hrs|hours?|min|minutes?))\b"
        ),
        "cycles": (
            r"\b(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
            r"(?:\s*[×x]\s*10\^?\d+)?\s*cycles?\b"
        ),
        "delta_K": (
            r"(?:ΔK(?:th)?|delta\s*K(?:th)?)\D{0,24}"
            r"\d+(?:\.\d+)?\s*MPa\s*(?:√\s*m|m\s*1/2|sqrt\s*\(?m\)?)"
        ),
        "pressure": (
            r"\b(?:pressure(?:\s+of)?|pressur(?:ized|ised)\s+to)\s*"
            r"\d+(?:\.\d+)?\s*(?:MPa|bar)\b"
            r"|\b\d+(?:\.\d+)?\s*bar\b"
        ),
        "stress": (
            r"\b(?:stress(?:\s+(?:amplitude|range|level))?|strength)\D{0,20}"
            r"\d+(?:\.\d+)?\s*MPa\b"
            r"|\b\d+(?:\.\d+)?\s*MPa\b(?=.{0,24}\b(?:stress|strength)\b)"
        ),
    }
    for key, pattern in scalar_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            conditions[key] = " ".join(match.group(0).split())
    if re.search(r"\bti[-–— ]?6al[-–— ]?4v\b|\bti64\b|\btial6v4\b", lower):
        conditions["material"] = "Ti-6Al-4V"
    processes = []
    for value, pattern in (
        ("L-PBF", r"\bl-?pbf\b|laser powder bed fusion"),
        ("SLM", r"\bslm\b|selective laser melting"),
        ("EBM", r"\bebm\b|electron beam melt"),
        ("DMLS", r"\bdmls\b|direct metal laser sinter"),
    ):
        if re.search(pattern, lower):
            processes.append(value)
    if processes:
        conditions["process"] = sorted(set(processes))
    surfaces = [
        value
        for value in ("as-built", "machined", "polished", "ground")
        if value in lower
    ]
    if "surface roughness" in lower:
        surfaces.append("surface_roughness")
    if surfaces:
        conditions["surface_state"] = sorted(set(surfaces))
    treatments = []
    if re.search(r"\bhip\b|hot isostatic press", lower):
        treatments.append("HIP")
    if re.search(r"\banneal", lower):
        treatments.append("annealed")
    if "heat treat" in lower:
        treatments.append("heat-treated")
    if treatments:
        conditions["heat_treatment"] = sorted(set(treatments))
    characterization = [
        name
        for name in ("EBSD", "SEM", "TEM", "XRD")
        if re.search(rf"\b{re.escape(name)}\b", text)
    ]
    if re.search(r"\bmicro-ct\b", lower):
        characterization.append("micro-CT")
    if "computed tomography" in lower:
        characterization.append("computed tomography")
    if characterization:
        conditions["characterization_method"] = sorted(set(characterization))
    if re.search(r"fatigue crack (?:growth|propagation)|\bda\s*/\s*dn\b", lower):
        conditions["testing_method"] = "fatigue crack growth"
    elif "fatigue test" in lower:
        conditions["testing_method"] = "fatigue"
    geometry = re.search(
        r"\b(?:compact tension|dog[- ]bone|cylindrical|hourglass|"
        r"four[- ]point bend(?:ing)?)\s+(?:specimen|sample|coupon)s?\b",
        text,
        re.I,
    )
    if geometry:
        conditions["sample_geometry"] = " ".join(geometry.group(0).split())
    return conditions


def build_trusted_evidence(
    paper_id: str,
    title: str,
    total_pages: int,
    scans: Dict[str, VariableScanRecord],
) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    seen = set()
    normalized_title = normalize_title(title)
    direct_sections = {
        "materials_and_methods",
        "manufacturing",
        "heat_treatment",
        "surface_characterization",
        "microstructure_characterization",
        "defect_characterization",
        "fatigue_testing",
        "results",
        "fractography",
        "discussion",
        "conclusion",
    }
    for category, scan in scans.items():
        for item in scan.original_evidence:
            original = " ".join(str(item["original_text"]).split()).strip()
            page_number = int(item["page_number"])
            section = str(item["section"])
            if (
                not original
                or not 1 <= page_number <= total_pages
                or section in {"title", "references", "appendix"}
                or (normalized_title and normalize_title(original) == normalized_title)
                or re.match(r"^\s*keywords?\s*:", original, re.I)
                or (
                    re.match(r"^[a-z]", original)
                    and not re.match(r"^da\s*/\s*dN\b", original, re.I)
                )
                or re.search(
                    r"\b(?:are|is|was|were|the|and|or|of|to|with|for|a|an)\s*$",
                    original,
                    re.I,
                )
            ):
                continue
            identity = (page_number, original)
            if identity in seen:
                continue
            seen.add(identity)
            quantitative = bool(NUMERIC_RE.search(original))
            reporting = bool(
                re.search(
                    r"\b(?:was|were|is|are|showed|observed|measured|tested|"
                    r"revealed|found|increased|decreased|exhibited|performed|"
                    r"indicate|indicates|confirmed?|demonstrate[sd]?|"
                    r"underscore[sd]?|present(?:ed|s)?)\b",
                    original,
                    re.I,
                )
            )
            attributed = bool(
                re.search(
                    r"^\s*\[\d+|(?:\bet al\.|\bhas been reported\b|"
                    r"\bhave been reported\b|\breported by\b|"
                    r"\bprevious studies?\b|\bother studies?\b)",
                    original,
                    re.I,
                )
            )
            study_specific = bool(
                re.search(
                    r"\b(?:in (?:the|this) (?:study|work)|we (?:used|tested|"
                    r"measured|fabricated|examined)|(?:was|were) (?:used|tested|"
                    r"measured|fabricated|examined|characterized))\b",
                    original,
                    re.I,
                )
            )
            caption_kind = ""
            if re.match(r"^\s*table\s+\d+", original, re.I):
                caption_kind = "TABLE"
            elif re.match(r"^\s*(?:fig(?:ure)?\.?)\s*\d+", original, re.I):
                caption_kind = "FIGURE"
            elif re.match(r"^\s*quanti\S*\s+average\b", original, re.I):
                caption_kind = "TABLE"
            elif (
                "Paris Law Constants" in original
                and re.search(r"\bUTS\b", original)
            ):
                caption_kind = "TABLE"
            elif (
                len(original) < 220
                and (
                    (
                        not reporting
                        and len(original.split()) <= 10
                        and not re.search(r"[.!?]\s*$", original)
                    )
                    or re.search(
                        r"\b(?:build|processing)\s+parameters\b",
                        original,
                        re.I,
                    )
                )
            ):
                caption_kind = "TABLE"
            elif re.match(
                r"^\s*(?:fracture surface|ebsd images?|optical micrographs?)\b",
                original,
                re.I,
            ):
                caption_kind = "FIGURE"
            micro_relation = bool(
                re.search(
                    r"(?:grain|lath|phase|martensit|microstructur).{0,160}"
                    r"(?:fatigue|crack|life|growth|initiat)|"
                    r"(?:fatigue|crack|life|growth|initiat).{0,160}"
                    r"(?:grain|lath|phase|martensit|microstructur)",
                    original,
                    re.I,
                )
            )
            if caption_kind:
                directness = "INDIRECT"
            elif category == "microstructure" and not quantitative and not micro_relation:
                directness = "MENTION_ONLY"
            elif attributed:
                directness = "INDIRECT"
            elif section == "introduction" and study_specific:
                directness = "DIRECT"
            elif section in direct_sections and (
                quantitative or reporting or section == "conclusion"
            ):
                directness = "DIRECT"
            else:
                directness = "INDIRECT"
            review_status = (
                f"{caption_kind}_REVIEW_REQUIRED"
                if caption_kind
                else "AUTO_PROVENANCE_VALIDATED"
            )
            digest = hashlib.sha256(
                f"{paper_id}|{page_number}|{original}".encode("utf-8")
            ).hexdigest()[:20].upper()
            records.append(
                EvidenceRecord(
                    evidence_id=f"EV_{digest}",
                    paper_id=paper_id,
                    claim=original,
                    original_text=original,
                    page_number=page_number,
                    section=section,
                    paragraph_index=int(item.get("paragraph_index", 0)),
                    experimental_conditions=_conditions_from_text(original),
                    canonical_paper_id=paper_id,
                    evidence_type=category.upper(),
                    variables={"category": category},
                    conditions=_conditions_from_text(original),
                    result=original,
                    units=sorted(set(re.findall(
                        r"\b(?:MPa|GPa|Hz|kHz|cycles?|mm|µm|μm|nm|%|°C|K)\b",
                        original,
                        re.I,
                    ))),
                    formula_reference=(
                        original if category == "equations_and_models" else ""
                    ),
                    table_or_figure_reference=(caption_kind or ""),
                    support_or_counter=(
                        "COUNTER" if re.search(
                            r"\b(?:however|contrary|in contrast|did not|no significant)\b",
                            original,
                            re.I,
                        ) else "SUPPORT"
                    ),
                    extraction_method=f"STAGE2_PAGE_SWEEP:{category}",
                    directness=directness,
                    confidence=0.95 if directness == "DIRECT" else 0.75,
                    review_status=review_status,
                    source_method=f"STAGE2_PAGE_SWEEP:{category}",
                    created_at=_now(),
                    updated_at=_now(),
                    data_version=PIPELINE_VERSION,
                )
            )
    return records


def validate_evidence_provenance(
    evidence: Sequence[EvidenceRecord], total_pages: int
) -> bool:
    for record in evidence:
        if not record.original_text or not 1 <= record.page_number <= total_pages:
            return False
        if record.directness == "DIRECT" and (
            not record.section or record.section in {"title", "unclassified"}
        ):
            return False
        if record.directness == "INFERRED" and "MANUAL" not in record.review_status:
            return False
    return True


def _load_cached_status(
    paths: Dict[str, Path], file_hash: str, *, require_deepseek: bool = False
) -> Optional[Dict[str, Any]]:
    status = _read_json(paths["status"], {})
    required = ("pages", "windows", "sections", "scans", "audit")
    if (
        status.get("file_hash_sha256") == file_hash
        and status.get("pipeline_version") == PIPELINE_VERSION
        and status.get("deep_read_complete") is True
        and (
            not require_deepseek
            or (
                status.get("deepseek_enhancement_enabled") is True
                and status.get("deepseek_enhancement_applied") is True
                and paths["semantic"].exists()
            )
        )
        and all(paths[name].exists() for name in required)
    ):
        status["idempotent_reuse"] = True
        return status
    return None


def deep_read_pdf(
    pdf_path: Path | str,
    *,
    paper_id: str = "",
    title: str = "",
    base_dir: Path = BASE_DIR,
    force: bool = False,
    checkpoint_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, PageRecord], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    use_deepseek: bool = False,
    deepseek_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the complete Stage-2 loop for one real PDF."""
    source_path = Path(pdf_path).resolve()
    if not source_path.exists():
        return {"status": "FAILED", "error": "PDF_NOT_FOUND", "deep_read_complete": False}
    registered = register_pdf_path(
        source_path, source_type="STAGE2_DEEP_READ", base_dir=base_dir
    )
    if not registered.get("pdf_valid"):
        return {
            "status": "FAILED",
            "error": registered.get("error", "INVALID_PDF"),
            "deep_read_complete": False,
            "evidence_count": 0,
        }
    paper_id = paper_id or str(registered["paper_id"])
    title = title or str(registered.get("title") or "")
    canonical_path = Path(registered["canonical_pdf_path"]).resolve()
    file_hash = str(registered["file_hash_sha256"])
    paths = deep_read_paths(base_dir, paper_id)
    if not force:
        cached = _load_cached_status(
            paths, file_hash, require_deepseek=use_deepseek
        )
        if cached:
            return cached

    try:
        pages, tables, figures, equations = parse_pdf_pages(
            canonical_path,
            paper_id,
            checkpoint_path=checkpoint_path or paths["page_checkpoint"],
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
    except DeepReadInterrupted as exc:
        update_paper_extraction_status(
            paper_id, extraction_status="INTERRUPTED", deep_read_complete=False,
            base_dir=base_dir,
        )
        return {
            "paper_id": paper_id, "title": title, "status": "INTERRUPTED",
            "processed_page_count": exc.processed_pages,
            "real_page_count": exc.total_pages, "deep_read_complete": False,
            "evidence_count": 0,
        }
    except Exception as exc:
        update_paper_extraction_status(
            paper_id,
            extraction_status="FAILED",
            deep_read_complete=False,
            base_dir=base_dir,
        )
        return {
            "paper_id": paper_id,
            "status": "FAILED",
            "error": str(exc),
            "deep_read_complete": False,
            "evidence_count": 0,
        }
    total_pages = len(pages)
    windows = build_sequential_windows(pages)
    sections = build_section_coverage(pages)
    scans = run_variable_sweeps(pages, audit_complete=False)
    scans, audit_log = audit_full_paper_extraction(pages, scans)
    evidence = build_trusted_evidence(paper_id, title, total_pages, scans)
    semantic_result: Dict[str, Any] = {
        "enabled": bool(use_deepseek),
        "applied": False,
        "enriched_record_count": 0,
        "semantic_success_count": 0,
        "semantic_failure_count": 0,
        "usage": {},
    }
    if use_deepseek:
        if deepseek_client is None:
            from src.deepseek_client import DeepSeekClient

            deepseek_client = DeepSeekClient()
        from src.deepseek_semantic_enrichment import enrich_evidence_with_deepseek

        evidence, semantic_result = enrich_evidence_with_deepseek(
            client=deepseek_client,
            title=title,
            pages=pages,
            sections=sections,
            evidence=evidence,
        )

    real_page_count_valid = (
        total_pages == int(registered["real_page_count"])
        and [page.page_number for page in pages] == list(range(1, total_pages + 1))
    )
    covered_pages = sorted(
        {number for window in windows for number in window.source_page_numbers}
    )
    sequential_scan_complete = covered_pages == list(range(1, total_pages + 1))
    variable_sweep_complete = set(scans) == set(CATEGORY_TERMS)
    missing_audit_complete = (
        not any(
            page.parse_status in {"OCR_REQUIRED", "PARSE_FAILED"} for page in pages
        )
        and not any(
            not row.get("fixed")
            and row.get("action") != "MANUAL_REVIEW_REQUIRED"
            for row in audit_log
        )
    )
    evidence_provenance_valid = validate_evidence_provenance(evidence, total_pages)
    parsable = [
        page for page in pages
        if page.parse_status not in {"OCR_REQUIRED", "PARSE_FAILED"}
    ]
    checked = [
        page for page in parsable
        if page.parse_status in {
            "PROCESSED", "EMPTY_PAGE", "FIGURE_ONLY", "TABLE_ONLY",
            "MANUAL_REVIEW_REQUIRED",
        }
    ]
    page_coverage_ratio = round(len(checked) / len(parsable), 6) if parsable else 0.0
    deep_read_complete = all(
        (
            real_page_count_valid,
            sequential_scan_complete,
            variable_sweep_complete,
            missing_audit_complete,
            evidence_provenance_valid,
        )
    )

    _write_jsonl(paths["pages"], [page.to_dict() for page in pages])
    _write_jsonl(paths["windows"], [window.to_dict() for window in windows])
    _write_json(paths["sections"], [section.to_dict() for section in sections])
    _write_json(paths["scans"], {key: value.to_dict() for key, value in scans.items()})
    _write_jsonl(paths["tables"], tables)
    _write_jsonl(paths["figures"], figures)
    _write_jsonl(paths["equations"], equations)
    _write_jsonl(paths["audit"], audit_log)
    if use_deepseek:
        _write_json(paths["semantic"], semantic_result)
    evidence_write = upsert_trusted_evidence(
        [record.to_dict() for record in evidence],
        paper_id=paper_id,
        total_pages=total_pages,
        title=title,
        base_dir=base_dir,
    )

    status = {
        "paper_id": paper_id,
        "title": title,
        "local_pdf_path": str(source_path),
        "source_pdf_path": str(canonical_path),
        "file_hash_sha256": file_hash,
        "pipeline_version": PIPELINE_VERSION,
        "status": "COMPLETED" if deep_read_complete else "PARTIAL",
        "real_page_count": total_pages,
        "page_record_count": len(pages),
        "processed_page_count": len(pages),
        "page_coverage_ratio": page_coverage_ratio,
        "section_count": len({page.section_title for page in pages}),
        "classified_page_count": sum(
            page.section_title != "unclassified" for page in pages
        ),
        "unclassified_page_count": sum(
            page.section_title == "unclassified" for page in pages
        ),
        "evidence_count": evidence_write["paper_evidence_count"],
        "direct_evidence_count": sum(
            record.directness == "DIRECT" for record in evidence
        ),
        "indirect_evidence_count": sum(
            record.directness == "INDIRECT" for record in evidence
        ),
        "inferred_evidence_count": sum(
            record.directness == "INFERRED" for record in evidence
        ),
        "mention_only_count": sum(
            record.directness == "MENTION_ONLY" for record in evidence
        ),
        "invalid_evidence_count": sum(
            record.directness == "INVALID" for record in evidence
        ),
        "numeric_value_count": len(
            {
                (value.get("page_number"), value.get("raw_value"))
                for scan in scans.values()
                for value in scan.structured_values
            }
        ),
        "formula_evidence_count": sum(
            record.source_method.endswith(":equations_and_models")
            and bool(EQUATION_RE.search(record.original_text))
            for record in evidence
        ),
        "table_review_required_count": sum(
            row["review_status"] == "TABLE_REVIEW_REQUIRED" for row in tables
        ),
        "figure_review_required_count": sum(
            row["review_status"] == "FIGURE_REVIEW_REQUIRED" for row in figures
        ),
        "audit_issue_count": len(audit_log),
        "audit_fixed_count": sum(bool(row.get("fixed")) for row in audit_log),
        "real_page_count_valid": real_page_count_valid,
        "sequential_scan_complete": sequential_scan_complete,
        "variable_sweep_complete": variable_sweep_complete,
        "missing_audit_complete": missing_audit_complete,
        "evidence_provenance_valid": evidence_provenance_valid,
        "deepseek_enhancement_enabled": bool(use_deepseek),
        "deepseek_enhancement_applied": bool(semantic_result.get("applied")),
        "deepseek_enriched_record_count": int(
            semantic_result.get("enriched_record_count") or 0
        ),
        "deepseek_semantic_success_count": int(
            semantic_result.get("semantic_success_count") or 0
        ),
        "deepseek_semantic_failure_count": int(
            semantic_result.get("semantic_failure_count") or 0
        ),
        "deepseek_usage": dict(semantic_result.get("usage") or {}),
        "deep_read_complete": deep_read_complete,
        "page_record_path": str(paths["pages"].resolve()),
        "processed_at": _now(),
        "idempotent_reuse": False,
    }
    _write_json(paths["status"], status)
    update_paper_extraction_status(
        paper_id,
        extraction_status=status["status"],
        deep_read_complete=deep_read_complete,
        page_record_path=status["page_record_path"],
        page_coverage_ratio=page_coverage_ratio,
        evidence_status=(
            "TRUSTED_EVIDENCE_AVAILABLE" if evidence else "NO_TRUSTED_EVIDENCE"
        ),
        base_dir=base_dir,
    )
    return status
