"""Stage-1 canonical data contracts for TitaniumFatigueChat.

These contracts define the stable storage boundary only.  They intentionally do
not claim that full-paper extraction, page-level evidence, or vector retrieval
has been implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


DATA_SCHEMA_VERSION = "stage1.0"

DIRECTNESS_VALUES = {
    "DIRECT",
    "INDIRECT",
    "INFERRED",
    "MENTION_ONLY",
    "INVALID",
}


@dataclass
class PaperRecord:
    paper_id: str
    title: str = ""
    authors: str = ""
    publication_date: str = ""
    doi: str = ""
    normalized_title: str = ""
    source_type: str = ""
    source_path: str = ""
    canonical_pdf_path: str = ""
    file_hash_sha256: str = ""
    real_page_count: int = 0
    pdf_valid: bool = False
    library_status: str = "CANDIDATE"
    rag_status: str = "NOT_INDEXED"
    evidence_status: str = "NOT_EXTRACTED"
    extraction_status: str = "NOT_STARTED"
    page_record_path: str = ""
    page_coverage_ratio: float = 0.0
    deep_read_complete: bool = False
    duplicate_status: str = "UNIQUE"
    duplicate_of: str = ""
    linked_versions: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    data_version: str = DATA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PdfFileRecord:
    pdf_file_id: str
    paper_id: str
    source_path: str
    canonical_pdf_path: str
    file_hash_sha256: str
    pdf_valid: bool
    real_page_count: int
    file_size: int
    ingest_status: str
    duplicate_status: str = "UNIQUE"
    created_at: str = ""
    updated_at: str = ""
    data_version: str = DATA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateRecord:
    duplicate_id: str
    source_record_id: str
    duplicate_of: str
    match_level: str
    match_value: str
    status: str
    reason: str
    requires_manual_review: bool = False
    created_at: str = ""
    data_version: str = DATA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    evidence_id: str
    paper_id: str
    claim: str = ""
    original_text: str = ""
    page_number: int = 0
    section: str = ""
    paragraph_index: int = 0
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    directness: str = "INVALID"
    confidence: float = 0.0
    review_status: str = "UNREVIEWED"
    source_method: str = ""
    canonical_paper_id: str = ""
    evidence_type: str = "SCIENTIFIC_CLAIM"
    variables: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    units: List[str] = field(default_factory=list)
    formula_reference: str = ""
    table_or_figure_reference: str = ""
    support_or_counter: str = "SUPPORT"
    extraction_method: str = ""
    created_at: str = ""
    updated_at: str = ""
    data_version: str = DATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.directness not in DIRECTNESS_VALUES:
            raise ValueError(
                f"Invalid directness {self.directness!r}; "
                f"expected one of {sorted(DIRECTNESS_VALUES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageRecord:
    paper_id: str
    page_number: int
    total_pages: int
    raw_text: str
    cleaned_text: str
    character_count: int
    section_title: str
    page_type: str
    parse_status: str
    contains_table: bool
    contains_figure_caption: bool
    contains_equation: bool
    ocr_required: bool
    source_pdf_path: str
    processed_at: str
    classification_confidence: float = 0.0
    data_version: str = "stage2.0"
    canonical_paper_id: str = ""
    page_text: str = ""
    text_length: int = 0
    extraction_status: str = ""
    image_count: int = 0
    table_candidate: bool = False
    formula_candidate: bool = False
    visual_review_status: str = "NOT_REQUIRED"

    def __post_init__(self) -> None:
        if not 1 <= self.page_number <= self.total_pages:
            raise ValueError("page_number must be inside the real PDF page range")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequentialWindowRecord:
    window_id: str
    paper_id: str
    start_page: int
    end_page: int
    source_page_numbers: List[int]
    window_text: str
    extracted_items: Dict[str, Any]
    processing_status: str
    data_version: str = "stage2.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectionCoverageRecord:
    section_name: str
    page_range: List[int]
    paragraph_count: int
    classification_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VariableScanRecord:
    category: str
    matched_pages: List[int]
    matched_sections: List[str]
    matched_terms: List[str]
    structured_values: List[Dict[str, Any]]
    original_evidence: List[Dict[str, Any]]
    missing_fields: List[str]
    scan_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationRecord:
    source_record: Dict[str, Any]
    target_record: Dict[str, Any]
    action: str
    reason: str
    status: str
    error: str = ""
    timestamp: str = ""
    data_version: str = DATA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
