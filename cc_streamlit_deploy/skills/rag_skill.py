"""Compatibility facade for the Stage-3 unified hybrid RAG.

Legacy chunk files are intentionally never opened here.  Existing callers may
keep using ``keyword_retrieve`` while scientific answers are routed through
the provenance-gated hybrid retriever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.stage1_store import BASE_DIR
from src.unified_rag import (
    answer_research_question,
    build_unified_rag,
    retrieve_counter_evidence,
    retrieve_research_evidence,
    retrieve_supporting_evidence,
)


def build_index(
    paper_ids: Sequence[str], *, base_dir: Path = BASE_DIR
) -> Dict[str, Any]:
    return build_unified_rag(paper_ids, base_dir=base_dir)


def keyword_retrieve(
    query: str, top_k: int = 5, *, base_dir: Path = BASE_DIR
) -> List[Dict[str, Any]]:
    """Backward-compatible name; executes both BM25 and vector retrieval."""
    return retrieve_research_evidence(
        query,
        task_type="legacy_compatibility_call",
        top_k=top_k,
        base_dir=base_dir,
    )["results"]


def get_evidence_text(
    query: str, top_k: int = 5, *, base_dir: Path = BASE_DIR
) -> str:
    results = keyword_retrieve(query, top_k=top_k, base_dir=base_dir)
    return "\n\n".join(
        f"[证据{index}] {item.get('title', '未知')} "
        f"(p.{item.get('page_number', '?')}, {item.get('section', '')})\n"
        f"{item.get('original_text', '')}"
        for index, item in enumerate(results, 1)
    )


def index_pdf_chunks(file_path: str, file_hash: str = "") -> int:
    raise RuntimeError(
        "Direct PDF chunk indexing is LEGACY and disabled. "
        "Run Stage-2 deep reading, then build_unified_rag()."
    )


def _load_chunks() -> List[Dict[str, Any]]:
    """Legacy private API retained as an explicit empty quarantine boundary."""
    return []


def _save_chunks(chunks: List[Dict[str, Any]]) -> None:
    raise RuntimeError("Legacy chunk writes are quarantined in Stage 3")


__all__ = [
    "answer_research_question",
    "build_index",
    "get_evidence_text",
    "keyword_retrieve",
    "retrieve_counter_evidence",
    "retrieve_research_evidence",
    "retrieve_supporting_evidence",
]
