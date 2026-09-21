"""Official scholarly source connectors used by v1.1 maintenance."""

from .base import LiteratureSource, SourceCandidate
from .arxiv import ARXIV_TOPIC_QUERIES, ArxivConnector, ArxivSource
from .core import CORESource
from .crossref import CrossrefSource
from .nasa_ntrs import NASANTRSSource
from .ncbi import NCBISource
from .openalex import OpenAlexSource
from .osti import OSTISource
from .semantic_scholar import SemanticScholarConnector, SemanticScholarSource
from .unpaywall import UnpaywallSource
from .web_of_science import WebOfScienceSource

__all__ = [
    "LiteratureSource", "SourceCandidate", "ArxivConnector", "ArxivSource",
    "ARXIV_TOPIC_QUERIES", "OpenAlexSource", "CrossrefSource", "UnpaywallSource",
    "CORESource", "OSTISource", "NASANTRSSource", "NCBISource", "WebOfScienceSource",
    "SemanticScholarConnector", "SemanticScholarSource",
]
