from pathlib import Path
from typing import List

import fitz


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text_parts = []

    try:
        doc = fitz.open(str(pdf_path))

        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)

        doc.close()

    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text: {str(e)}")

    text = "\n".join(text_parts).strip()

    # Basic cleanup
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split long text into overlapping chunks."""
    if not text:
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks