import hashlib
import json
from pathlib import Path


PDF_FILES_PATH = Path("data/pdf_files.jsonl")


def compute_file_hash(file_path: str) -> str:
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


def _load_index() -> list:
    if not PDF_FILES_PATH.exists():
        return []

    try:
        return [
            json.loads(line)
            for line in PDF_FILES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return []


def check_duplicate(file_hash: str) -> bool:
    index = _load_index()
    return any(item.get("file_hash_sha256") == file_hash for item in index)


def add_to_index(file_hash: str, file_name: str, saved_path: str) -> None:
    """Compatibility hook.

    PDF identity is registered by ``src.stage1_store`` before a literature card
    is saved.  This hook must not recreate the retired ``data/paper_index.json``
    store.
    """
    return None
