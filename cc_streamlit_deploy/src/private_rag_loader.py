"""Resolve a protected RAG bundle without placing it in the public repository.

Production may either mount an already extracted bundle through
``TFC_PRIVATE_RAG_ROOT`` or provide a private HTTPS zip artifact.  The zip is
accepted only when an expected SHA-256 is configured and is extracted into an
ephemeral cache outside the application checkout.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


PRIVATE_ROOT_ENV = "TFC_PRIVATE_RAG_ROOT"
PRIVATE_URL_ENV = "TFC_PRIVATE_RAG_BUNDLE_URL"
PRIVATE_SHA_ENV = "TFC_PRIVATE_RAG_BUNDLE_SHA256"
PRIVATE_TOKEN_ENV = "TFC_PRIVATE_RAG_BEARER_TOKEN"
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


class PrivateRagLoadError(RuntimeError):
    """Raised when the private RAG source is absent or fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_dir(root: Path) -> Path:
    """Accept either the bundle itself or a root containing data/cloud_bundle."""
    direct = root.resolve()
    nested = direct / "data" / "cloud_bundle"
    return nested if nested.is_dir() else direct


def _validate_zip_members(archive: zipfile.ZipFile) -> None:
    total = 0
    for member in archive.infolist():
        total += int(member.file_size)
        if total > MAX_EXTRACTED_BYTES:
            raise PrivateRagLoadError("Private RAG artifact exceeds extracted-size limit")
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise PrivateRagLoadError("Private RAG artifact contains an unsafe path")


def _download_private_artifact(
    url: str,
    expected_sha256: str,
    token: str,
    cache_root: Path,
) -> Path:
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "https":
        raise PrivateRagLoadError("Private RAG artifact URL must use HTTPS")
    if not expected_sha256 or len(expected_sha256) != 64:
        raise PrivateRagLoadError("Private RAG artifact SHA-256 is required")

    cache_key = hashlib.sha256(f"{url}|{expected_sha256}".encode("utf-8")).hexdigest()[:20]
    target = cache_root / cache_key
    ready = target / ".ready"
    if ready.is_file():
        return _bundle_dir(target)

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / f"{cache_key}.zip.part"
    request = urllib.request.Request(url, headers={"User-Agent": "TitaniumFatigueChat/1.0"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as output:
            copied = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                copied += len(block)
                if copied > MAX_ARCHIVE_BYTES:
                    raise PrivateRagLoadError("Private RAG artifact exceeds download-size limit")
                output.write(block)
        if _sha256(archive_path).casefold() != expected_sha256.casefold():
            raise PrivateRagLoadError("Private RAG artifact SHA-256 mismatch")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            _validate_zip_members(archive)
            archive.extractall(target)
        bundle = _bundle_dir(target)
        if not (bundle / "manifest.json").is_file():
            raise PrivateRagLoadError("Private RAG artifact does not contain a bundle manifest")
        ready.write_text(expected_sha256.casefold(), encoding="ascii")
        return bundle
    except PrivateRagLoadError:
        raise
    except Exception as exc:
        raise PrivateRagLoadError("Private RAG artifact could not be loaded") from exc
    finally:
        archive_path.unlink(missing_ok=True)


def resolve_private_rag_root(
    project_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return a private bundle root; never silently rebuild it from public data."""
    env = os.environ if environ is None else environ
    mounted = str(env.get(PRIVATE_ROOT_ENV) or "").strip()
    if mounted:
        root = _bundle_dir(Path(mounted).expanduser())
        if not (root / "manifest.json").is_file():
            raise PrivateRagLoadError("Configured private RAG root is invalid")
        return root

    url = str(env.get(PRIVATE_URL_ENV) or "").strip()
    if url:
        cache_value = str(env.get("TFC_PRIVATE_RAG_CACHE_ROOT") or "").strip()
        cache_root = (
            Path(cache_value).expanduser()
            if cache_value
            else Path(tempfile.gettempdir()) / "titanium-fatigue-private-rag"
        )
        return _download_private_artifact(
            url,
            str(env.get(PRIVATE_SHA_ENV) or "").strip(),
            str(env.get(PRIVATE_TOKEN_ENV) or "").strip(),
            cache_root,
        )

    local = Path(project_root) / "data" / "cloud_bundle"
    return local
