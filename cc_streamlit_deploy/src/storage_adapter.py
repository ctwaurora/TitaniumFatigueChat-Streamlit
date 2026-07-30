"""Storage boundary for the literature-library web workflow.

The adapter makes persistence semantics explicit without coupling the
scientific pipeline to a particular paid cloud provider.  Local runs write to
the project.  Streamlit Community Cloud runs use a prepared temporary working
copy unless an external persistent root is configured.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


LOCAL_PERSISTENT = "LOCAL_PERSISTENT"
CLOUD_TEMPORARY = "CLOUD_TEMPORARY"
EXTERNAL_PERSISTENT = "EXTERNAL_PERSISTENT"

CLOUD_TEMPORARY_WARNING = (
    "当前云端使用临时存储，上传文件和新索引在应用重启后可能丢失。"
)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_cloud_environment(environ: Mapping[str, str]) -> bool:
    explicit_mode = str(environ.get("LITERATURE_STORAGE_MODE") or "").upper()
    if explicit_mode == CLOUD_TEMPORARY:
        return True
    if any(
        _truthy(environ.get(key))
        for key in (
            "STREAMLIT_CLOUD",
            "IS_STREAMLIT_CLOUD",
            "STREAMLIT_SHARING_MODE",
        )
    ):
        return True
    runtime = str(environ.get("STREAMLIT_RUNTIME_ENVIRONMENT") or "").lower()
    hostname = str(environ.get("HOSTNAME") or "").lower()
    return runtime in {"cloud", "community_cloud"} or "streamlit" in hostname


@dataclass(frozen=True)
class StorageBackend:
    mode: str
    root: Path
    project_root: Path
    persistent: bool
    warning: str = ""

    def prepare(self) -> Path:
        """Create the working root and seed cloud-temporary metadata once."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.mode != CLOUD_TEMPORARY or self.root == self.project_root:
            return self.root

        marker = self.root / ".literature_storage_initialized"
        if marker.exists():
            return self.root

        # Seed only deployed runtime state.  PDF files are deliberately not
        # copied into or committed with a deployment package.
        source_data = self.project_root / "data"
        target_data = self.root / "data"
        if source_data.exists() and not target_data.exists():
            shutil.copytree(
                source_data,
                target_data,
                ignore=shutil.ignore_patterns("*.pdf", "*.pyc", "__pycache__"),
            )
        for relative in ("paper/pdfs", "tmp/pdfs/oa", "logs"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        marker.write_text(self.mode, encoding="utf-8")
        return self.root


def detect_storage_backend(
    project_root: Optional[Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> StorageBackend:
    project = (project_root or Path(__file__).resolve().parent.parent).resolve()
    env = os.environ if environ is None else environ
    explicit_mode = str(env.get("LITERATURE_STORAGE_MODE") or "").upper()
    external = str(env.get("LITERATURE_STORAGE_ROOT") or "").strip()

    if external or explicit_mode == EXTERNAL_PERSISTENT:
        if not external:
            raise ValueError(
                "LITERATURE_STORAGE_ROOT is required for EXTERNAL_PERSISTENT"
            )
        return StorageBackend(
            mode=EXTERNAL_PERSISTENT,
            root=Path(external).expanduser().resolve(),
            project_root=project,
            persistent=True,
        )

    if _is_cloud_environment(env):
        configured_temp = str(env.get("LITERATURE_TEMP_ROOT") or "").strip()
        if configured_temp:
            root = Path(configured_temp).expanduser().resolve()
        else:
            digest = hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:12]
            root = Path(tempfile.gettempdir()) / f"lpbf-literature-{digest}"
        return StorageBackend(
            mode=CLOUD_TEMPORARY,
            root=root.resolve(),
            project_root=project,
            persistent=False,
            warning=CLOUD_TEMPORARY_WARNING,
        )

    return StorageBackend(
        mode=LOCAL_PERSISTENT,
        root=project,
        project_root=project,
        persistent=True,
    )

