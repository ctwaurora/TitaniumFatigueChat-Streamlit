#!/usr/bin/env python3
"""Verify every deployment-manifest path is tracked by Git.

Run this from the GitHub working copy before pushing a deployment commit.
The check is read-only and deliberately ignores files outside the manifest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEPLOY_ROOT = REPO_ROOT / "cc_streamlit_deploy"
MANIFEST = DEPLOY_ROOT / "DEPLOY_MANIFEST.txt"


def _manifest_paths() -> list[str]:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    paths: list[str] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("TitaniumFatigueChat ") or value.startswith("file_count="):
            continue
        paths.append(value.replace("\\", "/"))
    return paths


def _tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", "cc_streamlit_deploy"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "cc_streamlit_deploy/"
    return {
        line[len(prefix) :].replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    }


def main() -> int:
    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}")
        return 2
    expected = set(_manifest_paths())
    tracked = _tracked_paths()
    missing = sorted(expected - tracked)
    if missing:
        print(f"ERROR: {len(missing)} manifest file(s) are not Git-tracked:")
        print("\n".join(f"- {path}" for path in missing))
        return 1
    print(f"OK: all {len(expected)} manifest files are Git-tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
