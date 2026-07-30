"""Create a secret-free Streamlit Community Cloud deployment copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}_streamlit_deploy"

ROOT_FILES = (
    "streamlit_app.py",
    "app.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "DEPLOY_STREAMLIT_COMMUNITY_CLOUD.md",
)

STREAMLIT_FILES = (
    ".streamlit/config.toml",
    ".streamlit/secrets.example.toml",
)

CODE_DIRECTORIES = ("src", "skills")
CODE_SUFFIXES = {".py"}
EXCLUDED_CODE_FILES = {
    "src/qwen_usage.py",
    "skills/qwen_skill.py",
}

CONFIG_FILES = ("config/task_profile.yaml",)
SCRIPT_FILES = ("scripts/export_streamlit_deploy.py",)

DATA_FILES = (
    "data/literature_database.csv",
    "data/candidate_papers.csv",
    "data/variable_relation_dataset.csv",
    "data/research_gap_dataset.csv",
    "data/hypothesis_dataset.csv",
    "data/fatigue_equation_library.csv",
    "data/equation_parameter_dataset.csv",
    "data/variable_mechanism.csv",
    "data/conflict_claims.csv",
    "data/condition_evidence_dataset.csv",
    "data/domain_dictionary.csv",
    "data/macro_micro_links.csv",
    "data/minimum_validation_dataset_schema.csv",
    "data/element_property_mechanism.csv",
    "data/experiment_design_dataset.csv",
    "data/equation_parameters.csv",
    "data/relevance_ranking.csv",
)

FORBIDDEN_NAMES = {
    ".git",
    ".env",
    "qwen_key.txt",
    "secrets.toml",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_PART_SEQUENCES = {
    ("outputs",),
    ("tmp",),
    ("logs",),
    ("data", "rag"),
    ("data", "tasks"),
    ("data", "deep_read"),
}
FORBIDDEN_SUFFIXES = {".pdf", ".pyc", ".pyo", ".jsonl", ".parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024

SECRET_PATTERNS = (
    ("API token", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    (
        "configured DEEPSEEK_API_KEY",
        re.compile(
            r"DEEPSEEK_API_KEY\s*=\s*[\"'](?!your-deepseek-api-key[\"'])[^\"']+[\"']",
            re.IGNORECASE,
        ),
    ),
    (
        "configured APP_PASSWORD",
        re.compile(
            r"APP_PASSWORD\s*=\s*[\"'](?!your-private-app-password[\"'])[^\"']+[\"']",
            re.IGNORECASE,
        ),
    ),
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")


class DeploymentExportError(RuntimeError):
    """Raised when the deployment copy fails a safety check."""


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_directory(project_root: Path, output_dir: Path) -> Path:
    project = project_root.resolve()
    destination = output_dir.resolve()
    if destination == project:
        raise DeploymentExportError("部署目录不能是主项目目录。")
    if _is_relative_to(project, destination):
        raise DeploymentExportError("部署目录不能是主项目的父目录。")
    if destination == destination.anchor or destination.parent == destination:
        raise DeploymentExportError("部署目录不能是文件系统根目录。")
    if destination.exists() and destination.is_symlink():
        raise DeploymentExportError("部署目录不能是符号链接。")
    return destination


def clean_output_directory(project_root: Path, output_dir: Path) -> Path:
    destination = validate_output_directory(project_root, output_dir)
    if destination.exists():
        def remove_readonly(function: object, path: str, _: object) -> None:
            Path(path).chmod(stat.S_IWRITE)
            function(path)

        shutil.rmtree(destination, onerror=remove_readonly)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _copy_file(project_root: Path, output_dir: Path, relative_path: str) -> bool:
    source = project_root / relative_path
    if not source.is_file():
        return False
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_code_directory(project_root: Path, output_dir: Path, name: str) -> None:
    source_root = project_root / name
    if not source_root.is_dir():
        return
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in CODE_SUFFIXES:
            continue
        relative = _relative_posix(source, project_root)
        if relative in EXCLUDED_CODE_FILES or "__pycache__" in source.parts:
            continue
        _copy_file(project_root, output_dir, relative)


def _copy_assets(project_root: Path, output_dir: Path) -> None:
    assets_root = project_root / "assets"
    if not assets_root.is_dir():
        return
    for source in sorted(assets_root.rglob("*")):
        if not source.is_file():
            continue
        relative = _relative_posix(source, project_root)
        if source.suffix.lower() in FORBIDDEN_SUFFIXES or source.stat().st_size > MAX_FILE_SIZE:
            continue
        _copy_file(project_root, output_dir, relative)


def copy_allowlist(project_root: Path, output_dir: Path) -> None:
    for relative in (
        *ROOT_FILES,
        *STREAMLIT_FILES,
        *CONFIG_FILES,
        *SCRIPT_FILES,
        *DATA_FILES,
    ):
        _copy_file(project_root, output_dir, relative)
    for directory in CODE_DIRECTORIES:
        _copy_code_directory(project_root, output_dir, directory)
    _copy_assets(project_root, output_dir)


def write_empty_cloud_library(project_root: Path, output_dir: Path) -> None:
    """Keep CSV schemas but never export local literature rows as cloud truth."""
    fallback_headers = {
        "data/candidate_papers.csv": (
            "candidate_id,title,authors,year,doi,source_url,metadata_source,"
            "oa_status,pdf_status,validation_status\n"
        ),
        "data/literature_database.csv": (
            "paper_id,title,authors,year,doi,source_url,metadata_source,"
            "pdf_status,evidence_status,rag_status\n"
        ),
    }
    for relative, fallback in fallback_headers.items():
        source = project_root / relative
        header = ""
        if source.exists():
            with source.open("r", encoding="utf-8-sig") as handle:
                header = handle.readline().strip()
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((header or fallback.strip()) + "\n", encoding="utf-8")


def _content_manifest_hash(output_dir: Path) -> str:
    digest = hashlib.sha256()
    excluded = {"DEPLOY_MANIFEST.txt", "DEPLOY_VERSION.json"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(output_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_deploy_version(project_root: Path, output_dir: Path) -> Path:
    payload = {
        "source_commit": source_commit(project_root),
        "export_time": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "deploy_manifest_hash": _content_manifest_hash(output_dir),
        "application_version": "3.2.0-cloud-oa",
    }
    target = output_dir / "DEPLOY_VERSION.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _contains_forbidden_parts(relative: Path) -> bool:
    lowered = tuple(part.lower() for part in relative.parts)
    if any(part in FORBIDDEN_NAMES or part.startswith(".env.") for part in lowered):
        return True
    for sequence in FORBIDDEN_PART_SEQUENCES:
        width = len(sequence)
        if any(lowered[index:index + width] == sequence for index in range(len(lowered) - width + 1)):
            return True
    return relative.suffix.lower() in FORBIDDEN_SUFFIXES


def scan_deployment(output_dir: Path) -> List[Tuple[str, str]]:
    violations: List[Tuple[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir)
        relative_name = relative.as_posix()
        if _contains_forbidden_parts(relative):
            violations.append((relative_name, "forbidden path or file type"))
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            violations.append((relative_name, "file exceeds 5 MiB"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                violations.append((relative_name, label))
        if WINDOWS_ABSOLUTE_PATH.search(text):
            violations.append((relative_name, "Windows absolute path"))
    return violations


def deployment_files(output_dir: Path) -> List[str]:
    return [
        path.relative_to(output_dir).as_posix()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]


def write_manifest(output_dir: Path, files: Sequence[str]) -> Path:
    manifest_path = output_dir / "DEPLOY_MANIFEST.txt"
    lines = [
        "TitaniumFatigueChat Streamlit deployment manifest",
        f"file_count={len(files) + 1}",
        "",
        *files,
        "DEPLOY_MANIFEST.txt",
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def export_deployment(
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[str]:
    project = project_root.resolve()
    destination = clean_output_directory(project, output_dir)
    copy_allowlist(project, destination)
    write_empty_cloud_library(project, destination)
    write_deploy_version(project, destination)

    violations = scan_deployment(destination)
    if violations:
        details = "; ".join(f"{path} ({reason})" for path, reason in violations)
        shutil.rmtree(destination)
        raise DeploymentExportError(f"部署安全扫描未通过：{details}")

    files = deployment_files(destination)
    write_manifest(destination, files)
    final_violations = scan_deployment(destination)
    if final_violations:
        shutil.rmtree(destination)
        raise DeploymentExportError("部署清单生成后安全扫描未通过。")
    return deployment_files(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="部署副本输出目录，默认位于主项目同级。",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = export_deployment(PROJECT_ROOT, args.output)
    except DeploymentExportError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[OK] Deployment copy: {args.output.resolve()}")
    print(f"[OK] Files: {len(files)}")
    for relative in files:
        print(relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
