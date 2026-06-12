"""Submission validation and packaging utilities."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

from .config import (
    ALLOWED_EDITABLE_FILES,
    IGNORED_VALIDATION_PARTS,
    MANIFEST_FILE,
    PARTICIPANT_AGGREGATOR_FILE,
)


def compute_locked_manifest(repo_root: Path) -> dict[str, str]:
    """Compute the SHA256 hashes of all locked files in the repository.

    Args:
        repo_root: Path to the repository root directory.

    Returns:
        A dictionary mapping relative file paths to their SHA256 hex digests.
    """
    manifest = {}
    for root, dirs, files in os.walk(repo_root, topdown=True, onerror=lambda _: None):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not _should_skip_dir(root_path, name)]
        for file_name in files:
            file_path = root_path / file_name
            relative_path = file_path.relative_to(repo_root)
            relative_text = relative_path.as_posix()
            if _should_skip_for_manifest(relative_path):
                continue
            manifest[relative_text] = _sha256(file_path)
    return manifest


def write_manifest(repo_root: Path) -> Path:
    """Compute the locked manifest and write it to the manifest JSON file.

    Args:
        repo_root: Path to the repository root directory.

    Returns:
        The path to the generated manifest file.
    """
    manifest_path = repo_root / MANIFEST_FILE
    payload = compute_locked_manifest(repo_root)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return manifest_path


def validate_submission_state(repo_root: Path) -> None:
    """Validate that the repository's locked files match the manifest.

    Ensures that no locked files have been added, removed, or modified.

    Args:
        repo_root: Path to the repository root directory.

    Raises:
        FileNotFoundError: If the manifest file is missing.
        ValueError: If there are modified, missing, or unexpected locked files.
    """
    manifest_path = repo_root / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        expected_manifest = json.load(handle)

    current_manifest = compute_locked_manifest(repo_root)
    missing = sorted(set(expected_manifest) - set(current_manifest))
    unexpected = sorted(set(current_manifest) - set(expected_manifest))
    changed = sorted(
        path
        for path in expected_manifest.keys() & current_manifest.keys()
        if expected_manifest[path] != current_manifest[path]
    )

    problems = []
    if missing:
        problems.append(f"Missing locked files: {missing}")
    if unexpected:
        problems.append(f"Unexpected locked files: {unexpected}")
    if changed:
        problems.append(f"Modified locked files: {changed}")

    if problems:
        raise ValueError("Submission validation failed. " + " | ".join(problems))


def package_submission(repo_root: Path, output_path: Path) -> Path:
    """Validate the submission and package the participant's files into a zip archive.

    Args:
        repo_root: Path to the repository root directory.
        output_path: Path where the output zip file should be written.

    Returns:
        The path to the created submission zip package.
    """
    validate_submission_state(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in (PARTICIPANT_AGGREGATOR_FILE,):
            archive.write(repo_root / relative_path, arcname=relative_path.as_posix())
    return output_path


def _sha256(file_path: Path) -> str:
    """Compute the SHA256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        The hex digest of the file checksum.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_skip_for_manifest(relative_path: Path) -> bool:
    """Check if a file should be excluded from the locked manifest.

    Args:
        relative_path: Path of the file relative to repo root.

    Returns:
        True if the file is editable/ignored/temporary, False otherwise.
    """
    if relative_path.as_posix() in ALLOWED_EDITABLE_FILES:
        return True
    if relative_path == MANIFEST_FILE:
        return True
    if any(part in IGNORED_VALIDATION_PARTS for part in relative_path.parts):
        return True
    if (
        relative_path.parts[:2] == ("assets", "checkpoints")
        and relative_path.suffix == ".pt"
    ):
        return True
    if relative_path.parts[:2] == ("data", "toy"):
        return True
    return False


def _should_skip_dir(root_path: Path, dir_name: str) -> bool:
    """Check if a directory should be skipped during repo walks.

    Args:
        root_path: The directory's parent path.
        dir_name: The name of the directory.

    Returns:
        True if the directory should be skipped, False otherwise.
    """
    if dir_name in IGNORED_VALIDATION_PARTS:
        return True
    if dir_name.startswith("tmp"):
        return True
    if dir_name.startswith("pytest-cache-files"):
        return True
    return False
