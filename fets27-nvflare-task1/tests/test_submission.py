from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from conftest import make_test_dir
from fets27_challenge.submission import compute_locked_manifest, package_submission, validate_submission_state


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "participant").mkdir(parents=True)
    (repo_root / "participant" / "aggregator.py").write_text("print('participant')\n", encoding="utf-8")
    (repo_root / "participant" / "site_hparams.yaml").write_text("cohorts: {}\n", encoding="utf-8")
    (repo_root / "README.md").write_text("locked\n", encoding="utf-8")
    manifest = compute_locked_manifest(repo_root)
    (repo_root / "challenge_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return repo_root


def test_package_submission_contains_only_allowed_files():
    repo_root = _make_repo(make_test_dir("submission-package"))
    output_path = repo_root / "submission.zip"
    package_submission(repo_root, output_path)

    with zipfile.ZipFile(output_path) as archive:
        assert sorted(archive.namelist()) == [
            "participant/aggregator.py",
            "participant/site_hparams.yaml",
        ]


def test_validate_submission_rejects_locked_file_changes():
    repo_root = _make_repo(make_test_dir("submission-locked-change"))
    (repo_root / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Modified locked files"):
        validate_submission_state(repo_root)
