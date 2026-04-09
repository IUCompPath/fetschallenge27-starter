from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import REPO_ROOT, make_test_dir
from fets27_challenge.runtime import run_challenge
from fets27_challenge.synthetic_data import prepare_assets


def _deps_available() -> bool:
    required = ("torch", "nvflare", "monai", "nibabel")
    return all(importlib.util.find_spec(name) is not None for name in required)


RUN_SMOKE = os.environ.get("FETS27_RUN_NVFLARE_SMOKE") == "1"


@pytest.mark.skipif(not (_deps_available() and RUN_SMOKE), reason="opt-in NVFLARE smoke test")
def test_smoke_single_cohort_simulation():
    temp_root = make_test_dir("smoke-single")
    data_root = temp_root / "toy_data"
    workspace_root = temp_root / "workspace"
    output_dir = temp_root / "outputs"
    prepare_assets(REPO_ROOT, data_root)

    json_path, csv_path, cohort_scores = run_challenge(
        repo_root=REPO_ROOT,
        mode="local",
        cohort_names=["glioma"],
        data_root=data_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        num_rounds=1,
        threads=2,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert len(cohort_scores) == 1
    assert cohort_scores[0].cohort == "glioma"


@pytest.mark.skipif(not (_deps_available() and RUN_SMOKE), reason="opt-in NVFLARE smoke test")
def test_smoke_all_cohort_summary():
    temp_root = make_test_dir("smoke-all")
    data_root = temp_root / "toy_data"
    workspace_root = temp_root / "workspace"
    output_dir = temp_root / "outputs"
    prepare_assets(REPO_ROOT, data_root)

    json_path, csv_path, cohort_scores = run_challenge(
        repo_root=REPO_ROOT,
        mode="local",
        cohort_names=["glioma", "meningioma", "sub_sahara"],
        data_root=data_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        num_rounds=1,
        threads=2,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert {score.cohort for score in cohort_scores} == {"glioma", "meningioma", "sub_sahara"}
