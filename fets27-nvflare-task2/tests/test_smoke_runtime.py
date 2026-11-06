from __future__ import annotations

import importlib.util
import json
import os

import pytest

from conftest import REPO_ROOT, make_test_dir

from fets27_classification.runtime import run_challenge
from fets27_classification.synthetic_data import prepare_assets


def _deps_available() -> bool:
    required = ("torch", "nvflare", "monai", "nibabel")
    return all(importlib.util.find_spec(name) is not None for name in required)


RUN_SMOKE = os.environ.get("FETS27_RUN_NVFLARE_SMOKE") == "1"


@pytest.mark.skipif(
    not (_deps_available() and RUN_SMOKE), reason="opt-in NVFLARE smoke test"
)
def test_smoke_single_cohort_simulation():
    temp_root = make_test_dir("smoke-single")
    data_root = temp_root / "toy_data"
    workspace_root = temp_root / "workspace"
    output_dir = temp_root / "outputs"
    prepare_assets(REPO_ROOT, data_root)

    json_path, csv_path, cohort_scores = run_challenge(
        repo_root=REPO_ROOT,
        mode="local",
        cohort_names=["brain_tumor"],
        data_root=data_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        num_rounds=1,
        threads=2,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert len(cohort_scores) == 1
    assert cohort_scores[0].cohort == "brain_tumor"
    assert cohort_scores[0].site_count == 2
    assert "mcc" in cohort_scores[0].metrics
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "local"
    assert payload["cohorts"][0]["metrics"]["mcc"] is not None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"data_loader_workers": 0}, "data_loader_workers"),
        ({"validation_limit": 0}, "validation_limit"),
    ],
)
def test_run_challenge_rejects_invalid_resource_limits(kwargs, message):
    temp_root = make_test_dir("smoke-invalid-limits")

    with pytest.raises(ValueError, match=message):
        run_challenge(
            repo_root=REPO_ROOT,
            mode="local",
            cohort_names=["brain_tumor"],
            data_root=temp_root / "toy_data",
            workspace_root=temp_root / "workspace",
            output_dir=temp_root / "outputs",
            num_rounds=1,
            threads=2,
            **kwargs,
        )
