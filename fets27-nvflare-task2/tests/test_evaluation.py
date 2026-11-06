from __future__ import annotations

import csv
import json
import math

from conftest import make_test_dir

from fets27_classification.evaluation import (
    CohortScore,
    resolve_best_model_path,
    write_summary,
)


def _cohort_score(metrics: dict | None = None) -> CohortScore:
    return CohortScore(
        cohort="brain_tumor",
        display_name="BraTS Glioma vs Meningioma",
        cohort_score=0.8,
        metrics=metrics
        or {
            "balanced_accuracy": 0.8,
            "accuracy": 0.81,
            "auroc": 0.9,
            "macro_f1": 0.79,
            "sensitivity": 0.7,
            "specificity": 0.9,
            "mcc": 0.62,
            "confusion_matrix": [[90, 10], [30, 70]],
        },
        site_count=25,
        validation_case_count=395,
        best_model_path="best_FL_global_model.pt",
        job_workspace="workspace",
    )


def test_resolve_best_model_path_prefers_best_filename():
    job_root = make_test_dir("evaluation-path") / "job"
    best_path = (
        job_root / "server" / "simulate_job" / "app_server" / "best_FL_global_model.pt"
    )
    fallback_path = (
        job_root / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_bytes(b"best")
    fallback_path.write_bytes(b"fallback")

    assert resolve_best_model_path(job_root) == best_path


def test_write_summary_reports_every_metric_including_mcc():
    output_dir = make_test_dir("evaluation-summary")

    json_path, csv_path = write_summary(
        output_dir, mode="local", cohort_scores=[_cohort_score()]
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "local"
    assert payload["overall_public_score"] == 0.8
    assert payload["cohorts"][0]["metrics"]["mcc"] == 0.62

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    for name in (
        "balanced_accuracy",
        "auroc",
        "macro_f1",
        "sensitivity",
        "specificity",
        "mcc",
    ):
        assert name in rows[0]
    assert rows[0]["mcc"] == "0.62"
    assert rows[0]["confusion_matrix"] == "[[90, 10], [30, 70]]"


def test_write_summary_serializes_non_finite_metrics_as_none():
    output_dir = make_test_dir("evaluation-nan")
    metrics = _cohort_score().metrics | {"auroc": float("nan")}
    score = _cohort_score(metrics)

    json_path, csv_path = write_summary(
        output_dir, mode="official", cohort_scores=[score]
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cohorts"][0]["metrics"]["auroc"] is None
    assert "nan" not in csv_path.read_text(encoding="utf-8").lower()
    assert math.isnan(score.metrics["auroc"])
