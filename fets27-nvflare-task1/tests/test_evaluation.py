from __future__ import annotations

from conftest import make_test_dir
from fets27_challenge.evaluation import resolve_best_model_path


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

    resolved = resolve_best_model_path(job_root)
    assert resolved == best_path
