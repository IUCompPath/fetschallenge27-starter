from __future__ import annotations

from pathlib import Path

import pytest
from conftest import REPO_ROOT, make_test_dir

from fets27_classification.client import (
    REPORTED_VALIDATION_METRICS,
    _PreloadedDataLoader,
    apply_site_hparams,
    resolve_site_datalist_path,
)
from fets27_classification.config import DEFAULT_KEY_METRIC
from fets27_classification.metrics import binary_metrics


class _RecordingLoader:
    def __init__(self):
        self.iterations = 0
        self.dataset = [1, 2]

    def __iter__(self):
        self.iterations += 1
        return iter([self.iterations])

    def __len__(self):
        return 1


def test_preloaded_data_loader_preserves_prefetched_iterator():
    loader = _RecordingLoader()
    wrapped = _PreloadedDataLoader(loader)

    wrapped.preload()

    assert loader.iterations == 1
    assert list(wrapped) == [1]
    assert loader.iterations == 1
    assert list(wrapped) == [2]
    assert loader.iterations == 2
    assert len(wrapped) == 1
    assert wrapped.dataset == [1, 2]


def test_reported_validation_metrics_cover_every_non_key_metric():
    metric_names = set(binary_metrics([0, 1], [0.2, 0.8]))

    assert DEFAULT_KEY_METRIC == "val_balanced_accuracy"
    assert set(REPORTED_VALIDATION_METRICS) == metric_names - {
        "balanced_accuracy",
        "confusion_matrix",
    }


def test_resolve_site_datalist_path_prefers_the_explicit_argument():
    resolved = resolve_site_datalist_path(
        datalist_json_path="D:/data/brain_tumor/datalist/site-7.json",
        datalist_dir="D:/ignored",
        client_name="site-7",
    )

    assert resolved == "D:/data/brain_tumor/datalist/site-7.json"


def test_resolve_site_datalist_path_falls_back_to_the_site_name():
    root = make_test_dir("client-datalist")
    datalist_dir = root / "datalist"
    datalist_dir.mkdir()
    (datalist_dir / "site-7.json").write_text("{}", encoding="utf-8")

    resolved = resolve_site_datalist_path(
        datalist_json_path=None, datalist_dir=str(datalist_dir), client_name="site-7"
    )

    assert Path(resolved) == datalist_dir / "site-7.json"


def test_resolve_site_datalist_path_requires_an_argument():
    with pytest.raises(ValueError, match="datalist_json_path or --datalist_dir"):
        resolve_site_datalist_path(
            datalist_json_path=None, datalist_dir=None, client_name="site-7"
        )


def test_resolve_site_datalist_path_reports_a_missing_site():
    root = make_test_dir("client-datalist-missing")

    with pytest.raises(FileNotFoundError, match="No datalist found for site"):
        resolve_site_datalist_path(
            datalist_json_path=None, datalist_dir=str(root), client_name="site-7"
        )


def test_apply_site_hparams_refines_the_received_defaults():
    class _Args:
        pass

    args = apply_site_hparams(
        _Args(),
        str(REPO_ROOT / "participant" / "site_hparams.yaml"),
        "brain_tumor",
        "site-1",
    )

    assert args.learning_rate == pytest.approx(1e-04)
    assert args.batch_size == 4
    assert args.cache_dataset == 0.0

    default_args = apply_site_hparams(
        _Args(),
        str(REPO_ROOT / "participant" / "site_hparams.yaml"),
        "brain_tumor",
        "site-99",
    )
    assert default_args.learning_rate == pytest.approx(1e-04)
