from __future__ import annotations

import json

import pytest

from conftest import make_test_dir

from fets27_classification.cohort_registry import get_cohort_spec
from fets27_classification.data_pipeline import (
    build_dataloaders,
    build_validation_loader,
    compute_class_weights,
    deduplicate_cases,
    evaluate_model,
    load_case_lists,
)
from fets27_classification.synthetic_data import generate_toy_dataset

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("nibabel")

SPATIAL_SIZE = (16, 16, 16)


def _toy_data_root():
    root = make_test_dir("data-pipeline") / "toy_data"
    generate_toy_dataset(root)
    return root


def _paths(root):
    spec = get_cohort_spec("brain_tumor")
    return (
        spec.dataset_dir(root),
        spec.datalist_dir(root) / "site-1.json",
    )


def test_build_dataloaders_shapes_and_class_weights():
    root = _toy_data_root()
    dataset_dir, datalist_path = _paths(root)

    train_loader, valid_loader, class_weights = build_dataloaders(
        dataset_base_dir=str(dataset_dir),
        datalist_json_path=str(datalist_path),
        spatial_size=SPATIAL_SIZE,
        batch_size=2,
        cache_rate=0.0,
        data_loader_workers=0,
    )

    train_batch = next(iter(train_loader))
    valid_batch = next(iter(valid_loader))
    assert train_batch["image"].shape == (2, 4, *SPATIAL_SIZE)
    assert train_batch["image"].dtype == torch.float32
    assert train_batch["label"].dtype == torch.int64
    assert valid_batch["image"].shape[1:] == (4, *SPATIAL_SIZE)
    assert class_weights.shape == (2,)


def test_load_case_lists_applies_validation_limit():
    root = _toy_data_root()
    dataset_dir, datalist_path = _paths(root)

    train_cases, valid_cases = load_case_lists(
        datalist_json_path=str(datalist_path),
        dataset_base_dir=str(dataset_dir),
        validation_limit=1,
    )

    assert len(train_cases) == 2
    assert len(valid_cases) == 1


def test_load_case_lists_rejects_invalid_label():
    root = _toy_data_root()
    dataset_dir, datalist_path = _paths(root)
    payload = json.loads(datalist_path.read_text(encoding="utf-8"))
    payload["training"][0]["label"] = 2
    datalist_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid label"):
        load_case_lists(
            datalist_json_path=str(datalist_path),
            dataset_base_dir=str(dataset_dir),
        )


def test_compute_class_weights_uses_inverse_frequency():
    weights = compute_class_weights([0, 0, 1])

    assert weights.tolist() == pytest.approx([0.75, 1.5])
    assert compute_class_weights([1, 1, 1]).tolist() == [1.0, 1.0]


def test_deduplicate_cases_keeps_first_seen_order():
    cases = [
        {"subject": "a", "label": 0},
        {"subject": "b", "label": 1},
        {"subject": "a", "label": 0},
    ]

    assert [case["subject"] for case in deduplicate_cases(cases)] == ["a", "b"]


def test_build_validation_loader_rejects_empty_case_list():
    with pytest.raises(ValueError, match="No validation cases"):
        build_validation_loader([], spatial_size=SPATIAL_SIZE)


def test_evaluate_model_returns_full_metric_dict():
    root = _toy_data_root()
    dataset_dir, datalist_path = _paths(root)
    _, valid_cases = load_case_lists(
        datalist_json_path=str(datalist_path),
        dataset_base_dir=str(dataset_dir),
    )
    loader = build_validation_loader(
        valid_cases, spatial_size=SPATIAL_SIZE, batch_size=2
    )
    model = torch.nn.Sequential(
        torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
    )

    metrics = evaluate_model(model, loader, torch.device("cpu"))

    assert set(metrics) >= {
        "balanced_accuracy",
        "accuracy",
        "auroc",
        "macro_f1",
        "sensitivity",
        "specificity",
        "mcc",
        "confusion_matrix",
    }
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
