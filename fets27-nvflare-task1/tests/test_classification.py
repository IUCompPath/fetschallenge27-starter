from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import REPO_ROOT, make_test_dir

from fets27_challenge import cli
from fets27_challenge.cohort_registry import COHORT_REGISTRY, get_cohort_spec
from fets27_challenge.evaluation import CohortScore, SiteScore, write_summary


def _tiny_models():
    from fets27_challenge.models import ResNet3D, ViT3D

    return [
        ResNet3D(in_channels=2, num_classes=3),
        ViT3D(in_channels=2, num_classes=3, img_size=(16, 16, 16)),
    ]


@pytest.mark.parametrize("index", [0, 1], ids=["resnet3d", "vit3d"])
def test_classification_models_return_logits_with_float_only_state(index):
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    model = _tiny_models()[index]

    assert model(torch.zeros(2, 2, 16, 16, 16)).shape == (2, 3)
    assert model.task == "classification"
    # An integer buffer (BatchNorm's num_batches_tracked) would be float-averaged.
    assert {v.dtype for v in model.state_dict().values()} == {torch.float32}


def test_model_wrapper_selects_architecture(monkeypatch):
    pytest.importorskip("monai")
    from fets27_challenge.models import ViT3D, create_model_for_cohort

    spec = get_cohort_spec("classification")
    monkeypatch.setitem(
        COHORT_REGISTRY, "classification", replace(spec, model_wrapper="ViT3D")
    )
    model = create_model_for_cohort("classification")

    assert isinstance(model, ViT3D)
    assert model.img_size == spec.roi_size  # NVFLARE rebuilds the model from this


def test_evaluate_model_reports_accuracy():
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    from fets27_challenge.classification import build_components
    from fets27_challenge.data_pipeline import evaluate_model

    class AlwaysClassOne(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 1.0]]).expand(x.shape[0], -1)

    _, _, inferer, post_transform, valid_metric = build_components(
        label_transform="", roi_size=(4, 4, 4), infer_roi_size=(4, 4, 4)
    )
    loader = [
        {"image": torch.zeros(1, 1, 4, 4, 4), "label": torch.tensor([label])}
        for label in (1, 1, 0, 1)
    ]

    score = evaluate_model(
        AlwaysClassOne(), loader, inferer, post_transform, valid_metric, "cpu"
    )
    assert score == pytest.approx(0.75)


def test_toy_classification_data_loads_and_baseline_local_train_is_valid():
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    pytest.importorskip("nibabel")
    from fets27_challenge.data_pipeline import build_dataloaders
    from fets27_challenge.local_training import (
        load_participant_local_train,
        parameter_spec,
        validate_local_train_result,
    )
    from fets27_challenge.models import ResNet3D
    from fets27_challenge.synthetic_data import generate_toy_dataset

    data_root = make_test_dir("toy-classification")
    generate_toy_dataset(data_root)
    spec = get_cohort_spec("classification")
    train_loader, *_ = build_dataloaders(
        task=spec.task,
        dataset_base_dir=str(spec.dataset_dir(data_root)),
        datalist_json_path=str(spec.datalist_dir(data_root) / "site-1.json"),
        label_transform=spec.label_transform,
        batch_size=2,
        cache_rate=0.0,
        roi_size=(16, 16, 16),
        infer_roi_size=(16, 16, 16),
        data_loader_workers=1,
    )
    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, spec.in_channels, 16, 16, 16)
    assert batch["label"].dtype == torch.int64
    assert set(batch["label"].tolist()) == {0, 1}

    class _Writer:
        def add_scalar(self, *args):
            pass

    model = ResNet3D(in_channels=spec.in_channels, num_classes=spec.out_channels)
    result = load_participant_local_train(REPO_ROOT / "participant" / "client.py")(
        model=model,
        train_loader=train_loader,
        device=torch.device("cpu"),
        current_round=0,
        aggregation_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        fedproxloss_mu=0.0,
        server_meta={},
        state={},
        summary_writer=_Writer(),
        client_name="site-1",
    )
    validate_local_train_result(result, parameter_spec(model.state_dict()))
    assert result.metrics["train_loss"] > 0


def test_write_summary_names_each_tasks_metric():
    out_dir = make_test_dir("summary-metrics")
    seg = CohortScore("glioma", "G", 0.5, "m.pt", "ws", [SiteScore("site-1", "val_dice", 0.5)])
    cls = CohortScore("classification", "C", 0.75, "m.pt", "ws", [SiteScore("site-1", "val_acc", 0.75)])

    json_path, csv_path = write_summary(out_dir, mode="local", cohort_scores=[cls])
    assert json.loads(json_path.read_text())["cohorts"][0]["site_scores"] == [
        {"site_name": "site-1", "val_acc": 0.75}
    ]
    row = next(csv.DictReader(csv_path.open()))
    assert (row["val_acc"], row["val_dice"]) == ("0.75", "")

    _, csv_path = write_summary(out_dir, mode="local", cohort_scores=[seg])
    row = next(csv.DictReader(csv_path.open()))
    assert (row["val_dice"], row["val_acc"]) == ("0.5", "")


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], ["glioma"]),  # unchanged default: the segmentation track
        (["--task", "classification"], ["classification"]),
        (["--cohort", "classification"], ["classification"]),
    ],
)
def test_cli_selects_cohorts_by_task(monkeypatch, extra_args, expected):
    captured = {}

    def fake_run_challenge(**kwargs):
        captured.update(kwargs)
        return Path("a.json"), Path("a.csv"), []

    monkeypatch.setattr(cli, "run_challenge", fake_run_challenge)
    cli.main(
        ["run-local", "--data-root", "d", "--workspace", "w", "--output-dir", "o", *extra_args]
    )
    assert captured["cohort_names"] == expected
