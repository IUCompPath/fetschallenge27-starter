from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml
from conftest import make_test_dir

from fets27_classification.cohort_registry import get_cohort_spec
from fets27_classification.participant_config import (
    build_per_site_config,
    build_site_train_args,
    load_site_hparams,
    validate_site_hparams,
)


def test_load_site_hparams_accepts_sample_file():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))

    assert "cohorts" in config
    assert set(config["cohorts"]) == {"brain_tumor"}


def test_invalid_key_rejected():
    bad_payload = {
        "cohorts": {
            "brain_tumor": {
                "defaults": {
                    "learning_rate": 1e-4,
                    "batch_size": 1,
                    "aggregation_epochs": 1,
                    "weight_decay": 1e-5,
                    "fedproxloss_mu": 0.0,
                    "cache_dataset": 0.0,
                    "not_allowed": 7,
                },
                "sites": {"site-1": {}},
            },
        }
    }
    temp_dir = make_test_dir("invalid-hparams")
    path = temp_dir / "site_hparams.yaml"
    path.write_text(yaml.safe_dump(bad_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported keys"):
        validate_site_hparams(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("aggregation_epochs", 0, "positive integer"),
        ("batch_size", 0, "positive integer"),
        ("learning_rate", 0.0, "greater than zero"),
        ("weight_decay", -1.0, "non-negative"),
        ("fedproxloss_mu", float("nan"), "finite number"),
        ("cache_dataset", 1.1, "between zero and one"),
    ],
)
def test_invalid_hparam_value_rejected(key, value, message):
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    config["cohorts"]["brain_tumor"]["defaults"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_site_hparams(config)


def test_build_per_site_config_merges_defaults_and_site_overrides():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("brain_tumor")
    per_site = build_per_site_config(
        config,
        cohort_spec,
        dataset_base_dir=Path("D:/data/brain_tumor/dataset"),
        site_datalist_paths={
            "site-0": Path("D:/data/brain_tumor/datalist/site-0.json"),
            "site-1": Path("D:/data/brain_tumor/datalist/site-1.json"),
        },
        data_loader_workers=3,
        validation_limit=8,
    )

    assert set(per_site) == {"site-0", "site-1"}
    assert "--aggregation_epochs 1" in per_site["site-0"]["train_args"]
    assert "--learning_rate 0.0001" in per_site["site-1"]["train_args"]
    assert "--spatial_size 96 96 96" in per_site["site-0"]["train_args"]
    assert "--data_loader_workers 3" in per_site["site-0"]["train_args"]
    assert "--validation_limit 8" in per_site["site-0"]["train_args"]


def test_train_args_change_only_allowed_hparams():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("brain_tumor")
    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=Path("D:/data/brain_tumor/dataset"),
        datalist_json_path=Path("D:/data/brain_tumor/datalist/site-1.json"),
        participant_client_file=Path("participant/client.py"),
        hparams={
            **config["cohorts"]["brain_tumor"]["defaults"],
            **config["cohorts"]["brain_tumor"]["sites"]["site-1"],
        },
    )

    assert "--cohort brain_tumor" in train_args
    assert "--num_classes 2" in train_args
    assert "--batch_size 4" in train_args
    assert "--weight_decay" in train_args
    assert "--participant_client_file" in train_args
    assert "--validation_limit" not in train_args
    assert "--unknown" not in train_args
    participant_token = train_args.split()[
        train_args.split().index("--participant_client_file") + 1
    ]
    assert Path(unquote(participant_token)) == Path("participant/client.py").resolve()


def test_train_args_support_site_resolved_datalists():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("brain_tumor")
    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=Path("D:/data/brain_tumor/dataset"),
        datalist_dir=Path("D:/data/brain_tumor/datalist"),
        hparams=config["cohorts"]["brain_tumor"]["defaults"],
        site_hparams_file=Path("participant/site_hparams.yaml"),
    )

    assert "--datalist_json_path" not in train_args
    assert "--datalist_dir" in train_args
    assert "--site_hparams_file" in train_args
    tokens = train_args.split()
    assert Path(unquote(tokens[tokens.index("--datalist_dir") + 1])) == Path(
        "D:/data/brain_tumor/datalist"
    ).resolve()
    assert Path(unquote(tokens[tokens.index("--site_hparams_file") + 1])) == Path(
        "participant/site_hparams.yaml"
    ).resolve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "datalist_json_path": Path("site-1.json"),
            "datalist_dir": Path("datalist"),
        },
    ],
)
def test_train_args_require_exactly_one_datalist_argument(kwargs):
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("brain_tumor")

    with pytest.raises(ValueError, match="exactly one of"):
        build_site_train_args(
            cohort_spec,
            dataset_base_dir=Path("D:/data/brain_tumor/dataset"),
            hparams=config["cohorts"]["brain_tumor"]["defaults"],
            **kwargs,
        )


def test_train_args_encode_paths_with_spaces_as_single_tokens():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("brain_tumor")
    dataset_path = Path(".test-artifacts/path with spaces/dataset")
    datalist_path = Path(".test-artifacts/path with spaces/site-1.json")
    participant_path = Path(".test-artifacts/path with spaces/client.py")

    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=dataset_path,
        datalist_json_path=datalist_path,
        participant_client_file=participant_path,
        hparams=config["cohorts"]["brain_tumor"]["defaults"],
    )
    tokens = train_args.split()

    for flag, expected in (
        ("--dataset_base_dir", dataset_path),
        ("--datalist_json_path", datalist_path),
        ("--participant_client_file", participant_path),
    ):
        token = tokens[tokens.index(flag) + 1]
        assert " " not in token
        assert Path(unquote(token)) == expected.resolve()
