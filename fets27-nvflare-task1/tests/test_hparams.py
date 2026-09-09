from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml
from conftest import make_test_dir

from fets27_challenge.cohort_registry import get_cohort_spec
from fets27_challenge.participant_config import (
    build_per_site_config,
    build_site_train_args,
    load_site_hparams,
    validate_site_hparams,
)


def test_load_site_hparams_accepts_sample_file():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    assert "cohorts" in config
    assert set(config["cohorts"]) == {"glioma"}


def test_invalid_key_rejected():
    bad_payload = {
        "cohorts": {
            "glioma": {
                "defaults": {
                    "learning_rate": 1e-4,
                    "batch_size": 1,
                    "aggregation_epochs": 1,
                    "weight_decay": 1e-5,
                    "fedproxloss_mu": 0.0,
                    "cache_dataset": 0.0,
                    "not_allowed": 7,
                },
                "sites": {"site-1": {}, "site-2": {}},
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
    config["cohorts"]["glioma"]["defaults"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_site_hparams(config)


def test_build_per_site_config_merges_defaults_and_site_overrides():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("glioma")
    per_site = build_per_site_config(
        config,
        cohort_spec,
        dataset_base_dir=Path("D:/data/glioma/dataset"),
        site_datalist_paths={
            "site-1": Path("D:/data/glioma/datalist/site-1.json"),
            "site-2": Path("D:/data/glioma/datalist/site-2.json"),
        },
        data_loader_workers=3,
    )

    assert set(per_site) == {"site-1", "site-2"}
    assert "--aggregation_epochs 1" in per_site["site-1"]["train_args"]
    assert "--learning_rate 8e-05" in per_site["site-2"]["train_args"]
    assert "--label_transform brats_multi_channel" in per_site["site-1"]["train_args"]
    assert "--data_loader_workers 3" in per_site["site-1"]["train_args"]


def test_train_args_change_only_allowed_hparams():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("glioma")
    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=Path("D:/data/glioma/dataset"),
        datalist_json_path=Path("D:/data/glioma/datalist/site-2.json"),
        participant_client_file=Path("participant/client.py"),
        hparams={
            **config["cohorts"]["glioma"]["defaults"],
            **config["cohorts"]["glioma"]["sites"]["site-2"],
        },
    )

    assert "--cohort glioma" in train_args
    assert "--out_channels 3" in train_args
    assert "--batch_size 2" in train_args
    assert "--roi_size 128 128 128" in train_args
    assert "--weight_decay" in train_args
    assert "--participant_client_file" in train_args
    participant_token = train_args.split()[
        train_args.split().index("--participant_client_file") + 1
    ]
    assert Path(unquote(participant_token)) == Path("participant/client.py").resolve()
    assert "--unknown" not in train_args


def test_train_args_encode_paths_with_spaces_as_single_tokens():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("glioma")
    dataset_path = Path(".test-artifacts/path with spaces/dataset")
    datalist_path = Path(".test-artifacts/path with spaces/site-1.json")
    participant_path = Path(".test-artifacts/path with spaces/client.py")

    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=dataset_path,
        datalist_json_path=datalist_path,
        participant_client_file=participant_path,
        hparams=config["cohorts"]["glioma"]["defaults"],
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
