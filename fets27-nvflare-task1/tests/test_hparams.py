from __future__ import annotations

from pathlib import Path

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
    )

    assert set(per_site) == {"site-1", "site-2"}
    assert "--aggregation_epochs 1" in per_site["site-1"]["train_args"]
    assert "--learning_rate 8e-05" in per_site["site-2"]["train_args"]
    assert "--label_transform brats_multi_channel" in per_site["site-1"]["train_args"]


def test_train_args_change_only_allowed_hparams():
    config = load_site_hparams(Path("participant/site_hparams.yaml"))
    cohort_spec = get_cohort_spec("glioma")
    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=Path("D:/data/glioma/dataset"),
        datalist_json_path=Path("D:/data/glioma/datalist/site-2.json"),
        hparams={
            **config["cohorts"]["glioma"]["defaults"],
            **config["cohorts"]["glioma"]["sites"]["site-2"],
        },
    )

    assert "--cohort glioma" in train_args
    assert "--out_channels 3" in train_args
    assert "--batch_size 1" in train_args
    assert "--weight_decay" in train_args
    assert "--unknown" not in train_args
