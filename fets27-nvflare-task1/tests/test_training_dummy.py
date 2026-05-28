from __future__ import annotations

import json

import pytest

from conftest import make_test_dir
from fets27_challenge.cli import main
from fets27_challenge.training_dummy import (
    discover_training_dummy_cases,
    prepare_training_dummy_layout,
)


def test_prepare_training_dummy_layout_writes_site_datalists():
    root = make_test_dir("training-dummy")
    source_root = root / "source"
    data_root = root / "prepared"
    for index in range(1, 6):
        _write_case(source_root, f"FeTS2022_{index:05d}")

    summary = prepare_training_dummy_layout(
        source_root=source_root,
        data_root=data_root,
        site_count=2,
        validation_count=1,
    )

    datalist_dir = data_root / "glioma" / "datalist"
    site_1 = _read_json(datalist_dir / "site-1.json")
    site_2 = _read_json(datalist_dir / "site-2.json")
    site_all = _read_json(datalist_dir / "site-All.json")

    assert summary["case_count"] == 5
    assert summary["training_count"] == 4
    assert summary["validation_count"] == 1
    assert len(site_1["training"]) == 2
    assert len(site_2["training"]) == 2
    assert len(site_all["training"]) == 4
    assert len(site_1["validation"]) == 1
    assert len(site_1["training"][0]["image"]) == 4
    assert site_1["training"][0]["label"].endswith("_seg.nii.gz")
    assert (data_root / "glioma" / "dataset" / "SOURCE.txt").exists()


def test_prepare_training_dummy_layout_can_copy_dataset_files():
    root = make_test_dir("training-dummy-copy")
    source_root = root / "source"
    data_root = root / "prepared"
    for index in range(1, 4):
        _write_case(source_root, f"FeTS2022_{index:05d}")

    prepare_training_dummy_layout(
        source_root=source_root,
        data_root=data_root,
        site_count=1,
        validation_count=1,
        file_mode="copy",
    )

    datalist = _read_json(data_root / "glioma" / "datalist" / "site-1.json")
    first_entry = datalist["training"][0]
    assert first_entry["image"][0] == "FeTS2022_00001/FeTS2022_00001_flair.nii.gz"
    assert (
        data_root
        / "glioma"
        / "dataset"
        / "FeTS2022_00001"
        / "FeTS2022_00001_flair.nii.gz"
    ).exists()


def test_prepare_training_dummy_cli_writes_runtime_layout(capsys):
    root = make_test_dir("training-dummy-cli")
    source_root = root / "source"
    data_root = root / "prepared"
    for index in range(1, 5):
        _write_case(source_root, f"FeTS2022_{index:05d}")

    main(
        [
            "prepare-training-dummy",
            "--source-root",
            str(source_root),
            "--data-root",
            str(data_root),
            "--site-count",
            "2",
            "--validation-count",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "Prepared training dummy layout" in output
    datalist_dir = data_root / "glioma" / "datalist"
    assert (datalist_dir / "site-1.json").exists()
    assert (datalist_dir / "site-2.json").exists()
    assert (datalist_dir / "site-All.json").exists()


def test_discover_training_dummy_cases_reports_missing_files():
    root = make_test_dir("training-dummy-missing")
    source_root = root / "source"
    case_dir = source_root / "FeTS2022_00001"
    case_dir.mkdir(parents=True)
    (case_dir / "FeTS2022_00001_flair.nii.gz").write_bytes(b"dummy")

    with pytest.raises(FileNotFoundError, match="missing required file"):
        discover_training_dummy_cases(source_root)


def test_prepare_training_dummy_layout_rejects_empty_sites():
    root = make_test_dir("training-dummy-empty-sites")
    source_root = root / "source"
    data_root = root / "prepared"
    for index in range(1, 4):
        _write_case(source_root, f"FeTS2022_{index:05d}")

    with pytest.raises(ValueError, match="empty training sites"):
        prepare_training_dummy_layout(
            source_root=source_root,
            data_root=data_root,
            site_count=3,
            validation_count=1,
        )


def _write_case(source_root, case_id: str):
    case_dir = source_root / case_id
    case_dir.mkdir(parents=True)
    for suffix in ("flair", "t1", "t1ce", "t2", "seg"):
        (case_dir / f"{case_id}_{suffix}.nii.gz").write_bytes(f"{case_id}-{suffix}".encode())


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
