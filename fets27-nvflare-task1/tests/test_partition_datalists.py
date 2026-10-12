from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from conftest import make_test_dir

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_partition_datalists.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_partition_datalists", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_case(training_dir: Path, case_id: str) -> None:
    case_dir = training_dir / case_id
    case_dir.mkdir(parents=True)
    for suffix in ("t1c", "t1n", "t2f", "t2w", "seg"):
        (case_dir / f"{case_id}_{suffix}.nii.gz").write_bytes(
            f"{case_id}-{suffix}".encode()
        )


def _write_partition(path: Path, mapping: dict[str, int]) -> None:
    rows = ["ID,Site_ID"] + [f"{case_id},{site_id}" for case_id, site_id in mapping.items()]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_layout_partitions_by_csv_and_holds_out_validation():
    root = make_test_dir("partition-datalists")
    training = root / "training"
    data_root = root / "prepared"
    case_ids = [f"FETS2027_Tr_{index:05d}" for index in range(5)]
    for case_id in case_ids:
        _write_case(training, case_id)

    csv_path = root / "partition.csv"
    _write_partition(
        csv_path, {case_id: (index % 2) + 1 for index, case_id in enumerate(case_ids)}
    )

    summary = MODULE.build_layout(
        partition_csv=csv_path,
        training_dir=training,
        data_root=data_root,
        validation_count=1,
    )

    datalist_dir = data_root / "glioma" / "datalist"
    site_1 = json.loads((datalist_dir / "site-1.json").read_text(encoding="utf-8"))
    site_2 = json.loads((datalist_dir / "site-2.json").read_text(encoding="utf-8"))
    site_all = json.loads((datalist_dir / "site-All.json").read_text(encoding="utf-8"))

    assert summary["site_count"] == 2
    assert summary["validation_ids"] == [case_ids[-1]]
    assert summary["training_count"] == 4
    assert summary["validation_count"] == 1
    assert len(site_1["validation"]) == 1
    assert len(site_2["validation"]) == 1
    assert len(site_all["training"]) == 4
    assert len(site_1["training"][0]["image"]) == 4
    assert site_1["training"][0]["label"].endswith("_seg.nii.gz")
    assert (data_root / "glioma" / "dataset" / "SOURCE.txt").exists()


def test_build_layout_copy_mode_copies_case_files():
    root = make_test_dir("partition-datalists-copy")
    training = root / "training"
    data_root = root / "prepared"
    case_ids = [f"FETS2027_Tr_{index:05d}" for index in range(3)]
    for case_id in case_ids:
        _write_case(training, case_id)

    csv_path = root / "partition.csv"
    _write_partition(csv_path, {case_id: 7 for case_id in case_ids})

    MODULE.build_layout(
        partition_csv=csv_path,
        training_dir=training,
        data_root=data_root,
        validation_count=1,
        file_mode="copy",
    )

    datalist = json.loads(
        (data_root / "glioma" / "datalist" / "site-7.json").read_text(encoding="utf-8")
    )
    first = datalist["training"][0]
    assert first["image"][0] == f"{case_ids[0]}/{case_ids[0]}_t1c.nii.gz"
    assert (
        data_root / "glioma" / "dataset" / case_ids[0] / f"{case_ids[0]}_t1c.nii.gz"
    ).exists()


def test_read_partition_csv_rejects_missing_columns():
    root = make_test_dir("partition-bad-header")
    csv_path = root / "bad.csv"
    csv_path.write_text("case,site\nFETS2027_Tr_00000,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain 'ID' and 'Site_ID'"):
        MODULE.read_partition_csv(csv_path)


def test_build_layout_requires_full_partition_coverage():
    root = make_test_dir("partition-missing-coverage")
    training = root / "training"
    case_ids = [f"FETS2027_Tr_{index:05d}" for index in range(3)]
    for case_id in case_ids:
        _write_case(training, case_id)

    csv_path = root / "partition.csv"
    _write_partition(csv_path, {case_ids[0]: 1, case_ids[1]: 1})

    with pytest.raises(ValueError, match="absent from partition CSV"):
        MODULE.build_layout(
            partition_csv=csv_path,
            training_dir=training,
            data_root=root / "prepared",
            validation_count=1,
        )


def test_build_layout_missing_training_case_raises():
    root = make_test_dir("partition-missing-case")
    training = root / "training"
    for case_id in ("FETS2027_Tr_00000", "FETS2027_Tr_00001"):
        _write_case(training, case_id)

    csv_path = root / "partition.csv"
    _write_partition(
        csv_path,
        {"FETS2027_Tr_00000": 1, "FETS2027_Tr_00001": 1, "FETS2027_Tr_00002": 1},
    )

    with pytest.raises(FileNotFoundError, match="missing from training dir"):
        MODULE.build_layout(
            partition_csv=csv_path,
            training_dir=training,
            data_root=root / "prepared",
            validation_count=1,
        )
