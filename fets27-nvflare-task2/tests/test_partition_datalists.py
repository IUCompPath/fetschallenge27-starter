from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from conftest import REPO_ROOT, make_test_dir

SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_partition_datalists.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_partition_datalists", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

TRAINING_ROWS = [
    ("FETS2027_Tr_00000", "3", "glioma"),
    ("FETS2027_Tr_00001", "3", "meningioma"),
    ("FETS2027_Tr_00002", "7", "glioma"),
    ("FETS2027_Tr_00003", "7", "meningioma"),
]
VALIDATION_ROWS = [
    ("FETS2027_Va_00000", "glioma"),
    ("FETS2027_Va_00001", "meningioma"),
]


def _write_case(case_dir: Path, case_id: str, *, missing_suffix: str | None = None) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("t1c", "t1n", "t2f", "t2w"):
        if suffix == missing_suffix:
            continue
        (case_dir / f"{case_id}_{suffix}.nii.gz").write_bytes(
            f"{case_id}-{suffix}".encode()
        )


def _write_csv(path: Path, header: list[str], rows: list[tuple[str, ...]]) -> None:
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_tree(
    root: Path,
    *,
    training_rows=TRAINING_ROWS,
    validation_rows=VALIDATION_ROWS,
    training_case_ids=None,
    extra_training_case: str | None = None,
    missing_suffix: str | None = None,
):
    training_dir = root / "fets2027-data" / "training"
    validation_dir = root / "fets2027-data" / "validation"
    training_ids = training_case_ids or [row[0] for row in training_rows]
    for case_id in training_ids:
        _write_case(training_dir / case_id, case_id, missing_suffix=missing_suffix)
    if extra_training_case:
        _write_case(training_dir / extra_training_case, extra_training_case)
    for case_id, _ in validation_rows:
        _write_case(validation_dir / case_id, case_id)

    training_csv = root / "Task2_Classification_Training_Partition.csv"
    validation_csv = root / "Task2_Classification_Validation_Labels.csv"
    _write_csv(training_csv, ["ID", "Site_ID", "Label"], [tuple(r) for r in training_rows])
    _write_csv(validation_csv, ["ID", "Label"], [tuple(r) for r in validation_rows])
    return training_csv, validation_csv, training_dir, validation_dir


def _build(root: Path, **kwargs) -> dict:
    training_csv, validation_csv, training_dir, validation_dir = _make_tree(root, **kwargs)
    return MODULE.build_layout(
        training_partition_csv=training_csv,
        validation_labels_csv=validation_csv,
        training_dir=training_dir,
        validation_dir=validation_dir,
        data_root=root / "prepared",
    )


def test_build_layout_partitions_sites_and_shares_validation():
    root = make_test_dir("partition-sites")
    summary = _build(root)

    datalist_dir = root / "prepared" / "brain_tumor" / "datalist"
    site_3 = json.loads((datalist_dir / "site-3.json").read_text(encoding="utf-8"))
    site_7 = json.loads((datalist_dir / "site-7.json").read_text(encoding="utf-8"))
    site_all = json.loads((datalist_dir / "site-All.json").read_text(encoding="utf-8"))

    assert summary["site_count"] == 2
    assert summary["training_count"] == 4
    assert summary["validation_count"] == 2
    assert summary["label_counts"] == {"0": 2, "1": 2}
    assert len(site_3["training"]) == 2
    assert len(site_7["training"]) == 2
    assert len(site_3["validation"]) == len(site_7["validation"]) == 2
    assert len(site_all["training"]) == 4
    assert (root / "prepared" / "brain_tumor" / "dataset" / "SOURCE.txt").exists()

    first = site_3["training"][0]
    assert first["subject"] == "FETS2027_Tr_00000"
    assert first["label"] == 0
    assert len(first["image"]) == 4
    assert Path(first["image"][0]).is_absolute()
    assert first["image"][0].endswith("FETS2027_Tr_00000_t1c.nii.gz")

    validation_labels = {case["subject"]: case["label"] for case in site_3["validation"]}
    assert validation_labels == {"FETS2027_Va_00000": 0, "FETS2027_Va_00001": 1}


def test_build_layout_copy_mode_copies_case_files():
    root = make_test_dir("partition-copy")
    training_csv, validation_csv, training_dir, validation_dir = _make_tree(root)

    MODULE.build_layout(
        training_partition_csv=training_csv,
        validation_labels_csv=validation_csv,
        training_dir=training_dir,
        validation_dir=validation_dir,
        data_root=root / "prepared",
        file_mode="copy",
    )

    dataset_dir = root / "prepared" / "brain_tumor" / "dataset"
    datalist = json.loads(
        (root / "prepared" / "brain_tumor" / "datalist" / "site-3.json").read_text(
            encoding="utf-8"
        )
    )
    first = datalist["training"][0]
    assert first["image"][0] == "FETS2027_Tr_00000/FETS2027_Tr_00000_t1c.nii.gz"
    assert (dataset_dir / first["image"][0]).exists()
    assert not (dataset_dir / "SOURCE.txt").exists()


def test_read_training_partition_rejects_missing_columns():
    root = make_test_dir("partition-bad-header")
    path = root / "bad.csv"
    _write_csv(path, ["ID", "Label"], [("FETS2027_Tr_00000", "glioma")])

    with pytest.raises(ValueError, match="missing required column"):
        MODULE.read_training_partition(path)


def test_build_layout_rejects_unknown_label():
    root = make_test_dir("partition-unknown-label")
    rows = list(TRAINING_ROWS)
    rows[0] = (rows[0][0], rows[0][1], "oligodendroglioma")

    with pytest.raises(ValueError, match="Unsupported label"):
        _build(root, training_rows=rows)


def test_build_layout_rejects_duplicate_case_ids():
    root = make_test_dir("partition-duplicate")
    rows = list(TRAINING_ROWS) + [TRAINING_ROWS[0]]

    with pytest.raises(ValueError, match="more than once"):
        _build(root, training_rows=rows)


def test_build_layout_rejects_unsafe_site_id():
    root = make_test_dir("partition-unsafe-site")
    rows = list(TRAINING_ROWS)
    rows[0] = (rows[0][0], "../evil", rows[0][2])

    with pytest.raises(ValueError, match="Invalid training site_id"):
        _build(root, training_rows=rows)


def test_build_layout_rejects_case_missing_from_tree():
    root = make_test_dir("partition-missing-case")
    rows = list(TRAINING_ROWS) + [("FETS2027_Tr_99999", "3", "glioma")]

    with pytest.raises(FileNotFoundError, match="missing from the case directory"):
        _build(
            root,
            training_rows=rows,
            training_case_ids=[row[0] for row in TRAINING_ROWS],
        )


def test_build_layout_rejects_case_absent_from_csv():
    root = make_test_dir("partition-extra-case")

    with pytest.raises(ValueError, match="absent from the CSV"):
        _build(root, extra_training_case="FETS2027_Tr_09999")


def test_build_layout_rejects_missing_modality():
    root = make_test_dir("partition-missing-modality")

    with pytest.raises(FileNotFoundError, match="missing required file"):
        _build(root, missing_suffix="t2w")


def test_build_layout_requires_both_classes():
    root = make_test_dir("partition-single-class")
    rows = [
        ("FETS2027_Tr_00000", "3", "glioma"),
        ("FETS2027_Tr_00001", "3", "glioma"),
        ("FETS2027_Tr_00002", "7", "glioma"),
        ("FETS2027_Tr_00003", "7", "glioma"),
    ]

    with pytest.raises(ValueError, match="must contain both classes"):
        _build(root, training_rows=rows)


def test_build_layout_rejects_partition_overlap():
    root = make_test_dir("partition-overlap")
    validation_rows = [("FETS2027_Tr_00000", "glioma"), VALIDATION_ROWS[1]]

    with pytest.raises(ValueError, match="appear in both partitions"):
        _build(root, validation_rows=validation_rows)
