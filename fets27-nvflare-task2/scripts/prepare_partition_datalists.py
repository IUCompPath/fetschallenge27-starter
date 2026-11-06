#!/usr/bin/env python3
"""Build per-site brain-tumor datalists from the released FeTS 2027 Task 2 files.

This organizer-side helper reads the Task 2 classification partition and
validation labels shipped under ``fets2027-data/``:

    Task2_Classification_Training_Partition.csv   (ID,Site_ID,Label)
    Task2_Classification_Validation_Labels.csv    (ID,Label)

and writes the runtime layout consumed by ``run-local``/``run-official``:

    <data-root>/brain_tumor/datalist/site-<id>.json
    <data-root>/brain_tumor/datalist/site-All.json
    <data-root>/brain_tumor/dataset/SOURCE.txt    (absolute mode only)

Every site datalist holds that site's training cases plus the official shared
``validation`` list, so all simulated clients evaluate the same cases. Only JSON
metadata (and copies of the images in ``--file-mode copy``) is written below
``--data-root``; the released data tree is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

DEFAULT_COHORT = "brain_tumor"
DEFAULT_MODALITY_SUFFIXES = ("t1c", "t1n", "t2f", "t2w")
LABEL_MAP = {"glioma": 0, "meningioma": 1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the partition datalist helper.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-partition-csv",
        type=Path,
        required=True,
        help="CSV with ID,Site_ID,Label columns (Task2_Classification_Training_Partition.csv)",
    )
    parser.add_argument(
        "--validation-labels-csv",
        type=Path,
        required=True,
        help="CSV with ID,Label columns (Task2_Classification_Validation_Labels.csv)",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="Directory of FETS2027_Tr_* case folders containing the four modalities",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory of FETS2027_Va_* case folders containing the four modalities",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Destination data root; writes <data-root>/brain_tumor/datalist and dataset",
    )
    parser.add_argument(
        "--cohort",
        default=DEFAULT_COHORT,
        help="Cohort directory name under the data root (default: brain_tumor)",
    )
    parser.add_argument(
        "--modality-suffixes",
        nargs="+",
        default=list(DEFAULT_MODALITY_SUFFIXES),
        help="Modality filename suffixes (default: t1c t1n t2f t2w)",
    )
    parser.add_argument(
        "--file-mode",
        choices=["absolute", "copy"],
        default="absolute",
        help="Reference source files in place or copy them into brain_tumor/dataset",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}; "
                f"found {sorted(fieldnames)}"
            )
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def read_training_partition(path: Path) -> dict[str, tuple[str, int]]:
    """Read a case-to-site-and-label mapping from the training partition CSV.

    Args:
        path: Path to a CSV with ``ID``, ``Site_ID``, and ``Label`` columns.

    Returns:
        A mapping of case ID to ``(site_id, label_index)``.

    Raises:
        ValueError: If columns are missing, rows are malformed, or a case is
            assigned twice.
    """
    mapping: dict[str, tuple[str, int]] = {}
    for row in _read_csv(Path(path), {"ID", "Site_ID", "Label"}):
        case_id = row["ID"]
        site_id = row["Site_ID"]
        if not case_id or not site_id:
            raise ValueError(f"{path} contains a row with an empty ID or Site_ID")
        if case_id in mapping:
            raise ValueError(f"{path} lists case {case_id!r} more than once")
        if not _is_safe_site_id(site_id):
            raise ValueError(f"Invalid training site_id {site_id!r} for {case_id!r}")
        mapping[case_id] = (site_id, _label_index(row["Label"], case_id))
    return mapping


def read_validation_labels(path: Path) -> dict[str, int]:
    """Read the official validation label mapping.

    Args:
        path: Path to a CSV with ``ID`` and ``Label`` columns.

    Returns:
        A mapping of validation case ID to label index.

    Raises:
        ValueError: If columns are missing or rows are malformed.
    """
    mapping: dict[str, int] = {}
    for row in _read_csv(Path(path), {"ID", "Label"}):
        case_id = row["ID"]
        if not case_id:
            raise ValueError(f"{path} contains a row with an empty ID")
        if case_id in mapping:
            raise ValueError(f"{path} lists case {case_id!r} more than once")
        mapping[case_id] = _label_index(row["Label"], case_id)
    return mapping


def discover_cases(
    case_dir: Path, modality_suffixes: tuple[str, ...]
) -> dict[str, tuple[Path, ...]]:
    """Discover case folders and their four modality paths.

    Args:
        case_dir: Directory containing ``FETS2027_*`` case folders.
        modality_suffixes: Modality filename suffixes expected per case.

    Returns:
        A mapping of case ID to its modality paths.

    Raises:
        FileNotFoundError: If the directory is missing or a case lacks a modality.
        ValueError: If no case folders are found.
    """
    root = Path(case_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {root}")

    cases: dict[str, tuple[Path, ...]] = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        case_id = folder.name
        modalities = tuple(
            folder / f"{case_id}_{suffix}.nii.gz" for suffix in modality_suffixes
        )
        missing = [path.name for path in modalities if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Case {case_id} is missing required file(s): {', '.join(missing)}"
            )
        cases[case_id] = modalities

    if not cases:
        raise FileNotFoundError(f"No case folders found under {root}")
    return cases


def build_layout(
    *,
    training_partition_csv: Path,
    validation_labels_csv: Path,
    training_dir: Path,
    validation_dir: Path,
    data_root: Path,
    cohort: str = DEFAULT_COHORT,
    modality_suffixes: tuple[str, ...] = DEFAULT_MODALITY_SUFFIXES,
    file_mode: str = "absolute",
) -> dict:
    """Write the runtime data-root layout from the released Task 2 files.

    Args:
        training_partition_csv: CSV mapping case IDs to site IDs and labels.
        validation_labels_csv: CSV mapping validation case IDs to labels.
        training_dir: Directory containing the training case folders.
        validation_dir: Directory containing the validation case folders.
        data_root: Destination data root.
        cohort: Cohort directory name under the data root.
        modality_suffixes: Modality filename suffixes expected per case.
        file_mode: ``absolute`` to reference files in place or ``copy`` to copy them.

    Returns:
        A summary dictionary describing the generated layout.

    Raises:
        ValueError: If labels, coverage, or class balance are invalid.
        FileNotFoundError: If cases or modalities listed in the CSV are missing.
    """
    if file_mode not in {"absolute", "copy"}:
        raise ValueError("file_mode must be 'absolute' or 'copy'.")
    if not cohort or Path(cohort).name != cohort or cohort in {".", ".."}:
        raise ValueError(f"Invalid cohort name {cohort!r}.")

    partition = read_training_partition(training_partition_csv)
    validation_labels = read_validation_labels(validation_labels_csv)
    overlap = sorted(set(partition) & set(validation_labels))
    if overlap:
        raise ValueError(
            f"{len(overlap)} case(s) appear in both partitions, e.g. {overlap[:5]}"
        )

    training_cases = discover_cases(training_dir, tuple(modality_suffixes))
    validation_cases = discover_cases(validation_dir, tuple(modality_suffixes))

    _require_full_coverage(partition, training_cases, "training")
    _require_full_coverage(validation_labels, validation_cases, "validation")
    _require_both_classes({case: label for case, (_, label) in partition.items()}, "training")
    _require_both_classes(validation_labels, "validation")

    cohort_root = Path(data_root) / cohort
    dataset_dir = cohort_root / "dataset"
    datalist_dir = cohort_root / "datalist"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datalist_dir.mkdir(parents=True, exist_ok=True)

    if file_mode == "copy":
        for case_id, modalities in training_cases.items():
            _copy_case(case_id, modalities, dataset_dir)
        for case_id, modalities in validation_cases.items():
            _copy_case(case_id, modalities, dataset_dir)

        def path_builder(case_id: str, path: Path) -> str:
            return f"{case_id}/{path.name}"

    else:
        (dataset_dir / "SOURCE.txt").write_text(
            "Datalists in ../datalist reference the released files in place:\n"
            f"training:   {Path(training_dir).resolve()}\n"
            f"validation: {Path(validation_dir).resolve()}\n",
            encoding="utf-8",
        )

        def path_builder(case_id: str, path: Path) -> str:
            return str(path.resolve())

    validation_entries = [
        _case_to_entry(case_id, validation_cases[case_id], label, path_builder)
        for case_id, label in sorted(validation_labels.items())
    ]

    site_cases: dict[str, list[str]] = {}
    for case_id, (site_id, _) in partition.items():
        site_cases.setdefault(site_id, []).append(case_id)

    all_training_entries = []
    site_counts = {}
    for site_id in sorted(site_cases, key=_site_sort_key):
        entries = [
            _case_to_entry(
                case_id, training_cases[case_id], partition[case_id][1], path_builder
            )
            for case_id in sorted(site_cases[site_id])
        ]
        _write_json(
            datalist_dir / f"site-{site_id}.json",
            {"training": entries, "validation": validation_entries},
        )
        all_training_entries.extend(entries)
        site_counts[f"site-{site_id}"] = len(entries)

    _write_json(
        datalist_dir / "site-All.json",
        {"training": all_training_entries, "validation": validation_entries},
    )

    label_counts = {
        str(index): sum(1 for _, label in partition.values() if label == index)
        for index in range(len(LABEL_MAP))
    }
    return {
        "training_partition_csv": str(Path(training_partition_csv).resolve()),
        "validation_labels_csv": str(Path(validation_labels_csv).resolve()),
        "training_dir": str(Path(training_dir).resolve()),
        "validation_dir": str(Path(validation_dir).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "cohort": cohort,
        "dataset_dir": str(dataset_dir),
        "datalist_dir": str(datalist_dir),
        "file_mode": file_mode,
        "training_count": len(all_training_entries),
        "validation_count": len(validation_entries),
        "site_count": len(site_counts),
        "label_counts": label_counts,
        "sites": site_counts,
    }


def _require_full_coverage(records: dict, cases: dict, split: str) -> None:
    """Require an exact, two-way match between CSV rows and case folders."""
    missing = sorted(set(records) - set(cases))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {split} case(s) listed in the CSV are missing from the "
            f"case directory, e.g. {missing[:5]}"
        )
    extra = sorted(set(cases) - set(records))
    if extra:
        raise ValueError(
            f"{len(extra)} {split} case(s) are absent from the CSV, e.g. {extra[:5]}"
        )


def _require_both_classes(labels: dict[str, int], split: str) -> None:
    present = sorted(set(labels.values()))
    if present != sorted(LABEL_MAP.values()):
        raise ValueError(
            f"The {split} partition must contain both classes "
            f"{sorted(LABEL_MAP.values())}; found {present}"
        )


def _label_index(raw_label: str, case_id: str) -> int:
    key = (raw_label or "").strip().casefold()
    if key not in LABEL_MAP:
        raise ValueError(
            f"Unsupported label {raw_label!r} for {case_id!r}; expected one of "
            f"{sorted(LABEL_MAP)}."
        )
    return LABEL_MAP[key]


def _is_safe_site_id(site_id: str) -> bool:
    """Return whether a site identifier is safe to use as a datalist filename."""
    return bool(site_id) and site_id not in {".", ".."} and Path(site_id).name == site_id


def _site_sort_key(site_id: str):
    stripped = site_id.lstrip("+-")
    return (0, int(site_id)) if stripped.isdigit() else (1, site_id)


def _case_to_entry(case_id: str, modalities, label: int, path_builder) -> dict:
    return {
        "image": [path_builder(case_id, path) for path in modalities],
        "label": int(label),
        "subject": case_id,
    }


def _copy_case(case_id: str, modalities, dataset_dir: Path) -> None:
    target_dir = dataset_dir / case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in modalities:
        target_path = target_dir / source_path.name
        if (
            not target_path.exists()
            or target_path.stat().st_size != source_path.stat().st_size
        ):
            shutil.copy2(source_path, target_path)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    """Run the partition datalist preparation CLI."""
    args = parse_args(argv)
    summary = build_layout(
        training_partition_csv=args.training_partition_csv,
        validation_labels_csv=args.validation_labels_csv,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        data_root=args.data_root,
        cohort=args.cohort,
        modality_suffixes=tuple(args.modality_suffixes),
        file_mode=args.file_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
