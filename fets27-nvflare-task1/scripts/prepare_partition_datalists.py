#!/usr/bin/env python3
"""Build per-site glioma datalists from the FeTS 2027 Task 1 partition CSV.

This organizer-side helper reads ``Task1_Segmentation_Partition.csv`` (columns
``ID,Site_ID``), matches every listed case against a FeTS training tree, holds
out a configurable number of labeled cases for validation, and writes the
runtime layout consumed by ``run-local``/``run-official``:

    <data-root>/glioma/datalist/site-<id>.json
    <data-root>/glioma/datalist/site-All.json
    <data-root>/glioma/dataset/SOURCE.txt   (absolute mode only)

Each site datalist holds that site's training cases plus a shared ``validation``
list, so every simulated client evaluates the same held-out cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

DEFAULT_MODALITY_SUFFIXES = ("t1c", "t1n", "t2f", "t2w")
SEGMENTATION_SUFFIX = "seg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the partition datalist helper.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-csv",
        type=Path,
        required=True,
        help="CSV with ID,Site_ID columns (e.g. Task1_Segmentation_Partition.csv)",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="Directory of FETS2027_Tr_* case folders containing modalities and _seg masks",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Destination data root; writes <data-root>/glioma/datalist and dataset",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=16,
        help="Number of labeled cases held out for the shared validation list",
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
        help="Reference source files in place or copy them into glioma/dataset",
    )
    return parser.parse_args(argv)


def read_partition_csv(path: Path) -> dict[str, str]:
    """Read a case-to-site partition mapping from a CSV file.

    Args:
        path: Path to a CSV with ``ID`` and ``Site_ID`` columns.

    Returns:
        A mapping of case ID to site ID.

    Raises:
        ValueError: If required columns are missing or rows are malformed.
    """
    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        if not {"ID", "Site_ID"}.issubset(fieldnames):
            raise ValueError(
                f"{path} must contain 'ID' and 'Site_ID' columns; found {sorted(fieldnames)}"
            )
        for row in reader:
            case_id = (row.get("ID") or "").strip()
            site_id = (row.get("Site_ID") or "").strip()
            if not case_id or not site_id:
                raise ValueError(f"{path} contains a row with an empty ID or Site_ID")
            if mapping.get(case_id, site_id) != site_id:
                raise ValueError(f"{path} assigns case {case_id!r} to multiple sites")
            mapping[case_id] = site_id
    if not mapping:
        raise ValueError(f"{path} contains no partition rows")
    return mapping


def discover_cases(
    training_dir: Path, modality_suffixes: tuple[str, ...]
) -> dict[str, tuple[tuple[Path, ...], Path]]:
    """Discover FeTS cases and their modality and label paths.

    Args:
        training_dir: Directory containing ``FETS2027_Tr_*`` case directories.
        modality_suffixes: Modality filename suffixes expected for every case.

    Returns:
        A mapping of case ID to ``(modality_paths, label_path)``.

    Raises:
        FileNotFoundError: If the training directory or required case files are missing.
        ValueError: If no cases are discovered.
    """
    root = Path(training_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {root}")

    cases: dict[str, tuple[tuple[Path, ...], Path]] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        modalities = tuple(
            case_dir / f"{case_id}_{suffix}.nii.gz" for suffix in modality_suffixes
        )
        label = case_dir / f"{case_id}_{SEGMENTATION_SUFFIX}.nii.gz"
        missing = [path.name for path in (*modalities, label) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Case {case_id} is missing required file(s): {', '.join(missing)}"
            )
        cases[case_id] = (modalities, label)

    if not cases:
        raise FileNotFoundError(f"No case folders found under {root}")
    return cases


def build_layout(
    *,
    partition_csv: Path,
    training_dir: Path,
    data_root: Path,
    validation_count: int = 16,
    modality_suffixes: tuple[str, ...] = DEFAULT_MODALITY_SUFFIXES,
    file_mode: str = "absolute",
) -> dict:
    """Write the runtime data-root layout from a partition CSV and training tree.

    Args:
        partition_csv: CSV mapping case IDs to site IDs.
        training_dir: Directory containing the source case folders.
        data_root: Destination data root for the ``glioma`` cohort.
        validation_count: Number of labeled cases held out for validation.
        modality_suffixes: Modality filename suffixes expected per case.
        file_mode: ``absolute`` to reference files in place or ``copy`` to copy them.

    Returns:
        A summary dictionary describing the generated layout.

    Raises:
        ValueError: If the configuration or partition coverage is invalid.
        FileNotFoundError: If partition cases are missing from the training tree.
    """
    if file_mode not in {"absolute", "copy"}:
        raise ValueError("file_mode must be 'absolute' or 'copy'.")
    if validation_count < 1:
        raise ValueError("validation_count must be at least 1.")

    partition = read_partition_csv(partition_csv)
    cases = discover_cases(training_dir, tuple(modality_suffixes))

    missing = sorted(set(partition) - set(cases))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} partition case(s) missing from training dir, "
            f"e.g. {missing[:5]}"
        )
    extra = sorted(set(cases) - set(partition))
    if extra:
        raise ValueError(
            f"{len(extra)} training case(s) absent from partition CSV, e.g. {extra[:5]}"
        )
    if validation_count >= len(cases):
        raise ValueError("validation_count must be smaller than the number of cases.")

    case_ids = sorted(cases)
    validation_ids = case_ids[-validation_count:]
    validation_set = set(validation_ids)

    cohort_root = Path(data_root) / "glioma"
    dataset_dir = cohort_root / "dataset"
    datalist_dir = cohort_root / "datalist"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datalist_dir.mkdir(parents=True, exist_ok=True)

    if file_mode == "copy":
        for case_id in case_ids:
            _copy_case(case_id, cases[case_id], dataset_dir)

        def path_builder(case_id: str, path: Path) -> str:
            return f"{case_id}/{path.name}"

    else:
        (dataset_dir / "SOURCE.txt").write_text(
            "Datalists in ../datalist reference the source files in place:\n"
            f"{Path(training_dir).resolve()}\n",
            encoding="utf-8",
        )

        def path_builder(case_id: str, path: Path) -> str:
            return str(path.resolve())

    validation_entries = [
        _case_to_entry(case_id, cases[case_id], path_builder)
        for case_id in validation_ids
    ]

    site_cases: dict[str, list[str]] = {}
    for case_id in case_ids:
        if case_id in validation_set:
            continue
        site_cases.setdefault(partition[case_id], []).append(case_id)

    all_training_entries = []
    site_counts = {}
    for site_id in sorted(site_cases, key=_site_sort_key):
        entries = [
            _case_to_entry(case_id, cases[case_id], path_builder)
            for case_id in site_cases[site_id]
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

    return {
        "partition_csv": str(Path(partition_csv).resolve()),
        "training_dir": str(Path(training_dir).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "dataset_dir": str(dataset_dir),
        "datalist_dir": str(datalist_dir),
        "file_mode": file_mode,
        "case_count": len(cases),
        "training_count": len(all_training_entries),
        "validation_count": len(validation_entries),
        "validation_ids": validation_ids,
        "site_count": len(site_counts),
        "sites": site_counts,
    }


def _site_sort_key(site_id: str):
    stripped = site_id.lstrip("+-")
    return (0, int(site_id)) if stripped.isdigit() else (1, site_id)


def _case_to_entry(case_id: str, case, path_builder) -> dict:
    modalities, label = case
    return {
        "image": [path_builder(case_id, path) for path in modalities],
        "label": path_builder(case_id, label),
    }


def _copy_case(case_id: str, case, dataset_dir: Path) -> None:
    modalities, label = case
    target_dir = dataset_dir / case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in (*modalities, label):
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
        partition_csv=args.partition_csv,
        training_dir=args.training_dir,
        data_root=args.data_root,
        validation_count=args.validation_count,
        modality_suffixes=tuple(args.modality_suffixes),
        file_mode=args.file_mode,
    )
    printable = {k: v for k, v in summary.items() if k != "validation_ids"}
    printable["validation_ids"] = f"{summary['validation_count']} held out"
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
