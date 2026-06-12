"""Prepare FeTS-style dummy folders for the local challenge runtime."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MODALITY_SUFFIXES = ("flair", "t1", "t1ce", "t2")
SEGMENTATION_SUFFIX = "seg"


@dataclass(frozen=True)
class DummyCase:
    """Represents a single dummy patient case containing images and label paths."""

    case_id: str
    modalities: tuple[Path, Path, Path, Path]
    label: Path


def prepare_training_dummy_layout(
    *,
    source_root: Path,
    data_root: Path,
    site_count: int = 2,
    validation_count: int = 2,
    file_mode: Literal["absolute", "copy"] = "absolute",
) -> dict:
    """Write a runtime-compatible glioma data root from FeTS case folders.

    This function discovers the cases in `source_root`, splits them into training and
    validation sets, partitions the training cases among a number of sites, and writes
    the corresponding data structures and datalist JSON files to `data_root`.

    Args:
        source_root: Path to the directory containing source FeTS case directories.
        data_root: Path to the target data root directory.
        site_count: Number of simulated training sites to partition the training cases into.
        validation_count: Number of cases to hold out for validation.
        file_mode: Either 'absolute' (datalist paths refer directly to source files) or
            'copy' (files are copied into the dataset subdirectory).

    Returns:
        A dictionary summarizing the layout configuration and case counts.

    Raises:
        ValueError: If configuration values are invalid or there are insufficient cases.
    """

    if site_count < 1:
        raise ValueError("site_count must be at least 1.")
    if validation_count < 1:
        raise ValueError("validation_count must be at least 1.")
    if file_mode not in {"absolute", "copy"}:
        raise ValueError("file_mode must be 'absolute' or 'copy'.")

    cases = discover_training_dummy_cases(source_root)
    if len(cases) <= validation_count:
        raise ValueError(
            f"Need more cases than validation_count={validation_count}; "
            f"found {len(cases)} case(s)."
        )

    training_count = len(cases) - validation_count
    if site_count > training_count:
        raise ValueError(
            f"site_count={site_count} would create empty training sites; "
            f"only {training_count} training case(s) are available after "
            f"holding out validation_count={validation_count}."
        )

    cohort_root = data_root / "glioma"
    dataset_dir = cohort_root / "dataset"
    datalist_dir = cohort_root / "datalist"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datalist_dir.mkdir(parents=True, exist_ok=True)

    validation_cases = cases[-validation_count:]
    training_cases = cases[:-validation_count]
    site_names = [f"site-{index}" for index in range(1, site_count + 1)]
    site_training_cases = {site_name: [] for site_name in site_names}
    for index, case in enumerate(training_cases):
        site_training_cases[site_names[index % site_count]].append(case)

    if file_mode == "copy":
        for case in cases:
            _copy_case(case, dataset_dir)
        path_builder = lambda case, path: _relative_case_path(case, path)
    else:
        _write_absolute_dataset_note(dataset_dir, source_root)
        path_builder = lambda case, path: str(path.resolve())

    validation_entries = [
        _case_to_entry(case, path_builder=path_builder) for case in validation_cases
    ]
    all_training_entries = []
    site_counts = {}

    for site_name, site_cases in site_training_cases.items():
        training_entries = [
            _case_to_entry(case, path_builder=path_builder) for case in site_cases
        ]
        payload = {"training": training_entries, "validation": validation_entries}
        _write_json(datalist_dir / f"{site_name}.json", payload)
        all_training_entries.extend(training_entries)
        site_counts[site_name] = len(training_entries)

    _write_json(
        datalist_dir / "site-All.json",
        {"training": all_training_entries, "validation": validation_entries},
    )

    return {
        "source_root": str(source_root),
        "data_root": str(data_root),
        "dataset_dir": str(dataset_dir),
        "datalist_dir": str(datalist_dir),
        "file_mode": file_mode,
        "case_count": len(cases),
        "training_count": len(training_cases),
        "validation_count": len(validation_cases),
        "sites": site_counts,
    }


def discover_training_dummy_cases(source_root: Path) -> list[DummyCase]:
    """Scan the source root directory for valid FeTS case folders.

    Args:
        source_root: Path to the directory containing case folders.

    Returns:
        A list of DummyCase instances representing the discovered cases.

    Raises:
        FileNotFoundError: If the source root does not exist, or case files are missing.
        NotADirectoryError: If the source root is not a directory.
    """
    source_root = source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dummy data source does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Dummy data source is not a directory: {source_root}")

    cases = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        modalities = tuple(
            case_dir / f"{case_id}_{suffix}.nii.gz" for suffix in MODALITY_SUFFIXES
        )
        label = case_dir / f"{case_id}_{SEGMENTATION_SUFFIX}.nii.gz"
        missing = [path.name for path in (*modalities, label) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Case {case_id} is missing required file(s): {', '.join(missing)}"
            )
        cases.append(DummyCase(case_id=case_id, modalities=modalities, label=label))

    if not cases:
        raise FileNotFoundError(f"No case folders found under {source_root}")
    return cases


def _case_to_entry(case: DummyCase, *, path_builder) -> dict:
    """Convert a DummyCase to a dictionary entry for datalists.

    Args:
        case: The DummyCase instance.
        path_builder: A callable that builds strings from paths.

    Returns:
        A dictionary with "image" (list of modal file paths) and "label" string paths.
    """
    return {
        "image": [path_builder(case, path) for path in case.modalities],
        "label": path_builder(case, case.label),
    }


def _copy_case(case: DummyCase, dataset_dir: Path):
    """Copy a case's modality and label files to a target dataset directory.

    Args:
        case: The DummyCase instance.
        dataset_dir: Destination parent directory.
    """
    target_dir = dataset_dir / case.case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in (*case.modalities, case.label):
        target_path = target_dir / source_path.name
        if not target_path.exists() or target_path.stat().st_size != source_path.stat().st_size:
            shutil.copy2(source_path, target_path)


def _relative_case_path(case: DummyCase, source_path: Path) -> str:
    """Generate a relative string path for a case file.

    Args:
        case: The DummyCase instance.
        source_path: The file path to convert.

    Returns:
        A string path relative to the dataset directory.
    """
    return f"{case.case_id}/{source_path.name}"


def _write_absolute_dataset_note(dataset_dir: Path, source_root: Path):
    """Write a text file noting that files are referenced in-place.

    Args:
        dataset_dir: Path to write the note.
        source_root: Path to the source root.
    """
    note_path = dataset_dir / "SOURCE.txt"
    note_path.write_text(
        "Datalists in ../datalist reference the source files in place:\n"
        f"{source_root.resolve()}\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict):
    """Serialize a dictionary to a JSON file with pretty formatting.

    Args:
        path: Path to the output file.
        payload: Dictionary to serialize.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
