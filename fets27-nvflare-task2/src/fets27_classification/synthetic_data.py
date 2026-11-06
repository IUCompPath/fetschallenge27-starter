"""Synthetic assets for smoke tests and onboarding."""

from __future__ import annotations

import json
from pathlib import Path

from .cohort_registry import COHORT_REGISTRY

MODALITIES = ("t1c", "t1n", "t2f", "t2w")


def prepare_assets(repo_root: Path, data_root: Path) -> dict[str, list[str]]:
    """Generate all baseline checkpoints and synthetic datasets needed for testing.

    Args:
        repo_root: Path to the repository root.
        data_root: Path to the data root directory.

    Returns:
        A dictionary containing lists of created checkpoint and dataset file paths.
    """
    created = {"checkpoints": [], "datasets": []}
    created["checkpoints"] = generate_baseline_checkpoints(repo_root)
    created["datasets"] = generate_toy_dataset(data_root)
    return created


def generate_baseline_checkpoints(repo_root: Path) -> list[str]:
    """Generate synthetic baseline checkpoints for each cohort.

    Args:
        repo_root: Path to the repository root.

    Returns:
        A list of paths to the created checkpoint files.
    """
    import torch  # pragma: no cover - runtime dependency path

    from .models import create_model_for_cohort

    created = []
    for index, cohort_name in enumerate(COHORT_REGISTRY.keys(), start=1):
        checkpoint_path = COHORT_REGISTRY[cohort_name].checkpoint_path(repo_root)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(index)
        model = create_model_for_cohort(cohort_name)
        torch.save(model.state_dict(), checkpoint_path)
        created.append(str(checkpoint_path))
    return created


def generate_toy_dataset(data_root: Path) -> list[str]:
    """Generate a synthetic classification dataset with site datalists.

    Args:
        data_root: Path to the data root directory.

    Returns:
        A list of paths to all created files (images and JSON datalists).
    """
    created = []
    data_root.mkdir(parents=True, exist_ok=True)

    for cohort_name, cohort_spec in COHORT_REGISTRY.items():
        cohort_root = data_root / cohort_name
        dataset_dir = cohort_root / "dataset"
        training_dir = dataset_dir / "training"
        validation_dir = dataset_dir / "validation"
        datalist_dir = cohort_root / "datalist"
        training_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)
        datalist_dir.mkdir(parents=True, exist_ok=True)

        site_training_entries: dict[str, list[dict]] = {}
        all_training_entries: list[dict] = []

        for site_index, site_name in enumerate(cohort_spec.sample_sites, start=1):
            entries = []
            for case_index in range(1, 3):
                case_id = f"{cohort_name}_{site_name}_train_{case_index}"
                label = (site_index + case_index) % 2
                image_paths = _write_case(
                    training_dir, case_id, label=label, seed=site_index * 100 + case_index
                )
                entry = {
                    "image": [str(path.relative_to(dataset_dir)) for path in image_paths],
                    "label": label,
                    "subject": case_id,
                }
                entries.append(entry)
                all_training_entries.append(entry)
                created.extend(str(path) for path in image_paths)
            site_training_entries[site_name] = entries

        validation_entries = []
        for case_index in range(1, 3):
            case_id = f"{cohort_name}_val_{case_index}"
            label = case_index % 2
            image_paths = _write_case(
                validation_dir, case_id, label=label, seed=900 + case_index
            )
            validation_entries.append(
                {
                    "image": [str(path.relative_to(dataset_dir)) for path in image_paths],
                    "label": label,
                    "subject": case_id,
                }
            )
            created.extend(str(path) for path in image_paths)

        for site_name, entries in site_training_entries.items():
            _write_json(
                datalist_dir / f"{site_name}.json",
                {"training": entries, "validation": validation_entries},
            )
            created.append(str(datalist_dir / f"{site_name}.json"))

        _write_json(
            datalist_dir / "site-All.json",
            {"training": all_training_entries, "validation": validation_entries},
        )
        created.append(str(datalist_dir / "site-All.json"))

    return created


def _write_case(case_dir: Path, case_id: str, *, label: int, seed: int) -> list[Path]:
    """Generate one synthetic four-modality case for a binary label.

    Args:
        case_dir: Directory the modality files are written to.
        case_id: Case identifier used as the filename prefix.
        label: Integer class label (0 glioma, 1 meningioma).
        seed: Random seed for the synthetic volumes.

    Returns:
        The list of written modality paths.
    """
    import nibabel as nib
    import numpy as np

    rng = np.random.default_rng(seed)
    shape = (32, 32, 32)
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for channel_index, modality in enumerate(MODALITIES):
        volume = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
        start = 8 + label * 4
        volume[start : start + 12, start : start + 12, start : start + 12] += (
            0.6 + 0.1 * channel_index + 0.4 * label
        )
        path = case_dir / f"{case_id}_{modality}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), path)
        paths.append(path)
    return paths


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
