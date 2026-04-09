"""Synthetic assets for smoke tests and onboarding."""

from __future__ import annotations

import json
from pathlib import Path

from .cohort_registry import COHORT_REGISTRY


def prepare_assets(repo_root: Path, data_root: Path) -> dict[str, list[str]]:
    created = {"checkpoints": [], "datasets": []}
    created["checkpoints"] = generate_baseline_checkpoints(repo_root)
    created["datasets"] = generate_toy_dataset(data_root)
    return created


def generate_baseline_checkpoints(repo_root: Path) -> list[str]:
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
    import nibabel as nib  # pragma: no cover - runtime dependency path
    import numpy as np

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

        site_training_entries = {}
        all_training_entries = []
        validation_entries = []

        for site_index, site_name in enumerate(cohort_spec.sample_sites, start=1):
            entries = []
            for case_index in range(1, 3):
                case_id = f"{cohort_name}_{site_name}_train_{case_index}"
                image_path = training_dir / f"{case_id}_image.nii.gz"
                label_path = training_dir / f"{case_id}_label.nii.gz"
                _write_case(
                    image_path=image_path,
                    label_path=label_path,
                    in_channels=cohort_spec.in_channels,
                    label_transform=cohort_spec.label_transform,
                    seed=site_index * 100 + case_index,
                )
                entry = {
                    "image": str(image_path.relative_to(dataset_dir)),
                    "label": str(label_path.relative_to(dataset_dir)),
                }
                entries.append(entry)
                all_training_entries.append(entry)
                created.extend([str(image_path), str(label_path)])
            site_training_entries[site_name] = entries

        for case_index in range(1, 3):
            case_id = f"{cohort_name}_val_{case_index}"
            image_path = validation_dir / f"{case_id}_image.nii.gz"
            label_path = validation_dir / f"{case_id}_label.nii.gz"
            _write_case(
                image_path=image_path,
                label_path=label_path,
                in_channels=cohort_spec.in_channels,
                label_transform=cohort_spec.label_transform,
                seed=900 + case_index,
            )
            entry = {
                "image": str(image_path.relative_to(dataset_dir)),
                "label": str(label_path.relative_to(dataset_dir)),
            }
            validation_entries.append(entry)
            created.extend([str(image_path), str(label_path)])

        for site_name, entries in site_training_entries.items():
            payload = {"training": entries, "validation": validation_entries}
            with (datalist_dir / f"{site_name}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        with (datalist_dir / "site-All.json").open("w", encoding="utf-8") as handle:
            json.dump({"training": all_training_entries, "validation": validation_entries}, handle, indent=2)

    return created


def _write_case(*, image_path: Path, label_path: Path, in_channels: int, label_transform: str, seed: int):
    import nibabel as nib
    import numpy as np

    rng = np.random.default_rng(seed)
    shape = (32, 32, 32)
    image = rng.normal(0.0, 0.1, size=shape + (in_channels,)).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    if label_transform == "binary_channel":
        label[10:22, 10:22, 10:22] = 1
        image[10:22, 10:22, 10:22, :] += 0.8
    else:
        label[8:24, 8:24, 8:24] = 2
        label[12:20, 12:20, 12:20] = 1
        label[14:18, 14:18, 14:18] = 4
        image[8:24, 8:24, 8:24, :] += 0.4
        image[12:20, 12:20, 12:20, :] += 0.3
        image[14:18, 14:18, 14:18, :] += 0.3

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), label_path)

