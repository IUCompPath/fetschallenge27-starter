"""Locked cohort registry for the challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SAMPLE_SITE_NAMES


@dataclass(frozen=True)
class CohortSpec:
    name: str
    display_name: str
    in_channels: int
    out_channels: int
    roi_size: tuple[int, int, int]
    infer_roi_size: tuple[int, int, int]
    label_transform: str
    model_wrapper: str
    checkpoint_relpath: str
    dataset_subdir: str = "dataset"
    datalist_subdir: str = "datalist"
    sample_sites: tuple[str, ...] = SAMPLE_SITE_NAMES

    def dataset_dir(self, data_root: Path) -> Path:
        return data_root / self.name / self.dataset_subdir

    def datalist_dir(self, data_root: Path) -> Path:
        return data_root / self.name / self.datalist_subdir

    def checkpoint_path(self, repo_root: Path) -> Path:
        return repo_root / self.checkpoint_relpath


COHORT_REGISTRY: dict[str, CohortSpec] = {
    "glioma": CohortSpec(
        name="glioma",
        display_name="BraTS Glioma",
        in_channels=4,
        out_channels=3,
        roi_size=(32, 32, 32),
        infer_roi_size=(32, 32, 32),
        label_transform="brats_multi_channel",
        model_wrapper="GliomaSegResNet",
        checkpoint_relpath="assets/checkpoints/glioma_baseline.pt",
    ),
    "meningioma": CohortSpec(
        name="meningioma",
        display_name="BraTS Meningioma",
        in_channels=4,
        out_channels=1,
        roi_size=(32, 32, 32),
        infer_roi_size=(32, 32, 32),
        label_transform="binary_channel",
        model_wrapper="MeningiomaSegResNet",
        checkpoint_relpath="assets/checkpoints/meningioma_baseline.pt",
    ),
    "sub_sahara": CohortSpec(
        name="sub_sahara",
        display_name="Sub-Sahara Glioma",
        in_channels=4,
        out_channels=3,
        roi_size=(32, 32, 32),
        infer_roi_size=(32, 32, 32),
        label_transform="brats_multi_channel",
        model_wrapper="SubSaharaSegResNet",
        checkpoint_relpath="assets/checkpoints/sub_sahara_baseline.pt",
    ),
}


def get_cohort_spec(cohort_name: str) -> CohortSpec:
    try:
        return COHORT_REGISTRY[cohort_name]
    except KeyError as exc:
        raise KeyError(f"Unknown cohort {cohort_name!r}.") from exc
