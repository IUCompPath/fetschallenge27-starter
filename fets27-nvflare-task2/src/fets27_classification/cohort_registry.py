"""Locked cohort registry for the challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SAMPLE_SITE_NAMES


@dataclass(frozen=True)
class CohortSpec:
    """Static task definition shared by the client, runtime, and evaluator."""

    name: str
    display_name: str
    in_channels: int
    num_classes: int
    spatial_size: tuple[int, int, int]
    model_wrapper: str
    checkpoint_relpath: str
    dataset_subdir: str = "dataset"
    datalist_subdir: str = "datalist"
    sample_sites: tuple[str, ...] = SAMPLE_SITE_NAMES

    def dataset_dir(self, data_root: Path) -> Path:
        """Get the absolute path to the dataset directory for this cohort.

        Args:
            data_root: Root directory where the challenge datasets are stored.

        Returns:
            The path to the dataset directory.
        """
        return data_root / self.name / self.dataset_subdir

    def datalist_dir(self, data_root: Path) -> Path:
        """Get the absolute path to the datalist directory for this cohort.

        Args:
            data_root: Root directory where the challenge datasets are stored.

        Returns:
            The path to the datalist directory containing site JSON files.
        """
        return data_root / self.name / self.datalist_subdir

    def checkpoint_path(self, repo_root: Path) -> Path:
        """Get the absolute path to the baseline checkpoint for this cohort.

        Args:
            repo_root: Root directory of the repository.

        Returns:
            The path to the baseline checkpoint file.
        """
        return repo_root / self.checkpoint_relpath


COHORT_REGISTRY: dict[str, CohortSpec] = {
    "brain_tumor": CohortSpec(
        name="brain_tumor",
        display_name="BraTS Glioma vs Meningioma",
        in_channels=4,
        num_classes=2,
        spatial_size=(96, 96, 96),
        model_wrapper="BrainTumorViT",
        checkpoint_relpath="assets/checkpoints/brain_tumor_baseline.pt",
    ),
}


def get_cohort_spec(cohort_name: str) -> CohortSpec:
    """Retrieve the CohortSpec details for a specific cohort.

    Args:
        cohort_name: The name of the cohort (e.g., 'brain_tumor').

    Returns:
        The matching CohortSpec.

    Raises:
        KeyError: If the cohort name is not registered.
    """
    try:
        return COHORT_REGISTRY[cohort_name]
    except KeyError as exc:
        raise KeyError(f"Unknown cohort {cohort_name!r}.") from exc
