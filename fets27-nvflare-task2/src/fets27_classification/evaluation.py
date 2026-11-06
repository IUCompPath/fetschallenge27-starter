"""Locked evaluation logic for public and official runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .cohort_registry import CohortSpec
from .config import DEFAULT_SAVE_FILENAME
from .data_pipeline import (
    build_validation_loader,
    deduplicate_cases,
    evaluate_model,
    get_torch_module,
    load_datalist_cases,
    require_runtime_dependencies,
)
from .models import create_model_for_cohort

EVAL_BATCH_SIZE = 4
METRIC_COLUMNS = (
    "balanced_accuracy",
    "accuracy",
    "auroc",
    "macro_f1",
    "sensitivity",
    "specificity",
    "mcc",
)


@dataclass
class CohortScore:
    """Represents the official validation metrics for one cohort."""

    cohort: str
    display_name: str
    cohort_score: float
    metrics: dict
    site_count: int
    validation_case_count: int
    best_model_path: str
    job_workspace: str

    def to_dict(self) -> dict:
        """Convert the cohort score into a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the cohort validation score.
        """
        data = asdict(self)
        data["metrics"] = _jsonable_metrics(self.metrics)
        return data


def discover_site_datalists(datalist_dir: Path) -> dict[str, Path]:
    """Find all site JSON datalists in the specified datalist directory.

    Excludes the aggregated 'site-All.json' list.

    Args:
        datalist_dir: Path to the datalist directory.

    Returns:
        A dictionary mapping site name keys (e.g. 'site-1') to their JSON datalist paths.
    """
    discovered = {}
    for json_path in sorted(datalist_dir.glob("site-*.json")):
        if json_path.name == "site-All.json":
            continue
        discovered[json_path.stem] = json_path
    return discovered


def resolve_best_model_path(
    job_workspace: Path, save_filename: str = DEFAULT_SAVE_FILENAME
) -> Path:
    """Locate the best global model checkpoint under the job workspace directory.

    Args:
        job_workspace: Path to the job's workspace.
        save_filename: The preferred filename of the checkpoint to find.

    Returns:
        The path to the resolved model checkpoint file.

    Raises:
        FileNotFoundError: If no matching checkpoint files can be found.
    """
    candidates = list(job_workspace.rglob(save_filename))
    if candidates:
        return sorted(candidates)[0]
    fallback = list(job_workspace.rglob("FL_global_model.pt"))
    if fallback:
        return sorted(fallback)[0]
    raise FileNotFoundError(f"Unable to find a global checkpoint under {job_workspace}")


def evaluate_best_checkpoint(
    cohort_spec: CohortSpec,
    *,
    data_root: Path,
    job_workspace: Path,
    data_loader_workers: int = 0,
) -> CohortScore:
    """Evaluate the best checkpoint on the official shared validation split.

    Every site datalist embeds the same validation list, so the cases are
    deduplicated and scored exactly once.

    Args:
        cohort_spec: Specification details of the cohort to evaluate.
        data_root: Path to the datasets' root.
        job_workspace: Path to the NVFLARE simulation job workspace.
        data_loader_workers: Worker processes used by the evaluation loader.

    Returns:
        A CohortScore instance containing the official validation metrics.
    """
    require_runtime_dependencies()

    torch = get_torch_module()
    checkpoint_path = resolve_best_model_path(job_workspace)
    model = create_model_for_cohort(cohort_spec.name)
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_base_dir = str(cohort_spec.dataset_dir(data_root))
    site_datalists = discover_site_datalists(cohort_spec.datalist_dir(data_root))
    validation_cases = []
    for datalist_path in site_datalists.values():
        validation_cases.extend(
            load_datalist_cases(
                datalist_json_path=str(datalist_path),
                dataset_base_dir=dataset_base_dir,
                data_list_key="validation",
            )
        )
    unique_cases = deduplicate_cases(validation_cases)
    valid_loader = build_validation_loader(
        unique_cases,
        spatial_size=cohort_spec.spatial_size,
        batch_size=EVAL_BATCH_SIZE,
        data_loader_workers=data_loader_workers,
    )
    metrics = evaluate_model(model, valid_loader, device)
    return CohortScore(
        cohort=cohort_spec.name,
        display_name=cohort_spec.display_name,
        cohort_score=float(metrics["balanced_accuracy"]),
        metrics=metrics,
        site_count=len(site_datalists),
        validation_case_count=len(unique_cases),
        best_model_path=str(checkpoint_path),
        job_workspace=str(job_workspace),
    )


def write_summary(
    output_dir: Path, *, mode: str, cohort_scores: list[CohortScore]
) -> tuple[Path, Path]:
    """Write evaluation summaries to JSON and CSV formats under the output directory.

    Args:
        output_dir: Destination folder path.
        mode: Run mode string (e.g. 'local' or 'official').
        cohort_scores: List of CohortScore instances to summarize.

    Returns:
        A tuple of (json_summary_path, csv_summary_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_score = sum(score.cohort_score for score in cohort_scores) / len(
        cohort_scores
    )
    payload = {
        "mode": mode,
        "overall_public_score": overall_score,
        "cohorts": [score.to_dict() for score in cohort_scores],
    }

    json_path = output_dir / f"{mode}_summary.json"
    csv_path = output_dir / f"{mode}_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "overall_public_score",
                "cohort",
                "display_name",
                "cohort_score",
                *METRIC_COLUMNS,
                "confusion_matrix",
                "site_count",
                "validation_case_count",
                "best_model_path",
                "job_workspace",
            ],
        )
        writer.writeheader()
        for cohort_score in cohort_scores:
            metrics = _jsonable_metrics(cohort_score.metrics)
            row = {
                "mode": mode,
                "overall_public_score": overall_score,
                "cohort": cohort_score.cohort,
                "display_name": cohort_score.display_name,
                "cohort_score": cohort_score.cohort_score,
                "confusion_matrix": metrics["confusion_matrix"],
                "site_count": cohort_score.site_count,
                "validation_case_count": cohort_score.validation_case_count,
                "best_model_path": cohort_score.best_model_path,
                "job_workspace": cohort_score.job_workspace,
            }
            for name in METRIC_COLUMNS:
                row[name] = metrics[name]
            writer.writerow(row)
    return json_path, csv_path


def _jsonable_metrics(metrics: dict) -> dict:
    """Return metrics with non-finite floats replaced by ``None``."""
    jsonable = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            jsonable[key] = None
        else:
            jsonable[key] = value
    return jsonable


def _load_state_dict(checkpoint_path: Path):
    """Load the state dict from a checkpoint file, handling wrapper structures.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The weights state dictionary.

    Raises:
        TypeError: If the checkpoint file's structure is unrecognized.
    """
    torch = get_torch_module()
    state = torch.load(checkpoint_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        return state["model"]
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")
