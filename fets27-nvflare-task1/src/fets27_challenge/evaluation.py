"""Locked evaluation logic for public and official runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .cohort_registry import CohortSpec
from .config import DEFAULT_SAVE_FILENAME
from .data_pipeline import build_dataloaders, evaluate_model, get_torch_module, require_runtime_dependencies
from .models import create_model_for_cohort


@dataclass
class SiteScore:
    site_name: str
    val_dice: float


@dataclass
class CohortScore:
    cohort: str
    display_name: str
    cohort_score: float
    best_model_path: str
    job_workspace: str
    site_scores: list[SiteScore]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["site_scores"] = [asdict(score) for score in self.site_scores]
        return data


def discover_site_datalists(datalist_dir: Path) -> dict[str, Path]:
    discovered = {}
    for json_path in sorted(datalist_dir.glob("site-*.json")):
        if json_path.name == "site-All.json":
            continue
        discovered[json_path.stem] = json_path
    return discovered


def resolve_best_model_path(job_workspace: Path, save_filename: str = DEFAULT_SAVE_FILENAME) -> Path:
    candidates = list(job_workspace.rglob(save_filename))
    if candidates:
        return sorted(candidates)[0]
    fallback = list(job_workspace.rglob("FL_global_model.pt"))
    if fallback:
        return sorted(fallback)[0]
    raise FileNotFoundError(f"Unable to find a global checkpoint under {job_workspace}")


def evaluate_best_checkpoint(cohort_spec: CohortSpec, *, data_root: Path, job_workspace: Path) -> CohortScore:
    require_runtime_dependencies()

    torch = get_torch_module()
    checkpoint_path = resolve_best_model_path(job_workspace)
    model = create_model_for_cohort(cohort_spec.name)
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    site_scores = []
    for site_name, datalist_path in discover_site_datalists(cohort_spec.datalist_dir(data_root)).items():
        _, valid_loader, inferer, post_transform, valid_metric = build_dataloaders(
            dataset_base_dir=str(cohort_spec.dataset_dir(data_root)),
            datalist_json_path=str(datalist_path),
            label_transform=cohort_spec.label_transform,
            batch_size=1,
            cache_rate=0.0,
            roi_size=cohort_spec.roi_size,
            infer_roi_size=cohort_spec.infer_roi_size,
        )
        score = evaluate_model(model, valid_loader, inferer, post_transform, valid_metric, device)
        site_scores.append(SiteScore(site_name=site_name, val_dice=float(score)))

    cohort_score = sum(score.val_dice for score in site_scores) / len(site_scores)
    return CohortScore(
        cohort=cohort_spec.name,
        display_name=cohort_spec.display_name,
        cohort_score=float(cohort_score),
        best_model_path=str(checkpoint_path),
        job_workspace=str(job_workspace),
        site_scores=site_scores,
    )


def write_summary(output_dir: Path, *, mode: str, cohort_scores: list[CohortScore]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_score = sum(score.cohort_score for score in cohort_scores) / len(cohort_scores)
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
                "site_name",
                "val_dice",
                "best_model_path",
                "job_workspace",
            ],
        )
        writer.writeheader()
        for cohort_score in cohort_scores:
            for site_score in cohort_score.site_scores:
                writer.writerow(
                    {
                        "mode": mode,
                        "overall_public_score": overall_score,
                        "cohort": cohort_score.cohort,
                        "display_name": cohort_score.display_name,
                        "cohort_score": cohort_score.cohort_score,
                        "site_name": site_score.site_name,
                        "val_dice": site_score.val_dice,
                        "best_model_path": cohort_score.best_model_path,
                        "job_workspace": cohort_score.job_workspace,
                    }
                )
    return json_path, csv_path


def _load_state_dict(checkpoint_path: Path):
    torch = get_torch_module()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")

