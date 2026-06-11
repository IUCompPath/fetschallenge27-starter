"""Challenge runtime orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .cohort_registry import get_cohort_spec
from .config import DEFAULT_KEY_METRIC, DEFAULT_SAVE_FILENAME, PARTICIPANT_HPARAM_FILE
from .evaluation import (
    CohortScore,
    discover_site_datalists,
    evaluate_best_checkpoint,
    write_summary,
)
from .models import create_model_for_cohort
from .participant_config import (
    build_per_site_config,
    build_site_train_args,
    load_site_hparams,
    resolve_site_hparams,
)
from .participant_loader import load_participant_aggregator


LOGGER = logging.getLogger(__name__)


def run_challenge(
    *,
    repo_root: Path,
    mode: str,
    cohort_names: list[str],
    data_root: Path,
    workspace_root: Path,
    output_dir: Path,
    num_rounds: int,
    threads: int | None = None,
    gpu: str | None = None,
) -> tuple[Path, Path, list[CohortScore]]:
    LOGGER.info(
        "starting challenge run: mode=%s cohorts=%s data_root=%s workspace=%s "
        "output_dir=%s rounds=%s threads=%s gpu=%s",
        mode,
        cohort_names,
        data_root,
        workspace_root,
        output_dir,
        num_rounds,
        threads,
        gpu,
    )
    cohort_scores = []
    for cohort_name in cohort_names:
        cohort_scores.append(
            run_single_cohort(
                repo_root=repo_root,
                cohort_name=cohort_name,
                data_root=data_root,
                workspace_root=workspace_root,
                num_rounds=num_rounds,
                threads=threads,
                gpu=gpu,
            )
        )
    json_path, csv_path = write_summary(
        output_dir, mode=mode, cohort_scores=cohort_scores
    )
    LOGGER.info("challenge summaries written: json=%s csv=%s", json_path, csv_path)
    return json_path, csv_path, cohort_scores


def run_single_cohort(
    *,
    repo_root: Path,
    cohort_name: str,
    data_root: Path,
    workspace_root: Path,
    num_rounds: int,
    threads: int | None = None,
    gpu: str | None = None,
) -> CohortScore:
    from nvflare.apis.dxo import DataKind  # pragma: no cover - runtime dependency path
    from nvflare.app_opt.pt.recipes.fedavg import (
        FedAvgRecipe,
    )  # pragma: no cover - runtime dependency path
    from nvflare.client.config import (
        TransferType,
    )  # pragma: no cover - runtime dependency path
    from nvflare.recipe import (
        SimEnv,
        add_experiment_tracking,
    )  # pragma: no cover - runtime dependency path

    cohort_spec = get_cohort_spec(cohort_name)
    participant_config = load_site_hparams(repo_root / PARTICIPANT_HPARAM_FILE)
    site_datalist_paths = discover_site_datalists(cohort_spec.datalist_dir(data_root))
    if not site_datalist_paths:
        raise FileNotFoundError(
            f"No site datalists found under {cohort_spec.datalist_dir(data_root)}"
        )
    LOGGER.info(
        "[server] preparing cohort=%s sites=%s rounds=%s datalist_dir=%s",
        cohort_name,
        sorted(site_datalist_paths),
        num_rounds,
        cohort_spec.datalist_dir(data_root),
    )

    dataset_base_dir = cohort_spec.dataset_dir(data_root)
    per_site_config = build_per_site_config(
        participant_config,
        cohort_spec,
        dataset_base_dir=dataset_base_dir,
        site_datalist_paths=site_datalist_paths,
    )

    first_site = next(iter(site_datalist_paths))
    default_hparams = resolve_site_hparams(participant_config, cohort_name, first_site)
    train_args = build_site_train_args(
        cohort_spec,
        dataset_base_dir=dataset_base_dir,
        datalist_json_path=site_datalist_paths[first_site],
        hparams=default_hparams,
    )

    aggregator = load_participant_aggregator(repo_root)
    recipe_name = f"fets27_{cohort_name}"
    job_workspace = workspace_root / recipe_name
    LOGGER.info(
        "[server] loaded aggregator=%s job_workspace=%s",
        aggregator.__class__.__name__,
        job_workspace,
    )

    recipe_kwargs = {
        "name": recipe_name,
        "min_clients": len(site_datalist_paths),
        "num_rounds": num_rounds,
        "model": create_model_for_cohort(cohort_name),
        "train_script": str(
            (repo_root / "src" / "fets27_challenge" / "client.py").resolve()
        ),
        "train_args": train_args,
        "aggregator": aggregator,
        "aggregator_data_kind": DataKind.WEIGHT_DIFF,
        "per_site_config": per_site_config,
        "params_transfer_type": TransferType.DIFF,
        "key_metric": DEFAULT_KEY_METRIC,
        "save_filename": DEFAULT_SAVE_FILENAME,
    }

    checkpoint_path = cohort_spec.checkpoint_path(repo_root)
    if checkpoint_path.exists():
        recipe_kwargs["initial_ckpt"] = str(checkpoint_path.resolve())
        LOGGER.info("[server] using initial checkpoint=%s", checkpoint_path)
    else:
        LOGGER.info("[server] no initial checkpoint found at %s", checkpoint_path)

    recipe = FedAvgRecipe(**recipe_kwargs)
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    sim_env_kwargs = {
        "clients": list(site_datalist_paths),
        "workspace_root": str(workspace_root.resolve()),
    }
    if threads is not None:
        sim_env_kwargs["num_threads"] = threads
    if gpu is not None:
        sim_env_kwargs["gpu_config"] = gpu
    env = SimEnv(**sim_env_kwargs)

    execute_start = time.perf_counter()
    LOGGER.info(
        "[server] starting NVFLARE simulation: recipe=%s min_clients=%s clients=%s",
        recipe_name,
        len(site_datalist_paths),
        list(site_datalist_paths),
    )
    recipe.execute(env)
    LOGGER.info(
        "[server] NVFLARE simulation finished in %.1fs; starting checkpoint evaluation",
        time.perf_counter() - execute_start,
    )
    eval_start = time.perf_counter()
    score = evaluate_best_checkpoint(
        cohort_spec, data_root=data_root, job_workspace=job_workspace
    )
    LOGGER.info(
        "[server] evaluation complete for cohort=%s in %.1fs: score=%s",
        cohort_name,
        time.perf_counter() - eval_start,
        score,
    )
    return score
