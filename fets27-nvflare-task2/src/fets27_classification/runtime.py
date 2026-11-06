"""Challenge runtime orchestration."""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path

from .cohort_registry import get_cohort_spec
from .config import (
    DEFAULT_DATA_LOADER_WORKERS,
    DEFAULT_KEY_METRIC,
    DEFAULT_SAVE_FILENAME,
    LOCKED_CLIENT_FILE,
    PARTICIPANT_CLIENT_FILE,
    PARTICIPANT_HPARAM_FILE,
)
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


def resolve_client_scripts(repo_root: Path) -> tuple[Path, Path]:
    """Resolve the locked runner and participant local-training module."""
    locked_client = repo_root / LOCKED_CLIENT_FILE
    participant_client = repo_root / PARTICIPANT_CLIENT_FILE
    for script in (locked_client, participant_client):
        if not script.is_file():
            raise FileNotFoundError(f"Missing client script: {script}")
    return locked_client.resolve(), participant_client.resolve()


def _initial_model_kwargs(
    recipe_parameters: set[str], *, cohort_name: str, checkpoint_path: Path
) -> dict:
    """Return the recipe keyword arguments that seed the server model.

    Uses the recipe's ``initial_ckpt`` path argument when the installed NVFLARE
    supports it. Otherwise the checkpoint is loaded into the model in-process so
    older releases still start from the same locked baseline.

    Args:
        recipe_parameters: Parameter names accepted by ``FedAvgRecipe.__init__``.
        cohort_name: Cohort whose locked model should be created.
        checkpoint_path: Path to the baseline checkpoint.

    Returns:
        Keyword arguments to merge into the recipe construction call.

    Raises:
        TypeError: If the recipe accepts no initial model argument at all.
    """
    model_key = next(
        (key for key in ("initial_model", "model") if key in recipe_parameters), None
    )
    if model_key is None:
        raise TypeError(
            "The installed NVFLARE FedAvgRecipe does not accept an initial model."
        )
    model = create_model_for_cohort(cohort_name)
    kwargs = {}
    if checkpoint_path.exists():
        if "initial_ckpt" in recipe_parameters:
            kwargs["initial_ckpt"] = str(checkpoint_path.resolve())
        else:
            from .evaluation import _load_state_dict

            model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
        LOGGER.info("[server] using initial checkpoint=%s", checkpoint_path)
    else:
        LOGGER.info("[server] no initial checkpoint found at %s", checkpoint_path)
    kwargs[model_key] = model
    return kwargs


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
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
    validation_limit: int | None = None,
) -> tuple[Path, Path, list[CohortScore]]:
    """Run the federated learning challenge across cohorts and evaluate them.

    Args:
        repo_root: Path to the repository root directory.
        mode: The challenge run mode (e.g. 'local' or 'official').
        cohort_names: A list of cohort names to execute.
        data_root: Path to the datasets' root directory.
        workspace_root: Path to save NVFLARE jobs and workspace data.
        output_dir: Folder where summary reports (JSON, CSV) will be written.
        num_rounds: The number of federation rounds to run.
        threads: Optional number of threads to configure for the simulation environment.
        gpu: Optional GPU device IDs configuration (e.g. '0').
        data_loader_workers: Worker processes used by each client data loader.
        validation_limit: Optional diagnostic cap on client-side validation cases.

    Returns:
        A tuple containing:
        - The path to the JSON summary file.
        - The path to the CSV summary file.
        - A list of CohortScore instances containing results for each cohort.
    """
    if data_loader_workers <= 0:
        raise ValueError("data_loader_workers must be a positive integer.")
    if validation_limit is not None and validation_limit < 1:
        raise ValueError("validation_limit must be a positive integer.")

    LOGGER.info(
        "starting challenge run: mode=%s cohorts=%s data_root=%s workspace=%s "
        "output_dir=%s rounds=%s threads=%s gpu=%s data_loader_workers=%s "
        "validation_limit=%s",
        mode,
        cohort_names,
        data_root,
        workspace_root,
        output_dir,
        num_rounds,
        threads,
        gpu,
        data_loader_workers,
        validation_limit,
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
                data_loader_workers=data_loader_workers,
                validation_limit=validation_limit,
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
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
    validation_limit: int | None = None,
) -> CohortScore:
    """Run the NVFLARE simulation for one cohort and evaluate the result.

    Args:
        repo_root: Path to the repository root directory.
        cohort_name: Name of the cohort to run.
        data_root: Path to the datasets' root directory.
        workspace_root: Path to save workspace data.
        num_rounds: Number of federation rounds.
        threads: Optional number of threads to allocate.
        gpu: Optional GPU configuration string.
        data_loader_workers: Worker processes used by each client data loader.
        validation_limit: Optional diagnostic cap on client-side validation cases.

    Returns:
        A CohortScore instance containing the evaluation results.

    Raises:
        FileNotFoundError: If the datalist directories or site JSONs cannot be found.
    """
    from nvflare.apis.dxo import DataKind  # pragma: no cover - runtime dependency path
    from nvflare.app_opt.pt.recipes.fedavg import (  # pragma: no cover - runtime dependency path
        FedAvgRecipe,
    )
    from nvflare.client.config import (  # pragma: no cover - runtime dependency path
        TransferType,
    )
    from nvflare.recipe import (  # pragma: no cover - runtime dependency path
        SimEnv,
        add_experiment_tracking,
    )

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
    locked_client_file, participant_client_file = resolve_client_scripts(repo_root)
    recipe_parameters = set(inspect.signature(FedAvgRecipe.__init__).parameters)
    supports_per_site_config = "per_site_config" in recipe_parameters

    first_site = next(iter(site_datalist_paths))
    default_hparams = resolve_site_hparams(participant_config, cohort_name, first_site)
    if supports_per_site_config:
        per_site_config = build_per_site_config(
            participant_config,
            cohort_spec,
            dataset_base_dir=dataset_base_dir,
            site_datalist_paths=site_datalist_paths,
            participant_client_file=participant_client_file,
            data_loader_workers=data_loader_workers,
            validation_limit=validation_limit,
        )
        train_args = build_site_train_args(
            cohort_spec,
            dataset_base_dir=dataset_base_dir,
            datalist_json_path=site_datalist_paths[first_site],
            hparams=default_hparams,
            participant_client_file=participant_client_file,
            data_loader_workers=data_loader_workers,
            validation_limit=validation_limit,
        )
    else:
        # Older NVFLARE recipes only support one shared train_args string. The
        # locked client then resolves its own site datalist and hyperparameters
        # from the site name reported by the simulator.
        per_site_config = None
        train_args = build_site_train_args(
            cohort_spec,
            dataset_base_dir=dataset_base_dir,
            datalist_dir=cohort_spec.datalist_dir(data_root),
            hparams=default_hparams,
            participant_client_file=participant_client_file,
            data_loader_workers=data_loader_workers,
            validation_limit=validation_limit,
            site_hparams_file=repo_root / PARTICIPANT_HPARAM_FILE,
        )
        LOGGER.info(
            "[server] installed NVFLARE FedAvgRecipe lacks per_site_config; "
            "clients resolve <datalist_dir>/<site_name>.json and site_hparams.yaml"
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
        "train_script": str(locked_client_file),
        "train_args": train_args,
        "aggregator": aggregator,
        "aggregator_data_kind": DataKind.WEIGHT_DIFF,
        "params_transfer_type": TransferType.DIFF,
    }
    if per_site_config is not None:
        recipe_kwargs["per_site_config"] = per_site_config
    if "key_metric" in recipe_parameters:
        recipe_kwargs["key_metric"] = DEFAULT_KEY_METRIC
    if "save_filename" in recipe_parameters:
        recipe_kwargs["save_filename"] = DEFAULT_SAVE_FILENAME
    recipe_kwargs.update(
        _initial_model_kwargs(
            recipe_parameters,
            cohort_name=cohort_name,
            checkpoint_path=cohort_spec.checkpoint_path(repo_root),
        )
    )

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
        cohort_spec,
        data_root=data_root,
        job_workspace=job_workspace,
        data_loader_workers=min(data_loader_workers, 2),
    )
    LOGGER.info(
        "[server] evaluation complete for cohort=%s in %.1fs: score=%s metrics=%s",
        cohort_name,
        time.perf_counter() - eval_start,
        score.cohort_score,
        score.metrics,
    )
    return score
