"""Validation and expansion of participant hyperparameter configuration."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from urllib.parse import quote

import yaml

from .cohort_registry import CohortSpec
from .config import ALLOWED_HPARAM_KEYS, COHORT_NAMES, DEFAULT_DATA_LOADER_WORKERS


def load_site_hparams(config_path: Path | str) -> dict:
    """Load and validate the participant hyperparameters from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The validated configuration dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    validate_site_hparams(payload)
    return payload


def validate_site_hparams(payload: dict):
    """Validate that the site hyperparameters payload has the correct schema.

    Args:
        payload: The configuration dictionary loaded from YAML.

    Raises:
        ValueError: If any validation checks fail (e.g. incorrect types, missing keys).
    """
    if not isinstance(payload, dict):
        raise ValueError("site_hparams.yaml must contain a mapping at the root.")

    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("site_hparams.yaml must contain a 'cohorts' mapping.")

    missing = set(COHORT_NAMES) - set(cohorts)
    if missing:
        raise ValueError(f"Missing cohort definitions: {sorted(missing)}")

    for cohort_name in COHORT_NAMES:
        cohort_payload = cohorts[cohort_name]
        if not isinstance(cohort_payload, dict):
            raise ValueError(f"cohorts.{cohort_name} must be a mapping.")

        defaults = cohort_payload.get("defaults")
        sites = cohort_payload.get("sites")
        if not isinstance(defaults, dict):
            raise ValueError(f"cohorts.{cohort_name}.defaults must be a mapping.")
        if not isinstance(sites, dict):
            raise ValueError(f"cohorts.{cohort_name}.sites must be a mapping.")

        _validate_hparam_mapping(defaults, f"cohorts.{cohort_name}.defaults")
        for site_name, site_mapping in sites.items():
            if not isinstance(site_mapping, dict):
                raise ValueError(
                    f"cohorts.{cohort_name}.sites.{site_name} must be a mapping."
                )
            _validate_hparam_mapping(
                site_mapping, f"cohorts.{cohort_name}.sites.{site_name}"
            )


def _validate_hparam_mapping(mapping: dict, context: str):
    """Check that all keys in a hyperparameter mapping are supported.

    Args:
        mapping: The hyperparameter mapping to validate.
        context: Context description for error messages.

    Raises:
        ValueError: If mapping contains unsupported hyperparameter keys.
    """
    invalid_keys = sorted(set(mapping) - set(ALLOWED_HPARAM_KEYS))
    if invalid_keys:
        raise ValueError(f"{context} contains unsupported keys: {invalid_keys}")

    for key, value in mapping.items():
        value_context = f"{context}.{key}"
        if key in {"aggregation_epochs", "batch_size"}:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{value_context} must be a positive integer.")
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{value_context} must be a finite number.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{value_context} must be a finite number.")
        if key == "learning_rate" and numeric_value <= 0:
            raise ValueError(f"{value_context} must be greater than zero.")
        if key in {"weight_decay", "fedproxloss_mu"} and numeric_value < 0:
            raise ValueError(f"{value_context} must be non-negative.")
        if key == "cache_dataset" and not 0 <= numeric_value <= 1:
            raise ValueError(f"{value_context} must be between zero and one.")


def resolve_site_hparams(config: dict, cohort_name: str, site_name: str) -> dict:
    """Resolve site-specific hyperparameters by merging defaults with site overrides.

    Args:
        config: The overall validated configuration dictionary.
        cohort_name: Name of the cohort.
        site_name: Name of the site to resolve parameters for.

    Returns:
        A dictionary containing the resolved hyperparameters for the site.
    """
    cohort_payload = config["cohorts"][cohort_name]
    resolved = dict(cohort_payload["defaults"])
    resolved.update(cohort_payload["sites"].get(site_name, {}))
    return resolved


def build_site_train_args(
    cohort_spec: CohortSpec,
    *,
    dataset_base_dir: Path,
    datalist_json_path: Path,
    hparams: dict,
    participant_client_file: Path | None = None,
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
) -> str:
    """Construct a string of command line arguments for running a client training script.

    Args:
        cohort_spec: Cohort spec details.
        dataset_base_dir: Path to the dataset directory.
        datalist_json_path: Path to the datalist JSON file.
        hparams: Dictionary of resolved hyperparameters.
        participant_client_file: Participant module containing ``local_train``.
        data_loader_workers: Worker processes used by each client data loader.

    Returns:
        The command line arguments string.
    """
    dataset_base_dir = dataset_base_dir.resolve()
    datalist_json_path = datalist_json_path.resolve()
    args = [
        "--cohort",
        cohort_spec.name,
        "--dataset_base_dir",
        _encode_path_arg(dataset_base_dir),
        "--datalist_json_path",
        _encode_path_arg(datalist_json_path),
        "--label_transform",
        cohort_spec.label_transform,
        "--in_channels",
        str(cohort_spec.in_channels),
        "--out_channels",
        str(cohort_spec.out_channels),
        "--roi_size",
        *(str(v) for v in cohort_spec.roi_size),
        "--infer_roi_size",
        *(str(v) for v in cohort_spec.infer_roi_size),
        "--data_loader_workers",
        str(data_loader_workers),
    ]
    if participant_client_file is not None:
        args.extend(
            [
                "--participant_client_file",
                _encode_path_arg(participant_client_file.resolve()),
            ]
        )
    for key in ALLOWED_HPARAM_KEYS:
        args.extend([f"--{key}", str(hparams[key])])
    return " ".join(args)


def build_per_site_config(
    config: dict,
    cohort_spec: CohortSpec,
    *,
    dataset_base_dir: Path,
    site_datalist_paths: dict[str, Path],
    participant_client_file: Path | None = None,
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
) -> dict[str, dict]:
    """Build train argument configurations for each participating site.

    Args:
        config: The overall configuration dictionary.
        cohort_spec: Cohort spec details.
        dataset_base_dir: Path to the dataset directory.
        site_datalist_paths: Dictionary mapping site names to their JSON datalist paths.
        participant_client_file: Participant module containing ``local_train``.
        data_loader_workers: Worker processes used by each client data loader.

    Returns:
        A dictionary mapping site names to dictionaries containing their "train_args" strings.
    """
    per_site_config = {}
    for site_name, datalist_path in site_datalist_paths.items():
        hparams = resolve_site_hparams(config, cohort_spec.name, site_name)
        per_site_config[site_name] = {
            "train_args": build_site_train_args(
                cohort_spec,
                dataset_base_dir=dataset_base_dir,
                datalist_json_path=datalist_path,
                hparams=hparams,
                participant_client_file=participant_client_file,
                data_loader_workers=data_loader_workers,
            )
        }
    return per_site_config


def _encode_path_arg(path_value: Path) -> str:
    """Encode a path as one token for NVFLARE's whitespace-split arguments.

    Args:
        path_value: The Path object.

    Returns:
        The URL-encoded path string.
    """
    return quote(str(path_value), safe="/:\\")
