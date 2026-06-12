"""Validation and expansion of participant hyperparameter configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from .cohort_registry import CohortSpec
from .config import ALLOWED_HPARAM_KEYS, COHORT_NAMES


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
) -> str:
    """Construct a string of command line arguments for running a client training script.

    Args:
        cohort_spec: Cohort spec details.
        dataset_base_dir: Path to the dataset directory.
        datalist_json_path: Path to the datalist JSON file.
        hparams: Dictionary of resolved hyperparameters.

    Returns:
        The command line arguments string.
    """
    dataset_base_dir = dataset_base_dir.resolve()
    datalist_json_path = datalist_json_path.resolve()
    args = [
        "--cohort",
        cohort_spec.name,
        "--dataset_base_dir",
        _quote(dataset_base_dir),
        "--datalist_json_path",
        _quote(datalist_json_path),
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
    ]
    for key in ALLOWED_HPARAM_KEYS:
        args.extend([f"--{key}", str(hparams[key])])
    return " ".join(args)


def build_per_site_config(
    config: dict,
    cohort_spec: CohortSpec,
    *,
    dataset_base_dir: Path,
    site_datalist_paths: dict[str, Path],
) -> dict[str, dict]:
    """Build train argument configurations for each participating site.

    Args:
        config: The overall configuration dictionary.
        cohort_spec: Cohort spec details.
        dataset_base_dir: Path to the dataset directory.
        site_datalist_paths: Dictionary mapping site names to their JSON datalist paths.

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
            )
        }
    return per_site_config


def _quote(path_value: Path) -> str:
    """Wrap a path string in quotes if it contains spaces.

    Args:
        path_value: The Path object.

    Returns:
        The quoted or unquoted path string.
    """
    text = str(path_value)
    if " " in text:
        return f'"{text}"'
    return text
