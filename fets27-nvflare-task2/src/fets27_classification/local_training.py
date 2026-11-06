"""Loading and validation helpers for participant local training."""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path

from .config import RESERVED_PARTICIPANT_META_KEYS
from .participant_loader import load_participant_module

try:
    from nvflare.app_common.abstract.fl_model import FLModel
except ModuleNotFoundError:  # pragma: no cover - used only without NVFLARE
    from .compat import FLModel


def load_participant_local_train(module_path: Path) -> Callable:
    """Load the participant-defined ``local_train`` function."""
    module = load_participant_module(module_path, "participant.client")

    local_train = getattr(module, "local_train", None)
    if not callable(local_train):
        raise AttributeError(
            "participant/client.py must expose callable local_train()."
        )
    return local_train


def parameter_spec(params: Mapping) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return the shape and dtype expected for each model parameter."""
    return {
        key: (tuple(int(value) for value in param.shape), _dtype_name(param.dtype))
        for key, param in params.items()
    }


def normalize_local_train_params(result: FLModel, torch_module) -> FLModel:
    """Detach and copy participant parameters into CPU torch tensors."""
    normalized = {}
    for key, value in result.params.items():
        try:
            normalized[key] = torch_module.as_tensor(value).detach().cpu().clone()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                f"Parameter {key!r} could not be converted to a CPU tensor."
            ) from exc
    result.params = normalized
    return result


def validate_local_train_result(
    result: FLModel,
    expected_spec: Mapping[str, tuple[tuple[int, ...], str]],
) -> FLModel:
    """Validate that ``local_train`` returned a compatible full model."""
    if not isinstance(result, FLModel):
        raise TypeError("local_train() must return an FLModel.")
    if not isinstance(result.params, Mapping) or not result.params:
        raise ValueError(
            "local_train() must return FLModel.params as a non-empty mapping."
        )

    params_type = getattr(result.params_type, "value", result.params_type)
    if params_type not in (None, "FULL"):
        raise ValueError(
            "local_train() must return full parameters; params_type must be FULL."
        )

    expected_keys = set(expected_spec)
    returned_keys = set(result.params)
    if returned_keys != expected_keys:
        missing = sorted(expected_keys - returned_keys)
        unexpected = sorted(returned_keys - expected_keys)
        raise ValueError(
            "local_train() returned incompatible parameter keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for key, value in result.params.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(
                f"Parameter {key!r} must be a tensor or array with shape and dtype."
            )
        returned = (
            tuple(int(size) for size in value.shape),
            _dtype_name(value.dtype),
        )
        if returned != expected_spec[key]:
            raise ValueError(
                f"Parameter {key!r} has shape/dtype {returned}; expected {expected_spec[key]}."
            )

    if result.meta is not None and not isinstance(result.meta, dict):
        raise TypeError("FLModel.meta returned by local_train() must be a dictionary.")
    if result.metrics is not None and not isinstance(result.metrics, dict):
        raise TypeError(
            "FLModel.metrics returned by local_train() must be a dictionary."
        )

    meta = result.meta or {}
    metrics = result.metrics or {}
    _validate_string_keys(meta, "FLModel.meta")
    _validate_string_keys(metrics, "FLModel.metrics")
    reserved_keys = sorted(set(meta) & RESERVED_PARTICIPANT_META_KEYS)
    if reserved_keys:
        raise ValueError(
            "FLModel.meta returned by local_train() contains organizer-reserved "
            f"keys: {reserved_keys}"
        )
    _validate_num_steps(meta)
    _validate_serializable(meta, "FLModel.meta")
    _validate_serializable(metrics, "FLModel.metrics")
    return result


def _dtype_name(dtype) -> str:
    """Normalize equivalent torch and NumPy dtype names."""
    name = str(dtype)
    return name.removeprefix("torch.").removeprefix("numpy.")


def _validate_string_keys(mapping: Mapping, label: str) -> None:
    invalid = [key for key in mapping if not isinstance(key, str)]
    if invalid:
        raise TypeError(f"{label} keys must be strings; invalid keys: {invalid!r}")


def _validate_num_steps(meta: Mapping) -> None:
    key = "NUM_STEPS_CURRENT_ROUND"
    if key not in meta:
        return
    value = meta[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a positive finite number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{key} must be a positive finite number.")


def _validate_serializable(value, label: str) -> None:
    """Fail before transport when participant metadata cannot be serialized."""
    try:
        from nvflare.fuel.utils import fobs

        serializer = fobs.dumps
    except ModuleNotFoundError:  # pragma: no cover - test-only fallback
        serializer = pickle.dumps

    try:
        serializer(value)
    except Exception as exc:
        raise ValueError(f"{label} must contain NVFLARE-serializable values.") from exc
