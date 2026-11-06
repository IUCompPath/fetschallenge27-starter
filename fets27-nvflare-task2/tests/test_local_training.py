from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fets27_classification.local_training import (
    FLModel,
    load_participant_local_train,
    normalize_local_train_params,
    parameter_spec,
    validate_local_train_result,
)
from fets27_classification.participant_loader import load_participant_module
from fets27_classification.runtime import resolve_client_scripts


def test_load_participant_local_train():
    local_train = load_participant_local_train(Path("participant/client.py"))

    assert callable(local_train)


def test_validate_local_train_result_accepts_compatible_full_model():
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    result = FLModel(
        params={"weight": np.ones((2, 3), dtype=np.float32)},
        params_type="FULL",
        meta={"participant_stat": 1.0},
    )

    assert validate_local_train_result(result, expected) is result


def test_validate_local_train_result_accepts_equivalent_torch_and_numpy_dtypes():
    torch = pytest.importorskip("torch")
    expected = parameter_spec({"weight": torch.zeros((2, 3), dtype=torch.float32)})
    result = FLModel(params={"weight": np.ones((2, 3), dtype=np.float32)})

    validate_local_train_result(result, expected)
    normalize_local_train_params(result, torch)

    assert isinstance(result.params["weight"], torch.Tensor)
    assert result.params["weight"].device.type == "cpu"
    assert result.params["weight"].dtype == torch.float32


def test_normalize_local_train_params_detaches_and_clones_tensors():
    torch = pytest.importorskip("torch")
    original = torch.ones((2, 3), requires_grad=True)
    result = FLModel(params={"weight": original})

    normalize_local_train_params(result, torch)

    normalized = result.params["weight"]
    assert normalized.device.type == "cpu"
    assert not normalized.requires_grad
    assert normalized.data_ptr() != original.data_ptr()


def test_validate_local_train_result_rejects_reserved_meta():
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    result = FLModel(
        params={"weight": np.ones((2, 3), dtype=np.float32)},
        meta={"initial_metrics": {"val_balanced_accuracy": 999.0}},
    )

    with pytest.raises(ValueError, match="organizer-reserved"):
        validate_local_train_result(result, expected)


def test_validate_local_train_result_rejects_nonserializable_metadata():
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    result = FLModel(
        params={"weight": np.ones((2, 3), dtype=np.float32)},
        meta={"callback": lambda: None},
    )

    with pytest.raises(ValueError, match="serializable"):
        validate_local_train_result(result, expected)


@pytest.mark.parametrize("num_steps", [0, -1, float("nan"), "two"])
def test_validate_local_train_result_rejects_invalid_num_steps(num_steps):
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})
    result = FLModel(
        params={"weight": np.ones((2, 3), dtype=np.float32)},
        meta={"NUM_STEPS_CURRENT_ROUND": num_steps},
    )

    with pytest.raises(ValueError, match="positive finite"):
        validate_local_train_result(result, expected)


@pytest.mark.parametrize(
    "params",
    [
        {"different": np.ones((2, 3), dtype=np.float32)},
        {"weight": np.ones((3, 2), dtype=np.float32)},
        {"weight": np.ones((2, 3), dtype=np.float64)},
    ],
)
def test_validate_local_train_result_rejects_incompatible_params(params):
    expected = parameter_spec({"weight": np.zeros((2, 3), dtype=np.float32)})

    with pytest.raises(ValueError, match="incompatible|shape/dtype"):
        validate_local_train_result(FLModel(params=params, params_type="FULL"), expected)


def test_validate_local_train_result_requires_flmodel():
    with pytest.raises(TypeError, match="must return an FLModel"):
        validate_local_train_result({}, {})


def test_resolve_client_scripts_returns_locked_and_participant_files():
    locked_client, participant_client = resolve_client_scripts(Path.cwd())

    assert (
        locked_client == (Path.cwd() / "src/fets27_classification/client.py").resolve()
    )
    assert participant_client == (Path.cwd() / "participant/client.py").resolve()


def test_failed_participant_import_does_not_remain_in_sys_modules():
    temp_dir = Path(".test-artifacts") / "failed-participant-import"
    temp_dir.mkdir(parents=True, exist_ok=True)
    module_path = temp_dir / "broken.py"
    module_path.write_text("raise RuntimeError('broken import')\n", encoding="utf-8")
    module_name = "participant.broken_test_module"
    sys.modules.pop(module_name, None)

    with pytest.raises(RuntimeError, match="broken import"):
        load_participant_module(module_path, module_name)

    assert module_name not in sys.modules
