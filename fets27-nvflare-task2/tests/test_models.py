from __future__ import annotations

import pytest

from fets27_classification.cohort_registry import get_cohort_spec
from fets27_classification.models import BrainTumorViT, create_model_for_cohort

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("einops")


def test_locked_model_is_vit_small():
    model = create_model_for_cohort("brain_tumor")

    assert isinstance(model, BrainTumorViT)
    assert (model.patch_size, model.hidden_size, model.num_layers, model.num_heads) == (
        16,
        384,
        12,
        6,
    )


def test_locked_model_state_dict_is_float_only():
    model = create_model_for_cohort("brain_tumor")

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )
    assert all(tensor.is_floating_point() for tensor in model.state_dict().values())


def test_locked_model_state_dict_round_trips_between_instances():
    first = create_model_for_cohort("brain_tumor")
    second = create_model_for_cohort("brain_tumor")

    second.load_state_dict(first.state_dict(), strict=True)

    assert set(first.state_dict()) == set(second.state_dict())


def test_locked_model_forward_returns_two_logits():
    model = create_model_for_cohort("brain_tumor")
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 4, 96, 96, 96))

    assert output.shape == (1, 2)


def test_cohort_spec_matches_model_inputs():
    spec = get_cohort_spec("brain_tumor")
    model = create_model_for_cohort(spec.name)

    assert spec.in_channels == model.in_channels
    assert spec.num_classes == model.num_classes
    assert spec.spatial_size == model.img_size
    assert all(size % model.patch_size == 0 for size in spec.spatial_size)


def test_unknown_cohort_raises():
    with pytest.raises(KeyError, match="Unknown cohort"):
        create_model_for_cohort("meningioma_only")
