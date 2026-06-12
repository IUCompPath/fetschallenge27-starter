"""Locked model definitions."""

from __future__ import annotations

from typing import Any

try:
    from monai.networks.nets.segresnet import SegResNet
except ImportError:  # pragma: no cover - exercised only with full runtime deps
    SegResNet = None


def _require_monai():
    """Ensure that MONAI is available, raising an ImportError if it is not."""
    if SegResNet is None:
        raise ImportError("MONAI is required to instantiate the challenge models.")


class SerializableSegResNet(SegResNet if SegResNet is not None else object):
    """SegResNet wrapper that preserves constructor arguments for NVFLARE."""

    def __init__(
        self,
        *,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: tuple[int, ...] = (1, 1, 1),
        init_filters: int = 8,
        in_channels: int = 4,
        out_channels: int = 3,
        dropout_prob: float = 0.2,
    ):
        """Initialize the SerializableSegResNet model.

        Preserves configuration attributes as instance attributes for serialization/inspection in NVFLARE.
        """
        _require_monai()
        super().__init__(
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob


class GliomaSegResNet(SerializableSegResNet):
    """Specific SegResNet configuration used for glioma segmentation."""

    def __init__(self):
        """Initialize the GliomaSegResNet model with preset hyperparameters."""
        super().__init__(
            in_channels=4, out_channels=3, init_filters=8, dropout_prob=0.2
        )


MODEL_CLASS_BY_COHORT = {
    "glioma": GliomaSegResNet,
}


def create_model_for_cohort(cohort_name: str) -> Any:
    """Create a new model instance appropriate for the specified cohort.

    Args:
        cohort_name: Name of the cohort (e.g. 'glioma').

    Returns:
        An instance of the neural network model.

    Raises:
        KeyError: If the cohort name is unknown or has no model registered.
    """
    try:
        model_cls = MODEL_CLASS_BY_COHORT[cohort_name]
    except KeyError as exc:
        raise KeyError(f"Unknown cohort {cohort_name!r}.") from exc
    return model_cls()
