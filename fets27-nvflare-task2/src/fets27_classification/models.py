"""Locked model definitions."""

from __future__ import annotations

from typing import Any

try:
    from monai.networks.nets.vit import ViT
except ImportError:  # pragma: no cover - exercised only with full runtime deps
    ViT = None


def _require_monai():
    """Ensure that MONAI is available, raising an ImportError if it is not."""
    if ViT is None:
        raise ImportError("MONAI is required to instantiate the challenge models.")


class SerializableViT(ViT if ViT is not None else object):
    """3D ViT classifier that preserves constructor arguments for NVFLARE.

    Returns logits of shape ``(batch, num_classes)``. ViT uses LayerNorm only, so
    the state dict holds no running statistics or integer buffers and can be
    averaged directly by NVFLARE's weight-difference transfer.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        num_classes: int = 2,
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: int = 16,
        hidden_size: int = 384,
        mlp_dim: int = 1536,
        num_layers: int = 12,
        num_heads: int = 6,
    ):
        """Initialize the ViT and keep its configuration as instance attributes."""
        _require_monai()
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            num_classes=num_classes,
            spatial_dims=3,
            post_activation="",  # raw logits for cross-entropy, not MONAI's Tanh head
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, inputs):
        """Return only the logits; MONAI's ViT also returns its hidden states."""
        return super().forward(inputs)[0]


class BrainTumorViT(SerializableViT):
    """ViT-Small/16 used for glioma vs meningioma classification.

    Four MRI modalities, two classes, 96^3 input (6x6x6 = 216 patch tokens).
    """

    def __init__(self):
        """Initialize the BrainTumorViT model with preset hyperparameters."""
        super().__init__(in_channels=4, num_classes=2, img_size=(96, 96, 96))


MODEL_CLASS_BY_COHORT = {
    "brain_tumor": BrainTumorViT,
}


def create_model_for_cohort(cohort_name: str) -> Any:
    """Create a new model instance appropriate for the specified cohort.

    Args:
        cohort_name: Name of the cohort (e.g. 'brain_tumor').

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
