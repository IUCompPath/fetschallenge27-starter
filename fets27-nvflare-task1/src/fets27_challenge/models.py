"""Locked model definitions."""

from __future__ import annotations

from typing import Any

from .cohort_registry import get_cohort_spec

try:
    from monai.networks.nets.resnet import ResNet
    from monai.networks.nets.segresnet import SegResNet
    from monai.networks.nets.vit import ViT
except ImportError:  # pragma: no cover - exercised only with full runtime deps
    ResNet = SegResNet = ViT = None


def _require_monai():
    """Ensure that MONAI is available, raising an ImportError if it is not."""
    if SegResNet is None:
        raise ImportError("MONAI is required to instantiate the challenge models.")


class SerializableSegResNet(SegResNet if SegResNet is not None else object):
    """SegResNet wrapper that preserves constructor arguments for NVFLARE."""

    task = "segmentation"

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


def _batch_to_group_norm(module, num_groups: int = 8) -> None:
    """Swap BatchNorm3d for GroupNorm in place.

    MONAI's ResNet ignores its ``norm`` argument in the residual blocks, so it cannot
    be configured at construction time.
    """
    from torch import nn

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            _batch_to_group_norm(child, num_groups)


class ResNet3D(ResNet if ResNet is not None else object):
    """ResNet-18 3D classifier returning logits of shape (batch, num_classes).

    GroupNorm (as in SegResNet) keeps the state dict float-only; BatchNorm's integer
    ``num_batches_tracked`` buffer would be float-averaged by the aggregator.
    """

    task = "classification"

    def __init__(self, *, in_channels: int, num_classes: int):
        """Initialize the model and keep constructor arguments for NVFLARE."""
        _require_monai()
        super().__init__(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )
        _batch_to_group_norm(self)
        self.in_channels = in_channels
        self.num_classes = num_classes


class ViT3D(ViT if ViT is not None else object):
    """ViT-Small/16 3D classifier returning logits of shape (batch, num_classes)."""

    task = "classification"

    def __init__(self, *, in_channels: int, num_classes: int, img_size: tuple[int, ...]):
        """Initialize the model and keep constructor arguments for NVFLARE."""
        _require_monai()
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=16,
            hidden_size=384,
            mlp_dim=1536,
            num_layers=12,
            num_heads=6,
            classification=True,
            num_classes=num_classes,
            spatial_dims=3,
            post_activation="",  # raw logits for cross-entropy, not MONAI's Tanh head
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        """Return only the logits; MONAI's ViT also returns its hidden states."""
        return super().forward(x)[0]


MODEL_FACTORY_BY_WRAPPER = {
    "GliomaSegResNet": lambda spec: GliomaSegResNet(),
    "ResNet3D": lambda spec: ResNet3D(
        in_channels=spec.in_channels, num_classes=spec.out_channels
    ),
    "ViT3D": lambda spec: ViT3D(
        in_channels=spec.in_channels,
        num_classes=spec.out_channels,
        img_size=spec.roi_size,
    ),
}


def create_model_for_cohort(cohort_name: str) -> Any:
    """Create a new model instance appropriate for the specified cohort.

    The architecture is the cohort's ``model_wrapper``; its shape comes from the cohort spec.

    Args:
        cohort_name: Name of the cohort (e.g. 'glioma').

    Returns:
        An instance of the neural network model.

    Raises:
        KeyError: If the cohort name is unknown or has no model registered.
    """
    spec = get_cohort_spec(cohort_name)
    return MODEL_FACTORY_BY_WRAPPER[spec.model_wrapper](spec)
