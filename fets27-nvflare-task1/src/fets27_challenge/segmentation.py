"""Segmentation task: MONAI transforms, sliding-window inference, and Dice scoring."""

from __future__ import annotations

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with full runtime deps
    import torch
    from monai.inferers import SlidingWindowInferer
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose,
        DivisiblePadd,
        EnsureChannelFirstd,
        LoadImaged,
        MapTransform,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        Spacingd,
    )
except ImportError:  # pragma: no cover - imported in tests without heavy deps
    torch = None
    SlidingWindowInferer = None
    DiceMetric = None
    Activations = None
    AsDiscrete = None
    Compose = None
    ConvertToMultiChannelBasedOnBratsEt3d = None
    DivisiblePadd = None
    EnsureChannelFirstd = None
    LoadImaged = None
    MapTransform = object
    NormalizeIntensityd = None
    Orientationd = None
    RandFlipd = None
    RandScaleIntensityd = None
    RandShiftIntensityd = None
    RandSpatialCropd = None
    Spacingd = None


def build_components(*, label_transform: str, roi_size, infer_roi_size):
    """Build the segmentation transforms and evaluation components.

    Args:
        label_transform: Name of the transform to apply to labels.
        roi_size: Spatial crop size for training.
        infer_roi_size: Crop size for sliding window validation inference.

    Returns:
        A tuple of (train_transform, valid_transform, inferer, post_transform, valid_metric).
    """
    return (
        _build_train_transform(label_transform, roi_size),
        _build_valid_transform(label_transform),
        SlidingWindowInferer(roi_size=infer_roi_size, sw_batch_size=1, overlap=0.5),
        Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]),
        DiceMetric(include_background=True, reduction="mean"),
    )


class BinaryChannelLabeld(MapTransform):  # pragma: no cover - runtime dependency path
    """Dictionary-based MONAI transform to binarize label images to 0 or 1."""

    def __call__(self, data):
        """Binarize label channels in a MONAI dictionary."""
        d = dict(data)
        for key in self.keys:
            label = np.asarray(d[key]).astype(np.float32)
            if label.ndim == 4 and label.shape[0] == 1:
                binary = (label > 0).astype(np.float32)
            elif label.ndim == 3:
                binary = (label > 0).astype(np.float32)[None, ...]
            else:
                binary = (label > 0).astype(np.float32)
                if binary.ndim == 3:
                    binary = binary[None, ...]
            d[key] = binary
        return d


class ConvertToMultiChannelBasedOnBratsEt3d(MapTransform):
    """Convert FeTS label 3 (and legacy 4) to BraTS TC/WT/ET channels."""

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self._legacy_label_warned = False

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            if hasattr(image, "ndim") and image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            if bool((image == 4).any()) and not self._legacy_label_warned:
                LOGGER.warning(
                    "%s contains legacy label 4. New FeTS challenge adopts BraTS "
                    "2023 convention: treating legacy label 4 as enhancing-tumor "
                    "label 3.",
                    key,
                )
                self._legacy_label_warned = True
            enhancing = (image == 3) | (image == 4)
            channels = [
                (image == 1) | enhancing,
                (image == 1) | (image == 2) | enhancing,
                enhancing,
            ]
            d[key] = (
                np.stack(channels, axis=0)
                if isinstance(image, np.ndarray)
                else torch.stack(channels, dim=0)
            )
        return d


def _build_train_transform(label_transform: str, roi_size: tuple[int, int, int]):
    """Build the MONAI transformation pipeline for training.

    Args:
        label_transform: Name of the label transform function.
        roi_size: Spatial dimensions to crop the input images to.

    Returns:
        A MONAI Compose transformation pipeline.
    """
    label_ops = _label_transform_ops(label_transform)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            *label_ops,
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def _build_valid_transform(label_transform: str):
    """Build the MONAI transformation pipeline for validation.

    Args:
        label_transform: Name of the label transform function.

    Returns:
        A MONAI Compose transformation pipeline.
    """
    label_ops = _label_transform_ops(label_transform)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            *label_ops,
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            DivisiblePadd(keys=["image", "label"], k=16),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def _label_transform_ops(label_transform: str):
    """Resolve the list of transforms to apply to label keys.

    Args:
        label_transform: Name of the label transform string.

    Returns:
        A list of transform operations to apply to labels.

    Raises:
        ValueError: If the label transform name is not recognized.
    """
    if label_transform == "brats_multi_channel":
        return [ConvertToMultiChannelBasedOnBratsEt3d(keys="label")]
    if label_transform == "binary_channel":
        return [BinaryChannelLabeld(keys=["label"])]
    raise ValueError(f"Unsupported label transform {label_transform!r}")
