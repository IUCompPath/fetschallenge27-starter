"""Classification task: 3D volume transforms and accuracy scoring.

Datalist entries are ``{"image": <path or list of paths>, "label": <class index>}``.
"""

from __future__ import annotations


def build_components(*, label_transform: str, roi_size, infer_roi_size):
    """Build the classification transforms and evaluation components.

    ``label_transform`` and ``infer_roi_size`` are segmentation-only and ignored.

    Args:
        label_transform: Unused.
        roi_size: Fixed spatial size every volume is padded or cropped to.
        infer_roi_size: Unused.

    Returns:
        A tuple of (train_transform, valid_transform, inferer, post_transform, valid_metric).
    """
    from monai.transforms import (  # pragma: no cover - runtime dependency path
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        ResizeWithPadOrCropd,
        Spacingd,
    )

    prepare = [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys="image", axcodes="RAS"),
        ResizeWithPadOrCropd(keys="image", spatial_size=roi_size),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    augment = [
        *(RandFlipd(keys="image", prob=0.5, spatial_axis=axis) for axis in range(3)),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
    return (
        Compose([*prepare, *augment]),
        Compose(prepare),
        lambda images, model: model(images),
        lambda logits: logits.argmax(dim=1),
        lambda y_pred, y: (y_pred == y).float(),  # per-sample 0/1, averaged by evaluate_model
    )
