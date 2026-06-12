"""Locked data loading and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only with full runtime deps
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
    from monai.inferers import SlidingWindowInferer
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
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
    CacheDataset = None
    DataLoader = None
    Dataset = None
    load_decathlon_datalist = None
    SlidingWindowInferer = None
    DiceMetric = None
    Activations = None
    AsDiscrete = None
    Compose = None
    ConvertToMultiChannelBasedOnBratsClassesd = None
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


def require_runtime_dependencies():
    """Verify that PyTorch, MONAI, and other runtime dependencies are installed.

    Raises:
        ImportError: If any of the required dependencies are missing.
    """
    if torch is None or Compose is None:
        raise ImportError(
            "Runtime dependencies are missing. Install PyTorch, MONAI, and NiBabel before running training."
        )


class BinaryChannelLabeld(MapTransform):  # pragma: no cover - runtime dependency path
    """Dictionary-based MONAI transform to binarize label images to 0 or 1."""

    def __call__(self, data):
        """Binarize the label channel(s) in the input dictionary.

        Args:
            data: Input dictionary containing label arrays.

        Returns:
            The modified dictionary with binarized labels.
        """
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


def build_dataloaders(
    *,
    dataset_base_dir: str,
    datalist_json_path: str,
    label_transform: str,
    batch_size: int,
    cache_rate: float,
    roi_size: tuple[int, int, int],
    infer_roi_size: tuple[int, int, int],
):
    """Build MONAI training and validation dataloaders and inference components.

    Args:
        dataset_base_dir: Base directory path containing dataset files.
        datalist_json_path: Path to the datalist JSON file specifying case files.
        label_transform: Name of the transform to apply to labels.
        batch_size: Batch size for the training loader.
        cache_rate: Fraction of dataset files to cache in memory.
        roi_size: Spatial crop size for training.
        infer_roi_size: Crop size for sliding window validation inference.

    Returns:
        A tuple of (train_loader, valid_loader, inferer, post_transform, valid_metric).
    """
    require_runtime_dependencies()

    train_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=dataset_base_dir,
    )
    valid_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=dataset_base_dir,
    )

    train_transform = _build_train_transform(label_transform, roi_size)
    valid_transform = _build_valid_transform(label_transform)

    if cache_rate > 0.0:
        train_dataset = CacheDataset(
            data=train_list,
            transform=train_transform,
            cache_rate=cache_rate,
            num_workers=1,
        )
        valid_dataset = CacheDataset(
            data=valid_list,
            transform=valid_transform,
            cache_rate=cache_rate,
            num_workers=1,
        )
    else:
        train_dataset = Dataset(data=train_list, transform=train_transform)
        valid_dataset = Dataset(data=valid_list, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
    inferer = SlidingWindowInferer(
        roi_size=infer_roi_size, sw_batch_size=1, overlap=0.5
    )
    post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    valid_metric = DiceMetric(include_background=True, reduction="mean")
    return train_loader, valid_loader, inferer, post_transform, valid_metric


def evaluate_model(
    model: Any, valid_loader, inferer, post_transform, valid_metric, device
) -> float:
    """Evaluate a segmentation model on validation data using a sliding window inferer.

    Args:
        model: The PyTorch neural network model.
        valid_loader: Validation data loader.
        inferer: MONAI SlidingWindowInferer instance.
        post_transform: Transformations to apply to model predictions before scoring.
        valid_metric: MONAI DiceMetric or similar metric evaluator.
        device: PyTorch device to run evaluation on (e.g. 'cuda:0' or 'cpu').

    Returns:
        The average validation metric score.

    Raises:
        ValueError: If validation metric calculations fail or produce no values.
    """
    require_runtime_dependencies()

    model.eval()
    with torch.no_grad():
        total_metric = 0.0
        count = 0
        for batch_data in valid_loader:
            val_images = batch_data["image"].to(device)
            val_labels = batch_data["label"].to(device)
            val_outputs = inferer(val_images, model)
            val_outputs = post_transform(val_outputs)
            metric_tensor = valid_metric(y_pred=val_outputs, y=val_labels)
            metric_array = metric_tensor.detach().cpu().numpy().reshape(-1)
            valid_values = metric_array[~np.isnan(metric_array)]
            total_metric += float(valid_values.sum())
            count += int(valid_values.size)
        if count == 0:
            raise ValueError("No valid validation Dice values were produced.")
        return total_metric / count


def get_torch_module():
    """Retrieve the imported PyTorch module.

    Returns:
        The PyTorch module.
    """
    require_runtime_dependencies()
    return torch


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
        return [ConvertToMultiChannelBasedOnBratsClassesd(keys="label")]
    if label_transform == "binary_channel":
        return [BinaryChannelLabeld(keys=["label"])]
    raise ValueError(f"Unsupported label transform {label_transform!r}")
