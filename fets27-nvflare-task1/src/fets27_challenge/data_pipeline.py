"""Locked data loading and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import classification, segmentation
from .config import DEFAULT_DATA_LOADER_WORKERS

try:  # pragma: no cover - exercised only with full runtime deps
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
except ImportError:  # pragma: no cover - imported in tests without heavy deps
    torch = None
    CacheDataset = None
    DataLoader = None
    Dataset = None
    load_decathlon_datalist = None

# Task plug-ins: each supplies (train_transform, valid_transform, inferer,
# post_transform, valid_metric); everything else in this module is shared.
TASK_COMPONENTS = {
    "segmentation": segmentation.build_components,
    "classification": classification.build_components,
}

_CACHE_WORKERS = 1


def require_runtime_dependencies():
    """Verify that PyTorch, MONAI, and other runtime dependencies are installed.

    Raises:
        ImportError: If any of the required dependencies are missing.
    """
    if torch is None or DataLoader is None:
        raise ImportError(
            "Runtime dependencies are missing. Install PyTorch, MONAI, and NiBabel before running training."
        )


def build_dataloaders(
    *,
    task: str,
    dataset_base_dir: str,
    datalist_json_path: str,
    label_transform: str,
    batch_size: int,
    cache_rate: float,
    roi_size: tuple[int, int, int],
    infer_roi_size: tuple[int, int, int],
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
):
    """Build MONAI training and validation dataloaders and inference components.

    Args:
        task: Task name, a key of ``TASK_COMPONENTS`` ('segmentation' or 'classification').
        dataset_base_dir: Base directory path containing dataset files.
        datalist_json_path: Path to the datalist JSON file specifying case files.
        label_transform: Name of the transform to apply to labels (segmentation only).
        batch_size: Batch size for the training loader.
        cache_rate: Fraction of dataset files to cache in memory.
        roi_size: Spatial crop size for training (fixed input size for classification).
        infer_roi_size: Crop size for sliding window validation inference (segmentation only).
        data_loader_workers: Worker processes used by each training and validation loader.

    Returns:
        A tuple of (train_loader, valid_loader, inferer, post_transform, valid_metric).
    """
    require_runtime_dependencies()

    # is_segmentation only joins "label" with base_dir when it is a path string;
    # integer class labels pass through unchanged.
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

    train_transform, valid_transform, inferer, post_transform, valid_metric = (
        TASK_COMPONENTS[task](
            label_transform=label_transform,
            roi_size=roi_size,
            infer_roi_size=infer_roi_size,
        )
    )

    if cache_rate > 0.0:
        train_dataset = CacheDataset(
            data=train_list,
            transform=train_transform,
            cache_rate=cache_rate,
            num_workers=_CACHE_WORKERS,
        )
        valid_dataset = CacheDataset(
            data=valid_list,
            transform=valid_transform,
            cache_rate=cache_rate,
            num_workers=_CACHE_WORKERS,
        )
    else:
        train_dataset = Dataset(data=train_list, transform=train_transform)
        valid_dataset = Dataset(data=valid_list, transform=valid_transform)

    loader_kwargs = {
        "num_workers": data_loader_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, valid_loader, inferer, post_transform, valid_metric


def evaluate_model(
    model: Any, valid_loader, inferer, post_transform, valid_metric, device
) -> float:
    """Evaluate a model on validation data with the task's inferer, post-transform, and metric.

    Args:
        model: The PyTorch neural network model.
        valid_loader: Validation data loader.
        inferer: Callable ``(images, model) -> outputs``, e.g. MONAI SlidingWindowInferer.
        post_transform: Transformations to apply to model predictions before scoring.
        valid_metric: Callable ``(y_pred=, y=)`` returning per-sample metric values.
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
            val_images = batch_data["image"].to(device, non_blocking=True)
            val_labels = batch_data["label"].to(device, non_blocking=True)
            val_outputs = inferer(val_images, model)
            val_outputs = post_transform(val_outputs)
            metric_tensor = valid_metric(y_pred=val_outputs, y=val_labels)
            metric_array = metric_tensor.detach().cpu().numpy().reshape(-1)
            valid_values = metric_array[~np.isnan(metric_array)]
            total_metric += float(valid_values.sum())
            count += int(valid_values.size)
        if count == 0:
            raise ValueError("No valid validation metric values were produced.")
        return total_metric / count


def get_torch_module():
    """Retrieve the imported PyTorch module.

    Returns:
        The PyTorch module.
    """
    require_runtime_dependencies()
    return torch
