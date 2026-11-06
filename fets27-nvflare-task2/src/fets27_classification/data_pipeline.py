"""Locked classification data loading and evaluation helpers."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_LOADER_WORKERS
from .metrics import binary_metrics

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with full runtime deps
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandScaleIntensityd,
        RandShiftIntensityd,
        Resized,
    )
except ImportError:  # pragma: no cover - imported in tests without heavy deps
    torch = None
    CacheDataset = None
    DataLoader = None
    Dataset = None
    load_decathlon_datalist = None
    Compose = None
    CropForegroundd = None
    EnsureTyped = None
    LoadImaged = None
    NormalizeIntensityd = None
    Orientationd = None
    RandAffined = None
    RandFlipd = None
    RandGaussianNoised = None
    RandScaleIntensityd = None
    RandShiftIntensityd = None
    Resized = None

_CACHE_WORKERS = 1
NUM_CLASSES = 2


def require_runtime_dependencies():
    """Verify that PyTorch, MONAI, and NiBabel are installed.

    Raises:
        ImportError: If any of the required dependencies are missing.
    """
    if torch is None or Compose is None:
        raise ImportError(
            "Runtime dependencies are missing. Install PyTorch, MONAI, and NiBabel "
            "before running training."
        )


def get_torch_module():
    """Retrieve the imported PyTorch module.

    Returns:
        The PyTorch module.
    """
    require_runtime_dependencies()
    return torch


def build_classification_transforms(
    spatial_size: tuple[int, int, int], *, train: bool
):
    """Build the locked MONAI image pipeline for classification cases.

    Args:
        spatial_size: Target size every case is resampled to.
        train: Whether to include the stochastic augmentation operators.

    Returns:
        A MONAI ``Compose`` transform applied to the ``image`` key.
    """
    require_runtime_dependencies()
    operations = [
        LoadImaged(keys="image", ensure_channel_first=True),
        Orientationd(keys="image", axcodes="RAS"),
        CropForegroundd(keys="image", source_key="image"),
        Resized(keys="image", spatial_size=spatial_size, mode="trilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if train:
        operations += [
            RandFlipd(keys="image", prob=0.5, spatial_axis=axis) for axis in range(3)
        ]
        operations += [
            RandAffined(
                keys="image",
                prob=0.25,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode="bilinear",
                padding_mode="border",
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.15, std=0.01),
        ]
    operations.append(EnsureTyped(keys="image", dtype=torch.float32))
    return Compose(operations)


def load_datalist_cases(
    *, datalist_json_path: str, dataset_base_dir: str, data_list_key: str
) -> list[dict[str, Any]]:
    """Load one case list from a site datalist written by the partition helper.

    Args:
        datalist_json_path: Path to the site JSON.
        dataset_base_dir: Base directory used to resolve relative image paths.
        data_list_key: Either ``training`` or ``validation``.

    Returns:
        The list of case dictionaries for that key, or an empty list when the
        datalist does not define it.

    Raises:
        FileNotFoundError: If the datalist file does not exist.
        ValueError: If the datalist is not a JSON mapping.
    """
    require_runtime_dependencies()
    datalist_path = Path(datalist_json_path)
    if not datalist_path.is_file():
        raise FileNotFoundError(f"Missing datalist file: {datalist_path}")
    with datalist_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Datalist {datalist_path} must contain a JSON mapping.")
    if not payload.get(data_list_key):
        return []
    return list(
        load_decathlon_datalist(
            data_list_file_path=datalist_json_path,
            is_segmentation=True,
            data_list_key=data_list_key,
            base_dir=dataset_base_dir,
        )
    )


def deduplicate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated cases while preserving first-seen order.

    Site datalists all embed the same shared validation list, so the official
    evaluator uses this to evaluate each validation case exactly once.
    """
    seen: set[Any] = set()
    unique: list[dict[str, Any]] = []
    for case in cases:
        key = case.get("subject")
        if key is None:
            key = tuple(case.get("image", ()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    return unique


def load_case_lists(
    *,
    datalist_json_path: str,
    dataset_base_dir: str,
    validation_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the training and shared validation case lists from a site datalist.

    Args:
        datalist_json_path: Path to the site JSON written by the partition helper.
        dataset_base_dir: Base directory used to resolve relative image paths.
        validation_limit: Optional cap on client-side validation cases used only
            as a diagnostic speed knob. The official evaluator never applies it.

    Returns:
        A ``(training_cases, validation_cases)`` tuple.

    Raises:
        ValueError: If either list is empty or contains an invalid label.
    """
    train_list = load_datalist_cases(
        datalist_json_path=datalist_json_path,
        dataset_base_dir=dataset_base_dir,
        data_list_key="training",
    )
    valid_list = load_datalist_cases(
        datalist_json_path=datalist_json_path,
        dataset_base_dir=dataset_base_dir,
        data_list_key="validation",
    )
    if not train_list:
        raise ValueError(f"No training cases found in {datalist_json_path}")
    if not valid_list:
        raise ValueError(f"No validation cases found in {datalist_json_path}")
    for case in (*train_list, *valid_list):
        label = case.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(
                f"Case {case.get('subject', '<unknown>')!r} has invalid label {label!r}; "
                "expected 0 (glioma) or 1 (meningioma)."
            )
    if validation_limit is not None:
        if validation_limit < 1:
            raise ValueError("validation_limit must be a positive integer.")
        valid_list = valid_list[:validation_limit]
    return list(train_list), list(valid_list)


def build_dataloaders(
    *,
    dataset_base_dir: str,
    datalist_json_path: str,
    spatial_size: tuple[int, int, int],
    batch_size: int,
    cache_rate: float,
    data_loader_workers: int = DEFAULT_DATA_LOADER_WORKERS,
    validation_limit: int | None = None,
):
    """Build the locked training and validation loaders for one site.

    Args:
        dataset_base_dir: Base directory used to resolve relative image paths.
        datalist_json_path: Path to the site JSON written by the partition helper.
        spatial_size: Target size every case is resampled to.
        batch_size: Batch size for both loaders.
        cache_rate: Fraction of cases cached in memory (0 disables caching).
        data_loader_workers: Worker processes used by each loader.
        validation_limit: Optional diagnostic cap on the validation list.

    Returns:
        A ``(train_loader, valid_loader, class_weights)`` tuple where
        ``class_weights`` is a float32 tensor of length two.
    """
    require_runtime_dependencies()
    train_list, valid_list = load_case_lists(
        datalist_json_path=datalist_json_path,
        dataset_base_dir=dataset_base_dir,
        validation_limit=validation_limit,
    )
    train_transform = build_classification_transforms(spatial_size, train=True)
    valid_transform = build_classification_transforms(spatial_size, train=False)

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

    loader_kwargs: dict[str, Any] = {
        "num_workers": data_loader_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if data_loader_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    class_weights = compute_class_weights([int(case["label"]) for case in train_list])
    return train_loader, valid_loader, class_weights


def build_validation_loader(
    validation_cases: list[dict[str, Any]],
    *,
    spatial_size: tuple[int, int, int],
    batch_size: int = 4,
    data_loader_workers: int = 0,
):
    """Build a validation-only loader for an explicit case list.

    Args:
        validation_cases: Cases to evaluate, already resolved to absolute paths.
        spatial_size: Target size every case is resampled to.
        batch_size: Evaluation batch size.
        data_loader_workers: Worker processes used by the loader.

    Returns:
        A MONAI ``DataLoader`` over the validation cases.

    Raises:
        ValueError: If no cases were supplied.
    """
    require_runtime_dependencies()
    if not validation_cases:
        raise ValueError("No validation cases were supplied for evaluation.")
    dataset = Dataset(
        data=list(validation_cases),
        transform=build_classification_transforms(spatial_size, train=False),
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": data_loader_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if data_loader_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )


def compute_class_weights(labels: list[int]):
    """Return inverse-frequency class weights for the supplied labels.

    Args:
        labels: Integer labels for one site's training cases.

    Returns:
        A float32 tensor of length two. Sites that contain a single class fall
        back to ``[1.0, 1.0]`` because inverse frequency is undefined there.
    """
    require_runtime_dependencies()
    counts = Counter(int(label) for label in labels)
    if any(counts.get(index, 0) == 0 for index in range(NUM_CLASSES)):
        LOGGER.warning(
            "Site training list contains a single class (%s); using uniform class weights.",
            dict(sorted(counts.items())),
        )
        return torch.ones(NUM_CLASSES, dtype=torch.float32)
    total = len(labels)
    return torch.tensor(
        [total / (NUM_CLASSES * counts[index]) for index in range(NUM_CLASSES)],
        dtype=torch.float32,
    )


def evaluate_model(
    model: Any, valid_loader, device, *, amp: bool = True
) -> dict[str, Any]:
    """Compute the official classification metrics on a validation loader.

    Args:
        model: The PyTorch classifier to evaluate.
        valid_loader: Validation data loader.
        device: Torch device to run evaluation on.
        amp: Whether to use CUDA autocast when running on a GPU.

    Returns:
        A dictionary of classification metrics including ``balanced_accuracy``
        and ``mcc``.

    Raises:
        ValueError: If no validation samples were evaluated.
    """
    require_runtime_dependencies()
    amp_enabled = bool(amp) and device.type == "cuda"
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in valid_loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            logits = logits[0] if isinstance(logits, tuple) else logits
            labels.extend(int(value) for value in batch["label"].tolist())
            probabilities.extend(
                float(value) for value in torch.softmax(logits, dim=1)[:, 1].cpu()
            )
    if not labels:
        raise ValueError("No validation samples were evaluated.")
    return binary_metrics(labels, probabilities)
