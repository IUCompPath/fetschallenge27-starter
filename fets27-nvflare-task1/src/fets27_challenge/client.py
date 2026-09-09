"""Organizer-owned NVFLARE client runner."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from urllib.parse import unquote

from fets27_challenge.data_pipeline import (
    build_dataloaders,
    evaluate_model,
    get_torch_module,
    require_runtime_dependencies,
)
from fets27_challenge.config import DEFAULT_DATA_LOADER_WORKERS
from fets27_challenge.local_training import (
    load_participant_local_train,
    normalize_local_train_params,
    parameter_spec,
    validate_local_train_result,
)
from fets27_challenge.models import create_model_for_cohort

LOGGER = logging.getLogger(__name__)


class _PreloadedDataLoader:
    """Proxy a loader while preserving one iterator preloaded before each round."""

    def __init__(self, loader):
        self._loader = loader
        self._preloaded_iterator = None

    def preload(self) -> None:
        self._preloaded_iterator = iter(self._loader)

    def __iter__(self):
        if self._preloaded_iterator is not None:
            iterator = self._preloaded_iterator
            self._preloaded_iterator = None
            return iterator
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _value_nbytes(value) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    return 0


def _summarize_params(params) -> tuple[int, int]:
    if not params:
        return 0, 0
    return len(params), sum(_value_nbytes(value) for value in params.values())


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def parse_args():
    """Parse command-line arguments for the organizer-owned client runner."""
    parser = argparse.ArgumentParser(description="FeTS27 Task 1 client runner.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--participant_client_file",
        type=lambda value: Path(unquote(value)),
        required=True,
    )
    parser.add_argument("--aggregation_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0)
    parser.add_argument("--cache_dataset", type=float, default=0.0)
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--datalist_json_path", type=str, required=True)
    parser.add_argument("--label_transform", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--out_channels", type=int, required=True)
    parser.add_argument("--roi_size", type=int, nargs=3, required=True)
    parser.add_argument("--infer_roi_size", type=int, nargs=3, required=True)
    parser.add_argument(
        "--data_loader_workers", type=int, default=DEFAULT_DATA_LOADER_WORKERS
    )
    return parser.parse_args()


def main():
    """Run the locked client lifecycle around participant-defined local training."""
    require_runtime_dependencies()

    import nvflare.client as flare
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.client.tracking import SummaryWriter

    torch = get_torch_module()
    args = parse_args()
    args.dataset_base_dir = unquote(args.dataset_base_dir)
    args.datalist_json_path = unquote(args.datalist_json_path)

    flare.init()
    system_info = flare.system_info()
    summary_writer = SummaryWriter()
    client_name = (
        system_info.get("site_name") or system_info.get("client_name") or "client"
    )
    local_train = load_participant_local_train(args.participant_client_file.resolve())

    LOGGER.info(
        "[%s] starting client runner: cohort=%s datalist=%s batch_size=%s "
        "data_loader_workers=%s participant_client=%s",
        client_name,
        args.cohort,
        args.datalist_json_path,
        args.batch_size,
        args.data_loader_workers,
        args.participant_client_file,
    )

    train_loader, valid_loader, inferer, post_transform, valid_metric = (
        build_dataloaders(
            dataset_base_dir=args.dataset_base_dir,
            datalist_json_path=args.datalist_json_path,
            label_transform=args.label_transform,
            batch_size=args.batch_size,
            cache_rate=args.cache_dataset,
            roi_size=tuple(args.roi_size),
            infer_roi_size=tuple(args.infer_roi_size),
            data_loader_workers=args.data_loader_workers,
        )
    )
    if len(train_loader) == 0:
        raise ValueError("Training data loader is empty.")
    train_loader = _PreloadedDataLoader(train_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = create_model_for_cohort(args.cohort).to(device)
    expected_spec = parameter_spec(model.state_dict())
    participant_state = {}

    while flare.is_running():
        train_loader.preload()
        LOGGER.info("[%s] waiting for global model", client_name)
        receive_start = time.perf_counter()
        input_model = flare.receive()
        tensor_count, payload_bytes = _summarize_params(input_model.params)
        LOGGER.info(
            "[%s] round %s received global model: tensors=%s approx_payload=%s "
            "(non-authoritative diagnostic; raw tensor bytes only) "
            "receive_wait_or_transfer=%.1fs",
            client_name,
            input_model.current_round,
            tensor_count,
            _format_bytes(payload_bytes),
            time.perf_counter() - receive_start,
        )

        model.load_state_dict(input_model.params, strict=True)
        model.to(device)
        valid_start = time.perf_counter()
        global_metric = evaluate_model(
            model, valid_loader, inferer, post_transform, valid_metric, device
        )
        LOGGER.info(
            "[%s] round %s official validation complete in %.1fs: val_dice=%.6f",
            client_name,
            input_model.current_round,
            time.perf_counter() - valid_start,
            global_metric,
        )
        summary_writer.add_scalar(
            "val_metric_global_model", global_metric, input_model.current_round
        )

        train_start = time.perf_counter()
        output_model = local_train(
            model=model,
            train_loader=train_loader,
            device=device,
            current_round=input_model.current_round,
            aggregation_epochs=args.aggregation_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fedproxloss_mu=args.fedproxloss_mu,
            server_meta=dict(input_model.meta or {}),
            state=participant_state,
            summary_writer=summary_writer,
            client_name=client_name,
        )
        validate_local_train_result(output_model, expected_spec)
        normalize_local_train_params(output_model, torch)

        output_model.metrics = dict(output_model.metrics or {})
        output_model.metrics["val_dice"] = global_metric
        output_model.meta = dict(output_model.meta or {})
        output_model.meta.setdefault(
            FLMetaKey.NUM_STEPS_CURRENT_ROUND,
            len(train_loader) * args.aggregation_epochs,
        )

        update_tensor_count, update_payload_bytes = _summarize_params(
            output_model.params
        )
        LOGGER.info(
            "[%s] round %s local_train returned in %.1fs: tensors=%s "
            "approx_payload=%s (non-authoritative diagnostic; raw tensor bytes only) "
            "metrics=%s meta_keys=%s",
            client_name,
            input_model.current_round,
            time.perf_counter() - train_start,
            update_tensor_count,
            _format_bytes(update_payload_bytes),
            sorted(output_model.metrics),
            sorted(output_model.meta),
        )
        flare.send(output_model)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
