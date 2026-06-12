"""Locked NVFLARE client training script."""

from __future__ import annotations

import argparse
import copy
import logging
import time

from fets27_challenge.data_pipeline import (
    build_dataloaders,
    evaluate_model,
    get_torch_module,
    require_runtime_dependencies,
)
from fets27_challenge.models import create_model_for_cohort


LOGGER = logging.getLogger(__name__)


def _value_nbytes(value) -> int:
    """Calculate the memory footprint of a tensor or numpy array in bytes.

    Args:
        value: The array or tensor object.

    Returns:
        The number of bytes.
    """
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    return 0


def _summarize_params(params) -> tuple[int, int]:
    """Summarize a parameters dictionary in terms of count and size in bytes.

    Args:
        params: Parameters dictionary.

    Returns:
        A tuple containing (total parameter count, total size in bytes).
    """
    if not params:
        return 0, 0
    total_bytes = sum(_value_nbytes(value) for value in params.values())
    return len(params), total_bytes


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count into a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        A formatted string (e.g. '1.5 MiB').
    """
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def parse_args():
    """Parse command line arguments for the client training run.

    Returns:
        The parsed Namespace object.
    """
    parser = argparse.ArgumentParser(description="FeTS27 Task 1 locked client script.")
    parser.add_argument("--cohort", required=True)
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
    return parser.parse_args()


def main():
    """Initialize the NVFLARE client, run validation and local training iterations."""
    require_runtime_dependencies()

    import nvflare.client as flare
    from monai.losses import DiceLoss
    from nvflare.client.tracking import SummaryWriter

    try:  # pragma: no cover - runtime path
        from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
    except ImportError:  # pragma: no cover - depends on NVFLARE extras
        PTFedProxLoss = None

    torch = get_torch_module()
    import torch.optim as optim

    args = parse_args()

    flare.init()
    system_info = flare.system_info()
    summary_writer = SummaryWriter()
    client_name = (
        system_info.get("site_name") or system_info.get("client_name") or "client"
    )
    LOGGER.info(
        "[%s] starting client script: cohort=%s datalist=%s batch_size=%s "
        "aggregation_epochs=%s lr=%s cache_dataset=%s",
        client_name,
        args.cohort,
        args.datalist_json_path,
        args.batch_size,
        args.aggregation_epochs,
        args.learning_rate,
        args.cache_dataset,
    )

    loader_start = time.perf_counter()
    train_loader, valid_loader, inferer, post_transform, valid_metric = (
        build_dataloaders(
            dataset_base_dir=args.dataset_base_dir,
            datalist_json_path=args.datalist_json_path,
            label_transform=args.label_transform,
            batch_size=args.batch_size,
            cache_rate=args.cache_dataset,
            roi_size=tuple(args.roi_size),
            infer_roi_size=tuple(args.infer_roi_size),
        )
    )
    LOGGER.info(
        "[%s] dataloaders ready in %.1fs: train_cases=%s train_batches=%s "
        "valid_cases=%s valid_batches=%s dataset_base_dir=%s",
        client_name,
        time.perf_counter() - loader_start,
        len(train_loader.dataset),
        len(train_loader),
        len(valid_loader.dataset),
        len(valid_loader),
        args.dataset_base_dir,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("[%s] using device=%s", client_name, device)
    model = create_model_for_cohort(args.cohort).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True
    )
    criterion_prox = None
    if args.fedproxloss_mu > 0:
        if PTFedProxLoss is None:
            raise ImportError(
                "FedProx support is unavailable in the installed NVFLARE package."
            )
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    while flare.is_running():
        LOGGER.info("[%s] waiting for global model", client_name)
        receive_start = time.perf_counter()
        input_model = flare.receive()
        receive_elapsed = time.perf_counter() - receive_start
        round_start = time.perf_counter()
        tensor_count, payload_bytes = _summarize_params(input_model.params)
        LOGGER.info(
            "[%s] round %s received global model: tensors=%s approx_payload=%s "
            "receive_wait_or_transfer=%.1fs params_type=%s",
            client_name,
            input_model.current_round,
            tensor_count,
            _format_bytes(payload_bytes),
            receive_elapsed,
            input_model.params_type,
        )
        load_start = time.perf_counter()
        model.load_state_dict(input_model.params, strict=True)
        model.to(device)
        LOGGER.info(
            "[%s] round %s global model loaded onto %s in %.1fs",
            client_name,
            input_model.current_round,
            device,
            time.perf_counter() - load_start,
        )

        valid_start = time.perf_counter()
        LOGGER.info(
            "[%s] round %s starting validation: batches=%s",
            client_name,
            input_model.current_round,
            len(valid_loader),
        )
        global_metric = evaluate_model(
            model, valid_loader, inferer, post_transform, valid_metric, device
        )
        LOGGER.info(
            "[%s] round %s validation complete in %.1fs: val_dice=%.6f",
            client_name,
            input_model.current_round,
            time.perf_counter() - valid_start,
            global_metric,
        )
        summary_writer.add_scalar(
            "val_metric_global_model", global_metric, input_model.current_round
        )

        model_global = None
        if criterion_prox is not None:
            model_global = copy.deepcopy(model)
            for parameter in model_global.parameters():
                parameter.requires_grad = False

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.aggregation_epochs
        log_interval = max(1, min(50, steps_per_epoch // 10 or 1))
        LOGGER.info(
            "[%s] round %s starting local training: epochs=%s steps_per_epoch=%s "
            "total_steps=%s log_interval=%s",
            client_name,
            input_model.current_round,
            args.aggregation_epochs,
            steps_per_epoch,
            total_steps,
            log_interval,
        )

        for epoch in range(args.aggregation_epochs):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            for batch_index, batch_data in enumerate(train_loader, start=1):
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if criterion_prox is not None:
                    loss += criterion_prox(model, model_global)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                should_log_step = (
                    batch_index == 1
                    or batch_index == steps_per_epoch
                    or batch_index % log_interval == 0
                )
                if should_log_step:
                    LOGGER.info(
                        "[%s] round %s epoch %s/%s step %s/%s "
                        "avg_loss=%.6f elapsed=%.1fs",
                        client_name,
                        input_model.current_round,
                        epoch + 1,
                        args.aggregation_epochs,
                        batch_index,
                        steps_per_epoch,
                        running_loss / batch_index,
                        time.perf_counter() - epoch_start,
                    )

            if len(train_loader) == 0:
                raise ValueError("Training data loader is empty.")
            avg_loss = running_loss / len(train_loader)
            global_step = input_model.current_round * max(total_steps, 1) + epoch
            summary_writer.add_scalar("train_loss", avg_loss, global_step)
            LOGGER.info(
                "[%s] round %s epoch %s/%s complete in %.1fs: avg_loss=%.6f",
                client_name,
                input_model.current_round,
                epoch + 1,
                args.aggregation_epochs,
                time.perf_counter() - epoch_start,
                avg_loss,
            )

        state_dict_start = time.perf_counter()
        params = model.cpu().state_dict()
        update_tensor_count, update_payload_bytes = _summarize_params(params)
        LOGGER.info(
            "[%s] round %s prepared update: tensors=%s approx_payload=%s "
            "total_steps=%s preparation=%.1fs round_elapsed=%.1fs",
            client_name,
            input_model.current_round,
            update_tensor_count,
            _format_bytes(update_payload_bytes),
            total_steps,
            time.perf_counter() - state_dict_start,
            time.perf_counter() - round_start,
        )
        output_model = flare.FLModel(
            params=params,
            metrics={"val_dice": global_metric},
            meta={"NUM_STEPS_CURRENT_ROUND": total_steps},
        )
        send_start = time.perf_counter()
        LOGGER.info(
            "[%s] round %s sending update to server",
            client_name,
            input_model.current_round,
        )
        flare.send(output_model)
        LOGGER.info(
            "[%s] round %s update sent in %.1fs",
            client_name,
            input_model.current_round,
            time.perf_counter() - send_start,
        )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
