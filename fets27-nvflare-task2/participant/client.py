"""Participant-editable local training for the FeTS27 NVFLARE challenge.

Implement ``local_train`` to update the organizer-provided model and return an
``FLModel`` containing full, compatible model parameters plus any metadata the
participant aggregator needs. The organizer-owned client runner performs
``flare.receive``/``flare.send`` and computes the official validation metrics.
"""

from __future__ import annotations

import copy
import logging
import time

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
except ModuleNotFoundError:  # pragma: no cover - used only without NVFLARE
    from fets27_classification.compat import FLMetaKey, FLModel


LOGGER = logging.getLogger(__name__)


def local_train(
    *,
    model,
    train_loader,
    device,
    current_round: int,
    aggregation_epochs: int,
    learning_rate: float,
    weight_decay: float,
    fedproxloss_mu: float,
    class_weights,
    server_meta: dict,
    state: dict,
    summary_writer,
    client_name: str,
) -> FLModel:
    """Train locally and return full updated parameters and optional metadata.

    ``class_weights`` is the organizer-computed inverse-frequency class weight
    vector for this site's training cases. ``state`` persists for this client
    across rounds and can hold optimizer, scheduler, or algorithm-specific state.
    ``server_meta`` contains metadata returned by the participant aggregator in
    the previous round.
    """
    try:  # pragma: no cover - runtime dependency path
        from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
    except ImportError:  # pragma: no cover - depends on NVFLARE extras
        PTFedProxLoss = None

    import torch
    import torch.optim as optim

    model.to(device)
    optimizer = state.get("optimizer")
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        state["optimizer"] = optimizer

    criterion = state.get("criterion")
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        state["criterion"] = criterion

    amp_enabled = device.type == "cuda"
    scaler = state.get("scaler")
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        state["scaler"] = scaler

    criterion_prox = None
    model_global = None
    if fedproxloss_mu > 0:
        if PTFedProxLoss is None:
            raise ImportError(
                "FedProx support is unavailable in the installed NVFLARE package."
            )
        criterion_prox = PTFedProxLoss(mu=fedproxloss_mu)
        model_global = copy.deepcopy(model)
        for parameter in model_global.parameters():
            parameter.requires_grad = False

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Training data loader is empty.")
    total_steps = steps_per_epoch * aggregation_epochs
    log_interval = max(1, min(50, steps_per_epoch // 10))
    last_avg_loss = 0.0
    last_balanced_accuracy = 0.0
    LOGGER.info(
        "[%s] round %s starting participant local_train: epochs=%s "
        "steps_per_epoch=%s class_weights=%s server_meta_keys=%s",
        client_name,
        current_round,
        aggregation_epochs,
        steps_per_epoch,
        [float(value) for value in class_weights],
        sorted(server_meta),
    )

    for epoch in range(aggregation_epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        true_positive = false_positive = true_negative = false_negative = 0
        for batch_index, batch_data in enumerate(train_loader, start=1):
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(outputs, labels)
                if criterion_prox is not None:
                    loss = loss + criterion_prox(model, model_global)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            with torch.no_grad():
                predictions = outputs.detach().argmax(dim=1)
                true_positive += int(((labels == 1) & (predictions == 1)).sum())
                false_positive += int(((labels == 0) & (predictions == 1)).sum())
                true_negative += int(((labels == 0) & (predictions == 0)).sum())
                false_negative += int(((labels == 1) & (predictions == 0)).sum())

            if (
                batch_index == 1
                or batch_index == steps_per_epoch
                or batch_index % log_interval == 0
            ):
                LOGGER.info(
                    "[%s] round %s epoch %s/%s step %s/%s avg_loss=%.6f",
                    client_name,
                    current_round,
                    epoch + 1,
                    aggregation_epochs,
                    batch_index,
                    steps_per_epoch,
                    running_loss / batch_index,
                )

        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        last_balanced_accuracy = (sensitivity + specificity) / 2
        last_avg_loss = running_loss / steps_per_epoch
        summary_writer.add_scalar(
            "train_loss",
            last_avg_loss,
            current_round * max(total_steps, 1) + epoch,
        )
        summary_writer.add_scalar(
            "train_balanced_accuracy",
            last_balanced_accuracy,
            current_round * max(total_steps, 1) + epoch,
        )
        LOGGER.info(
            "[%s] round %s epoch %s/%s complete in %.1fs: avg_loss=%.6f "
            "train_balanced_accuracy=%.6f",
            client_name,
            current_round,
            epoch + 1,
            aggregation_epochs,
            time.perf_counter() - epoch_start,
            last_avg_loss,
            last_balanced_accuracy,
        )

    params = {key: value.detach() for key, value in model.state_dict().items()}
    return FLModel(
        params=params,
        metrics={
            "train_loss": last_avg_loss,
            "train_balanced_accuracy": last_balanced_accuracy,
        },
        meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: total_steps},
    )
