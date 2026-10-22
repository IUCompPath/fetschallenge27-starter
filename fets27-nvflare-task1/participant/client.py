"""Participant-editable local training for the FeTS27 NVFLARE challenge.

Implement ``local_train`` to update the organizer-provided model and return an
``FLModel`` containing full, compatible model parameters plus any metadata the
participant aggregator needs. The organizer-owned client runner performs
``flare.receive``/``flare.send`` and computes the official validation metric.
"""

from __future__ import annotations

import copy
import logging
import time

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
except ModuleNotFoundError:  # pragma: no cover - used only without NVFLARE
    from fets27_challenge.compat import FLMetaKey, FLModel


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
    server_meta: dict,
    state: dict,
    summary_writer,
    client_name: str,
) -> FLModel:
    """Train locally and return full updated parameters and optional metadata.

    ``state`` persists for this client across rounds and can hold optimizer,
    scheduler, or algorithm-specific state. ``server_meta`` contains metadata
    returned by the participant aggregator in the previous round.

    ``model.task`` is ``"segmentation"`` (batch label: multi-channel mask, model
    output: mask logits) or ``"classification"`` (batch label: class indices of
    shape ``(B,)``, model output: logits of shape ``(B, num_classes)``).
    """
    try:  # pragma: no cover - runtime dependency path
        from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
    except ImportError:  # pragma: no cover - depends on NVFLARE extras
        PTFedProxLoss = None

    import torch.nn as nn
    import torch.optim as optim

    model.to(device)
    optimizer = state.get("optimizer")
    if optimizer is None:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        state["optimizer"] = optimizer

    criterion = state.get("criterion")
    if criterion is None:
        if model.task == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            from monai.losses import DiceLoss

            criterion = DiceLoss(
                smooth_nr=0,
                smooth_dr=1e-5,
                squared_pred=True,
                to_onehot_y=False,
                sigmoid=True,
            )
        state["criterion"] = criterion

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
    LOGGER.info(
        "[%s] round %s starting participant local_train: epochs=%s "
        "steps_per_epoch=%s server_meta_keys=%s",
        client_name,
        current_round,
        aggregation_epochs,
        steps_per_epoch,
        sorted(server_meta),
    )

    for epoch in range(aggregation_epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for batch_index, batch_data in enumerate(train_loader, start=1):
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if criterion_prox is not None:
                loss += criterion_prox(model, model_global)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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

        last_avg_loss = running_loss / steps_per_epoch
        summary_writer.add_scalar(
            "train_loss",
            last_avg_loss,
            current_round * max(total_steps, 1) + epoch,
        )
        LOGGER.info(
            "[%s] round %s epoch %s/%s complete in %.1fs: avg_loss=%.6f",
            client_name,
            current_round,
            epoch + 1,
            aggregation_epochs,
            time.perf_counter() - epoch_start,
            last_avg_loss,
        )

    params = {key: value.detach() for key, value in model.state_dict().items()}
    return FLModel(
        params=params,
        metrics={"train_loss": last_avg_loss},
        meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: total_steps},
    )
