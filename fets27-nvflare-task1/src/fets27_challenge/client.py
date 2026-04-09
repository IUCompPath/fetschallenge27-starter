"""Locked NVFLARE client training script."""

from __future__ import annotations

import argparse
import copy

from fets27_challenge.data_pipeline import (
    build_dataloaders,
    evaluate_model,
    get_torch_module,
    require_runtime_dependencies,
)
from fets27_challenge.models import create_model_for_cohort


def parse_args():
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

    train_loader, valid_loader, inferer, post_transform, valid_metric = build_dataloaders(
        dataset_base_dir=args.dataset_base_dir,
        datalist_json_path=args.datalist_json_path,
        label_transform=args.label_transform,
        batch_size=args.batch_size,
        cache_rate=args.cache_dataset,
        roi_size=tuple(args.roi_size),
        infer_roi_size=tuple(args.infer_roi_size),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model_for_cohort(args.cohort).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    criterion_prox = None
    if args.fedproxloss_mu > 0:
        if PTFedProxLoss is None:
            raise ImportError("FedProx support is unavailable in the installed NVFLARE package.")
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params, strict=True)
        model.to(device)

        global_metric = evaluate_model(model, valid_loader, inferer, post_transform, valid_metric, device)
        summary_writer.add_scalar("val_metric_global_model", global_metric, input_model.current_round)

        model_global = None
        if criterion_prox is not None:
            model_global = copy.deepcopy(model)
            for parameter in model_global.parameters():
                parameter.requires_grad = False

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.aggregation_epochs

        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            for batch_data in train_loader:
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

            if len(train_loader) == 0:
                raise ValueError("Training data loader is empty.")
            avg_loss = running_loss / len(train_loader)
            global_step = input_model.current_round * max(total_steps, 1) + epoch
            summary_writer.add_scalar("train_loss", avg_loss, global_step)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"val_dice": global_metric},
            meta={"NUM_STEPS_CURRENT_ROUND": total_steps},
        )
        flare.send(output_model)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
