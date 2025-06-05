import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, MeanMetric
from tqdm import tqdm

__all__ = ["train_model", "evaluate_model", "set_random_state"]


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer_factory: Callable[..., torch.optim.Optimizer],
    metrics: MetricCollection,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | None = None,
    num_epochs: int = 10,
    train_metric_log_interval: int = 1000,
    log_progress: bool = True,
    scheduler_factory: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    scheduler_monitor_metric: str | None = None,
    checkpoint_best_metric_name: str = "val_f1",
    checkpoint_metric_mode: str = "max",
    checkpoint_path: Path | None = None,
):
    assert checkpoint_metric_mode in ["max", "min"]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optimizer_factory(model.parameters())
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

    criterion.to(device)
    model.to(device)
    total_batches_per_epoch = len(train_loader) + len(val_loader)

    metrics = metrics.to(device)
    train_metrics = metrics.clone(prefix="train_")
    val_metrics = metrics.clone(prefix="val_")
    train_metrics_dict = {}
    val_metrics_dict = {}

    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)

    # model saving stuff
    best_metric_value = float("-inf") if checkpoint_metric_mode == "max" else float("inf")
    last_model_save_path: Path | None = None

    for epoch in range(num_epochs):
        with tqdm(
            total=total_batches_per_epoch,
            desc=f"Epoch {epoch + 1:2d}/{num_epochs}",
            unit="batch",
            disable=not log_progress,
        ) as progress_bar:
            # ----------------------- #
            # Training phase
            # ----------------------- #
            model.train()
            log_dict: dict[str, str] = {"phase": "train"}
            for idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # metrics
                train_loss_metric.update(loss)
                train_metrics.update(outputs.sigmoid(), labels)

                # logging
                train_metrics_dict.update({"train_loss": train_loss_metric.compute()})
                if idx % train_metric_log_interval == 0 or idx == len(train_loader) - 1:
                    train_metrics_dict.update(train_metrics.compute())
                    log_dict.update(
                        {
                            k: f"{v:.4f}" if k.endswith("_loss") else f"{v:.2f}"
                            for k, v in train_metrics_dict.items()
                            if "matrix" not in k
                        }
                    )
                log_dict.update({"lr": f"{optimizer.param_groups[0]['lr']:.1e}"})
                progress_bar.update(1)
                progress_bar.set_postfix(**log_dict)

            # ------------------- #
            # Validation phase
            # ------------------- #
            log_dict["phase"] = "val"
            progress_bar.set_postfix(**log_dict)
            model.eval()
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)

                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)

                    val_loss_metric.update(loss)
                    val_metrics.update(outputs.sigmoid(), labels)

                    progress_bar.update(1)

            # logging
            val_metrics_dict.update({"val_loss": val_loss_metric.compute()})
            val_metrics_dict.update(val_metrics.compute())
            log_dict.update(
                {
                    k: f"{v:.4f}" if k.endswith("_loss") else f"{v:.2f}"
                    for k, v in val_metrics_dict.items()
                    if "matrix" not in k
                }
            )
            log_dict.update({"lr": f"{optimizer.param_groups[0]['lr']:.1e}"})
            progress_bar.set_postfix(**log_dict)

            if scheduler is not None:
                if scheduler_monitor_metric is not None:
                    scheduler.step(val_metrics_dict[scheduler_monitor_metric])
                else:
                    scheduler.step()

            # save the best model best on our metric
            if checkpoint_path is not None:
                if (
                    checkpoint_metric_mode == "max"
                    and val_metrics_dict[checkpoint_best_metric_name] > best_metric_value
                ) or (
                    checkpoint_metric_mode == "min"
                    and val_metrics_dict[checkpoint_best_metric_name] < best_metric_value
                ):
                    if last_model_save_path and last_model_save_path.exists():
                        last_model_save_path.unlink()
                    checkpoint_dir = checkpoint_path.parent
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_metric_value = float(val_metrics_dict[checkpoint_best_metric_name])
                    last_model_save_path = checkpoint_dir / checkpoint_path.name.format(
                        **{"epoch": epoch, checkpoint_best_metric_name: best_metric_value}
                    )
                    torch.save(model.state_dict(), last_model_save_path)

            train_loss_metric.reset()
            train_metrics.reset()

            val_loss_metric.reset()
            val_metrics.reset()

    return (
        train_metrics_dict,
        val_metrics_dict,
        {
            "metric_name": checkpoint_best_metric_name,
            "metric_value": best_metric_value,
            "checkpoint_path": last_model_save_path,
        },
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metrics: MetricCollection,
):
    model.eval()
    model.to(device)
    criterion.to(device)

    loss_metric = MeanMetric().to(device)
    metrics = metrics.clone(prefix="test_").to(device)

    for features, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)

        loss_metric.update(loss)
        metrics.update(outputs.sigmoid(), labels)

    test_metrics_dict: dict = metrics.compute()
    test_metrics_dict["test_loss"] = loss_metric.compute()
    return test_metrics_dict


def set_random_state(random_state):
    """Set random state for reproducibility."""
    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
