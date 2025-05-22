from functools import partial
from pathlib import Path

import rootutils
import torch
import torch.nn as nn
import typer
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection, AUROC
from torchinfo import summary
from src.data import dataset_factory
from src.models import model_factory
from src.trainer import train_model, evaluate_model, set_random_state
from src.feature_selection import *

root_dir = rootutils.find_root(__file__, [".project-root"])
app = typer.Typer()


@app.command()
def train(
    dataset_type: str = "mqttset",
    model_type: str = "mlp",
    feature_indices: list[int] | None = None,
    lr: float = 1e-3,
    num_epochs: int = 1,
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    train_metric_log_interval: int = 1000,
    threshold: float = 0.5,
):
    set_random_state(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, extras = dataset_factory(
        dataset=dataset_type, feature_indices=feature_indices, random_state=random_state
    )
    train_dataset, val_dataset, test_dataset = datasets["train"], datasets["val"], datasets["test"]
    pos_weight = extras["pos_weight"]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # prepare model
    input_dim = train_dataset[0][0].shape[0]
    model = model_factory(model_type, input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = MetricCollection(
        {
            "acc": Accuracy(task="binary", threshold=threshold),
            "prec": Precision(task="binary", threshold=threshold),
            "rec": Recall(task="binary", threshold=threshold),
            "f1": F1Score(task="binary", threshold=threshold),
            "auroc": AUROC(task="binary"),
        }
    )

    summary(
        model,
        input_size=(batch_size, input_dim),
        device=device,
        col_names=["input_size", "output_size", "num_params"],
    )

    train_metrics_dict, val_metrics_dict, best_model = train_model(
        model=model,
        criterion=criterion,
        optimizer_factory=partial(torch.optim.AdamW, lr=lr),
        scheduler_factory=partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.5, patience=2
        ),
        metrics=metrics,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        train_metric_log_interval=train_metric_log_interval,
        scheduler_monitor_metric="val_f1",
        checkpoint_best_metric_name="val_f1",
        checkpoint_metric_mode="max",
        checkpoint_path=root_dir
        / "checkpoints"
        / f"{model_type}-epoch={{epoch:02d}}-val_f1={{val_f1:.4f}}.pt",
    )

    logger.info("Train Results: {0}", {k: f"{v:.6f}" for k, v in train_metrics_dict.items()})
    logger.info("Validation Results: {0}", {k: f"{v:.6f}" for k, v in val_metrics_dict.items()})
    logger.info(f"Best validation F1 score: {best_model['metric_value']:.4f}")
    logger.info(f"Model saved to: {best_model['checkpoint_path']}")

    logger.info("Evaluating on test set...")
    test_metrics_dict = evaluate_model(model, criterion, test_loader, device, metrics)
    logger.info("Test Results: {0}", {k: f"{v:.6f}" for k, v in test_metrics_dict.items()})


@app.command()
def evaluate(
    model_checkpoint: str,
    dataset_type: str = "mqttset",
    feature_indices: list[int] | None = None,
    model_type: str = "mlp",
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    threshold: float = 0.5,
):
    set_random_state(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, extras = dataset_factory(
        dataset=dataset_type, feature_indices=feature_indices, random_state=random_state
    )
    test_dataset = datasets["test"]
    pos_weight = extras["pos_weight"]
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    input_dim = test_dataset[0][0].shape[0]
    model = model_factory(model_type, input_dim)

    summary(
        model,
        input_size=(batch_size, input_dim),
        device=device,
        col_names=["input_size", "output_size", "num_params"],
    )

    model_checkpoint = Path(model_checkpoint)
    if not model_checkpoint.exists():
        if (root_dir / model_checkpoint).exists():
            model_checkpoint = root_dir / model_checkpoint
        elif (root_dir / "checkpoints" / model_checkpoint).exists():
            model_checkpoint = root_dir / "checkpoints" / model_checkpoint
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
    logger.info(f"Evaluating model: {model_checkpoint}")
    model.load_state_dict(torch.load(str(model_checkpoint), map_location=device))

    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="binary", threshold=threshold),
            "precision": Precision(task="binary", threshold=threshold),
            "recall": Recall(task="binary", threshold=threshold),
            "f1_score": F1Score(task="binary", threshold=threshold),
            "auroc": AUROC(task="binary"),
        }
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_metrics_dict = evaluate_model(model, criterion, test_loader, device, metrics)
    logger.info("Test Results: {0}", {k: f"{v:.6f}" for k, v in test_metrics_dict.items()})


@app.command()
def feature_selection(
    dataset_type: str = "mqttset",
    model_type: str = "mlp",
    lr: float = 5e-3,
    num_epochs: int = 2,
    batch_size: int = 64,
    num_train_samples: int = 500,
    num_val_samples: int = 200,
    num_workers: int = 0,
    random_state: int | None = 42,
    classification_threshold: float = 0.5,
    population_size: int = 50,
    num_generations: int = 30,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    feature_penalty_factor: float = 0.1,
    verbose: bool = True,
):
    set_random_state(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, extras = dataset_factory(dataset_type, random_state)
    train_dataset, val_dataset = datasets["train"], datasets["val"]
    pos_weight = extras["pos_weight"]
    num_total_features = train_dataset[0][0].shape[0]
    feature_names = train_dataset.feature_names

    def _model_factory(input_dim):
        return model_factory(model_type, input_dim)

    fitness_function = partial(
        f1_score_based_fitness_evaluator,
        model_factory=_model_factory,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_epochs=num_epochs,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        optimizer_factory=partial(torch.optim.Adam, lr=lr),
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        feature_penalty_factor=feature_penalty_factor,
        classification_threshold=classification_threshold,
    )

    toolbox = initialize_toolbox(
        num_total_features=num_total_features, fitness_function=fitness_function
    )

    selected_indices, metrics = run_genetic_algorithm(
        toolbox=toolbox,
        population_size=population_size,
        num_generations=num_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        verbose=verbose,
    )
    logger.info(f"Selected feature indices: {selected_indices}")
    logger.info(f"Genetic algorithm metrics: {metrics}")
    if selected_indices:
        logger.info(f"Selected feature names: {[feature_names[i] for i in selected_indices]}")


if __name__ == "__main__":
    app()
