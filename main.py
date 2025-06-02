import cyclopts
import polars as pl
import random
import rootutils
import torch
import torch.nn as nn
from functools import partial
from loguru import logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection, AUROC
from typing import Literal

from src.data import dataset_factory, load_data, NetworkTrafficDataset
from src.feature_selection import *
from src.models import model_factory
from src.trainer import train_model, evaluate_model, set_random_state

root_dir = rootutils.find_root(__file__, [".project-root"])
app = cyclopts.App()

_SEQUENCE_LENGTHS = {
    "mqttset": 10,
    "iiotset": 10,
    "x-iiotd": 1,
}


@app.command()
def train(
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"] = "mqttset",
    data_dir: Path = root_dir / "data",
    model_type: Literal["cnn1d", "lstm", "gru"] = "cnn1d",
    features: list[str] | None = None,
    lr: float = 1e-3,
    num_epochs: int = 5,
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    train_metric_log_interval: int = 1000,
    sequence_length: Literal["auto"] | int = "auto",
    threshold: float = 0.5,
    debug: bool = False,
):
    set_random_state(random_state)

    s_length = (
        _SEQUENCE_LENGTHS[dataset_type]
        if isinstance(sequence_length, str) and sequence_length == "auto"
        else sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, extras = dataset_factory(
        dataset=dataset_type,
        data_dir=data_dir,
        features=features,
        sequence_length=s_length,
        random_state=random_state,
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
    input_dim = train_dataset[0][0].shape[-1]
    model = model_factory(model_type, input_dim=input_dim, sequence_length=s_length)
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
        input_size=(batch_size, s_length, input_dim),
        device=device,
        col_names=["input_size", "output_size", "num_params"],
    )

    if debug:
        return

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
        / f"{model_type}-{dataset_type}-epoch={{epoch:02d}}-val_f1={{val_f1:.4f}}.pt",
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
    data_dir: Path = root_dir / "data",
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"] = "mqttset",
    features: list[str] | None = None,
    model_type: Literal["cnn1d", "lstm", "gru"] = "cnn1d",
    sequence_length: Literal["auto"] | int = "auto",
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    threshold: float = 0.5,
):
    set_random_state(random_state)

    s_length = (
        _SEQUENCE_LENGTHS[dataset_type]
        if isinstance(sequence_length, str) and sequence_length == "auto"
        else sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, extras = dataset_factory(
        dataset=dataset_type,
        features=features,
        random_state=random_state,
        sequence_length=s_length,
        data_dir=data_dir,
    )
    test_dataset = datasets["test"]
    pos_weight = extras["pos_weight"]
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    input_dim = test_dataset[0][0].shape[-1]
    model = model_factory(model_type, input_dim=input_dim, sequence_length=s_length)

    summary(
        model,
        input_size=(batch_size, s_length, input_dim),
        device=device,
        col_names=["input_size", "output_size", "num_params"],
    )

    model_checkpoint: Path = Path(model_checkpoint)
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
    method: Literal["ga", "gp"] = "gp",
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"] = "mqttset",
    model_type: Literal["cnn1d", "lstm", "gru"] = "cnn1d",
    sequence_length: Literal["auto"] | int = "auto",
    data_dir: Path = root_dir / "data",
    lr: float = 5e-3,
    num_epochs: int = 5,
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
    dataset_split_method: str = "balanced",
    verbose: bool = True,
):
    assert dataset_split_method in ["stratified", "balanced", "random"]
    set_random_state(random_state)

    s_length = (
        _SEQUENCE_LENGTHS[dataset_type]
        if isinstance(sequence_length, str) and sequence_length == "auto"
        else sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _model_factory(input_dim):
        return model_factory(model_type, input_dim=input_dim, sequence_length=s_length)

    def _sample(df: pl.DataFrame, max_size, split_method=dataset_split_method):
        if split_method == "stratified":
            _, indices = train_test_split(
                list(range(len(df))),
                test_size=max_size,
                stratify=df["attack_label"].to_numpy(),
            )
            return df[indices]
        elif split_method == "balanced":
            half_size = max_size // 2

            class_0 = df.filter(pl.col("attack_label") == 0)
            class_1 = df.filter(pl.col("attack_label") == 1)

            if len(class_0) < half_size or len(class_1) < half_size:
                raise ValueError("Not enough samples to create balanced dataset")

            sampled_0 = class_0.sample(n=half_size, seed=random_state, shuffle=True)
            sampled_1 = class_1.sample(n=half_size, seed=random_state, shuffle=True)

            return pl.concat([sampled_0, sampled_1])
        else:
            indices = random.sample(range(len(df)), max_size)
            return df[indices]

    data, extras = load_data(
        dataset_type, data_dir=data_dir, sequence_length=s_length, random_state=random_state
    )
    train_data = data.filter(pl.col("split") == "train")
    val_data = data.filter(pl.col("split") == "val")

    train_data = _sample(train_data, num_train_samples)
    val_data = _sample(val_data, num_val_samples)
    pos_weight = extras["pos_weight"]

    def _dataset_factory(split: str, features: list[int]):
        return NetworkTrafficDataset(
            train_data if split == "train" else val_data, s_length, split, features
        )

    dataset_train = _dataset_factory("train", features=[])

    num_features = dataset_train.num_features
    fitness_function = partial(
        f1_score_based_fitness_evaluator,
        num_features=num_features,
        model_factory=_model_factory,
        dataset_factory=_dataset_factory,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        optimizer_factory=partial(torch.optim.Adam, lr=lr),
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        classification_threshold=classification_threshold,
    )

    if method == "ga":
        toolbox = init_toolbox_ga(num_features=num_features, fitness_function=fitness_function)
        selected_indices, metrics = run_genetic_algorithm(
            toolbox=toolbox,
            population_size=population_size,
            num_generations=num_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            verbose=verbose,
        )
    else:
        toolbox = initialize_toolbox_gp(
            num_features=num_features, fitness_function=fitness_function
        )
        selected_indices, metrics = run_genetic_programming(
            num_features=num_features,
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
        logger.info(
            f"Selected feature names: {[dataset_train.feature_names[i] for i in selected_indices]}"
        )


if __name__ == "__main__":
    app()
