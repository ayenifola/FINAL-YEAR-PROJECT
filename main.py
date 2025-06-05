import random
from functools import partial
from pathlib import Path
from typing import Literal

import cyclopts
import polars as pl
import rootutils
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MetricCollection,
    AUROC,
    ConfusionMatrix,
)

from src.data import dataset_factory, load_data, NetworkTrafficDataset
from src.feature_selection import *
from src.models import model_factory
from src.trainer import train_model, evaluate_model, set_random_state

root_dir = rootutils.find_root(__file__, [".project-root"])
app = cyclopts.App()


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
    s_length: int = 10,
    threshold: float = 0.5,
    debug: bool = False,
):
    set_random_state(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = _execute_train(
        dataset_type,
        model_type,
        s_length,
        device,
        data_dir,
        features,
        lr,
        num_epochs,
        batch_size,
        num_workers,
        random_state,
        train_metric_log_interval,
        threshold,
        debug,
    )
    if results is None:
        return

    train_metrics_dict, val_metrics_dict, test_metrics_dict, best_model = results

    logger.info("Train Results: {0}", {k: f"{v:.6f}" for k, v in train_metrics_dict.items()})
    logger.info("Validation Results: {0}", {k: f"{v:.6f}" for k, v in val_metrics_dict.items()})
    logger.info(f"Best validation F1 score: {best_model['metric_value']:.4f}")
    logger.info(f"Model saved to: {best_model['checkpoint_path']}")

    logger.info("Evaluating on test set...")
    logger.info("Test Results: {0}", {k: f"{v:.6f}" for k, v in test_metrics_dict.items()})


@app.command()
def evaluate(
    model_checkpoint: str,
    data_dir: Path = root_dir / "data",
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"] = "mqttset",
    features: list[str] | None = None,
    model_type: Literal["cnn1d", "lstm", "gru"] = "cnn1d",
    s_length: int = 10,
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    threshold: float = 0.5,
):
    set_random_state(random_state)

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
    s_length: int = 10,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_features, metrics = _execute_feature_selection(
        method,
        dataset_type,
        model_type,
        s_length,
        device,
        data_dir,
        lr,
        num_epochs,
        batch_size,
        num_train_samples,
        num_val_samples,
        num_workers,
        random_state,
        classification_threshold,
        population_size,
        num_generations,
        crossover_prob,
        mutation_prob,
        dataset_split_method,
        verbose,
    )

    logger.info(f"Genetic algorithm metrics: {metrics}")
    logger.info(f"Selected feature names: {selected_features}")


@app.command()
def run_experiments(
    # Parameters for feature selection part
    gp_num_epochs: int = 5,
    gp_batch_size: int = 64,
    gp_population_size: int = 50,
    gp_num_generations: int = 15,
    gp_lr: float = 5e-3,
    gp_crossover_prob: float = 0.7,
    gp_mutation_prob: float = 0.2,
    gp_num_train_samples: int = 500,
    gp_num_val_samples: int = 200,
    gp_dataset_split_method: str = "balanced",
    # Parameters for model training part
    train_lr: float = 1e-3,
    train_num_epochs: int = 5,
    train_batch_size: int = 128,
    # Common parameters
    data_dir: Path = root_dir / "data",
    num_workers: int = 8,
    random_state: int | None = 42,
    classification_threshold: float = 0.5,
    s_length: int = 10,
    results_path: Path = root_dir / "checkpoints" / "experiment_results_gp.parquet",
    checkpoints_root_dir: Path = root_dir / "checkpoints" / "experiments_gp",
    debug: bool = False,
):
    """
    Runs a series of experiments: GP feature selection followed by model training.
    Saves results to a CSV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting batch of experiments. Results will be saved to {results_path}")
    if debug:
        logger.warning("PIPELINE DEBUG MODE ENABLED: Using dummy outputs for FS and Training.")

    all_results = []

    models_to_run: list[Literal["cnn1d", "lstm", "gru"]] = ["cnn1d", "lstm", "gru"]
    datasets_to_run: list[Literal["mqttset", "iiotset", "x-iiotd"]] = [
        "mqttset",
        "iiotset",
        "x-iiotd",
    ]

    # Create root directory for experiment checkpoints
    checkpoints_root_dir.mkdir(parents=True, exist_ok=True)

    experiment_counter = 0
    for dataset_type in datasets_to_run:
        for model_type in models_to_run:
            experiment_counter += 1
            current_random_state = (
                random_state + experiment_counter if random_state is not None else None
            )
            # Set seed for this specific experiment iteration
            set_random_state(current_random_state)

            logger.info(
                f"--- Experiment {experiment_counter}: Model={model_type}, Dataset={dataset_type}, RS={current_random_state} ---"
            )

            # --- 1. Feature Selection using GP ---
            logger.info("Step 1: Feature Selection using GP...")
            selected_features, stats = _execute_feature_selection(
                method="gp",
                dataset_type=dataset_type,
                model_type=model_type,
                s_length=s_length,
                data_dir=data_dir,
                device=device,
                lr=gp_lr,
                num_epochs=gp_num_epochs,
                batch_size=gp_batch_size,
                population_size=gp_population_size,
                num_generations=gp_num_generations,
                crossover_prob=gp_crossover_prob,
                mutation_prob=gp_mutation_prob,
                num_train_samples=gp_num_train_samples,
                num_val_samples=gp_num_val_samples,
                num_workers=0,
                random_state=current_random_state,
                classification_threshold=classification_threshold,
                dataset_split_method=gp_dataset_split_method,
                verbose=True,
            )

            current_row = {
                "model": model_type,
                "dataset": dataset_type,
                "sequence_length": s_length,
                "random_state": current_random_state,
                "best_fitness_gp": None,
                "features_count_gp": 0,
                "features_gp": [],
                "test_accuracy": None,
                "test_precision": None,
                "test_recall": None,
                "test_f1": None,
                "test_auroc": None,
                "test_confusion_matrix": None,
            }

            if stats:
                current_row["best_fitness_gp"] = stats["best_fitness"]
                current_row["features_count_gp"] = stats["num_selected_features"]

            current_row["features_gp"] = selected_features
            logger.info(f"GP selected features ({len(selected_features)}): {selected_features}")

            # --- 2. Train, Test, and Evaluate Model with GP features ---
            logger.info("Step 2: Training and evaluating model with GP-selected features...")

            train_checkpoint_dir = (
                checkpoints_root_dir / f"{model_type}_{dataset_type}_RS{current_random_state}"
            )

            train_results = _execute_train(
                dataset_type=dataset_type,
                data_dir=data_dir,
                device=device,
                model_type=model_type,
                features=selected_features,
                lr=train_lr,
                num_epochs=train_num_epochs,
                batch_size=train_batch_size,
                num_workers=num_workers,
                random_state=current_random_state,
                s_length=s_length,
                threshold=classification_threshold,
                checkpoint_dir=train_checkpoint_dir,
                debug=debug,
            )
            assert train_results is not None
            _, _, test_metrics, _ = train_results

            current_row.update(
                {
                    "test_accuracy": float(test_metrics["test_acc"]),
                    "test_precision": float(test_metrics["test_prec"]),
                    "test_recall": float(test_metrics["test_rec"]),
                    "test_f1": float(test_metrics["test_f1"]),
                    "test_auroc": float(test_metrics["test_auroc"]),
                    "test_confusion_matrix": test_metrics["test_confusion_matrix"]
                    .cpu()
                    .detach()
                    .numpy()
                    .flatten()
                    .tolist(),
                }
            )
            logger.info(f"Test metrics for {model_type}-{dataset_type}: {test_metrics}")

            all_results.append(current_row)
            df_results = pl.DataFrame(
                all_results,
                schema={
                    "model": pl.Utf8,
                    "dataset": pl.Utf8,
                    "sequence_length": pl.Int64,
                    "random_state": pl.Int64,
                    "best_fitness_gp": pl.List(pl.Float64),
                    "features_count_gp": pl.Int64,
                    "features_gp": pl.List(pl.Utf8),
                    "test_accuracy": pl.Float64,
                    "test_precision": pl.Float64,
                    "test_recall": pl.Float64,
                    "test_f1": pl.Float64,
                    "test_auroc": pl.Float64,
                    "test_confusion_matrix": pl.List(pl.Float64),
                },
            )
            df_results.write_parquet(results_path)
            logger.info(f"Intermediate results saved to {results_path}")

    logger.info("All experiments completed.")
    if all_results:
        logger.info(f"Final results summary:\n{pl.DataFrame(all_results)}")
    else:
        logger.info("No experiments were run or no results to save.")


def _execute_train(
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"],
    model_type: Literal["cnn1d", "lstm", "gru"],
    s_length: int,
    device: torch.device,
    data_dir: Path = root_dir / "data",
    checkpoint_dir: Path = root_dir / "checkpoints",
    features: list[str] | None = None,
    lr: float = 1e-3,
    num_epochs: int = 5,
    batch_size: int = 128,
    num_workers: int = 8,
    random_state: int | None = 42,
    train_metric_log_interval: int = 1000,
    threshold: float = 0.5,
    debug: bool = False,
):
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
            "confusion_matrix": ConfusionMatrix(task="binary", num_classes=2),
        }
    )

    summary(
        model,
        input_size=(batch_size, s_length, input_dim),
        device=device,
        col_names=["input_size", "output_size", "num_params"],
    )

    if debug:
        return None

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
        checkpoint_path=checkpoint_dir
        / f"{model_type}-{dataset_type}-epoch={{epoch:02d}}-val_f1={{val_f1:.4f}}.pt",
    )

    test_metrics_dict = evaluate_model(model, criterion, test_loader, device, metrics)

    return train_metrics_dict, val_metrics_dict, test_metrics_dict, best_model


def _execute_feature_selection(
    method: Literal["ga", "gp"],
    dataset_type: Literal["mqttset", "iiotset", "x-iiotd"],
    model_type: Literal["cnn1d", "lstm", "gru"],
    s_length: int,
    device: torch.device,
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
    def _model_factory(input_dim):
        return model_factory(model_type, input_dim=input_dim, sequence_length=s_length)

    def _sample(df: pl.DataFrame, max_size, split_method=dataset_split_method):
        if split_method == "stratified":
            _, indices = train_test_split(
                list(range(len(df))),
                test_size=max_size,
                stratify=df["sequence_label"].to_numpy(),
            )
            return df[indices]
        elif split_method == "balanced":
            half_size = max_size // 2

            class_0 = df.filter(pl.col("sequence_label") == 0)["sequence_id"].unique()
            class_1 = df.filter(pl.col("sequence_label") == 1)["sequence_id"].unique()

            if len(class_0) < half_size or len(class_1) < half_size:
                raise ValueError("Not enough samples to create balanced dataset")

            sampled_0 = class_0.sample(n=half_size, seed=random_state, shuffle=True)
            sampled_1 = class_1.sample(n=half_size, seed=random_state, shuffle=True)

            sequence_ids = pl.concat([sampled_0, sampled_1])

            return df.filter(pl.col("sequence_id").is_in(sequence_ids)).sort("sequence_id")
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

    selected_features = [dataset_train.feature_names[i] for i in selected_indices]
    return selected_features, metrics


if __name__ == "__main__":
    app()
