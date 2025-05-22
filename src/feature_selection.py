import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Precision, Recall, F1Score, MetricCollection

from src.data import AnomalyDetectionDataset
from src.trainer import train_model

__all__ = ["f1_score_based_fitness_evaluator", "initialize_toolbox", "run_genetic_algorithm"]


def f1_score_based_fitness_evaluator(
    individual_chromosome: list[int],
    model_factory: Callable[..., nn.Module],
    train_dataset: AnomalyDetectionDataset,
    val_dataset: AnomalyDetectionDataset,
    device: torch.device,
    optimizer_factory: Callable[..., torch.optim.Optimizer],
    criterion: nn.Module = nn.BCEWithLogitsLoss(),
    num_train_samples: int = 500,
    num_val_samples: int = 200,
    num_epochs: int = 5,
    num_workers: int = 0,
    batch_size: int = 32,
    feature_penalty_factor: float = 0.0,
    classification_threshold: float = 0.5,
) -> tuple[float]:
    """Evaluates the fitness of an individual (feature subset) by training
    and evaluating a neural network.

    The neural network class and its initialization arguments are passed flexibly,
    allowing for different model architectures to be used within the GA.
    A SubsetFeatureDataset is used internally to handle dynamic feature selection
    based on the individual's chromosome.

    Args:
        individual_chromosome: A binary list representing the feature subset
                                      (1 for selected, 0 for not selected).
        model_factory: Function that takes input_feature_dim to create class of the PyTorch neural network model to be trained.
        train_dataset: The full PyTorch Dataset for training, containing all features.
        val_dataset: The full PyTorch Dataset for validation, containing all features.
        device: The device (e.g., 'cuda' or 'cpu') to train and evaluate the model on.
        num_epochs: Number of epochs to train the temporary NN for fitness evaluation.
                                       Defaults to 5.
        batch_size: Batch size for training the temporary NN. Defaults to 32.
        optimizer_factory: The optimizer factory.
        criterion: The loss function. Defaults to nn.BCEWithLogitsLoss().
        feature_penalty_factor: A factor to penalize the fitness score based on the
                                                  number of selected features. Higher values penalize more.
                                                  Defaults to 0.0 (no penalty).

    Returns:
        tuple: A tuple containing a single float value representing the fitness
               (e.g., F1-score minus penalty). DEAP expects a tuple.
    """
    selected_feature_indices = [i for i, bit in enumerate(individual_chromosome) if bit == 1]

    if not selected_feature_indices:
        return (0.0,)

    train_indices = random.sample(range(len(train_dataset)), num_train_samples)
    current_train_loader = DataLoader(
        _SubsetFeatureDataset(train_dataset, selected_feature_indices, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_indices = random.sample(range(len(val_dataset)), num_val_samples)
    current_val_loader = DataLoader(
        _SubsetFeatureDataset(val_dataset, selected_feature_indices, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    input_dim_subset = len(selected_feature_indices)
    if input_dim_subset == 0:  # Should be caught by the initial check
        return (0.0,)

    _, val_metrics_dict, _ = train_model(
        model=model_factory(input_dim_subset),
        criterion=criterion,
        optimizer_factory=optimizer_factory,
        scheduler_factory=None,
        metrics=MetricCollection(
            {
                "prec": Precision(task="binary", threshold=classification_threshold),
                "rec": Recall(task="binary", threshold=classification_threshold),
                "f1": F1Score(task="binary", threshold=classification_threshold),
            }
        ),
        train_loader=current_train_loader,
        val_loader=current_val_loader,
        device=device,
        num_epochs=num_epochs,
        scheduler_monitor_metric="val_f1",
        checkpoint_metric_mode="max",
        log_progress=False,
    )

    selected_ratio = sum(individual_chromosome) / len(individual_chromosome)
    fitness_value = float(val_metrics_dict["val_f1"]) - (feature_penalty_factor * selected_ratio)

    return (fitness_value,)


class _SubsetFeatureDataset(Dataset):
    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        feature_indices: list[int],
        sample_indices: list[int],
    ):
        self._dataset = dataset
        self._sample_indices = sample_indices
        self._feature_indices = feature_indices
        # Validate that feature_indices are within bounds of original_dataset features
        if dataset and len(dataset) > 0:
            # Get one sample to check feature dimension
            sample_features, _ = dataset[0]
            if max(feature_indices) >= sample_features.shape[0]:
                raise ValueError("Feature index out of bounds for the original dataset.")

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx):
        idx = self._sample_indices[idx]
        features, label = self._dataset[idx]
        subset_features = features[self._feature_indices]
        return subset_features, label


def initialize_toolbox(
    num_total_features: int, fitness_function: Callable[[list[int]], tuple[float]]
) -> base.Toolbox:
    """
    Initializes DEAP creator and toolbox, registering the flexible evaluation function.
    """
    creator.FROZEN = False  # Allow re-creation if script is run multiple times in same session
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_total_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_genetic_algorithm(
    toolbox: base.Toolbox,
    population_size: int,
    num_generations: int,
    crossover_prob: float,
    mutation_prob: float,
    verbose: bool = True,
):
    """Runs the genetic algorithm using the provided DEAP toolbox and evolutionary parameters.

    This function orchestrates the evolution process, including selection, crossover,
    and mutation, over a specified number of generations. It uses a Hall of Fame
    to keep track of the best individual found.

    Args:
        toolbox: The DEAP toolbox configured with genetic operators
                                     and the fitness evaluation function.
        population_size: The number of individuals (chromosomes) in each generation.
        num_generations: The total number of generations to run the evolution for.
        crossover_prob: The probability of mating two individuals.
        mutation_prob: The probability of mutating an individual.
        verbose: Whether to print statistics during the evolution process.
                                  Defaults to True.

    Returns:
        tuple: A tuple containing:
            - best_individual_chromosome (list): The best feature subset found (binary list).
            - selected_feature_indices (list): A list of indices of the selected features.
    """
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logger.info(
        "Starting Genetic Algorithm: {num_generations} generations, Pop size: {population_size}",
        num_generations=num_generations,
        population_size=population_size,
    )
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    best_individual_chromosome = hof[0]
    best_fitness = best_individual_chromosome.fitness.values[0]
    selected_feature_indices = [i for i, bit in enumerate(best_individual_chromosome) if bit == 1]

    return selected_feature_indices, {
        "best_individual_chromosome": best_individual_chromosome,
        "best_fitness": best_fitness,
        "num_selected_features": sum(best_individual_chromosome),
    }
