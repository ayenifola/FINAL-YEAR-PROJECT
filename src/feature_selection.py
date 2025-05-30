import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Precision, Recall, F1Score, MetricCollection

from src.trainer import train_model

__all__ = ["f1_score_based_fitness_evaluator", "initialize_toolbox", "run_genetic_algorithm"]


def f1_score_based_fitness_evaluator(
    individual_chromosome: list[int],
    model_factory: Callable[..., nn.Module],
    dataset_factory: Callable[..., Dataset],
    device: torch.device,
    optimizer_factory: Callable[..., torch.optim.Optimizer],
    criterion: nn.Module = nn.BCEWithLogitsLoss(),
    num_epochs: int = 5,
    num_workers: int = 0,
    batch_size: int = 32,
    classification_threshold: float = 0.5,
) -> tuple[float, float]:
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
        dataset_factory: Function that takes dataset-split and selected feature indices to create a Dataset.
        device: The device (e.g., 'cuda' or 'cpu') to train and evaluate the model on.
        num_epochs: Number of epochs to train the temporary NN for fitness evaluation.
                                       Defaults to 5.
        batch_size: Batch size for training the temporary NN. Defaults to 32.
        optimizer_factory: The optimizer factory.
        criterion: The loss function. Defaults to nn.BCEWithLogitsLoss().

    Returns:
        tuple: A tuple containing a single float value representing the fitness
               (e.g., F1-score minus penalty). DEAP expects a tuple.
    """
    selected_feature_indices = [i for i, bit in enumerate(individual_chromosome) if bit == 1]

    if not selected_feature_indices:
        return 0.0, float("inf")  # Worst possible feature count

    train_dataset = dataset_factory("train", selected_feature_indices)
    current_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataset = dataset_factory("val", selected_feature_indices)
    current_val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    input_dim_subset = val_dataset[0][0].shape[0]
    if input_dim_subset == 0:  # Should be caught by the initial check
        return (0.0, float("inf"))

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

    selected_ratio = sum(individual_chromosome)  # / len(individual_chromosome)
    fitness_value = float(val_metrics_dict["val_f1"])

    return fitness_value, selected_ratio


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

    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)

    # Multi-objective: maximize F1, minimize number of features
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

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
    # hof = tools.HallOfFame(1)
    hof = tools.ParetoFront()
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
    # pop, logbook = algorithms.eaSimple(
    #     pop,
    #     toolbox,
    #     cxpb=crossover_prob,
    #     mutpb=mutation_prob,
    #     ngen=num_generations,
    #     stats=stats,
    #     halloffame=hof,
    #     verbose=verbose,
    # )
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=population_size,
        lambda_=population_size,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )
    for ind in hof:
        f1, num_features = ind.fitness.values
        logger.info(f"F1: {f1:.4f}, Features used: {int(num_features)}")

    best_individual_chromosome = sorted(hof, key=lambda ind: -ind.fitness.values[0])[0]
    best_fitness = best_individual_chromosome.fitness.values[0]
    selected_feature_indices = [i for i, bit in enumerate(best_individual_chromosome) if bit == 1]

    return selected_feature_indices, {
        "best_individual_chromosome": best_individual_chromosome,
        "best_fitness": best_fitness,
        "num_selected_features": sum(best_individual_chromosome),
    }
