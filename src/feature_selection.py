import operator
import random
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms, gp
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Precision, Recall, F1Score, MetricCollection

from src.trainer import train_model

__all__ = [
    "f1_score_based_fitness_evaluator",
    "init_toolbox_ga",
    "initialize_toolbox_gp",
    "run_genetic_algorithm",
    "run_genetic_programming",
]


def f1_score_based_fitness_evaluator(
    individual: list[int] | gp.PrimitiveTree,
    num_features: int,
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
    """Evaluates the fitness of a Genetic Algorithm (GA) or Genetic Programming (GP) individual (feature subset) by training
    and evaluating a neural network.

    Args:
        individual: This is either
            - a binary list representing the feature subset (1 for selected, 0 for not selected) for genetic algorithm, or
            - a GP tree representing feature selection logic, for genetic programming.
        num_features: Total number of available features.
        model_factory: Function that takes input_feature_dim to create PyTorch neural network model to be trained.
        dataset_factory: Function that takes dataset-split and selected feature indices to create a Dataset.
        device: The device (e.g., 'cuda' or 'cpu') to train and evaluate the model on.
        num_epochs: Number of epochs to train the temporary NN for fitness evaluation. Defaults to 5.
        num_workers: DataLoader workers
        batch_size: Batch size for training the temporary NN. Defaults to 32.
        optimizer_factory: The optimizer factory.
        criterion: The loss function. Defaults to nn.BCEWithLogitsLoss().
        classification_threshold: Classification threshold for metrics

    Returns:
        tuple: A tuple containing a single float value representing the fitness
               (e.g., F1-score minus penalty). DEAP expects a tuple.
    """
    feature_mask: list[int] = (
        _evaluate_gp_tree_to_feature_mask(individual, num_features)
        if isinstance(individual, gp.PrimitiveTree)
        else individual
    )
    selected_feature_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]

    if not selected_feature_indices:
        return 0.0, float("inf")

    train_dataset = dataset_factory("train", selected_feature_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataset = dataset_factory("val", selected_feature_indices)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    input_dim = val_dataset[0][0].shape[-1]
    # TODO: Maybe remove this check. Should be caught by the initial check
    if input_dim == 0:
        return 0.0, float("inf")

    _, val_metrics_dict, _ = train_model(
        model=model_factory(input_dim),
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
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        scheduler_monitor_metric="val_f1",
        checkpoint_metric_mode="max",
        log_progress=False,
    )

    num_selected_features = sum(feature_mask)
    f1_score = float(val_metrics_dict["val_f1"])

    return f1_score, num_selected_features


def init_toolbox_ga(
    num_features: int, fitness_function: Callable[[list[int]], tuple[float, ...]]
) -> base.Toolbox:
    """
    Initializes DEAP creator and toolbox, registering the flexible evaluation function.
    """
    _clear_creators()

    # Multi-objective: maximize F1, minimize number of features
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def initialize_toolbox_gp(
    num_features: int, fitness_function: Callable[[gp.PrimitiveTree], tuple[float, float, float]]
) -> base.Toolbox:
    """Initialize DEAP toolbox for genetic programming feature selection.

    Args:
        num_features: Total number of available features
        fitness_function: Function to evaluate GP individuals

    Returns:
        Configured toolbox and primitive set
    """
    _clear_creators()

    # Create fitness class (maximize F1, minimize features)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, pset=None)

    # Create primitive set
    pset = _create_primitive_set(num_features)
    creator.Individual.pset = pset

    # Initialize toolbox
    toolbox = base.Toolbox()

    # GP-specific registration
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Genetic operators
    toolbox.register("evaluate", fitness_function)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Bloat control: limit the height of trees generated by crossover and mutation
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


def run_genetic_algorithm(
    toolbox: base.Toolbox,
    population_size: int,
    num_generations: int,
    crossover_prob: float,
    mutation_prob: float,
    verbose: bool = True,
) -> tuple[list[int], dict[str, Any]]:
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

    best_individual = sorted(hof, key=lambda ind: -ind.fitness.values[0])[0]
    best_fitness = best_individual.fitness.values
    selected_feature_indices = [i for i, bit in enumerate(best_individual) if bit == 1]

    return selected_feature_indices, {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "num_selected_features": sum(best_individual),
    }


def run_genetic_programming(
    toolbox: base.Toolbox,
    num_features: int,
    population_size: int,
    num_generations: int,
    crossover_prob: float,
    mutation_prob: float,
    verbose: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Run genetic programming for feature selection.

    Args:
        toolbox: Configured DEAP toolbox
        num_features: Total number of available features
        population_size: Size of population
        num_generations: Number of generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        verbose: Whether to print progress

    Returns:
        Tuple of (selected_feature_indices, results_dict)
    """
    # Initialize population
    pop = toolbox.population(n=population_size)

    # Statistics and hall of fame
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logger.info(
        "Starting Genetic Programming: {num_generations} generations, Pop size: {population_size}",
        num_generations=num_generations,
        population_size=population_size,
    )

    # Run evolution
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

    # Log top few best individuals from Pareto front
    for i, ind in enumerate(hof[:5]):
        f1, num_selected_features = ind.fitness.values
        logger.info(f"Pareto #{i + 1}: F1={f1:.4f}, Features={int(num_selected_features)}")

    # Select the best individual (highest F1 score)
    best_individual = max(hof, key=lambda ind_: ind_.fitness.values[0])
    best_fitness = best_individual.fitness.values

    feature_mask = _evaluate_gp_tree_to_feature_mask(best_individual, num_features)
    selected_feature_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]

    return selected_feature_indices, {
        "best_individual": best_individual,
        "best_individual_str": str(best_individual),
        "best_fitness": best_fitness,
        "num_selected_features": len(selected_feature_indices),
        "pareto_front": hof,
        "logbook": logbook,
    }


def _clear_creators():
    # Clear existing creator classes if they exist
    creator.FROZEN = False
    for attr_name in ["FitnessMulti", "Individual"]:
        if hasattr(creator, attr_name):
            delattr(creator, attr_name)


def _create_primitive_set(num_features: int) -> gp.PrimitiveSet:
    """Create a primitive set for genetic programming feature selection.

    Args:
        num_features: Total number of available features

    Returns:
        PrimitiveSet configured for feature selection
    """
    # arity=0 as tree itself doesn't take runtime inputs
    pset = gp.PrimitiveSet("MAIN", arity=0)

    # Logical operators for combining feature presence or absence tokens
    pset.addPrimitive(operator.and_, 2)
    pset.addPrimitive(operator.or_, 2)
    pset.addPrimitive(operator.not_, 1)

    # Add terminals for each feature
    for i in range(num_features):
        pset.addTerminal(True, name=f"F{i}")  # Feature i is selected
        pset.addTerminal(False, name=f"N{i}")  # Feature i is not selected

    return pset


def _evaluate_gp_tree_to_feature_mask(
    individual: gp.PrimitiveTree, num_features: int
) -> list[int]:
    """Convert a GP tree to a binary feature selection mask.

    Args:
        individual: GP tree representing feature selection logic
        num_features: Total number of features

    Returns:
        Binary list where 1 means a feature is selected, 0 means not selected
    """
    # Compile the tree into a callable function
    func = gp.compile(individual, pset=individual.pset if hasattr(individual, "pset") else None)

    feature_mask = []
    tree_str = str(individual)

    for i in range(num_features):
        # Create a context where we evaluate if feature i should be selected
        # This is a simplified approach - in practice, you might want more sophisticated logic

        feature_selected = f"F{i}" in tree_str
        feature_ignored = f"N{i}" in tree_str

        if feature_ignored:
            feature_mask.append(0)
        elif feature_selected:
            feature_mask.append(1)
        else:
            # If both or neither appear, use random selection biased towards selection
            feature_mask.append(1 if random.random() > 0.3 else 0)

    return feature_mask
