import numpy as np
import random
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque


class OptimizationAlgorithm(Enum):
    BAYESIAN = "bayesian"
    RANDOM = "random"
    GRID = "grid"
    GENETIC = "genetic"
    PSO = "pso"


class AcquisitionFunction(Enum):
    UCB = "ucb"
    EI = "ei"
    PI = "pi"
    LCB = "lcb"


@dataclass
class HyperparameterConfig:
    name: str
    param_type: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    log_scale: bool = False
    choices: Optional[List[Any]] = None


@dataclass
class OptimizationResult:
    parameters: Dict[str, float]
    score: float
    iteration: int
    timestamp: float
    algorithm: str


@dataclass
class Particle:
    position: Dict[str, float]
    velocity: Dict[str, float]
    best_position: Dict[str, float]
    best_score: float


class GaussianProcess:
    def __init__(self, noise: float = 1e-4):
        self.noise = noise
        self.X = []
        self.y = []
        self.alpha = 1.0
        self.length_scale = 1.0

    def fit(self, X: List[List[float]], y: List[float]):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(X)
        n_samples = len(X)
        means = np.zeros(n_samples)
        stds = np.ones(n_samples) * 0.1

        if len(self.X) > 0:
            for i in range(n_samples):
                distances = np.linalg.norm(self.X - X[i], axis=1)
                kernel_vals = np.exp(-0.5 * distances ** 2 / self.length_scale ** 2)
                weighted_mean = np.sum(kernel_vals * self.y) / np.sum(kernel_vals) if np.sum(kernel_vals) > 0 else 0.5
                means[i] = weighted_mean

        return means, stds

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        distance = np.linalg.norm(x1 - x2)
        return self.alpha * np.exp(-0.5 * distance ** 2 / self.length_scale ** 2)


class BayesianOptimizer:
    def __init__(self, configs: List[HyperparameterConfig],
                 acquisition_func: AcquisitionFunction = AcquisitionFunction.UCB,
                 n_initial: int = 5, max_iter: int = 50):
        self.configs = configs
        self.acquisition_func = acquisition_func
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.gp = GaussianProcess()
        self.history: List[OptimizationResult] = []
        self._iteration = 0

    def _sample_random(self) -> Dict[str, float]:
        params = {}
        for config in self.configs:
            if config.choices:
                params[config.name] = random.choice(config.choices)
            elif config.log_scale:
                log_min = np.log10(config.min_value)
                log_max = np.log10(config.max_value)
                params[config.name] = 10 ** random.uniform(log_min, log_max)
            else:
                params[config.name] = random.uniform(config.min_value, config.max_value)
        return params

    def _to_vector(self, params: Dict[str, float]) -> List[float]:
        return [params[c.name] for c in self.configs]

    def _from_vector(self, vector: List[float]) -> Dict[str, float]:
        return {c.name: vector[i] for i, c in enumerate(self.configs)}

    def _acquisition(self, mean: float, std: float, best_score: float) -> float:
        if self.acquisition_func == AcquisitionFunction.UCB:
            return mean + 2 * std
        elif self.acquisition_func == AcquisitionFunction.EI:
            improvement = best_score - mean
            if std == 0:
                return 0
            z = improvement / std
            return improvement * self._norm_cdf(z) + std * self._norm_pdf(z)
        elif self.acquisition_func == AcquisitionFunction.PI:
            improvement = best_score - mean
            if std == 0:
                return 0
            z = improvement / std
            return self._norm_cdf(z)
        elif self.acquisition_func == AcquisitionFunction.LCB:
            return mean - 2 * std
        return mean

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + np.tanh(np.sqrt(np.pi / 2) * x))

    def _norm_pdf(self, x: float) -> float:
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        for _ in range(self.n_initial):
            params = self._sample_random()
            score = objective_func(params)
            self.history.append(OptimizationResult(
                parameters=params, score=score, iteration=self._iteration,
                timestamp=time.time(), algorithm="bayesian"
            ))
            self._iteration += 1

        X = [self._to_vector(r.parameters) for r in self.history]
        y = [r.score for r in self.history]
        self.gp.fit(X, y)

        best_result = max(self.history, key=lambda r: r.score)

        for _ in range(self.max_iter - self.n_initial):
            candidates = [self._sample_random() for _ in range(50)]
            vectors = [self._to_vector(c) for c in candidates]
            means, stds = self.gp.predict(vectors)

            best_score = best_result.score
            acquisition_values = [
                self._acquisition(means[i], stds[i], best_score)
                for i in range(len(candidates))
            ]

            best_idx = np.argmax(acquisition_values)
            params = candidates[best_idx]
            score = objective_func(params)

            self.history.append(OptimizationResult(
                parameters=params, score=score, iteration=self._iteration,
                timestamp=time.time(), algorithm="bayesian"
            ))
            self._iteration += 1

            if score > best_result.score:
                best_result = self.history[-1]

            X = [self._to_vector(r.parameters) for r in self.history]
            y = [r.score for r in self.history]
            self.gp.fit(X, y)

        return best_result

    def get_history(self) -> List[Dict[str, Any]]:
        return [{
            "parameters": r.parameters,
            "score": r.score,
            "iteration": r.iteration,
            "timestamp": r.timestamp
        } for r in self.history]


class RandomSearchOptimizer:
    def __init__(self, configs: List[HyperparameterConfig], max_iter: int = 100):
        self.configs = configs
        self.max_iter = max_iter
        self.history: List[OptimizationResult] = []
        self._iteration = 0

    def _sample_random(self) -> Dict[str, float]:
        params = {}
        for config in self.configs:
            if config.choices:
                params[config.name] = random.choice(config.choices)
            elif config.log_scale:
                log_min = np.log10(config.min_value)
                log_max = np.log10(config.max_value)
                params[config.name] = 10 ** random.uniform(log_min, log_max)
            else:
                params[config.name] = random.uniform(config.min_value, config.max_value)
        return params

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        best_result = None

        for _ in range(self.max_iter):
            params = self._sample_random()
            score = objective_func(params)

            result = OptimizationResult(
                parameters=params, score=score, iteration=self._iteration,
                timestamp=time.time(), algorithm="random"
            )
            self.history.append(result)
            self._iteration += 1

            if best_result is None or score > best_result.score:
                best_result = result

        return best_result


class GridSearchOptimizer:
    def __init__(self, configs: List[HyperparameterConfig], grid_points: int = 5):
        self.configs = configs
        self.grid_points = grid_points
        self.history: List[OptimizationResult] = []
        self._iteration = 0

    def _generate_grid(self) -> List[Dict[str, float]]:
        grids = []
        for config in self.configs:
            if config.choices:
                grids.append(config.choices)
            else:
                if config.log_scale:
                    log_min = np.log10(config.min_value)
                    log_max = np.log10(config.max_value)
                    points = np.logspace(log_min, log_max, self.grid_points)
                else:
                    points = np.linspace(config.min_value, config.max_value, self.grid_points)
                grids.append(points)

        from itertools import product
        combinations = list(product(*grids))
        return [{c.name: combinations[i][j] for j, c in enumerate(self.configs)}
                for i in range(len(combinations))]

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        grid = self._generate_grid()
        best_result = None

        for params in grid:
            score = objective_func(params)

            result = OptimizationResult(
                parameters=params, score=score, iteration=self._iteration,
                timestamp=time.time(), algorithm="grid"
            )
            self.history.append(result)
            self._iteration += 1

            if best_result is None or score > best_result.score:
                best_result = result

        return best_result


class GeneticAlgorithmOptimizer:
    def __init__(self, configs: List[HyperparameterConfig],
                 population_size: int = 20, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.configs = configs
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.history: List[OptimizationResult] = []
        self._iteration = 0

    def _create_individual(self) -> Dict[str, float]:
        params = {}
        for config in self.configs:
            if config.choices:
                params[config.name] = random.choice(config.choices)
            elif config.log_scale:
                log_min = np.log10(config.min_value)
                log_max = np.log10(config.max_value)
                params[config.name] = 10 ** random.uniform(log_min, log_max)
            else:
                params[config.name] = random.uniform(config.min_value, config.max_value)
        return params

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        mutated = dict(individual)
        for config in self.configs:
            if random.random() < self.mutation_rate:
                mutated[config.name] = self._create_individual()[config.name]
        return mutated

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        child = {}
        for config in self.configs:
            if random.random() < self.crossover_rate:
                child[config.name] = parent1[config.name]
            else:
                child[config.name] = parent2[config.name]
        return child

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        population = [self._create_individual() for _ in range(self.population_size)]
        best_result = None

        for gen in range(self.generations):
            scores = [(ind, objective_func(ind)) for ind in population]
            scores.sort(key=lambda x: x[1], reverse=True)

            for ind, score in scores:
                result = OptimizationResult(
                    parameters=ind, score=score, iteration=self._iteration,
                    timestamp=time.time(), algorithm="genetic"
                )
                self.history.append(result)
                self._iteration += 1

                if best_result is None or score > best_result.score:
                    best_result = result

            top_half = [s[0] for s in scores[:self.population_size // 2]]
            new_population = top_half.copy()

            while len(new_population) < self.population_size:
                parent1 = random.choice(top_half)
                parent2 = random.choice(top_half)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        return best_result


class ParticleSwarmOptimizer:
    def __init__(self, configs: List[HyperparameterConfig],
                 num_particles: int = 30, max_iter: int = 50,
                 inertia_weight: float = 0.7, cognitive_coeff: float = 1.4,
                 social_coeff: float = 1.4):
        self.configs = configs
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.history: List[OptimizationResult] = []
        self._iteration = 0

    def _create_particle(self) -> Particle:
        position = {}
        velocity = {}
        for config in self.configs:
            if config.choices:
                position[config.name] = random.choice(config.choices)
                velocity[config.name] = 0
            elif config.log_scale:
                log_min = np.log10(config.min_value)
                log_max = np.log10(config.max_value)
                position[config.name] = 10 ** random.uniform(log_min, log_max)
                velocity[config.name] = (10 ** random.uniform(log_min, log_max) - position[config.name]) * 0.1
            else:
                position[config.name] = random.uniform(config.min_value, config.max_value)
                velocity[config.name] = (config.max_value - config.min_value) * (random.random() - 0.5) * 0.1

        return Particle(
            position=position,
            velocity=velocity,
            best_position=dict(position),
            best_score=-float('inf')
        )

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        particles = [self._create_particle() for _ in range(self.num_particles)]
        global_best_position = None
        global_best_score = -float('inf')
        best_result = None

        for _ in range(self.max_iter):
            for particle in particles:
                score = objective_func(particle.position)
                result = OptimizationResult(
                    parameters=particle.position, score=score, iteration=self._iteration,
                    timestamp=time.time(), algorithm="pso"
                )
                self.history.append(result)
                self._iteration += 1

                if best_result is None or score > best_result.score:
                    best_result = result

                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = dict(particle.position)

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = dict(particle.position)

            for particle in particles:
                for config in self.configs:
                    if config.choices:
                        continue

                    cognitive_term = self.cognitive_coeff * random.random()
                    cognitive_term *= (particle.best_position[config.name] - particle.position[config.name])

                    social_term = self.social_coeff * random.random()
                    social_term *= (global_best_position[config.name] - particle.position[config.name])

                    particle.velocity[config.name] = (
                        self.inertia_weight * particle.velocity[config.name] +
                        cognitive_term + social_term
                    )

                    particle.position[config.name] += particle.velocity[config.name]

                    if not config.log_scale:
                        particle.position[config.name] = max(
                            config.min_value, min(config.max_value, particle.position[config.name])
                        )

        return best_result


class HyperparameterOptimizer:
    def __init__(self, algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN):
        self.algorithm = algorithm
        self.optimizers: Dict[OptimizationAlgorithm, Any] = {}
        self.results: List[OptimizationResult] = []
        self._current_configs: List[HyperparameterConfig] = []

    def configure(self, configs: List[HyperparameterConfig]):
        self._current_configs = configs
        self._initialize_optimizers(configs)

    def _initialize_optimizers(self, configs: List[HyperparameterConfig]):
        self.optimizers[OptimizationAlgorithm.BAYESIAN] = BayesianOptimizer(configs)
        self.optimizers[OptimizationAlgorithm.RANDOM] = RandomSearchOptimizer(configs)
        self.optimizers[OptimizationAlgorithm.GRID] = GridSearchOptimizer(configs)
        self.optimizers[OptimizationAlgorithm.GENETIC] = GeneticAlgorithmOptimizer(configs)
        self.optimizers[OptimizationAlgorithm.PSO] = ParticleSwarmOptimizer(configs)

    def optimize(self, objective_func: Callable[[Dict[str, float]], float],
                 algorithm: Optional[OptimizationAlgorithm] = None) -> OptimizationResult:
        algo = algorithm or self.algorithm
        optimizer = self.optimizers.get(algo)

        if not optimizer:
            raise ValueError(f"Unknown optimization algorithm: {algo}")

        result = optimizer.optimize(objective_func)
        self.results.append(result)

        return result

    def compare_algorithms(self, objective_func: Callable[[Dict[str, float]], float]) -> Dict[str, Any]:
        results = {}
        for algo in OptimizationAlgorithm:
            if algo in self.optimizers:
                result = self.optimizers[algo].optimize(objective_func)
                results[algo.value] = {
                    "best_score": result.score,
                    "best_params": result.parameters,
                    "iterations": len(self.optimizers[algo].history),
                    "algorithm": result.algorithm
                }

        best_algo = max(results.keys(), key=lambda k: results[k]["best_score"])
        return {
            "comparison": results,
            "best_algorithm": best_algo,
            "best_score": results[best_algo]["best_score"],
            "best_params": results[best_algo]["best_params"]
        }

    def get_history(self, algorithm: Optional[OptimizationAlgorithm] = None) -> List[Dict[str, Any]]:
        if algorithm and algorithm in self.optimizers:
            return self.optimizers[algorithm].get_history()
        return [r for opt in self.optimizers.values() for r in opt.get_history()]

    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        for algo, optimizer in self.optimizers.items():
            if optimizer.history:
                scores = [r.score for r in optimizer.history]
                stats[algo.value] = {
                    "total_iterations": len(optimizer.history),
                    "best_score": max(scores),
                    "avg_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "convergence_iteration": np.argmax(scores)
                }
        return stats

    def set_algorithm(self, algorithm: OptimizationAlgorithm):
        self.algorithm = algorithm

    def set_acquisition_function(self, func: AcquisitionFunction):
        if OptimizationAlgorithm.BAYESIAN in self.optimizers:
            self.optimizers[OptimizationAlgorithm.BAYESIAN].acquisition_func = func