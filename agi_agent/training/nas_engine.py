import numpy as np
import random
import time
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque


class ModelType(Enum):
    CNN = "cnn"
    TRANSFORMER = "transformer"
    MLP = "mlp"
    HYBRID = "hybrid"


class LayerType(Enum):
    CONV = "conv"
    TRANSFORMER_BLOCK = "transformer_block"
    DENSE = "dense"
    POOLING = "pooling"
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"
    ATTENTION = "attention"
    LAYER_NORM = "layer_norm"
    FEED_FORWARD = "feed_forward"
    EMBEDDING = "embedding"


class SearchStrategy(Enum):
    RANDOM_SEARCH = "random_search"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"


@dataclass
class ArchitectureSpace:
    model_type: ModelType
    max_layers: int = 10
    min_layers: int = 2
    max_units: int = 1024
    min_units: int = 16
    supported_layers: List[LayerType] = field(default_factory=list)


@dataclass
class LayerConfig:
    layer_type: LayerType
    units: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: str = "same"
    dropout_rate: float = 0.0
    num_heads: int = 8
    activation: str = "relu"
    normalization: bool = True


@dataclass
class ArchitectureCandidate:
    architecture_id: str
    model_type: ModelType
    layers: List[LayerConfig]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    train_time: float = 0.0
    params_count: int = 0


@dataclass
class NASResult:
    best_architecture: ArchitectureCandidate
    search_history: List[ArchitectureCandidate]
    total_iterations: int
    search_time: float
    strategy: str


class ArchitectureGenerator:
    def __init__(self, space: ArchitectureSpace):
        self.space = space

    def generate_random(self) -> ArchitectureCandidate:
        num_layers = random.randint(self.space.min_layers, self.space.max_layers)
        layers = []

        for _ in range(num_layers):
            layer_type = random.choice(self.space.supported_layers)
            layers.append(self._generate_layer(layer_type))

        return ArchitectureCandidate(
            architecture_id=f"arch_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            model_type=self.space.model_type,
            layers=layers,
            hyperparameters=self._generate_hyperparameters()
        )

    def _generate_layer(self, layer_type: LayerType) -> LayerConfig:
        units = random.randint(self.space.min_units, self.space.max_units)
        units = int(np.power(2, np.round(np.log2(units))))

        config = LayerConfig(layer_type=layer_type, units=units)

        if layer_type == LayerType.CONV:
            config.kernel_size = random.choice([1, 3, 5])
            config.stride = random.choice([1, 2])
            config.padding = random.choice(["same", "valid"])
            config.activation = random.choice(["relu", "gelu", "swish"])

        elif layer_type == LayerType.TRANSFORMER_BLOCK:
            config.num_heads = random.choice([4, 8, 16])
            config.units = random.choice([64, 128, 256, 512])
            config.activation = random.choice(["relu", "gelu"])
            config.dropout_rate = random.uniform(0.0, 0.3)

        elif layer_type == LayerType.DENSE:
            config.activation = random.choice(["relu", "gelu", "tanh", "sigmoid"])
            config.dropout_rate = random.uniform(0.0, 0.5)

        elif layer_type == LayerType.POOLING:
            config.kernel_size = random.choice([2, 3])
            config.stride = random.choice([1, 2])

        elif layer_type == LayerType.ATTENTION:
            config.num_heads = random.choice([4, 8, 16])
            config.dropout_rate = random.uniform(0.0, 0.3)

        elif layer_type == LayerType.FEED_FORWARD:
            config.units = random.choice([128, 256, 512, 1024])
            config.activation = random.choice(["relu", "gelu"])

        elif layer_type == LayerType.EMBEDDING:
            config.units = random.choice([64, 128, 256])

        return config

    def _generate_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": 10 ** random.uniform(-6, -3),
            "batch_size": random.choice([16, 32, 64, 128]),
            "optimizer": random.choice(["adam", "adamw", "sgd"]),
            "weight_decay": random.uniform(0, 1e-4),
            "epochs": random.randint(10, 100)
        }


class EvolutionaryNAS:
    def __init__(self, space: ArchitectureSpace,
                 population_size: int = 20, generations: int = 30,
                 mutation_rate: float = 0.2, crossover_rate: float = 0.5):
        self.space = space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generator = ArchitectureGenerator(space)
        self.history: List[ArchitectureCandidate] = []

    def _mutate(self, candidate: ArchitectureCandidate) -> ArchitectureCandidate:
        new_layers = []
        for layer in candidate.layers:
            if random.random() < self.mutation_rate:
                new_layer = self.generator._generate_layer(layer.layer_type)
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)

        if random.random() < self.mutation_rate:
            if len(new_layers) < self.space.max_layers and random.random() < 0.5:
                new_layers.append(self.generator._generate_layer(
                    random.choice(self.space.supported_layers)
                ))
            elif len(new_layers) > self.space.min_layers:
                new_layers.pop(random.randint(0, len(new_layers) - 1))

        new_hp = dict(candidate.hyperparameters)
        if random.random() < self.mutation_rate:
            new_hp["learning_rate"] = 10 ** random.uniform(-6, -3)
        if random.random() < self.mutation_rate:
            new_hp["batch_size"] = random.choice([16, 32, 64, 128])

        return ArchitectureCandidate(
            architecture_id=f"arch_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            model_type=self.space.model_type,
            layers=new_layers,
            hyperparameters=new_hp
        )

    def _crossover(self, parent1: ArchitectureCandidate, parent2: ArchitectureCandidate) -> ArchitectureCandidate:
        min_len = min(len(parent1.layers), len(parent2.layers))
        split_point = random.randint(0, min_len)

        new_layers = parent1.layers[:split_point] + parent2.layers[split_point:]

        new_layers = new_layers[:self.space.max_layers]

        if len(new_layers) < self.space.min_layers:
            for _ in range(self.space.min_layers - len(new_layers)):
                new_layers.append(self.generator._generate_layer(
                    random.choice(self.space.supported_layers)
                ))

        new_hp = {}
        for key in parent1.hyperparameters:
            if random.random() < self.crossover_rate:
                new_hp[key] = parent1.hyperparameters[key]
            else:
                new_hp[key] = parent2.hyperparameters.get(key, parent1.hyperparameters[key])

        return ArchitectureCandidate(
            architecture_id=f"arch_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            model_type=self.space.model_type,
            layers=new_layers,
            hyperparameters=new_hp
        )

    def search(self, evaluate_func: Callable[[ArchitectureCandidate], float]) -> NASResult:
        start_time = time.time()
        population = [self.generator.generate_random() for _ in range(self.population_size)]
        best_candidate = None

        for gen in range(self.generations):
            for candidate in population:
                candidate.score = evaluate_func(candidate)
                self.history.append(candidate)

                if best_candidate is None or candidate.score > best_candidate.score:
                    best_candidate = candidate

            scores = [(c, c.score) for c in population]
            scores.sort(key=lambda x: x[1], reverse=True)

            top_performers = [s[0] for s in scores[:self.population_size // 2]]
            new_population = top_performers.copy()

            while len(new_population) < self.population_size:
                parent1 = random.choice(top_performers)
                parent2 = random.choice(top_performers)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        search_time = time.time() - start_time

        return NASResult(
            best_architecture=best_candidate,
            search_history=self.history,
            total_iterations=self.generations * self.population_size,
            search_time=search_time,
            strategy="evolutionary"
        )


class ReinforcementLearningNAS:
    def __init__(self, space: ArchitectureSpace,
                 episodes: int = 100, max_steps: int = 50,
                 learning_rate: float = 0.01, discount_factor: float = 0.99):
        self.space = space
        self.episodes = episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.generator = ArchitectureGenerator(space)
        self.history: List[ArchitectureCandidate] = []
        self._policy: Dict[str, float] = {}

    def _get_state_key(self, candidate: ArchitectureCandidate) -> str:
        layer_types = "-".join([l.layer_type.value for l in candidate.layers])
        units = "-".join([str(l.units) for l in candidate.layers])
        return f"{layer_types}_{units}"

    def _select_action(self, state_key: str) -> bool:
        if state_key not in self._policy:
            self._policy[state_key] = 0.5
        return random.random() < self._policy[state_key]

    def search(self, evaluate_func: Callable[[ArchitectureCandidate], float]) -> NASResult:
        start_time = time.time()
        best_candidate = None

        for episode in range(self.episodes):
            candidate = self.generator.generate_random()
            state_key = self._get_state_key(candidate)

            for step in range(self.max_steps):
                if self._select_action(state_key):
                    candidate = self.generator._mutate(candidate)
                    state_key = self._get_state_key(candidate)

                candidate.score = evaluate_func(candidate)
                self.history.append(candidate)

                if best_candidate is None or candidate.score > best_candidate.score:
                    best_candidate = candidate

                reward = candidate.score

                if state_key in self._policy:
                    self._policy[state_key] += self.learning_rate * reward

        search_time = time.time() - start_time

        return NASResult(
            best_architecture=best_candidate,
            search_history=self.history,
            total_iterations=self.episodes * self.max_steps,
            search_time=search_time,
            strategy="reinforcement_learning"
        )


class BayesianNAS:
    def __init__(self, space: ArchitectureSpace, n_initial: int = 10, max_iter: int = 100):
        self.space = space
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.generator = ArchitectureGenerator(space)
        self.history: List[ArchitectureCandidate] = []
        self._gp_data: List[Tuple[List[float], float]] = []

    def _encode_architecture(self, candidate: ArchitectureCandidate) -> List[float]:
        encoding = []
        for layer in candidate.layers:
            encoding.append(layer.layer_type.value.__hash__() % 10)
            encoding.append(layer.units / 1024)
            if layer.layer_type == LayerType.CONV:
                encoding.append(layer.kernel_size / 5)
                encoding.append(layer.stride / 2)
            elif layer.layer_type == LayerType.TRANSFORMER_BLOCK:
                encoding.append(layer.num_heads / 16)

        encoding += [0.0] * (50 - len(encoding))
        return encoding[:50]

    def search(self, evaluate_func: Callable[[ArchitectureCandidate], float]) -> NASResult:
        start_time = time.time()
        best_candidate = None

        for _ in range(self.n_initial):
            candidate = self.generator.generate_random()
            candidate.score = evaluate_func(candidate)
            self.history.append(candidate)
            self._gp_data.append((self._encode_architecture(candidate), candidate.score))

            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate

        for _ in range(self.max_iter - self.n_initial):
            candidates = [self.generator.generate_random() for _ in range(20)]
            encodings = [self._encode_architecture(c) for c in candidates]

            acquisition_values = []
            for i, encoding in enumerate(encodings):
                distances = [np.linalg.norm(np.array(e) - np.array(encoding)) for e, _ in self._gp_data]
                if distances:
                    min_dist = min(distances)
                    exploitation = sum(s for _, s in self._gp_data) / len(self._gp_data)
                    acquisition = 0.7 * exploitation + 0.3 * (1 / (min_dist + 1e-6))
                else:
                    acquisition = random.random()
                acquisition_values.append(acquisition)

            best_idx = np.argmax(acquisition_values)
            candidate = candidates[best_idx]
            candidate.score = evaluate_func(candidate)
            self.history.append(candidate)
            self._gp_data.append((self._encode_architecture(candidate), candidate.score))

            if candidate.score > best_candidate.score:
                best_candidate = candidate

        search_time = time.time() - start_time

        return NASResult(
            best_architecture=best_candidate,
            search_history=self.history,
            total_iterations=self.max_iter,
            search_time=search_time,
            strategy="bayesian"
        )


class NeuralArchitectureSearch:
    def __init__(self, model_type: ModelType, strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY):
        self.model_type = model_type
        self.strategy = strategy
        self.space = self._create_search_space(model_type)
        self.engines: Dict[SearchStrategy, Any] = {}
        self._initialize_engines()

    def _create_search_space(self, model_type: ModelType) -> ArchitectureSpace:
        if model_type == ModelType.CNN:
            return ArchitectureSpace(
                model_type=model_type,
                max_layers=12,
                min_layers=3,
                max_units=512,
                min_units=16,
                supported_layers=[LayerType.CONV, LayerType.POOLING, LayerType.DENSE,
                                  LayerType.BATCH_NORM, LayerType.DROPOUT]
            )
        elif model_type == ModelType.TRANSFORMER:
            return ArchitectureSpace(
                model_type=model_type,
                max_layers=8,
                min_layers=2,
                max_units=1024,
                min_units=64,
                supported_layers=[LayerType.TRANSFORMER_BLOCK, LayerType.ATTENTION,
                                  LayerType.FEED_FORWARD, LayerType.LAYER_NORM,
                                  LayerType.DROPOUT, LayerType.EMBEDDING]
            )
        elif model_type == ModelType.MLP:
            return ArchitectureSpace(
                model_type=model_type,
                max_layers=6,
                min_layers=2,
                max_units=2048,
                min_units=32,
                supported_layers=[LayerType.DENSE, LayerType.BATCH_NORM, LayerType.DROPOUT]
            )
        else:
            return ArchitectureSpace(
                model_type=model_type,
                max_layers=10,
                min_layers=3,
                max_units=1024,
                min_units=32,
                supported_layers=[LayerType.CONV, LayerType.TRANSFORMER_BLOCK,
                                  LayerType.DENSE, LayerType.POOLING,
                                  LayerType.BATCH_NORM, LayerType.DROPOUT,
                                  LayerType.ATTENTION, LayerType.FEED_FORWARD]
            )

    def _initialize_engines(self):
        self.engines[SearchStrategy.EVOLUTIONARY] = EvolutionaryNAS(self.space)
        self.engines[SearchStrategy.REINFORCEMENT_LEARNING] = ReinforcementLearningNAS(self.space)
        self.engines[SearchStrategy.BAYESIAN] = BayesianNAS(self.space)

    def search(self, evaluate_func: Callable[[ArchitectureCandidate], float],
               strategy: Optional[SearchStrategy] = None) -> NASResult:
        strat = strategy or self.strategy
        engine = self.engines.get(strat)

        if not engine:
            raise ValueError(f"Unknown search strategy: {strat}")

        return engine.search(evaluate_func)

    def compare_strategies(self, evaluate_func: Callable[[ArchitectureCandidate], float]) -> Dict[str, Any]:
        results = {}
        for strat in self.engines:
            result = self.engines[strat].search(evaluate_func)
            results[strat.value] = {
                "best_score": result.best_architecture.score,
                "search_time": result.search_time,
                "iterations": result.total_iterations,
                "layers": len(result.best_architecture.layers),
                "strategy": result.strategy
            }

        best_strat = max(results.keys(), key=lambda k: results[k]["best_score"])
        return {
            "comparison": results,
            "best_strategy": best_strat,
            "best_score": results[best_strat]["best_score"]
        }

    def generate_architecture(self) -> ArchitectureCandidate:
        return ArchitectureGenerator(self.space).generate_random()

    def set_strategy(self, strategy: SearchStrategy):
        self.strategy = strategy

    def configure_space(self, max_layers: int = None, min_layers: int = None,
                        max_units: int = None, min_units: int = None):
        if max_layers:
            self.space.max_layers = max_layers
        if min_layers:
            self.space.min_layers = min_layers
        if max_units:
            self.space.max_units = max_units
        if min_units:
            self.space.min_units = min_units
        self._initialize_engines()

    def get_search_space(self) -> Dict[str, Any]:
        return {
            "model_type": self.space.model_type.value,
            "max_layers": self.space.max_layers,
            "min_layers": self.space.min_layers,
            "max_units": self.space.max_units,
            "min_units": self.space.min_units,
            "supported_layers": [l.value for l in self.space.supported_layers]
        }

    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        for strat, engine in self.engines.items():
            if engine.history:
                scores = [c.score for c in engine.history]
                stats[strat.value] = {
                    "total_evaluations": len(engine.history),
                    "best_score": max(scores),
                    "avg_score": np.mean(scores),
                    "std_score": np.std(scores)
                }
        return stats