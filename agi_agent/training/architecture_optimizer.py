import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class OptimizationDimension(Enum):
    WIDTH = "width"
    DEPTH = "depth"
    CONNECTIVITY = "connectivity"
    MODULAR = "modular"


class OptimizationAction(Enum):
    GROW = "grow"
    PRUNE = "prune"
    RECONNECT = "reconnect"
    SPLIT_MODULE = "split_module"
    MERGE_MODULE = "merge_module"


@dataclass
class ArchitectureState:
    hidden_dim: int = 64
    num_layers: int = 3
    connection_density: float = 0.8
    module_count: int = 5
    parameter_count: int = 0
    capacity_utilization: float = 0.5
    last_change_step: int = 0
    change_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationCriterion:
    dimension: OptimizationDimension
    action: OptimizationAction
    trigger_metric: str
    trigger_threshold: float
    cooldown_steps: int = 100
    description: str = ""


class ArchitectureOptimizer:
    def __init__(self, initial_hidden_dim: int = 64, min_hidden_dim: int = 16, max_hidden_dim: int = 512):
        self.state = ArchitectureState(hidden_dim=initial_hidden_dim)
        self.min_hidden_dim = min_hidden_dim
        self.max_hidden_dim = max_hidden_dim
        self.min_layers = 1
        self.max_layers = 8
        self.growth_step = 8
        self.prune_step = 4

        self.criteria: List[OptimizationCriterion] = []
        self.optimization_history: deque = deque(maxlen=100)
        self._last_optimization_step = 0
        self._cooldown_counters: Dict[str, int] = {}

        self._init_default_criteria()

        self.grow_callbacks: List[Callable] = []
        self.prune_callbacks: List[Callable] = []

    def _init_default_criteria(self):
        self.criteria.append(OptimizationCriterion(
            dimension=OptimizationDimension.WIDTH,
            action=OptimizationAction.GROW,
            trigger_metric="capacity_utilization",
            trigger_threshold=0.85,
            cooldown_steps=200,
            description="当容量利用率超过85%时增加网络宽度"
        ))

        self.criteria.append(OptimizationCriterion(
            dimension=OptimizationDimension.WIDTH,
            action=OptimizationAction.PRUNE,
            trigger_metric="neuron_activation_rate",
            trigger_threshold=0.05,
            cooldown_steps=500,
            description="当神经元激活率持续低于5%时剪枝"
        ))

        self.criteria.append(OptimizationCriterion(
            dimension=OptimizationDimension.DEPTH,
            action=OptimizationAction.GROW,
            trigger_metric="feature_complexity",
            trigger_threshold=0.8,
            cooldown_steps=1000,
            description="当特征复杂度超过80%时增加网络深度"
        ))

        self.criteria.append(OptimizationCriterion(
            dimension=OptimizationDimension.CONNECTIVITY,
            action=OptimizationAction.PRUNE,
            trigger_metric="weight_magnitude",
            trigger_threshold=0.01,
            cooldown_steps=300,
            description="当权重幅值持续低于阈值时稀疏化连接"
        ))

    def register_grow_callback(self, callback: Callable):
        self.grow_callbacks.append(callback)

    def register_prune_callback(self, callback: Callable):
        self.prune_callbacks.append(callback)

    def check_and_optimize(self, current_step: int, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        optimizations = []

        self._update_state_from_metrics(metrics)

        for criterion in self.criteria:
            key = f"{criterion.dimension.value}_{criterion.action.value}"
            if key in self._cooldown_counters:
                self._cooldown_counters[key] -= 1
                if self._cooldown_counters[key] > 0:
                    continue

            metric_value = metrics.get(criterion.trigger_metric, 0.0)
            should_act = False

            if criterion.action in (OptimizationAction.GROW, OptimizationAction.SPLIT_MODULE):
                should_act = metric_value >= criterion.trigger_threshold
            else:
                should_act = metric_value <= criterion.trigger_threshold

            if should_act:
                result = self._apply_optimization(criterion, current_step, metric_value)
                if result:
                    optimizations.append(result)
                    self._cooldown_counters[key] = criterion.cooldown_steps

        return optimizations

    def _update_state_from_metrics(self, metrics: Dict[str, float]):
        if "capacity_utilization" in metrics:
            self.state.capacity_utilization = metrics["capacity_utilization"]
        if "hidden_dim" in metrics:
            self.state.hidden_dim = int(metrics["hidden_dim"])
        if "num_layers" in metrics:
            self.state.num_layers = int(metrics["num_layers"])
        if "parameter_count" in metrics:
            self.state.parameter_count = int(metrics["parameter_count"])

    def _apply_optimization(self, criterion: OptimizationCriterion, step: int,
                            trigger_value: float) -> Optional[Dict[str, Any]]:
        old_state = {
            "hidden_dim": self.state.hidden_dim,
            "num_layers": self.state.num_layers,
            "connection_density": self.state.connection_density,
            "module_count": self.state.module_count
        }

        success = False
        details = {}

        if criterion.dimension == OptimizationDimension.WIDTH:
            if criterion.action == OptimizationAction.GROW:
                success = self._grow_width()
                details = {"growth_step": self.growth_step}
            elif criterion.action == OptimizationAction.PRUNE:
                success = self._prune_width()
                details = {"prune_step": self.prune_step}

        elif criterion.dimension == OptimizationDimension.DEPTH:
            if criterion.action == OptimizationAction.GROW:
                success = self._grow_depth()
                details = {"new_layer": True}
            elif criterion.action == OptimizationAction.PRUNE:
                success = self._prune_depth()
                details = {"removed_layer": True}

        elif criterion.dimension == OptimizationDimension.CONNECTIVITY:
            if criterion.action == OptimizationAction.RECONNECT:
                success = self._reconnect_weights()
                details = {"reconnection_ratio": 0.1}
            elif criterion.action == OptimizationAction.PRUNE:
                success = self._sparsify_connections()
                details = {"sparsification_rate": 0.05}

        if success:
            new_state = {
                "hidden_dim": self.state.hidden_dim,
                "num_layers": self.state.num_layers,
                "connection_density": self.state.connection_density,
                "module_count": self.state.module_count
            }

            record = {
                "step": step,
                "dimension": criterion.dimension.value,
                "action": criterion.action.value,
                "trigger_metric": criterion.trigger_metric,
                "trigger_value": trigger_value,
                "old_state": old_state,
                "new_state": new_state,
                "details": details
            }

            self.optimization_history.append(record)
            self.state.last_change_step = step
            self.state.change_history.append(record)

            self._notify_callbacks(criterion.action, old_state, new_state)

            return record

        return None

    def _grow_width(self) -> bool:
        new_dim = self.state.hidden_dim + self.growth_step
        if new_dim > self.max_hidden_dim:
            return False
        self.state.hidden_dim = new_dim
        return True

    def _prune_width(self) -> bool:
        new_dim = self.state.hidden_dim - self.prune_step
        if new_dim < self.min_hidden_dim:
            return False
        self.state.hidden_dim = new_dim
        return True

    def _grow_depth(self) -> bool:
        if self.state.num_layers >= self.max_layers:
            return False
        self.state.num_layers += 1
        return True

    def _prune_depth(self) -> bool:
        if self.state.num_layers <= self.min_layers:
            return False
        self.state.num_layers -= 1
        return True

    def _sparsify_connections(self) -> bool:
        new_density = max(0.3, self.state.connection_density - 0.05)
        if abs(new_density - self.state.connection_density) < 0.01:
            return False
        self.state.connection_density = new_density
        return True

    def _reconnect_weights(self) -> bool:
        if self.state.connection_density >= 0.95:
            return False
        self.state.connection_density = min(0.95, self.state.connection_density + 0.05)
        return True

    def _notify_callbacks(self, action: OptimizationAction, old_state: dict, new_state: dict):
        callbacks = []
        if action in (OptimizationAction.GROW, OptimizationAction.SPLIT_MODULE):
            callbacks = self.grow_callbacks
        elif action in (OptimizationAction.PRUNE, OptimizationAction.MERGE_MODULE):
            callbacks = self.prune_callbacks

        for callback in callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                pass

    def get_optimization_stats(self) -> Dict[str, Any]:
        action_counts = {}
        dim_counts = {}

        for opt in self.optimization_history:
            action = opt["action"]
            dim = opt["dimension"]
            action_counts[action] = action_counts.get(action, 0) + 1
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        return {
            "current_state": {
                "hidden_dim": self.state.hidden_dim,
                "num_layers": self.state.num_layers,
                "connection_density": self.state.connection_density,
                "module_count": self.state.module_count,
                "capacity_utilization": self.state.capacity_utilization
            },
            "total_optimizations": len(self.optimization_history),
            "action_distribution": action_counts,
            "dimension_distribution": dim_counts,
            "last_change_step": self.state.last_change_step,
            "min_hidden_dim": self.min_hidden_dim,
            "max_hidden_dim": self.max_hidden_dim,
            "growth_step": self.growth_step,
            "prune_step": self.prune_step
        }

    def set_growth_step(self, step: int):
        self.growth_step = max(1, step)
        self.prune_step = max(1, step // 2)

    def set_dimension_bounds(self, min_dim: int, max_dim: int):
        self.min_hidden_dim = max(4, min_dim)
        self.max_hidden_dim = max(self.min_hidden_dim, max_dim)
