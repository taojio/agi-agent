import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ParameterType(Enum):
    LEARNING_RATE = "learning_rate"
    EXPLORATION_RATE = "exploration_rate"
    TEMPERATURE = "temperature"


class AdjustmentStrategy(Enum):
    REAL_TIME = "real_time"
    PERIODIC = "periodic"
    HYBRID = "hybrid"


class PerformanceMetric:
    def __init__(self):
        self.loss_history: deque = deque(maxlen=100)
        self.accuracy_history: deque = deque(maxlen=100)
        self.reward_history: deque = deque(maxlen=100)
        self.task_complexity: float = 0.5
        self.environment_uncertainty: float = 0.5

    def update(self, loss: float, accuracy: float, reward: float = 0.0):
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.reward_history.append(reward)

    def set_task_complexity(self, complexity: float):
        self.task_complexity = max(0.0, min(1.0, complexity))

    def set_environment_uncertainty(self, uncertainty: float):
        self.environment_uncertainty = max(0.0, min(1.0, uncertainty))

    def get_loss_trend(self) -> float:
        if len(self.loss_history) < 5:
            return 0.0
        recent = list(self.loss_history)[-5:]
        earlier = list(self.loss_history)[-10:-5] if len(self.loss_history) >= 10 else recent
        return float(np.mean(recent) - np.mean(earlier))

    def get_accuracy_trend(self) -> float:
        if len(self.accuracy_history) < 5:
            return 0.0
        recent = list(self.accuracy_history)[-5:]
        earlier = list(self.accuracy_history)[-10:-5] if len(self.accuracy_history) >= 10 else recent
        return float(np.mean(recent) - np.mean(earlier))

    def get_avg_loss(self) -> float:
        return float(np.mean(self.loss_history)) if self.loss_history else 0.0

    def get_avg_accuracy(self) -> float:
        return float(np.mean(self.accuracy_history)) if self.accuracy_history else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_loss": self.get_avg_loss(),
            "avg_accuracy": self.get_avg_accuracy(),
            "loss_trend": self.get_loss_trend(),
            "accuracy_trend": self.get_accuracy_trend(),
            "task_complexity": self.task_complexity,
            "environment_uncertainty": self.environment_uncertainty,
            "history_length": len(self.loss_history)
        }


class HyperparameterController:
    def __init__(self, adjustment_strategy: AdjustmentStrategy = AdjustmentStrategy.HYBRID):
        self.strategy = adjustment_strategy
        self.parameters: Dict[ParameterType, float] = {
            ParameterType.LEARNING_RATE: 0.001,
            ParameterType.EXPLORATION_RATE: 0.3,
            ParameterType.TEMPERATURE: 1.0
        }
        self.default_parameters: Dict[ParameterType, float] = {
            ParameterType.LEARNING_RATE: 0.001,
            ParameterType.EXPLORATION_RATE: 0.3,
            ParameterType.TEMPERATURE: 1.0
        }
        self.bounds: Dict[ParameterType, Tuple[float, float]] = {
            ParameterType.LEARNING_RATE: (1e-6, 1.0),
            ParameterType.EXPLORATION_RATE: (0.01, 0.9),
            ParameterType.TEMPERATURE: (0.1, 10.0)
        }
        self.performance_metric = PerformanceMetric()
        self.adjustment_history: deque = deque(maxlen=500)
        self._periodic_counter = 0
        self._periodic_interval = 50
        self._last_adjustment_time = 0

    def set_parameter(self, param_type: ParameterType, value: float):
        bounds = self.bounds[param_type]
        self.parameters[param_type] = max(bounds[0], min(bounds[1], value))

    def get_parameter(self, param_type: ParameterType) -> float:
        return self.parameters[param_type]

    def adjust_learning_rate(self, force_adjust: bool = False) -> float:
        if self.strategy == AdjustmentStrategy.PERIODIC and not force_adjust:
            self._periodic_counter += 1
            if self._periodic_counter < self._periodic_interval:
                return self.parameters[ParameterType.LEARNING_RATE]
            self._periodic_counter = 0

        current_lr = self.parameters[ParameterType.LEARNING_RATE]
        loss_trend = self.performance_metric.get_loss_trend()
        accuracy_trend = self.performance_metric.get_accuracy_trend()
        complexity = self.performance_metric.task_complexity
        avg_loss = self.performance_metric.get_avg_loss()

        adjustment_factor = 1.0

        if loss_trend > 0.01:
            adjustment_factor = 0.85
        elif loss_trend < -0.01:
            adjustment_factor = 1.15
        elif accuracy_trend > 0.02:
            adjustment_factor = 1.1
        elif accuracy_trend < -0.02:
            adjustment_factor = 0.9

        complexity_factor = 1.0 / (1.0 + complexity * 0.5)
        adjustment_factor *= complexity_factor

        if avg_loss > 1.0:
            adjustment_factor *= 1.2
        elif avg_loss < 0.1:
            adjustment_factor *= 0.9

        new_lr = current_lr * adjustment_factor
        bounds = self.bounds[ParameterType.LEARNING_RATE]
        new_lr = max(bounds[0], min(bounds[1], new_lr))

        self.parameters[ParameterType.LEARNING_RATE] = new_lr

        self._record_adjustment(
            ParameterType.LEARNING_RATE,
            current_lr,
            new_lr,
            {"loss_trend": loss_trend, "accuracy_trend": accuracy_trend, "complexity": complexity}
        )

        return new_lr

    def adjust_exploration_rate(self, force_adjust: bool = False) -> float:
        if self.strategy == AdjustmentStrategy.PERIODIC and not force_adjust:
            self._periodic_counter += 1
            if self._periodic_counter < self._periodic_interval:
                return self.parameters[ParameterType.EXPLORATION_RATE]
            self._periodic_counter = 0

        current_er = self.parameters[ParameterType.EXPLORATION_RATE]
        uncertainty = self.performance_metric.environment_uncertainty
        accuracy_trend = self.performance_metric.get_accuracy_trend()
        avg_accuracy = self.performance_metric.get_avg_accuracy()

        target_er = 0.3 + uncertainty * 0.4

        if accuracy_trend > 0.02 and avg_accuracy > 0.8:
            target_er *= 0.7
        elif accuracy_trend < -0.02 or avg_accuracy < 0.4:
            target_er *= 1.3

        new_er = current_er * 0.9 + target_er * 0.1
        bounds = self.bounds[ParameterType.EXPLORATION_RATE]
        new_er = max(bounds[0], min(bounds[1], new_er))

        self.parameters[ParameterType.EXPLORATION_RATE] = new_er

        self._record_adjustment(
            ParameterType.EXPLORATION_RATE,
            current_er,
            new_er,
            {"uncertainty": uncertainty, "accuracy_trend": accuracy_trend, "avg_accuracy": avg_accuracy}
        )

        return new_er

    def adjust_temperature(self, task_type: str = "classification", force_adjust: bool = False) -> float:
        if self.strategy == AdjustmentStrategy.PERIODIC and not force_adjust:
            self._periodic_counter += 1
            if self._periodic_counter < self._periodic_interval:
                return self.parameters[ParameterType.TEMPERATURE]
            self._periodic_counter = 0

        current_temp = self.parameters[ParameterType.TEMPERATURE]
        avg_accuracy = self.performance_metric.get_avg_accuracy()
        loss_trend = self.performance_metric.get_loss_trend()

        base_temp = {
            "classification": 1.0,
            "regression": 0.5,
            "reinforcement": 1.5,
            "generation": 2.0,
            "few_shot": 0.8
        }.get(task_type, 1.0)

        temp_factor = 1.0
        if avg_accuracy > 0.9:
            temp_factor = 0.8
        elif avg_accuracy < 0.5:
            temp_factor = 1.3

        if loss_trend > 0.05:
            temp_factor *= 0.9
        elif loss_trend < -0.05:
            temp_factor *= 1.1

        new_temp = base_temp * temp_factor
        bounds = self.bounds[ParameterType.TEMPERATURE]
        new_temp = max(bounds[0], min(bounds[1], new_temp))

        self.parameters[ParameterType.TEMPERATURE] = new_temp

        self._record_adjustment(
            ParameterType.TEMPERATURE,
            current_temp,
            new_temp,
            {"task_type": task_type, "avg_accuracy": avg_accuracy, "loss_trend": loss_trend}
        )

        return new_temp

    def adjust_all(self, task_type: str = "classification", force_adjust: bool = False) -> Dict[str, float]:
        lr = self.adjust_learning_rate(force_adjust)
        er = self.adjust_exploration_rate(force_adjust)
        temp = self.adjust_temperature(task_type, force_adjust)

        return {
            "learning_rate": lr,
            "exploration_rate": er,
            "temperature": temp
        }

    def update_performance(self, loss: float, accuracy: float, reward: float = 0.0):
        self.performance_metric.update(loss, accuracy, reward)

    def set_task_context(self, task_complexity: float, environment_uncertainty: float):
        self.performance_metric.set_task_complexity(task_complexity)
        self.performance_metric.set_environment_uncertainty(environment_uncertainty)

    def _record_adjustment(self, param_type: ParameterType, old_value: float,
                          new_value: float, context: Dict[str, Any]):
        self.adjustment_history.append({
            "parameter_type": param_type.value,
            "old_value": old_value,
            "new_value": new_value,
            "delta": new_value - old_value,
            "context": context,
            "timestamp": np.random.randint(1000000)
        })

    def reset(self):
        self.parameters = self.default_parameters.copy()
        self.performance_metric = PerformanceMetric()
        self.adjustment_history.clear()
        self._periodic_counter = 0

    def get_parameter_summary(self) -> Dict[str, Any]:
        return {
            "learning_rate": {
                "current": self.parameters[ParameterType.LEARNING_RATE],
                "range": self.bounds[ParameterType.LEARNING_RATE],
                "default": self.default_parameters[ParameterType.LEARNING_RATE]
            },
            "exploration_rate": {
                "current": self.parameters[ParameterType.EXPLORATION_RATE],
                "range": self.bounds[ParameterType.EXPLORATION_RATE],
                "default": self.default_parameters[ParameterType.EXPLORATION_RATE]
            },
            "temperature": {
                "current": self.parameters[ParameterType.TEMPERATURE],
                "range": self.bounds[ParameterType.TEMPERATURE],
                "default": self.default_parameters[ParameterType.TEMPERATURE]
            },
            "adjustment_strategy": self.strategy.value,
            "adjustment_count": len(self.adjustment_history)
        }

    def get_adjustment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.adjustment_history)[-limit:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        return self.performance_metric.to_dict()