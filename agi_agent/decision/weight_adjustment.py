"""
decision/weight_adjustment.py - 决策因素权重动态调整机制

实现基于机器学习的权重自动学习算法、权重调整的实时性与稳定性平衡机制、
权重调整的安全边界与异常检测机制
"""
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque


class LearningMode(Enum):
    ONLINE = "online"
    BATCH = "batch"
    HYBRID = "hybrid"


class AdjustmentMode(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"


class BoundaryType(Enum):
    HARD = "hard"
    SOFT = "soft"
    ADAPTIVE = "adaptive"


@dataclass
class WeightBound:
    weight_name: str
    min_value: float = 0.0
    max_value: float = 1.0
    boundary_type: BoundaryType = BoundaryType.SOFT
    violation_penalty: float = 0.1

    def clamp(self, value: float) -> float:
        if self.boundary_type == BoundaryType.HARD:
            return max(self.min_value, min(self.max_value, value))
        elif self.boundary_type == BoundaryType.SOFT:
            if value < self.min_value:
                return self.min_value + (self.min_value - value) * (1 - self.violation_penalty)
            elif value > self.max_value:
                return self.max_value - (value - self.max_value) * (1 - self.violation_penalty)
            return value
        else:
            margin = 0.1 * (self.max_value - self.min_value)
            return max(self.min_value - margin, min(self.max_value + margin, value))

    def is_violated(self, value: float) -> bool:
        return value < self.min_value or value > self.max_value


@dataclass
class WeightUpdate:
    weight_name: str
    old_value: float
    new_value: float
    delta: float
    reason: str = ""
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_name": self.weight_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class LearningSample:
    weights: Dict[str, float]
    inputs: Dict[str, Any]
    output: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "inputs": self.inputs,
            "output": self.output,
            "timestamp": self.timestamp,
        }


@dataclass
class StabilityConstraint:
    max_single_adjustment: float = 0.1
    max_total_adjustment: float = 0.3
    adjustment_cooldown: float = 0.0
    momentum_factor: float = 0.9
    smoothing_factor: float = 0.1


class WeightAdjustmentSystem:
    def __init__(self, learning_mode: LearningMode = LearningMode.HYBRID,
                 adjustment_mode: AdjustmentMode = AdjustmentMode.GRADIENT_DESCENT):
        self._weights: Dict[str, float] = {}
        self._weight_bounds: Dict[str, WeightBound] = {}
        self._learning_samples: deque = deque(maxlen=5000)
        self._weight_history: Dict[str, deque] = {}
        self._learning_mode = learning_mode
        self._adjustment_mode = adjustment_mode
        self._stability_constraint = StabilityConstraint()
        self._last_adjustment_time: Dict[str, float] = {}
        self._gradient_cache: Dict[str, float] = {}
        self._learning_rate: Dict[str, float] = {}
        self._reward_history: deque = deque(maxlen=200)
        self._anomaly_detections: deque = deque(maxlen=100)
        self._stats = {
            "total_adjustments": 0,
            "successful_adjustments": 0,
            "boundary_violations": 0,
            "anomaly_detections": 0,
            "avg_adjustment_magnitude": 0.0,
            "avg_confidence": 0.5,
        }

    def register_weight(self, name: str, initial_value: float = 0.5,
                        min_value: float = 0.0, max_value: float = 1.0,
                        boundary_type: BoundaryType = BoundaryType.SOFT):
        self._weights[name] = initial_value
        self._weight_bounds[name] = WeightBound(
            weight_name=name,
            min_value=min_value,
            max_value=max_value,
            boundary_type=boundary_type,
        )
        self._weight_history[name] = deque(maxlen=100)
        self._weight_history[name].append(initial_value)
        self._learning_rate[name] = 0.01
        self._last_adjustment_time[name] = 0.0

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def set_weight(self, name: str, value: float) -> bool:
        if name not in self._weights:
            return False

        bound = self._weight_bounds[name]
        clamped_value = bound.clamp(value)

        if bound.is_violated(value):
            self._stats["boundary_violations"] += 1

        self._weights[name] = clamped_value
        self._weight_history[name].append(clamped_value)

        return True

    def add_learning_sample(self, weights: Dict[str, float], inputs: Dict[str, Any], output: float):
        sample = LearningSample(weights=weights, inputs=inputs, output=output)
        self._learning_samples.append(sample)
        self._reward_history.append(output)

    def adjust_weights(self, feedback: float, context: Dict[str, Any] = None) -> List[WeightUpdate]:
        updates = []

        if self._adjustment_mode == AdjustmentMode.GRADIENT_DESCENT:
            updates = self._gradient_descent_adjustment(feedback, context)
        elif self._adjustment_mode == AdjustmentMode.REINFORCEMENT_LEARNING:
            updates = self._reinforcement_learning_adjustment(feedback, context)
        elif self._adjustment_mode == AdjustmentMode.GENETIC_ALGORITHM:
            updates = self._genetic_algorithm_adjustment(feedback, context)
        elif self._adjustment_mode == AdjustmentMode.BAYESIAN_OPTIMIZATION:
            updates = self._bayesian_optimization_adjustment(feedback, context)

        for update in updates:
            self._apply_weight_update(update)

        return updates

    def _gradient_descent_adjustment(self, feedback: float, context: Dict[str, Any] = None) -> List[WeightUpdate]:
        updates = []
        now = time.time()

        for name, weight in self._weights.items():
            if now - self._last_adjustment_time[name] < self._stability_constraint.adjustment_cooldown:
                continue

            gradient = self._estimate_gradient(name, feedback)
            momentum = self._stability_constraint.momentum_factor * self._gradient_cache.get(name, 0)
            effective_gradient = momentum + (1 - self._stability_constraint.momentum_factor) * gradient

            lr = self._learning_rate[name]
            delta = lr * effective_gradient

            delta = self._apply_stability_constraints(name, delta)

            new_value = weight + delta
            bound = self._weight_bounds[name]
            new_value = bound.clamp(new_value)

            if abs(new_value - weight) > 1e-6:
                updates.append(WeightUpdate(
                    weight_name=name,
                    old_value=weight,
                    new_value=new_value,
                    delta=new_value - weight,
                    reason=f"Gradient descent adjustment (gradient={gradient:.4f})",
                    confidence=min(1.0, 0.5 + abs(gradient) * 2),
                ))

            self._gradient_cache[name] = effective_gradient

        return updates

    def _estimate_gradient(self, weight_name: str, feedback: float) -> float:
        samples = list(self._learning_samples)[-50:]
        if len(samples) < 10:
            return feedback * 0.1

        weight_values = []
        outputs = []

        for sample in samples:
            if weight_name in sample.weights:
                weight_values.append(sample.weights[weight_name])
                outputs.append(sample.output)

        if len(weight_values) < 5:
            return feedback * 0.1

        weight_np = np.array(weight_values)
        outputs_np = np.array(outputs)

        if np.std(weight_np) < 1e-10:
            return feedback * 0.1

        gradient = np.cov(weight_np, outputs_np)[0, 1] / np.var(weight_np)
        return float(gradient * feedback)

    def _reinforcement_learning_adjustment(self, feedback: float, context: Dict[str, Any] = None) -> List[WeightUpdate]:
        updates = []
        now = time.time()

        reward = feedback
        exploration_rate = max(0.01, 0.3 - len(self._reward_history) * 0.001)

        for name, weight in self._weights.items():
            if now - self._last_adjustment_time[name] < self._stability_constraint.adjustment_cooldown:
                continue

            delta = 0.0
            if reward > 0:
                delta = self._learning_rate[name] * reward * (0.5 + np.random.rand() * exploration_rate)
            else:
                delta = -self._learning_rate[name] * abs(reward) * (0.5 + np.random.rand() * exploration_rate)

            delta = self._apply_stability_constraints(name, delta)

            new_value = weight + delta
            bound = self._weight_bounds[name]
            new_value = bound.clamp(new_value)

            if abs(new_value - weight) > 1e-6:
                updates.append(WeightUpdate(
                    weight_name=name,
                    old_value=weight,
                    new_value=new_value,
                    delta=new_value - weight,
                    reason=f"RL adjustment (reward={reward:.4f})",
                    confidence=min(1.0, 0.5 + abs(reward)),
                ))

        return updates

    def _genetic_algorithm_adjustment(self, feedback: float, context: Dict[str, Any] = None) -> List[WeightUpdate]:
        updates = []

        if len(self._learning_samples) < 20:
            return updates

        samples = list(self._learning_samples)[-50:]
        samples.sort(key=lambda s: s.output, reverse=True)

        elite_samples = samples[:int(len(samples) * 0.2)]
        avg_weights = {}
        for name in self._weights.keys():
            values = [s.weights.get(name, 0.5) for s in elite_samples]
            avg_weights[name] = np.mean(values)

        mutation_rate = 0.1
        for name, weight in self._weights.items():
            target_weight = avg_weights.get(name, weight)
            mutation = (np.random.rand() - 0.5) * 2 * mutation_rate * (1 - abs(feedback))
            new_value = target_weight + mutation

            bound = self._weight_bounds[name]
            new_value = bound.clamp(new_value)

            delta = new_value - weight
            delta = self._apply_stability_constraints(name, delta)
            new_value = weight + delta

            if abs(new_value - weight) > 1e-6:
                updates.append(WeightUpdate(
                    weight_name=name,
                    old_value=weight,
                    new_value=new_value,
                    delta=new_value - weight,
                    reason=f"GA adjustment (elite avg={target_weight:.4f})",
                    confidence=min(1.0, 0.5 + feedback * 0.5),
                ))

        return updates

    def _bayesian_optimization_adjustment(self, feedback: float, context: Dict[str, Any] = None) -> List[WeightUpdate]:
        updates = []

        if len(self._learning_samples) < 10:
            return updates

        for name, weight in self._weights.items():
            samples = [s for s in self._learning_samples if name in s.weights]
            if len(samples) < 5:
                continue

            values = np.array([s.weights[name] for s in samples])
            outputs = np.array([s.output for s in samples])

            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val < 1e-10:
                continue

            improvement_mask = outputs > np.mean(outputs)
            if np.sum(improvement_mask) == 0:
                continue

            optimal_region = values[improvement_mask]
            target_weight = np.mean(optimal_region)
            uncertainty = np.std(optimal_region)

            acquisition = (target_weight - weight) + 0.1 * uncertainty

            delta = self._learning_rate[name] * acquisition
            delta = self._apply_stability_constraints(name, delta)

            new_value = weight + delta
            bound = self._weight_bounds[name]
            new_value = bound.clamp(new_value)

            if abs(new_value - weight) > 1e-6:
                updates.append(WeightUpdate(
                    weight_name=name,
                    old_value=weight,
                    new_value=new_value,
                    delta=new_value - weight,
                    reason=f"Bayesian optimization (acquisition={acquisition:.4f})",
                    confidence=min(1.0, 0.5 + (1 - uncertainty)),
                ))

        return updates

    def _apply_stability_constraints(self, name: str, delta: float) -> float:
        max_single = self._stability_constraint.max_single_adjustment
        delta = max(-max_single, min(max_single, delta))

        current_total_delta = sum(
            abs(self._weights[n] - self._weight_history[n][0])
            for n in self._weights
        )
        if current_total_delta + abs(delta) > self._stability_constraint.max_total_adjustment:
            remaining = self._stability_constraint.max_total_adjustment - current_total_delta
            delta = np.sign(delta) * min(abs(delta), max(0, remaining))

        return delta

    def _apply_weight_update(self, update: WeightUpdate):
        self.set_weight(update.weight_name, update.new_value)
        self._last_adjustment_time[update.weight_name] = time.time()
        self._stats["total_adjustments"] += 1
        self._stats["successful_adjustments"] += 1
        self._stats["avg_adjustment_magnitude"] = (
            self._stats["avg_adjustment_magnitude"] *
            (self._stats["total_adjustments"] - 1) + abs(update.delta)
        ) / self._stats["total_adjustments"]
        self._stats["avg_confidence"] = (
            self._stats["avg_confidence"] *
            (self._stats["total_adjustments"] - 1) + update.confidence
        ) / self._stats["total_adjustments"]

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        anomalies = []

        for name, history in self._weight_history.items():
            if len(history) < 20:
                continue

            recent = np.array(history[-10:])
            earlier = np.array(history[-20:-10])

            recent_mean = np.mean(recent)
            earlier_mean = np.mean(earlier)
            recent_std = np.std(recent)
            earlier_std = np.std(earlier)

            if earlier_std > 1e-10 and recent_std / earlier_std > 3.0:
                anomalies.append({
                    "weight_name": name,
                    "type": "variance_spike",
                    "severity": "high",
                    "description": f"Variance increased by {recent_std/earlier_std:.1f}x",
                    "recent_std": recent_std,
                    "earlier_std": earlier_std,
                    "timestamp": time.time(),
                })
                self._stats["anomaly_detections"] += 1

            if abs(recent_mean - earlier_mean) > 0.3 * (self._weight_bounds[name].max_value - self._weight_bounds[name].min_value):
                anomalies.append({
                    "weight_name": name,
                    "type": "drift",
                    "severity": "medium",
                    "description": f"Weight drifted by {abs(recent_mean - earlier_mean):.2f}",
                    "recent_mean": recent_mean,
                    "earlier_mean": earlier_mean,
                    "timestamp": time.time(),
                })

        self._anomaly_detections.extend(anomalies)
        return anomalies

    def enforce_bounds(self) -> List[WeightUpdate]:
        updates = []

        for name, weight in self._weights.items():
            bound = self._weight_bounds[name]
            if bound.is_violated(weight):
                new_value = bound.clamp(weight)
                updates.append(WeightUpdate(
                    weight_name=name,
                    old_value=weight,
                    new_value=new_value,
                    delta=new_value - weight,
                    reason="Boundary enforcement",
                    confidence=0.9,
                ))
                self._apply_weight_update(updates[-1])

        return updates

    def smooth_weights(self):
        for name, history in self._weight_history.items():
            if len(history) < 3:
                continue

            smoothed = np.convolve(list(history), [0.2, 0.3, 0.5], mode='valid')
            if len(smoothed) > 0:
                self._weights[name] = smoothed[-1]
                self._weight_history[name].append(smoothed[-1])

    def set_stability_constraint(self, constraint: StabilityConstraint):
        self._stability_constraint = constraint

    def set_learning_rate(self, weight_name: str, rate: float):
        if weight_name in self._learning_rate:
            self._learning_rate[weight_name] = max(0.0001, min(0.1, rate))

    def get_weight_history(self, weight_name: str, limit: int = 50) -> List[Dict[str, float]]:
        history = self._weight_history.get(weight_name, [])
        result = []
        for i, value in enumerate(list(history)[-limit:]):
            result.append({"index": i, "value": value})
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "weight_count": len(self._weights),
            "learning_mode": self._learning_mode.value,
            "adjustment_mode": self._adjustment_mode.value,
            "sample_count": len(self._learning_samples),
            "anomaly_count": len(self._anomaly_detections),
        }

    def get_boundary_info(self) -> Dict[str, Dict[str, float]]:
        info = {}
        for name, bound in self._weight_bounds.items():
            info[name] = {
                "current": self._weights[name],
                "min": bound.min_value,
                "max": bound.max_value,
                "type": bound.boundary_type.value,
            }
        return info