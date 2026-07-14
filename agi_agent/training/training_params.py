import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class LRScheduleType(Enum):
    FIXED = "fixed"
    META_LEARNING = "meta_learning"
    ADAPTIVE = "adaptive"
    COSINE_ANNEALING = "cosine_annealing"
    WARMUP = "warmup"
    STEP_DECAY = "step_decay"


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


@dataclass
class ParamSpec:
    name: str
    default_value: float
    min_value: float
    max_value: float
    description: str
    trainable: bool = True
    schedule_type: Optional[LRScheduleType] = None
    current_value: float = 0.0

    def __post_init__(self):
        if self.current_value == 0.0:
            self.current_value = self.default_value

    def clip(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


@dataclass
class TrainingParams:
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    initial_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-1
    learning_rate_pool: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3])

    weight_decay: float = 0.01
    l2_regularization: float = 1e-4
    dropout_rate: float = 0.1
    gradient_clip_threshold: float = 1.0

    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 100

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 0.4,
        "kl_divergence": 0.3,
        "confidence": 0.1,
        "sparsity": 0.1,
        "consistency": 0.05,
        "safety": 0.05
    })

    def get_param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec("initial_learning_rate", self.initial_learning_rate, 1e-6, 1e-1,
                      "初始学习率", schedule_type=LRScheduleType.META_LEARNING),
            ParamSpec("weight_decay", self.weight_decay, 1e-6, 0.1,
                      "权重衰减系数"),
            ParamSpec("l2_regularization", self.l2_regularization, 1e-6, 1e-2,
                      "L2正则化系数"),
            ParamSpec("dropout_rate", self.dropout_rate, 0.0, 0.5,
                      "Dropout随机失活率"),
            ParamSpec("gradient_clip_threshold", self.gradient_clip_threshold, 0.1, 10.0,
                      "梯度裁剪阈值"),
            ParamSpec("beta1", self.beta1, 0.8, 0.999,
                      "Adam beta1参数"),
            ParamSpec("beta2", self.beta2, 0.9, 0.9999,
                      "Adam beta2参数"),
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type.value,
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "max_learning_rate": self.max_learning_rate,
            "learning_rate_pool": self.learning_rate_pool,
            "weight_decay": self.weight_decay,
            "l2_regularization": self.l2_regularization,
            "dropout_rate": self.dropout_rate,
            "gradient_clip_threshold": self.gradient_clip_threshold,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "loss_weights": self.loss_weights.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingParams':
        params = cls()
        for key, value in data.items():
            if hasattr(params, key):
                if key == "optimizer_type":
                    setattr(params, key, OptimizerType(value))
                else:
                    setattr(params, key, value)
        return params


class LearningRateScheduler:
    def __init__(self, params: TrainingParams):
        self.params = params
        self.current_lr = params.initial_learning_rate
        self.global_step = 0
        self.warmup_steps = params.warmup_steps

        self.schedule_type = LRScheduleType.META_LEARNING
        self.lr_history: deque = deque(maxlen=1000)

        self._arm_rewards: Dict[float, float] = {lr: 0.0 for lr in params.learning_rate_pool}
        self._arm_counts: Dict[float, int] = {lr: 0 for lr in params.learning_rate_pool}
        self.exploration_rate = 0.2

        self.cosine_period = 1000
        self.cosine_restart_steps = 0

    def step(self, step: int, metrics: Dict[str, float] = None) -> float:
        self.global_step = step

        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            self.current_lr = self.params.initial_learning_rate * warmup_factor
        else:
            if self.schedule_type == LRScheduleType.FIXED:
                self.current_lr = self.params.initial_learning_rate

            elif self.schedule_type == LRScheduleType.META_LEARNING:
                self.current_lr = self._meta_learning_step(metrics)

            elif self.schedule_type == LRScheduleType.ADAPTIVE:
                self.current_lr = self._adaptive_step(metrics)

            elif self.schedule_type == LRScheduleType.COSINE_ANNEALING:
                self.current_lr = self._cosine_annealing_step(step)

            elif self.schedule_type == LRScheduleType.STEP_DECAY:
                self.current_lr = self._step_decay_step(step)

        self.current_lr = max(self.params.min_learning_rate,
                              min(self.params.max_learning_rate, self.current_lr))
        self.lr_history.append({"step": step, "lr": self.current_lr})

        return self.current_lr

    def _meta_learning_step(self, metrics: Dict[str, float]) -> float:
        if metrics is None:
            return self.current_lr

        free_energy = metrics.get("free_energy", 0.5)
        convergence_speed = metrics.get("convergence_speed", 0.0)

        reward = convergence_speed * (1.0 / (free_energy + 1e-8))

        if np.random.random() < self.exploration_rate:
            best_idx = np.random.randint(len(self.params.learning_rate_pool))
        else:
            rewards = [self._arm_rewards[lr] / max(1, self._arm_counts[lr])
                       for lr in self.params.learning_rate_pool]
            best_idx = np.argmax(rewards)

        best_lr = self.params.learning_rate_pool[best_idx]

        self._arm_rewards[best_lr] += reward
        self._arm_counts[best_lr] += 1

        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

        return best_lr

    def _adaptive_step(self, metrics: Dict[str, float]) -> float:
        if metrics is None:
            return self.current_lr

        free_energy = metrics.get("free_energy", 0.5)
        fe_trend = metrics.get("free_energy_trend", 0.0)

        lr = self.current_lr

        if fe_trend > 0.01:
            lr = lr * 0.95
        elif fe_trend < -0.01:
            lr = lr * 1.02
        elif free_energy > 0.5:
            lr = lr * 1.01
        else:
            lr = lr * 0.995

        return lr

    def _cosine_annealing_step(self, step: int) -> float:
        steps_since_restart = step - self.cosine_restart_steps
        progress = min(1.0, steps_since_restart / self.cosine_period)

        lr = (self.params.min_learning_rate +
              0.5 * (self.params.max_learning_rate - self.params.min_learning_rate) *
              (1 + math.cos(math.pi * progress)))

        if steps_since_restart >= self.cosine_period:
            self.cosine_restart_steps = step
            self.cosine_period = int(self.cosine_period * 1.2)

        return lr

    def _step_decay_step(self, step: int, decay_rate: float = 0.5,
                         decay_steps: int = 10000) -> float:
        decay_count = step // decay_steps
        lr = self.params.initial_learning_rate * (decay_rate ** decay_count)
        return lr

    def set_schedule_type(self, schedule_type: LRScheduleType):
        self.schedule_type = schedule_type

    def get_lr_stats(self) -> Dict[str, Any]:
        recent_lrs = [h["lr"] for h in self.lr_history]
        if not recent_lrs:
            recent_lrs = [self.current_lr]

        return {
            "current_lr": self.current_lr,
            "schedule_type": self.schedule_type.value,
            "global_step": self.global_step,
            "lr_mean": float(np.mean(recent_lrs)),
            "lr_std": float(np.std(recent_lrs)),
            "lr_min": float(np.min(recent_lrs)),
            "lr_max": float(np.max(recent_lrs)),
            "exploration_rate": self.exploration_rate,
            "arm_counts": {str(k): v for k, v in self._arm_counts.items()},
            "history_size": len(self.lr_history)
        }

    def reset(self):
        self.current_lr = self.params.initial_learning_rate
        self.global_step = 0
        self.lr_history.clear()
        self._arm_rewards = {lr: 0.0 for lr in self.params.learning_rate_pool}
        self._arm_counts = {lr: 0 for lr in self.params.learning_rate_pool}
        self.exploration_rate = 0.2
        self.cosine_restart_steps = 0
