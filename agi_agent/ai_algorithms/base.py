"""
ai_algorithms/base.py - AI 算法组件基类

所有 AI 算法组件的抽象基类，定义统一接口规范。
符合 UPG-015 规格的 AIAlgorithmComponent 接口。
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AlgorithmStatus(Enum):
    """算法状态"""
    IDLE = "idle"
    TRAINING = "training"
    TRAINED = "trained"
    PREDICTING = "predicting"
    ERROR = "error"


@dataclass
class AlgorithmMetrics:
    """算法性能指标"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    sample_count: int = 0
    feature_count: int = 0
    custom: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "training_time": float(self.training_time),
            "prediction_time": float(self.prediction_time),
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "custom": dict(self.custom),
        }


class AIAlgorithmComponent(ABC):
    """AI 算法组件基类

    所有 AI 算法组件必须继承此类，实现统一接口。

    Attributes:
        name: 组件名称
        status: 当前状态
        params: 算法参数
        metrics: 性能指标
        is_trained: 是否已训练
    """

    def __init__(self, name: str, **params):
        self.name = name
        self.status = AlgorithmStatus.IDLE
        self.params: Dict[str, Any] = dict(params)
        self.metrics = AlgorithmMetrics()
        self.is_trained = False
        self._train_count = 0
        self._predict_count = 0
        self._created_at = time.time()

    @property
    @abstractmethod
    def component_type(self) -> str:
        """组件类型标识"""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AIAlgorithmComponent":
        """训练模型

        Args:
            X: 训练数据，shape=(n_samples, n_features) 或 (n_samples,) 时序
            y: 标签（监督学习时）

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> AlgorithmMetrics:
        """评估模型性能（默认实现，子类可覆盖）"""
        if not self.is_trained:
            raise RuntimeError(f"{self.name} is not trained")
        predictions = self.predict(X)
        return self._compute_metrics(predictions, y)

    def _compute_metrics(self, predictions: np.ndarray,
                          actual: np.ndarray) -> AlgorithmMetrics:
        """计算性能指标"""
        metrics = AlgorithmMetrics()
        predictions = np.asarray(predictions, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)

        # 回归指标
        if predictions.shape == actual.shape:
            errors = predictions - actual
            metrics.rmse = float(np.sqrt(np.mean(errors ** 2)))
            metrics.mae = float(np.mean(np.abs(errors)))
            # R²
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
            metrics.accuracy = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics.sample_count = len(actual)
        metrics.prediction_time = 0.0
        return metrics

    def set_params(self, **params) -> "AIAlgorithmComponent":
        """设置算法参数"""
        self.params.update(params)
        return self

    def get_params(self) -> Dict[str, Any]:
        """获取算法参数"""
        return dict(self.params)

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "name": self.name,
            "component_type": self.component_type,
            "status": self.status.value,
            "is_trained": self.is_trained,
            "train_count": self._train_count,
            "predict_count": self._predict_count,
            "params": self.params,
            "metrics": self.metrics.to_dict(),
            "created_at": self._created_at,
        }

    def reset(self) -> None:
        """重置组件"""
        self.status = AlgorithmStatus.IDLE
        self.is_trained = False
        self.metrics = AlgorithmMetrics()
        self._train_count = 0
        self._predict_count = 0

    def _start_training(self) -> float:
        self.status = AlgorithmStatus.TRAINING
        return time.time()

    def _end_training(self, start_time: float) -> None:
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True
        self.status = AlgorithmStatus.TRAINED
        self._train_count += 1

    def _start_prediction(self) -> float:
        self.status = AlgorithmStatus.PREDICTING
        return time.time()

    def _end_prediction(self, start_time: float) -> None:
        self.metrics.prediction_time = time.time() - start_time
        self._predict_count += 1
        self.status = AlgorithmStatus.TRAINED

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"type='{self.component_type}', trained={self.is_trained})")
