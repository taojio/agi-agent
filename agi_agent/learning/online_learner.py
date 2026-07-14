"""
learning/online_learner.py - 在线学习模块

支持增量学习、反馈闭环、概念漂移检测、自适应学习率。
与传统批处理学习不同，数据流式到达，模型持续更新。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class LearningMode(Enum):
    """学习模式"""
    SUPERVISED = "supervised"      # 有监督
    REINFORCEMENT = "reinforcement"  # 强化
    UNSUPERVISED = "unsupervised"  # 无监督
    SELF_SUPERVISED = "self_supervised"  # 自监督


class DriftType(Enum):
    """概念漂移类型"""
    NO_DRIFT = "no_drift"
    GRADUAL = "gradual"        # 渐变
    ABRUPT = "abrupt"          # 突变
    INCREMENTAL = "incremental"  # 增量
    RECURRING = "recurring"    # 反复


@dataclass
class LearningSample:
    """学习样本"""
    sample_id: str
    timestamp: float
    features: Dict[str, float]
    label: Optional[Any] = None
    reward: Optional[float] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningUpdate:
    """学习更新记录"""
    update_id: str
    timestamp: float
    mode: LearningMode
    samples_processed: int
    loss_before: float
    loss_after: float
    improvement: float
    learning_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_improvement(self) -> bool:
        return self.improvement > 0


@dataclass
class DriftDetection:
    """漂移检测结果"""
    drift_type: DriftType
    detected_at: float
    severity: float        # 0-1
    affected_features: List[str]
    confidence: float


class OnlineLearner:
    """在线学习器

    支持流式数据增量学习，自动检测概念漂移，动态调整学习率。

    Attributes:
        weights: 当前模型权重
        learning_rate: 当前学习率
        drift_detector: 概念漂移检测器
    """

    def __init__(self, feature_names: List[str],
                 mode: LearningMode = LearningMode.SUPERVISED,
                 learning_rate: float = 0.01,
                 max_history: int = 1000):
        self.feature_names = list(feature_names)
        self.mode = mode
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.max_history = max_history

        # 模型参数：线性权重 + 偏置
        self.weights = np.zeros(len(feature_names), dtype=np.float64)
        self.bias = 0.0

        # 学习历史
        self.sample_history: deque = deque(maxlen=max_history)
        self.update_history: deque = deque(maxlen=200)
        self.loss_history: deque = deque(maxlen=100)

        # 自适应参数
        self._gradient_momentum = np.zeros(len(feature_names))
        self._gradient_velocity = np.zeros(len(feature_names))
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._epsilon = 1e-8
        self._update_count = 0

        # 概念漂移检测
        self._baseline_loss: Optional[float] = None
        self._loss_window: deque = deque(maxlen=30)
        self._drift_history: deque = deque(maxlen=20)

        # 性能追踪
        self._correct_predictions = 0
        self._total_predictions = 0

        # 特征统计（用于归一化）
        self._feature_means = np.zeros(len(feature_names))
        self._feature_stds = np.ones(len(feature_names))
        self._feature_counts = np.zeros(len(feature_names))

    def _to_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """将特征字典转为向量"""
        vec = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            vec[i] = float(features.get(name, 0.0))
        return vec

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """特征归一化"""
        return (vec - self._feature_means) / (self._feature_stds + self._epsilon)

    def _update_feature_stats(self, vec: np.ndarray) -> None:
        """增量更新特征统计"""
        self._feature_counts += 1
        delta = vec - self._feature_means
        self._feature_means += delta / self._feature_counts
        delta2 = vec - self._feature_means
        # Welford 算法
        if not hasattr(self, "_m2"):
            self._m2 = np.zeros(len(vec))
        self._m2 += delta * delta2
        if (self._feature_counts > 1).all():
            variance = self._m2 / (self._feature_counts - 1)
            self._feature_stds = np.sqrt(np.maximum(variance, self._epsilon))

    def predict(self, features: Dict[str, float]) -> float:
        """预测"""
        vec = self._to_feature_vector(features)
        normalized = self._normalize(vec)
        return float(np.dot(self.weights, normalized) + self.bias)

    def update(self, sample: LearningSample) -> LearningUpdate:
        """单样本增量更新

        Args:
            sample: 学习样本

        Returns:
            LearningUpdate: 更新记录
        """
        self._update_count += 1
        vec = self._to_feature_vector(sample.features)
        self._update_feature_stats(vec)
        normalized = self._normalize(vec)

        # 前向计算
        prediction = float(np.dot(self.weights, normalized) + self.bias)

        # 计算损失与梯度
        if self.mode == LearningMode.SUPERVISED:
            if sample.label is None:
                raise ValueError("Supervised learning requires label")
            target = float(sample.label)
            loss = 0.5 * (prediction - target) ** 2
            gradient = (prediction - target) * normalized
            bias_gradient = prediction - target
        elif self.mode == LearningMode.REINFORCEMENT:
            if sample.reward is None:
                raise ValueError("Reinforcement learning requires reward")
            target = float(sample.reward)
            loss = -target * prediction if prediction != 0 else 0.0
            gradient = -target * normalized * 0.01
            bias_gradient = -target * 0.01
        else:
            # 无监督：自监督重建损失
            target = float(np.mean(normalized))
            loss = 0.5 * (prediction - target) ** 2
            gradient = (prediction - target) * normalized
            bias_gradient = prediction - target

        loss_before = float(loss)

        # 应用样本权重
        gradient *= sample.weight
        bias_gradient *= sample.weight

        # Adam 优化器
        self._gradient_momentum = self._beta1 * self._gradient_momentum + \
                                   (1 - self._beta1) * gradient
        self._gradient_velocity = self._beta2 * self._gradient_velocity + \
                                   (1 - self._beta2) * (gradient ** 2)

        m_hat = self._gradient_momentum / (1 - self._beta1 ** self._update_count)
        v_hat = self._gradient_velocity / (1 - self._beta2 ** self._update_count)

        self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self._epsilon)
        self.bias -= self.learning_rate * bias_gradient

        # 重新计算损失
        new_prediction = float(np.dot(self.weights, normalized) + self.bias)
        if self.mode == LearningMode.SUPERVISED:
            loss_after = 0.5 * (new_prediction - target) ** 2
        elif self.mode == LearningMode.REINFORCEMENT:
            loss_after = -target * new_prediction if new_prediction != 0 else 0.0
        else:
            loss_after = 0.5 * (new_prediction - target) ** 2

        improvement = float(loss_before - loss_after)

        # 记录历史
        self.sample_history.append(sample)
        self.loss_history.append(loss_after)

        # 预测准确性
        if self.mode == LearningMode.SUPERVISED and sample.label is not None:
            self._total_predictions += 1
            if abs(new_prediction - float(sample.label)) < 0.5:
                self._correct_predictions += 1

        # 概念漂移检测
        self._detect_drift(loss_after)

        # 自适应学习率
        self._adjust_learning_rate()

        update = LearningUpdate(
            update_id=f"upd_{self._update_count}",
            timestamp=time.time(),
            mode=self.mode,
            samples_processed=1,
            loss_before=loss_before,
            loss_after=float(loss_after),
            improvement=improvement,
            learning_rate=self.learning_rate,
            metadata={"prediction": new_prediction, "target": target},
        )
        self.update_history.append(update)
        return update

    def batch_update(self, samples: List[LearningSample]) -> LearningUpdate:
        """批量更新（仍在样本级别应用，但聚合统计）"""
        if not samples:
            raise ValueError("No samples to update")

        total_loss_before = 0.0
        total_loss_after = 0.0
        for sample in samples:
            update = self.update(sample)
            total_loss_before += update.loss_before
            total_loss_after += update.loss_after

        return LearningUpdate(
            update_id=f"batch_{self._update_count}",
            timestamp=time.time(),
            mode=self.mode,
            samples_processed=len(samples),
            loss_before=total_loss_before / len(samples),
            loss_after=total_loss_after / len(samples),
            improvement=(total_loss_before - total_loss_after) / len(samples),
            learning_rate=self.learning_rate,
            metadata={"batch_size": len(samples)},
        )

    def _detect_drift(self, current_loss: float) -> Optional[DriftDetection]:
        """概念漂移检测"""
        self._loss_window.append(current_loss)

        if self._baseline_loss is None and len(self._loss_window) >= 10:
            self._baseline_loss = float(np.mean(list(self._loss_window)[:10]))
            return None

        if self._baseline_loss is None or len(self._loss_window) < 10:
            return None

        recent_avg = float(np.mean(list(self._loss_window)[-10:]))
        baseline = self._baseline_loss

        # 漂移阈值
        if recent_avg > baseline * 1.5:
            # 检测漂移类型
            window_list = list(self._loss_window)
            recent_half = window_list[-len(window_list) // 2:]
            old_half = window_list[:len(window_list) // 2]

            if len(old_half) < 5 or len(recent_half) < 5:
                return None

            recent_mean = float(np.mean(recent_half))
            old_mean = float(np.mean(old_half))

            if abs(recent_mean - old_mean) / max(old_mean, 1e-6) > 0.5:
                drift_type = DriftType.ABRUPT
            elif recent_mean > old_mean * 1.2:
                drift_type = DriftType.GRADUAL
            else:
                drift_type = DriftType.INCREMENTAL

            severity = min(1.0, (recent_avg - baseline) / max(baseline, 1e-6))

            detection = DriftDetection(
                drift_type=drift_type,
                detected_at=time.time(),
                severity=severity,
                affected_features=self._identify_affected_features(),
                confidence=min(1.0, len(self._loss_window) / 30.0),
            )
            self._drift_history.append(detection)

            # 漂移后重置基线
            self._baseline_loss = recent_avg
            return detection

        return None

    def _identify_affected_features(self) -> List[str]:
        """识别受影响的特征（梯度幅度大的）"""
        if not hasattr(self, "_gradient_momentum") or \
           not np.any(self._gradient_momentum):
            return []
        magnitudes = np.abs(self._gradient_momentum)
        threshold = float(np.mean(magnitudes) + np.std(magnitudes))
        affected = [self.feature_names[i]
                    for i in range(len(magnitudes))
                    if magnitudes[i] > threshold]
        return affected[:5]

    def _adjust_learning_rate(self) -> None:
        """自适应学习率调整"""
        if len(self.loss_history) < 10:
            return

        recent_losses = list(self.loss_history)[-10:]
        recent_mean = float(np.mean(recent_losses))
        recent_std = float(np.std(recent_losses))

        # 损失下降则增大学习率，上升则减小
        if len(self.loss_history) >= 20:
            older = list(self.loss_history)[-20:-10]
            older_mean = float(np.mean(older))
            if recent_mean < older_mean * 0.95:
                # 改善中，可适度增大
                self.learning_rate = min(self.base_learning_rate * 5,
                                          self.learning_rate * 1.05)
            elif recent_mean > older_mean * 1.05:
                # 退化中，减小学习率
                self.learning_rate = max(self.base_learning_rate * 0.1,
                                          self.learning_rate * 0.95)

        # 损失震荡时减小学习率
        if recent_std > recent_mean * 0.5 and recent_mean > 0:
            self.learning_rate = max(self.base_learning_rate * 0.1,
                                      self.learning_rate * 0.9)

    def get_accuracy(self) -> float:
        """获取预测准确率"""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return {name: float(self.weights[i])
                for i, name in enumerate(self.feature_names)}

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """获取漂移检测历史"""
        return [
            {
                "drift_type": d.drift_type.value,
                "detected_at": d.detected_at,
                "severity": d.severity,
                "affected_features": d.affected_features,
                "confidence": d.confidence,
            }
            for d in self._drift_history
        ]

    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        recent_losses = list(self.loss_history)[-20:]
        return {
            "mode": self.mode.value,
            "total_samples": len(self.sample_history),
            "total_updates": self._update_count,
            "current_learning_rate": float(self.learning_rate),
            "base_learning_rate": float(self.base_learning_rate),
            "recent_avg_loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
            "recent_loss_std": float(np.std(recent_losses)) if recent_losses else 0.0,
            "accuracy": self.get_accuracy(),
            "drift_count": len(self._drift_history),
            "feature_count": len(self.feature_names),
        }

    def reset(self) -> None:
        """重置学习器（保留特征配置）"""
        self.weights = np.zeros(len(self.feature_names))
        self.bias = 0.0
        self._gradient_momentum = np.zeros(len(self.feature_names))
        self._gradient_velocity = np.zeros(len(self.feature_names))
        self._update_count = 0
        self._correct_predictions = 0
        self._total_predictions = 0
        self._baseline_loss = None
        self.learning_rate = self.base_learning_rate
        self.sample_history.clear()
        self.update_history.clear()
        self.loss_history.clear()
        self._loss_window.clear()
        self._drift_history.clear()
