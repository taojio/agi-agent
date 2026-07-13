"""
ai_algorithms/anomaly_detection.py - 异常检测组件

支持多种异常检测算法：
- 孤立森林 (Isolation Forest)
- Z-score 统计检测
- IQR (四分位距) 检测
- AutoEncoder 重构异常检测（简化实现）
- DBSCAN 密度异常检测
- 组合检测器
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


class AnomalyType(Enum):
    """异常类型"""
    POINT = "point"           # 点异常
    CONTEXTUAL = "contextual" # 上下文异常
    COLLECTIVE = "collective" # 集体异常


class AnomalySeverity(Enum):
    """异常严重度"""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AnomalyResult:
    """异常检测结果"""
    is_anomaly: bool
    anomaly_score: float          # 异常分数 (0-1, 越高越异常)
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    detected_by: str              # 检测器名称
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": float(self.anomaly_score),
            "severity": self.severity.name,
            "anomaly_type": self.anomaly_type.value,
            "detected_by": self.detected_by,
            "details": self.details,
        }


@dataclass
class AnomalyReport:
    """异常检测报告"""
    total_samples: int
    anomaly_count: int
    anomaly_rate: float
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": float(self.anomaly_rate),
            "anomalies": self.anomalies,
            "severity_distribution": self.severity_distribution,
            "method": self.method,
        }


class ZScoreDetector(AIAlgorithmComponent):
    """Z-score 统计异常检测器

    基于均值和标准差的异常检测。
    """

    def __init__(self, name: str = "zscore", threshold: float = 3.0):
        super().__init__(name, threshold=threshold)
        self.threshold = threshold
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "anomaly_detection"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ZScoreDetector":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)
            # 避免除零
            self._std = np.where(self._std == 0, 1e-8, self._std)
            self.metrics.sample_count = len(data)
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回异常标签 (1=异常, 0=正常)"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            z_scores = np.abs((data - self._mean) / self._std)
            anomalies = (z_scores > self.threshold).any(axis=1).astype(int)
            self._end_prediction(start)
            return anomalies
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """返回异常分数 (0-1)"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        z_scores = np.abs((data - self._mean) / self._std)
        max_scores = np.max(z_scores, axis=1)
        # 归一化到 0-1
        return np.clip(max_scores / (self.threshold * 2), 0, 1)

    def detect(self, X: np.ndarray) -> List[AnomalyResult]:
        """检测并返回详细结果"""
        scores = self.score_samples(X)
        labels = self.predict(X)
        results = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            severity = self._score_to_severity(score)
            results.append(AnomalyResult(
                is_anomaly=bool(label),
                anomaly_score=float(score),
                severity=severity,
                anomaly_type=AnomalyType.POINT,
                detected_by=self.name,
                details={"z_score": float(score * self.threshold * 2)},
            ))
        return results

    @staticmethod
    def _score_to_severity(score: float) -> AnomalySeverity:
        if score < 0.5:
            return AnomalySeverity.NORMAL
        elif score < 0.65:
            return AnomalySeverity.LOW
        elif score < 0.8:
            return AnomalySeverity.MEDIUM
        elif score < 0.95:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


class IQRDetector(AIAlgorithmComponent):
    """IQR (四分位距) 异常检测器"""

    def __init__(self, name: str = "iqr", multiplier: float = 1.5):
        super().__init__(name, multiplier=multiplier)
        self.multiplier = multiplier
        self._q1: Optional[np.ndarray] = None
        self._q3: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None
        self._lower: Optional[np.ndarray] = None
        self._upper: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "anomaly_detection"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IQRDetector":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self._q1 = np.percentile(data, 25, axis=0)
            self._q3 = np.percentile(data, 75, axis=0)
            self._iqr = self._q3 - self._q1
            self._lower = self._q1 - self.multiplier * self._iqr
            self._upper = self._q3 + self.multiplier * self._iqr
            self.metrics.sample_count = len(data)
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            lower_mask = data < self._lower
            upper_mask = data > self._upper
            anomalies = (lower_mask | upper_mask).any(axis=1).astype(int)
            self._end_prediction(start)
            return anomalies
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # 计算超出 IQR 范围的程度
        lower_dist = np.maximum(0, self._lower - data) / (self._iqr + 1e-8)
        upper_dist = np.maximum(0, data - self._upper) / (self._iqr + 1e-8)
        total_dist = np.max(lower_dist + upper_dist, axis=1)
        return np.clip(total_dist / 3.0, 0, 1)

    def detect(self, X: np.ndarray) -> List[AnomalyResult]:
        scores = self.score_samples(X)
        labels = self.predict(X)
        return [
            AnomalyResult(
                is_anomaly=bool(label),
                anomaly_score=float(score),
                severity=ZScoreDetector._score_to_severity(score),
                anomaly_type=AnomalyType.POINT,
                detected_by=self.name,
            )
            for label, score in zip(labels, scores)
        ]


class IsolationForestDetector(AIAlgorithmComponent):
    """孤立森林异常检测器

    通过随机划分隔离数据点，异常点更容易被隔离（路径更短）。

    参数:
        n_trees: 树数量
        max_samples: 每棵树的样本数
        max_depth: 最大深度
    """

    def __init__(self, name: str = "isolation_forest",
                 n_trees: int = 100, max_samples: int = 256,
                 max_depth: int = 10):
        super().__init__(name, n_trees=n_trees, max_samples=max_samples,
                         max_depth=max_depth)
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_depth = max_depth
        self._trees: List[Dict] = []

    @property
    def component_type(self) -> str:
        return "anomaly_detection"

    def _build_tree(self, data: np.ndarray, depth: int = 0) -> Dict:
        """递归构建孤立树"""
        n_samples, n_features = data.shape
        if depth >= self.max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # 随机选择特征和分割点
        feature_idx = np.random.randint(n_features)
        feature_min = data[:, feature_idx].min()
        feature_max = data[:, feature_idx].max()
        if feature_min == feature_max:
            return {"type": "leaf", "size": n_samples}

        split = np.random.uniform(feature_min, feature_max)
        left_mask = data[:, feature_idx] < split
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"type": "leaf", "size": n_samples}

        return {
            "type": "node",
            "feature": feature_idx,
            "split": split,
            "left": self._build_tree(data[left_mask], depth + 1),
            "right": self._build_tree(data[right_mask], depth + 1),
        }

    def _path_length(self, tree: Dict, point: np.ndarray) -> float:
        """计算点在树中的路径长度"""
        depth = 0
        while tree["type"] == "node":
            if point[tree["feature"]] < tree["split"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
            depth += 1
        # 使用叶节点大小调整
        leaf_size = tree.get("size", 1)
        # 平均路径长度调整
        if leaf_size > 2:
            depth += 2 * (np.log(leaf_size - 1) + 0.5772156649) - 2 * (leaf_size - 1) / leaf_size
        return depth

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples = len(data)
            sample_size = min(self.max_samples, n_samples)

            self._trees = []
            for _ in range(self.n_trees):
                # 随机采样
                indices = np.random.choice(n_samples, sample_size, replace=False)
                sample = data[indices]
                tree = self._build_tree(sample)
                self._trees.append(tree)

            self.metrics.sample_count = n_samples
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = self.score_samples(X)
        return (scores > threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """返回异常分数 (0-1, 越高越异常)"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n = len(data)
            scores = np.zeros(n)
            sample_size = min(self.max_samples, self.metrics.sample_count)

            # 计算平均路径长度
            c_n = 2 * (np.log(sample_size - 1) + 0.5772156649) - \
                  2 * (sample_size - 1) / sample_size if sample_size > 2 else 1.0

            for i in range(n):
                avg_path = np.mean([self._path_length(tree, data[i])
                                   for tree in self._trees])
                # 异常分数：s = 2^(-E[h(x)] / c(n))
                scores[i] = 2 ** (-avg_path / c_n)

            self._end_prediction(start)
            return scores
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def detect(self, X: np.ndarray, threshold: float = 0.6) -> List[AnomalyResult]:
        scores = self.score_samples(X)
        labels = (scores > threshold).astype(int)
        return [
            AnomalyResult(
                is_anomaly=bool(label),
                anomaly_score=float(score),
                severity=ZScoreDetector._score_to_severity(score),
                anomaly_type=AnomalyType.POINT,
                detected_by=self.name,
                details={"path_length": float(-np.log2(max(score, 1e-10)))},
            )
            for label, score in zip(labels, scores)
        ]


class AutoEncoderDetector(AIAlgorithmComponent):
    """AutoEncoder 异常检测器（简化实现）

    使用简单的单层自编码器，通过重构误差检测异常。
    """

    def __init__(self, name: str = "autoencoder",
                 hidden_dim: int = 4, learning_rate: float = 0.01,
                 epochs: int = 100, threshold_percentile: float = 95.0):
        super().__init__(name, hidden_dim=hidden_dim,
                         learning_rate=learning_rate, epochs=epochs,
                         threshold_percentile=threshold_percentile)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile
        self._weights_encode: Optional[np.ndarray] = None
        self._bias_encode: Optional[np.ndarray] = None
        self._weights_decode: Optional[np.ndarray] = None
        self._bias_decode: Optional[np.ndarray] = None
        self._threshold: float = 0.0

    @property
    def component_type(self) -> str:
        return "anomaly_detection"

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoEncoderDetector":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples, n_features = data.shape

            # 归一化
            data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

            # 初始化权重
            np.random.seed(42)
            self._weights_encode = np.random.randn(n_features, self.hidden_dim) * 0.1
            self._bias_encode = np.zeros(self.hidden_dim)
            self._weights_decode = np.random.randn(self.hidden_dim, n_features) * 0.1
            self._bias_decode = np.zeros(n_features)

            # 训练（梯度下降）
            for epoch in range(self.epochs):
                # 前向
                hidden = self._relu(data_norm @ self._weights_encode + self._bias_encode)
                reconstructed = self._sigmoid(hidden @ self._weights_decode + self._bias_decode)

                # 重构误差
                error = reconstructed - data_norm
                loss = float(np.mean(error ** 2))

                # 反向传播（简化）
                grad_output = error * reconstructed * (1 - reconstructed)
                grad_hidden = grad_output @ self._weights_decode.T
                grad_hidden[hidden <= 0] = 0  # ReLU 梯度

                # 更新权重
                self._weights_decode -= self.learning_rate * hidden.T @ grad_output / n_samples
                self._bias_decode -= self.learning_rate * np.mean(grad_output, axis=0)
                self._weights_encode -= self.learning_rate * data_norm.T @ grad_hidden / n_samples
                self._bias_encode -= self.learning_rate * np.mean(grad_hidden, axis=0)

            # 计算训练集重构误差，设置阈值
            hidden = self._relu(data_norm @ self._weights_encode + self._bias_encode)
            reconstructed = self._sigmoid(hidden @ self._weights_decode + self._bias_decode)
            errors = np.mean((data_norm - reconstructed) ** 2, axis=1)
            self._threshold = float(np.percentile(errors, self.threshold_percentile))

            self.metrics.rmse = float(np.sqrt(np.mean(errors ** 2)))
            self.metrics.mae = float(np.mean(np.abs(errors)))
            self.metrics.sample_count = n_samples
            self.metrics.feature_count = n_features
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            # 归一化（使用训练时的统计量近似）
            data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
            hidden = self._relu(data_norm @ self._weights_encode + self._bias_encode)
            reconstructed = self._sigmoid(hidden @ self._weights_decode + self._bias_decode)
            errors = np.mean((data_norm - reconstructed) ** 2, axis=1)
            self._end_prediction(start)
            return errors
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def detect(self, X: np.ndarray) -> List[AnomalyResult]:
        scores = self.score_samples(X)
        labels = (scores > self._threshold).astype(int)
        normalized_scores = np.clip(scores / (self._threshold * 2), 0, 1)
        return [
            AnomalyResult(
                is_anomaly=bool(label),
                anomaly_score=float(score),
                severity=ZScoreDetector._score_to_severity(score),
                anomaly_type=AnomalyType.POINT,
                detected_by=self.name,
                details={"reconstruction_error": float(err)},
            )
            for label, score, err in zip(labels, normalized_scores, scores)
        ]


class AnomalyDetector:
    """异常检测统一接口

    提供多种异常检测算法的统一访问入口。

    Usage:
        detector = AnomalyDetector()
        report = detector.detect(data, method="isolation_forest")
    """

    METHODS = ["zscore", "iqr", "isolation_forest", "autoencoder", "ensemble", "auto"]

    def __init__(self):
        self._detectors: Dict[str, AIAlgorithmComponent] = {}

    def detect(self, data: np.ndarray, method: str = "auto",
<<<<<<< HEAD
                threshold: Optional[float] = None, **kwargs) -> AnomalyReport:
=======
                threshold: float = 0.6, **kwargs) -> AnomalyReport:
>>>>>>> 207ffbffe36779557c4a4cca69837167f09cfdbc
        """检测异常

        Args:
            data: 输入数据 (n_samples, n_features) 或 (n_samples,)
            method: 检测方法
<<<<<<< HEAD
            threshold: 异常阈值（None 表示使用算法默认值）
=======
            threshold: 异常阈值
>>>>>>> 207ffbffe36779557c4a4cca69837167f09cfdbc
            **kwargs: 算法参数

        Returns:
            AnomalyReport: 检测报告
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if method == "auto":
            method = self._select_method(data, **kwargs)

<<<<<<< HEAD
        # 方法相关的默认阈值
        if threshold is None:
            threshold = 3.0 if method == "zscore" else 0.6

        if method == "ensemble":
            return self._ensemble_detect(data, threshold)

        detector = self._get_or_create_detector(method, threshold=threshold, **kwargs)
=======
        if method == "ensemble":
            return self._ensemble_detect(data, threshold)

        detector = self._get_or_create_detector(method, **kwargs)
>>>>>>> 207ffbffe36779557c4a4cca69837167f09cfdbc
        detector.fit(data)
        results = self._call_detect(detector, data, threshold)

        return self._build_report(results, method)

    def _get_or_create_detector(self, method: str, **kwargs) -> AIAlgorithmComponent:
        if method not in self._detectors:
            if method == "zscore":
                self._detectors[method] = ZScoreDetector(threshold=kwargs.get("threshold", 3.0))
            elif method == "iqr":
                self._detectors[method] = IQRDetector(multiplier=kwargs.get("multiplier", 1.5))
            elif method == "isolation_forest":
                self._detectors[method] = IsolationForestDetector(
                    n_trees=kwargs.get("n_trees", 100),
                    max_samples=kwargs.get("max_samples", 256),
                )
            elif method == "autoencoder":
                self._detectors[method] = AutoEncoderDetector(
                    hidden_dim=kwargs.get("hidden_dim", 4),
                    epochs=kwargs.get("epochs", 100),
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        return self._detectors[method]

    def _call_detect(self, detector: AIAlgorithmComponent, data: np.ndarray,
                      threshold: float) -> List[AnomalyResult]:
        """调用检测器的 detect 方法"""
        if isinstance(detector, IsolationForestDetector):
            return detector.detect(data, threshold=threshold)
        return detector.detect(data)

    def _ensemble_detect(self, data: np.ndarray,
                          threshold: float) -> AnomalyReport:
        """组合检测：多个检测器投票"""
        detectors_to_use = ["zscore", "iqr", "isolation_forest"]
        all_results: List[List[AnomalyResult]] = []
        for method in detectors_to_use:
            detector = self._get_or_create_detector(method)
            detector.fit(data)
            results = self._call_detect(detector, data, threshold)
            all_results.append(results)

        # 投票：多数检测器判定为异常才算异常
        n_samples = len(data)
        final_results = []
        for i in range(n_samples):
            votes = sum(1 for results in all_results if results[i].is_anomaly)
            is_anomaly = votes >= 2  # 至少 2 票
            avg_score = float(np.mean([results[i].anomaly_score
                                       for results in all_results]))
            final_results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=avg_score,
                severity=ZScoreDetector._score_to_severity(avg_score),
                anomaly_type=AnomalyType.POINT,
                detected_by="ensemble",
                details={"votes": votes, "total": len(detectors_to_use)},
            ))
        return self._build_report(final_results, "ensemble")

    @staticmethod
    def _select_method(data: np.ndarray, **kwargs) -> str:
        """自动选择检测方法"""
        n_samples = len(data)
        if n_samples < 50:
            return "zscore"
        elif n_samples < 200:
            return "iqr"
        else:
            return "isolation_forest"

    @staticmethod
    def _build_report(results: List[AnomalyResult], method: str) -> AnomalyReport:
        """构建检测报告"""
        total = len(results)
        anomalies = [r for r in results if r.is_anomaly]
        severity_dist: Dict[str, int] = {}
        for r in anomalies:
            severity_dist[r.severity.name] = severity_dist.get(r.severity.name, 0) + 1

        return AnomalyReport(
            total_samples=total,
            anomaly_count=len(anomalies),
            anomaly_rate=len(anomalies) / total if total > 0 else 0.0,
            anomalies=[r.to_dict() for r in anomalies],
            severity_distribution=severity_dist,
            method=method,
        )

    def compare_methods(self, data: np.ndarray,
                          threshold: float = 0.6) -> Dict[str, Any]:
        """对比所有方法的检测结果"""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        results = {}
        for method in ["zscore", "iqr", "isolation_forest", "autoencoder"]:
            try:
                report = self.detect(data, method=method, threshold=threshold)
                results[method] = {
                    "anomaly_count": report.anomaly_count,
                    "anomaly_rate": report.anomaly_rate,
                    "training_time": self._detectors.get(method,
                                                          type("", (), {})()).metrics.training_time
                        if method in self._detectors else 0.0,
                }
            except Exception as e:
                results[method] = {"error": str(e)}
        return results
