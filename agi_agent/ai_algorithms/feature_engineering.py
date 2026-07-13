"""
ai_algorithms/feature_engineering.py - 特征工程组件

支持：
- PCA 降维
- 互信息特征选择
- 特征重要性评估
- 自动特征生成
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


@dataclass
class FeatureResult:
    """特征工程结果"""
    transformed_data: np.ndarray
    selected_features: List[int]
    feature_scores: Dict[str, float]
    n_components: int
    method: str
    explained_variance: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformed_data": self.transformed_data.tolist(),
            "selected_features": self.selected_features,
            "feature_scores": {k: float(v) for k, v in self.feature_scores.items()},
            "n_components": self.n_components,
            "method": self.method,
            "explained_variance": self.explained_variance.tolist()
                if self.explained_variance is not None else None,
        }


class PCAComponent(AIAlgorithmComponent):
    """PCA 主成分分析降维"""

    def __init__(self, name: str = "pca", n_components: Optional[int] = None,
                 variance_threshold: float = 0.95):
        super().__init__(name, n_components=n_components,
                         variance_threshold=variance_threshold)
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self._components: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "feature_engineering"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAComponent":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples, n_features = data.shape

            # 中心化
            self._mean = np.mean(data, axis=0)
            centered = data - self._mean

            # 协方差矩阵的特征分解
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # 降序排序
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # 解释方差比
            total_var = np.sum(eigenvalues)
            self._explained_variance_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
            self._explained_variance = eigenvalues

            # 选择主成分数量
            if self.n_components is None:
                cumulative = np.cumsum(self._explained_variance_ratio)
                self.n_components = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            self.n_components = min(self.n_components, n_features)

            self._components = eigenvectors[:, :self.n_components]
            self.metrics.sample_count = n_samples
            self.metrics.feature_count = n_features
            self.metrics.custom["total_variance_explained"] = float(
                np.sum(self._explained_variance_ratio[:self.n_components]))
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """transform 的别名"""
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            centered = data - self._mean
            transformed = centered @ self._components
            self._end_prediction(start)
            return transformed
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_result(self) -> FeatureResult:
        return FeatureResult(
            transformed_data=np.array([]),  # 需要外部调用 transform
            selected_features=list(range(self.n_components)),
            feature_scores={f"pc_{i}": float(self._explained_variance_ratio[i])
                           for i in range(self.n_components)},
            n_components=self.n_components,
            method="pca",
            explained_variance=self._explained_variance_ratio[:self.n_components],
        )


class MutualInfoSelector(AIAlgorithmComponent):
    """互信息特征选择"""

    def __init__(self, name: str = "mutual_info", k: int = 5,
                 threshold: float = 0.0):
        super().__init__(name, k=k, threshold=threshold)
        self.k = k
        self.threshold = threshold
        self._scores: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "feature_engineering"

    def _estimate_mutual_info(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """估计每个特征与目标的互信息（基于直方图）"""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        n_bins = min(20, max(5, int(np.sqrt(len(y)))))

        for f in range(n_features):
            # 离散化
            x = X[:, f]
            x_bins = np.histogram(x, bins=n_bins)[0] / len(x)
            y_bins = np.histogram(y, bins=n_bins)[0] / len(y)

            # 联合分布
            joint = np.histogram2d(x, y, bins=n_bins)[0] / len(y)
            # 互信息
            with np.errstate(divide="ignore", invalid="ignore"):
                mi = joint * np.log2(joint / (x_bins[:, None] * y_bins[None, :]) + 1e-10)
            mi = np.nan_to_num(mi)
            mi_scores[f] = 0.5 * np.sum(mi)

        return mi_scores

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MutualInfoSelector":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if y is None:
                raise ValueError("MutualInfoSelector requires target y")

            y = np.asarray(y, dtype=np.float64)
            self._scores = self._estimate_mutual_info(data, y)

            # 选择 top-k
            k = min(self.k, data.shape[1])
            self._selected_indices = np.argsort(self._scores)[::-1][:k]
            # 过滤阈值
            above_threshold = self._scores[self._selected_indices] >= self.threshold
            self._selected_indices = self._selected_indices[above_threshold]

            self.metrics.sample_count = len(data)
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data[:, self._selected_indices]

    def get_result(self) -> FeatureResult:
        return FeatureResult(
            transformed_data=np.array([]),
            selected_features=self._selected_indices.tolist(),
            feature_scores={f"feature_{i}": float(self._scores[i])
                           for i in range(len(self._scores))},
            n_components=len(self._selected_indices),
            method="mutual_info",
        )


class FeatureImportanceEvaluator(AIAlgorithmComponent):
    """特征重要性评估器

    基于方差、相关性、信息增益综合评估特征重要性。
    """

    def __init__(self, name: str = "feature_importance"):
        super().__init__(name)
        self._importance_scores: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    @property
    def component_type(self) -> str:
        return "feature_engineering"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
             feature_names: Optional[List[str]] = None) -> "FeatureImportanceEvaluator":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_features = data.shape[1]
            self._importance_scores = np.zeros(n_features)
            self._feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

            # 方差评分（标准化）
            variances = np.var(data, axis=0)
            var_scores = variances / (variances.max() + 1e-8)

            if y is not None:
                y = np.asarray(y, dtype=np.float64)
                # 相关性评分
                correlations = np.array([abs(np.corrcoef(data[:, f], y)[0, 1])
                                        if np.std(data[:, f]) > 0 else 0
                                        for f in range(n_features)])
                corr_scores = correlations / (correlations.max() + 1e-8)
                # 综合评分
                self._importance_scores = 0.5 * var_scores + 0.5 * corr_scores
            else:
                self._importance_scores = var_scores

            self.metrics.sample_count = len(data)
            self.metrics.feature_count = n_features
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def transform(self, X: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if top_k is None:
            top_k = data.shape[1]
        top_indices = np.argsort(self._importance_scores)[::-1][:top_k]
        return data[:, top_indices]

    def get_importance(self) -> Dict[str, float]:
        if self._importance_scores is None:
            return {}
        return {name: float(score) for name, score
                in zip(self._feature_names, self._importance_scores)}

    def get_result(self, top_k: Optional[int] = None) -> FeatureResult:
        if top_k is None:
            top_k = len(self._importance_scores)
        top_indices = np.argsort(self._importance_scores)[::-1][:top_k].tolist()
        return FeatureResult(
            transformed_data=np.array([]),
            selected_features=top_indices,
            feature_scores=self.get_importance(),
            n_components=top_k,
            method="feature_importance",
        )


class FeatureEngineer:
    """特征工程统一接口"""

    def __init__(self):
        self._components: Dict[str, AIAlgorithmComponent] = {}

    def reduce_dimensions(self, data: np.ndarray,
                            n_components: Optional[int] = None,
                            variance_threshold: float = 0.95) -> FeatureResult:
        """PCA 降维"""
        pca = PCAComponent(n_components=n_components,
                            variance_threshold=variance_threshold)
        pca.fit(data)
        transformed = pca.transform(data)
        result = pca.get_result()
        result.transformed_data = transformed
        self._components["pca"] = pca
        return result

    def select_features(self, data: np.ndarray, y: np.ndarray,
                          k: int = 5) -> FeatureResult:
        """互信息特征选择"""
        selector = MutualInfoSelector(k=k)
        selector.fit(data, y)
        transformed = selector.transform(data)
        result = selector.get_result()
        result.transformed_data = transformed
        self._components["mutual_info"] = selector
        return result

    def evaluate_importance(self, data: np.ndarray, y: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None,
                              top_k: Optional[int] = None) -> FeatureResult:
        """评估特征重要性"""
        evaluator = FeatureImportanceEvaluator()
        evaluator.fit(data, y, feature_names=feature_names)
        self._components["importance"] = evaluator
        return evaluator.get_result(top_k=top_k)

    def auto_pipeline(self, data: np.ndarray, y: Optional[np.ndarray] = None,
                       target_components: int = 2) -> np.ndarray:
        """自动特征工程流水线"""
        # 1. 评估重要性
        if y is not None:
            importance = self.evaluate_importance(data, y)
            top_k = min(target_components * 3, data.shape[1])
            top_indices = importance.selected_features[:top_k]
            data = data[:, top_indices]
        # 2. PCA 降维
        pca = PCAComponent(n_components=min(target_components, data.shape[1]))
        transformed = pca.fit_transform(data)
        self._components["pipeline_pca"] = pca
        return transformed
