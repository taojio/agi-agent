"""
ai_algorithms/clustering.py - 自动聚类组件

支持多种聚类算法：
- K-Means 聚类
- DBSCAN 密度聚类
- 层次聚类（自底向上凝聚式）
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


@dataclass
class ClusterResult:
    """聚类结果"""
    labels: np.ndarray              # 每个样本的簇标签
    centers: Optional[np.ndarray]   # 簇中心（K-Means）
    n_clusters: int                 # 簇数量
    silhouette_score: float = 0.0   # 轮廓系数
    method: str = ""
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    noise_count: int = 0            # 噪声点数（DBSCAN）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels.tolist(),
            "centers": self.centers.tolist() if self.centers is not None else None,
            "n_clusters": self.n_clusters,
            "silhouette_score": float(self.silhouette_score),
            "method": self.method,
            "cluster_sizes": dict(self.cluster_sizes),
            "noise_count": self.noise_count,
        }


class KMeansClusterer(AIAlgorithmComponent):
    """K-Means 聚类器"""

    def __init__(self, name: str = "kmeans", k: int = 3,
                 max_iters: int = 100, tolerance: float = 1e-4,
                 random_state: int = 42):
        super().__init__(name, k=k, max_iters=max_iters, tolerance=tolerance)
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state
        self._centers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "clustering"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KMeansClusterer":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples = len(data)
            if n_samples < self.k:
                raise ValueError(f"Need at least {self.k} samples")

            np.random.seed(self.random_state)
            # K-Means++ 初始化
            self._centers = self._kmeans_plusplus_init(data)
            self._labels = np.zeros(n_samples, dtype=int)

            for iteration in range(self.max_iters):
                # 分配簇
                distances = self._compute_distances(data, self._centers)
                new_labels = np.argmin(distances, axis=1)

                # 更新中心
                new_centers = np.zeros_like(self._centers)
                for k in range(self.k):
                    mask = new_labels == k
                    if mask.sum() > 0:
                        new_centers[k] = data[mask].mean(axis=0)
                    else:
                        new_centers[k] = self._centers[k]

                # 收敛检查
                shift = np.sum(np.abs(new_centers - self._centers))
                self._centers = new_centers
                self._labels = new_labels
                if shift < self.tolerance:
                    break

            # 计算轮廓系数
            self.metrics.custom["silhouette"] = self._compute_silhouette(data, self._labels)
            self.metrics.sample_count = n_samples
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _kmeans_plusplus_init(self, data: np.ndarray) -> np.ndarray:
        """K-Means++ 初始化"""
        n_samples = len(data)
        centers = np.zeros((self.k, data.shape[1]))
        # 第一个中心随机选择
        centers[0] = data[np.random.randint(n_samples)]
        # 后续中心按距离概率选择
        for i in range(1, self.k):
            distances = self._compute_distances(data, centers[:i])
            min_distances = np.min(distances, axis=1)
            probs = min_distances / (min_distances.sum() + 1e-8)
            centers[i] = data[np.random.choice(n_samples, p=probs)]
        return centers

    @staticmethod
    def _compute_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """计算欧氏距离"""
        # (n, 1, d) - (1, k, d) -> (n, k)
        diff = data[:, np.newaxis] - centers[np.newaxis, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _compute_silhouette(self, data: np.ndarray,
                              labels: np.ndarray) -> float:
        """计算轮廓系数"""
        n = len(data)
        if n < 2 or len(set(labels)) < 2:
            return 0.0
        distances = self._compute_distances(data, data)
        silhouette_values = []
        for i in range(n):
            same_cluster = labels == labels[i]
            same_cluster[i] = False
            if same_cluster.sum() == 0:
                continue
            a_i = np.mean(distances[i, same_cluster])
            # 找最近的簇
            other_clusters = set(labels) - {labels[i]}
            b_i = min(np.mean(distances[i, labels == c]) for c in other_clusters)
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            silhouette_values.append(s_i)
        return float(np.mean(silhouette_values)) if silhouette_values else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            distances = self._compute_distances(data, self._centers)
            labels = np.argmin(distances, axis=1)
            self._end_prediction(start)
            return labels
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def get_result(self) -> ClusterResult:
        if self._labels is None:
            raise RuntimeError("Model not trained")
        cluster_sizes = {int(k): int((self._labels == k).sum())
                        for k in range(self.k)}
        return ClusterResult(
            labels=self._labels,
            centers=self._centers,
            n_clusters=self.k,
            silhouette_score=self.metrics.custom.get("silhouette", 0.0),
            method="kmeans",
            cluster_sizes=cluster_sizes,
        )


class DBSCANClusterer(AIAlgorithmComponent):
    """DBSCAN 密度聚类器"""

    def __init__(self, name: str = "dbscan", eps: float = 0.5,
                 min_samples: int = 5):
        super().__init__(name, eps=eps, min_samples=min_samples)
        self.eps = eps
        self.min_samples = min_samples
        self._labels: Optional[np.ndarray] = None
        self._core_indices: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "clustering"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DBSCANClusterer":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples = len(data)
            self._labels = np.full(n_samples, -1)  # -1 = 噪声

            # 计算距离矩阵
            distances = np.sqrt(((data[:, np.newaxis] - data[np.newaxis, :]) ** 2).sum(axis=2))

            # 找核心点
            neighbors = [np.where(distances[i] <= self.eps)[0] for i in range(n_samples)]
            core_mask = np.array([len(n) >= self.min_samples for n in neighbors])
            self._core_indices = np.where(core_mask)[0]

            # 扩展簇
            cluster_id = 0
            visited = np.zeros(n_samples, dtype=bool)
            for i in range(n_samples):
                if visited[i] or not core_mask[i]:
                    continue
                # BFS 扩展
                queue = [i]
                visited[i] = True
                self._labels[i] = cluster_id
                while queue:
                    current = queue.pop(0)
                    for neighbor in neighbors[current]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            if core_mask[neighbor]:
                                queue.append(neighbor)
                            if self._labels[neighbor] == -1:
                                self._labels[neighbor] = cluster_id
                cluster_id += 1

            self.metrics.sample_count = n_samples
            self.metrics.feature_count = data.shape[1]
            self.metrics.custom["n_clusters"] = cluster_id
            self.metrics.custom["noise_count"] = int((self._labels == -1).sum())
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN 不支持对新点预测，返回训练标签"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self._labels.copy()

    def get_result(self) -> ClusterResult:
        if self._labels is None:
            raise RuntimeError("Model not trained")
        unique_labels = set(self._labels) - {-1}
        n_clusters = len(unique_labels)
        cluster_sizes = {int(k): int((self._labels == k).sum())
                        for k in unique_labels}
        noise_count = int((self._labels == -1).sum())
        return ClusterResult(
            labels=self._labels,
            centers=None,
            n_clusters=n_clusters,
            silhouette_score=0.0,  # DBSCAN 不直接计算
            method="dbscan",
            cluster_sizes=cluster_sizes,
            noise_count=noise_count,
        )


class HierarchicalClusterer(AIAlgorithmComponent):
    """层次聚类器（凝聚式，自底向上）"""

    def __init__(self, name: str = "hierarchical",
                 n_clusters: int = 3, linkage: str = "ward"):
        super().__init__(name, n_clusters=n_clusters, linkage=linkage)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self._labels: Optional[np.ndarray] = None
        self._linkage_matrix: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "clustering"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HierarchicalClusterer":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_samples = len(data)

            # 每个点初始为一个簇
            clusters = [[i] for i in range(n_samples)]
            distances = np.sqrt(((data[:, np.newaxis] - data[np.newaxis, :]) ** 2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            linkage_matrix = []

            while len(clusters) > self.n_clusters:
                # 找最近的两个簇
                min_dist = np.inf
                merge_i, merge_j = 0, 1
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        dist = self._cluster_distance(clusters[i], clusters[j],
                                                       distances, data)
                        if dist < min_dist:
                            min_dist = dist
                            merge_i, merge_j = i, j

                # 合并
                new_cluster = clusters[merge_i] + clusters[merge_j]
                # 记录 linkage
                linkage_matrix.append([merge_i, merge_j, min_dist, len(new_cluster)])
                # 移除被合并的簇（从大到小索引）
                clusters = [c for k, c in enumerate(clusters) if k not in (merge_i, merge_j)]
                clusters.append(new_cluster)

            # 生成标签
            self._labels = np.zeros(n_samples, dtype=int)
            for cluster_id, cluster in enumerate(clusters):
                for idx in cluster:
                    self._labels[idx] = cluster_id

            self._linkage_matrix = np.array(linkage_matrix) if linkage_matrix else None
            self.metrics.sample_count = n_samples
            self.metrics.feature_count = data.shape[1]
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _cluster_distance(self, c1: List[int], c2: List[int],
                            distance_matrix: np.ndarray,
                            data: np.ndarray) -> float:
        """计算簇间距离"""
        if self.linkage == "single":
            return min(distance_matrix[i, j] for i in c1 for j in c2)
        elif self.linkage == "complete":
            return max(distance_matrix[i, j] for i in c1 for j in c2)
        elif self.linkage == "average":
            return float(np.mean([distance_matrix[i, j] for i in c1 for j in c2]))
        else:  # ward
            mean1 = np.mean(data[c1], axis=0)
            mean2 = np.mean(data[c2], axis=0)
            return float(np.sum((mean1 - mean2) ** 2) * len(c1) * len(c2) / (len(c1) + len(c2)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self._labels.copy()

    def get_result(self) -> ClusterResult:
        if self._labels is None:
            raise RuntimeError("Model not trained")
        unique_labels = set(self._labels)
        cluster_sizes = {int(k): int((self._labels == k).sum())
                        for k in unique_labels}
        return ClusterResult(
            labels=self._labels,
            centers=None,
            n_clusters=len(unique_labels),
            silhouette_score=0.0,
            method=f"hierarchical_{self.linkage}",
            cluster_sizes=cluster_sizes,
        )


class AutoClusterer:
    """自动聚类器

    自动选择最佳聚类算法和参数。
    """

    def __init__(self):
        self._clusterers: Dict[str, AIAlgorithmComponent] = {}

    def cluster(self, data: np.ndarray, method: str = "auto",
                 k: Optional[int] = None, **kwargs) -> ClusterResult:
        """聚类

        Args:
            data: 输入数据
            method: 聚类方法 ("auto" 自动选择)
            k: 簇数量（K-Means/层次聚类）

        Returns:
            ClusterResult: 聚类结果
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if method == "auto":
            method = self._select_method(data)

        if method == "kmeans":
            if k is None:
                k = self._estimate_k(data)
            clusterer = KMeansClusterer(k=k, **kwargs)
        elif method == "dbscan":
            eps = kwargs.pop("eps", self._estimate_eps(data))
            clusterer = DBSCANClusterer(eps=eps, **kwargs)
        elif method == "hierarchical":
            if k is None:
                k = self._estimate_k(data)
            clusterer = HierarchicalClusterer(n_clusters=k, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        clusterer.fit(data)
        result = clusterer.get_result()
        self._clusterers[method] = clusterer
        return result

    def _select_method(self, data: np.ndarray) -> str:
        n_samples = len(data)
        if n_samples < 100:
            return "hierarchical"
        else:
            return "kmeans"

    def _estimate_k(self, data: np.ndarray) -> int:
        """使用肘部法估计最佳 K"""
        max_k = min(10, len(data) - 1)
        if max_k < 2:
            return 2
        inertias = []
        for k in range(2, max_k + 1):
            try:
                clusterer = KMeansClusterer(k=k, max_iters=20)
                clusterer.fit(data)
                # 计算惯性（簇内距离和）
                inertia = self._compute_inertia(data, clusterer._centers)
                inertias.append(inertia)
            except Exception:
                inertias.append(float("inf"))

        # 简化：选择轮廓系数最高的 K
        best_k = 2
        best_score = -1
        for k in range(2, max_k + 1):
            try:
                clusterer = KMeansClusterer(k=k, max_iters=50)
                clusterer.fit(data)
                score = clusterer.metrics.custom.get("silhouette", 0)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k

    @staticmethod
    def _compute_inertia(data: np.ndarray, centers: np.ndarray) -> float:
        distances = KMeansClusterer._compute_distances(data, centers)
        return float(np.sum(np.min(distances, axis=1) ** 2))

    @staticmethod
    def _estimate_eps(data: np.ndarray) -> float:
        """估计 DBSCAN 的 eps 参数"""
        n = len(data)
        distances = np.sqrt(((data[:, np.newaxis] - data[np.newaxis, :]) ** 2).sum(axis=2))
        k = min(5, n - 1)
        k_distances = np.sort(np.partition(distances, k, axis=1)[:, k])
        # 取拐点
        return float(np.percentile(k_distances, 75))
