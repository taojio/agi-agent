import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT = "reinforcement"
    GENERATION = "generation"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"
    CLUSTERING = "clustering"
    DETECTION = "detection"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    HIGH_COMPLEX = "high_complex"


class DataDistribution(Enum):
    IID = "iid"
    NON_IID = "non_iid"
    SPARSE = "sparse"
    IMBALANCED = "imbalanced"


class StrategyType(Enum):
    MAML = "maml"
    REPTILE = "reptile"
    PROTO_NET = "proto_net"
    MATCHING_NET = "matching_net"
    META_SGD = "meta_sgd"
    TRANSFER_LEARNING = "transfer_learning"
    FINE_TUNE = "fine_tune"
    SELF_SUPERVISED = "self_supervised"


class TaskFeatureVector:
    def __init__(self, task_id: str, task_type: TaskType,
                 complexity: TaskComplexity, data_distribution: DataDistribution,
                 input_dim: int = 0, output_dim: int = 0, num_samples: int = 0):
        self.task_id = task_id
        self.task_type = task_type
        self.complexity = complexity
        self.data_distribution = data_distribution
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        self.vector: Optional[np.ndarray] = None
        self.timestamp = np.random.randint(1000000)

    def compute_vector(self) -> np.ndarray:
        type_encoding = {
            TaskType.CLASSIFICATION: [1, 0, 0, 0, 0, 0, 0, 0],
            TaskType.REGRESSION: [0, 1, 0, 0, 0, 0, 0, 0],
            TaskType.REINFORCEMENT: [0, 0, 1, 0, 0, 0, 0, 0],
            TaskType.GENERATION: [0, 0, 0, 1, 0, 0, 0, 0],
            TaskType.FEW_SHOT: [0, 0, 0, 0, 1, 0, 0, 0],
            TaskType.ZERO_SHOT: [0, 0, 0, 0, 0, 1, 0, 0],
            TaskType.CLUSTERING: [0, 0, 0, 0, 0, 0, 1, 0],
            TaskType.DETECTION: [0, 0, 0, 0, 0, 0, 0, 1],
        }[self.task_type]

        complexity_encoding = {
            TaskComplexity.SIMPLE: [1, 0, 0, 0],
            TaskComplexity.MEDIUM: [0, 1, 0, 0],
            TaskComplexity.COMPLEX: [0, 0, 1, 0],
            TaskComplexity.HIGH_COMPLEX: [0, 0, 0, 1],
        }[self.complexity]

        dist_encoding = {
            DataDistribution.IID: [1, 0, 0, 0],
            DataDistribution.NON_IID: [0, 1, 0, 0],
            DataDistribution.SPARSE: [0, 0, 1, 0],
            DataDistribution.IMBALANCED: [0, 0, 0, 1],
        }[self.data_distribution]

        normalized_dim = min(self.input_dim / 1000, 1.0) if self.input_dim > 0 else 0.5
        normalized_output = min(self.output_dim / 100, 1.0) if self.output_dim > 0 else 0.5
        normalized_samples = min(self.num_samples / 10000, 1.0) if self.num_samples > 0 else 0.3

        self.vector = np.array(
            type_encoding + complexity_encoding + dist_encoding +
            [normalized_dim, normalized_output, normalized_samples]
        )

        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

        return self.vector

    def get_vector(self) -> np.ndarray:
        if self.vector is None:
            self.compute_vector()
        return self.vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "data_distribution": self.data_distribution.value,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_samples": self.num_samples,
            "vector_dim": len(self.get_vector()),
            "timestamp": self.timestamp
        }


class StrategyPerformanceRecord:
    def __init__(self, strategy_type: StrategyType, task_type: TaskType):
        self.strategy_type = strategy_type
        self.task_type = task_type
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.train_time_ms: List[float] = []
        self.sample_efficiency: List[float] = []
        self.success_rate: float = 0.0
        self.usage_count: int = 0
        self.success_count: int = 0

    def update(self, accuracy: float, loss: float, train_time_ms: float,
               sample_efficiency: float, success: bool):
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.train_time_ms.append(train_time_ms)
        self.sample_efficiency.append(sample_efficiency)
        self.usage_count += 1
        if success:
            self.success_count += 1
        if self.usage_count > 0:
            self.success_rate = self.success_count / self.usage_count

    def get_summary(self) -> Dict[str, Any]:
        if not self.accuracy_history:
            return {}

        return {
            "strategy_type": self.strategy_type.value,
            "task_type": self.task_type.value,
            "avg_accuracy": float(np.mean(self.accuracy_history)),
            "avg_loss": float(np.mean(self.loss_history)),
            "avg_train_time_ms": float(np.mean(self.train_time_ms)),
            "avg_sample_efficiency": float(np.mean(self.sample_efficiency)),
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "best_accuracy": max(self.accuracy_history),
            "worst_accuracy": min(self.accuracy_history)
        }


class StrategyVector:
    def __init__(self, strategy_type: StrategyType):
        self.strategy_type = strategy_type
        self.vector: Optional[np.ndarray] = None
        self.performance_records: Dict[str, StrategyPerformanceRecord] = {}

    def compute_vector(self) -> np.ndarray:
        strategy_encoding = {
            StrategyType.MAML: [1, 0, 0, 0, 0, 0, 0, 0],
            StrategyType.REPTILE: [0, 1, 0, 0, 0, 0, 0, 0],
            StrategyType.PROTO_NET: [0, 0, 1, 0, 0, 0, 0, 0],
            StrategyType.MATCHING_NET: [0, 0, 0, 1, 0, 0, 0, 0],
            StrategyType.META_SGD: [0, 0, 0, 0, 1, 0, 0, 0],
            StrategyType.TRANSFER_LEARNING: [0, 0, 0, 0, 0, 1, 0, 0],
            StrategyType.FINE_TUNE: [0, 0, 0, 0, 0, 0, 1, 0],
            StrategyType.SELF_SUPERVISED: [0, 0, 0, 0, 0, 0, 0, 1],
        }[self.strategy_type]

        avg_performance = self._compute_avg_performance()

        self.vector = np.array(strategy_encoding + [avg_performance])

        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

        return self.vector

    def _compute_avg_performance(self) -> float:
        if not self.performance_records:
            return 0.5
        avg_acc = np.mean([r.get_summary().get("avg_accuracy", 0.5)
                          for r in self.performance_records.values()])
        return float(avg_acc)

    def get_vector(self) -> np.ndarray:
        if self.vector is None:
            self.compute_vector()
        return self.vector

    def add_performance_record(self, task_type: TaskType, accuracy: float,
                               loss: float, train_time_ms: float,
                               sample_efficiency: float, success: bool):
        key = task_type.value
        if key not in self.performance_records:
            self.performance_records[key] = StrategyPerformanceRecord(self.strategy_type, task_type)
        self.performance_records[key].update(accuracy, loss, train_time_ms, sample_efficiency, success)
        self.vector = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type.value,
            "vector_dim": len(self.get_vector()),
            "performance_records": {k: v.get_summary() for k, v in self.performance_records.items()},
            "avg_performance": self._compute_avg_performance()
        }


class TaskStrategyKnowledgeBase:
    def __init__(self):
        self.task_features: Dict[str, TaskFeatureVector] = {}
        self.strategy_vectors: Dict[str, StrategyVector] = {}
        self.task_strategy_mappings: Dict[str, List[Dict[str, Any]]] = {}
        self.similarity_cache: Dict[str, float] = {}
        self.update_history: deque = deque(maxlen=500)
        self._auto_update_interval = 100
        self._update_counter = 0

    def register_task(self, task_id: str, task_type: TaskType,
                      complexity: TaskComplexity, data_distribution: DataDistribution,
                      input_dim: int = 0, output_dim: int = 0, num_samples: int = 0) -> TaskFeatureVector:
        feature_vector = TaskFeatureVector(
            task_id, task_type, complexity, data_distribution,
            input_dim, output_dim, num_samples
        )
        feature_vector.compute_vector()
        self.task_features[task_id] = feature_vector
        return feature_vector

    def register_strategy(self, strategy_type: StrategyType) -> StrategyVector:
        key = strategy_type.value
        if key not in self.strategy_vectors:
            self.strategy_vectors[key] = StrategyVector(strategy_type)
        return self.strategy_vectors[key]

    def update_strategy_performance(self, strategy_type: StrategyType, task_type: TaskType,
                                    accuracy: float, loss: float, train_time_ms: float,
                                    sample_efficiency: float, success: bool):
        key = strategy_type.value
        if key not in self.strategy_vectors:
            self.register_strategy(strategy_type)
        self.strategy_vectors[key].add_performance_record(
            task_type, accuracy, loss, train_time_ms, sample_efficiency, success
        )
        self._record_update("performance_update", {
            "strategy_type": strategy_type.value,
            "task_type": task_type.value,
            "accuracy": accuracy
        })

    def compute_task_strategy_similarity(self, task_vector: np.ndarray,
                                        strategy_vector: np.ndarray) -> float:
        if task_vector.shape != strategy_vector.shape:
            task_len, strat_len = len(task_vector), len(strategy_vector)
            max_len = max(task_len, strat_len)
            padded_task = np.zeros(max_len)
            padded_strat = np.zeros(max_len)
            padded_task[:task_len] = task_vector
            padded_strat[:strat_len] = strategy_vector
            task_vector, strategy_vector = padded_task, padded_strat

        norm = np.linalg.norm(task_vector) * np.linalg.norm(strategy_vector)
        if norm == 0:
            return 0.0
        return float(np.dot(task_vector, strategy_vector) / norm)

    def recommend_strategies(self, task_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if task_id not in self.task_features:
            return []

        task_feature = self.task_features[task_id]
        task_vector = task_feature.get_vector()

        recommendations = []
        for strategy_key, strategy_vector in self.strategy_vectors.items():
            strat_vector = strategy_vector.get_vector()
            similarity = self.compute_task_strategy_similarity(task_vector, strat_vector)

            perf_summary = strategy_vector.performance_records.get(
                task_feature.task_type.value, None
            )

            recommendations.append({
                "strategy_type": strategy_key,
                "similarity": similarity,
                "performance": perf_summary.get_summary() if perf_summary else {},
                "confidence": min(1.0, similarity * 0.7 + 0.3)
            })

        recommendations.sort(key=lambda x: x["similarity"], reverse=True)

        return recommendations[:top_k]

    def find_similar_tasks(self, task_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if task_id not in self.task_features:
            return []

        task_feature = self.task_features[task_id]
        task_vector = task_feature.get_vector()

        similarities = []
        for other_id, other_feature in self.task_features.items():
            if other_id == task_id:
                continue
            other_vector = other_feature.get_vector()
            similarity = self.compute_task_strategy_similarity(task_vector, other_vector)
            similarities.append({
                "task_id": other_id,
                "task_type": other_feature.task_type.value,
                "complexity": other_feature.complexity.value,
                "similarity": similarity
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def transfer_strategy(self, source_task_id: str, target_task_id: str) -> Dict[str, Any]:
        if source_task_id not in self.task_features or target_task_id not in self.task_features:
            return {"success": False, "reason": "Task not found"}

        source_feature = self.task_features[source_task_id]
        target_feature = self.task_features[target_task_id]

        similarity = self.compute_task_strategy_similarity(
            source_feature.get_vector(), target_feature.get_vector()
        )

        source_recommendations = self.recommend_strategies(source_task_id, top_k=1)

        if source_recommendations:
            best_strategy = source_recommendations[0]
            transfer_effectiveness = best_strategy["similarity"] * similarity
        else:
            best_strategy = None
            transfer_effectiveness = similarity * 0.5

        return {
            "success": True,
            "source_task": source_task_id,
            "target_task": target_task_id,
            "task_similarity": similarity,
            "recommended_strategy": best_strategy,
            "transfer_effectiveness": min(1.0, transfer_effectiveness),
            "confidence": min(1.0, similarity * 0.8 + 0.2)
        }

    def auto_update(self):
        self._update_counter += 1
        if self._update_counter >= self._auto_update_interval:
            self._update_counter = 0
            self._refresh_strategy_vectors()
            self._record_update("auto_update", {
                "task_count": len(self.task_features),
                "strategy_count": len(self.strategy_vectors)
            })

    def _refresh_strategy_vectors(self):
        for strategy_vector in self.strategy_vectors.values():
            strategy_vector.compute_vector()

    def _record_update(self, update_type: str, details: Dict[str, Any]):
        self.update_history.append({
            "type": update_type,
            "details": details,
            "timestamp": np.random.randint(1000000)
        })

    def get_task_feature(self, task_id: str) -> Optional[Dict[str, Any]]:
        feature = self.task_features.get(task_id)
        return feature.to_dict() if feature else None

    def get_strategy_summary(self, strategy_type: StrategyType) -> Optional[Dict[str, Any]]:
        key = strategy_type.value
        strategy = self.strategy_vectors.get(key)
        return strategy.to_dict() if strategy else None

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tasks": len(self.task_features),
            "total_strategies": len(self.strategy_vectors),
            "total_mappings": sum(len(v) for v in self.task_strategy_mappings.values()),
            "update_history_count": len(self.update_history),
            "auto_update_interval": self._auto_update_interval
        }

    def get_update_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.update_history)[-limit:]

    def export_knowledge(self) -> Dict[str, Any]:
        return {
            "tasks": {k: v.to_dict() for k, v in self.task_features.items()},
            "strategies": {k: v.to_dict() for k, v in self.strategy_vectors.items()},
            "mappings": self.task_strategy_mappings,
            "summary": self.get_summary()
        }

    def import_knowledge(self, knowledge: Dict[str, Any]):
        self.task_features.clear()
        self.strategy_vectors.clear()
        self.task_strategy_mappings.clear()

        for task_id, task_data in knowledge.get("tasks", {}).items():
            self.register_task(
                task_id,
                TaskType(task_data["task_type"]),
                TaskComplexity(task_data["complexity"]),
                DataDistribution(task_data["data_distribution"]),
                task_data.get("input_dim", 0),
                task_data.get("output_dim", 0),
                task_data.get("num_samples", 0)
            )

        for strategy_key, strategy_data in knowledge.get("strategies", {}).items():
            self.register_strategy(StrategyType(strategy_key))

        self.task_strategy_mappings = knowledge.get("mappings", {})