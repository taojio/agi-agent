import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..meta_learning.maml_algorithm import MAMLAlgorithm, MAMLMode, MAMLTask
from ..meta_learning.hyperparameter_controller import HyperparameterController, ParameterType
from ..meta_learning.task_strategy_knowledge import (
    TaskStrategyKnowledgeBase, TaskType, TaskComplexity, DataDistribution
)


class MetaEnhancedFeatureExtractor:
    def __init__(self, input_dim: int, feature_dim: int = 128):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.maml = MAMLAlgorithm(input_dim, feature_dim, hidden_dim=256)
        self.hp_controller = HyperparameterController()
        self.knowledge_base = TaskStrategyKnowledgeBase()
        self.feature_history: List[Tuple[np.ndarray, float]] = []
        self.adaptation_count = 0

    def extract_features(self, data: np.ndarray, task_context: Dict[str, Any] = None) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(1, -1)

        features = self.maml.model.forward(data)

        if task_context:
            self.hp_controller.set_task_context(
                task_context.get("complexity", 0.5),
                task_context.get("uncertainty", 0.5)
            )

        return features

    def adapt_to_task(self, support_data: Tuple[np.ndarray, np.ndarray],
                      query_data: Tuple[np.ndarray, np.ndarray],
                      task_id: str = "perception_task") -> Dict[str, Any]:
        task = MAMLTask(task_id, "feature_extraction", support_data, query_data)

        learning_rate = self.hp_controller.get_parameter(ParameterType.LEARNING_RATE)
        result = self.maml.adapt_to_task(task, learning_rate=learning_rate)

        self.adaptation_count += 1
        self.hp_controller.update_performance(
            result.get_best_metrics()["train_loss"],
            result.get_best_metrics()["query_accuracy"]
        )
        self.hp_controller.adjust_all(task_type="classification")

        return {
            "task_id": task_id,
            "adaptation_result": result.to_dict(),
            "learning_rate": learning_rate,
            "adaptation_count": self.adaptation_count
        }

    def register_task(self, task_id: str, task_type: TaskType,
                      complexity: TaskComplexity, data_distribution: DataDistribution,
                      input_dim: int, output_dim: int, num_samples: int):
        self.knowledge_base.register_task(
            task_id, task_type, complexity, data_distribution,
            input_dim, output_dim, num_samples
        )

    def recommend_strategy(self, task_id: str) -> List[Dict[str, Any]]:
        return self.knowledge_base.recommend_strategies(task_id)

    def update_strategy_performance(self, task_id: str, accuracy: float, loss: float,
                                    train_time_ms: float, sample_efficiency: float, success: bool):
        task_feature = self.knowledge_base.task_features.get(task_id)
        if task_feature:
            from ..meta_learning.task_strategy_knowledge import StrategyType
            self.knowledge_base.update_strategy_performance(
                StrategyType.MAML, task_feature.task_type,
                accuracy, loss, train_time_ms, sample_efficiency, success
            )

    def get_status(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "adaptation_count": self.adaptation_count,
            "maml_mode": self.maml.mode.value,
            "hp_summary": self.hp_controller.get_parameter_summary(),
            "knowledge_summary": self.knowledge_base.get_summary()
        }


class MetaEnhancedPatternRecognizer:
    def __init__(self, num_patterns: int = 100, feature_dim: int = 128):
        self.num_patterns = num_patterns
        self.feature_dim = feature_dim
        self.patterns: Dict[str, np.ndarray] = {}
        self.pattern_confidence: Dict[str, float] = {}
        self.maml = MAMLAlgorithm(feature_dim, num_patterns, hidden_dim=256)
        self.hp_controller = HyperparameterController()

    def add_pattern(self, pattern_id: str, pattern: np.ndarray):
        if pattern.ndim == 1:
            pattern = pattern.reshape(1, -1)
        self.patterns[pattern_id] = pattern
        self.pattern_confidence[pattern_id] = 0.5

    def recognize_pattern(self, input_data: np.ndarray) -> List[Dict[str, Any]]:
        features = self.maml.model.forward(input_data)

        results = []
        for pattern_id, pattern in self.patterns.items():
            pattern_features = self.maml.model.forward(pattern)
            similarity = float(np.dot(features.flatten(), pattern_features.flatten()) /
                              (np.linalg.norm(features) * np.linalg.norm(pattern_features)))
            results.append({
                "pattern_id": pattern_id,
                "similarity": similarity,
                "confidence": self.pattern_confidence[pattern_id]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def adapt_recognizer(self, training_data: List[Tuple[np.ndarray, str]],
                        validation_data: List[Tuple[np.ndarray, str]]):
        support_x = np.vstack([data[0] for data in training_data])
        support_y = np.array([list(self.patterns.keys()).index(data[1])
                             for data in training_data])

        query_x = np.vstack([data[0] for data in validation_data])
        query_y = np.array([list(self.patterns.keys()).index(data[1])
                           for data in validation_data])

        task = MAMLTask("pattern_recognition", "classification",
                        (support_x, support_y), (query_x, query_y))

        result = self.maml.adapt_to_task(task)

        for pattern_id in self.patterns:
            self.pattern_confidence[pattern_id] = min(1.0,
                self.pattern_confidence[pattern_id] + 0.1 * result.get_best_metrics()["accuracy"])

        return {
            "adaptation_result": result.to_dict(),
            "updated_confidences": self.pattern_confidence
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_patterns": len(self.patterns),
            "feature_dim": self.feature_dim,
            "maml_mode": self.maml.mode.value,
            "hp_summary": self.hp_controller.get_parameter_summary()
        }


class PerceptionMetaIntegration:
    def __init__(self, input_dim: int = 256, feature_dim: int = 128):
        self.feature_extractor = MetaEnhancedFeatureExtractor(input_dim, feature_dim)
        self.pattern_recognizer = MetaEnhancedPatternRecognizer(feature_dim=feature_dim)
        self.integration_history: List[Dict[str, Any]] = []

    def process(self, data: np.ndarray, task_context: Dict[str, Any] = None) -> Dict[str, Any]:
        features = self.feature_extractor.extract_features(data, task_context)
        patterns = self.pattern_recognizer.recognize_pattern(features)

        result = {
            "features": features.tolist(),
            "recognized_patterns": patterns,
            "feature_dim": features.shape[-1],
            "timestamp": np.random.randint(1000000)
        }

        self.integration_history.append(result)

        return result

    def adapt(self, support_data: Tuple[np.ndarray, np.ndarray],
              query_data: Tuple[np.ndarray, np.ndarray],
              task_id: str = "perception_adapt") -> Dict[str, Any]:
        feature_result = self.feature_extractor.adapt_to_task(support_data, query_data, task_id)

        return {
            "feature_extractor": feature_result,
            "pattern_recognizer": self.pattern_recognizer.get_status(),
            "overall_status": self.get_status()
        }

    def register_task(self, task_id: str, task_type: str, complexity: str,
                      data_distribution: str, input_dim: int, output_dim: int, num_samples: int):
        from ..meta_learning.task_strategy_knowledge import TaskType, TaskComplexity, DataDistribution
        self.feature_extractor.register_task(
            task_id,
            TaskType(task_type),
            TaskComplexity(complexity),
            DataDistribution(data_distribution),
            input_dim, output_dim, num_samples
        )

    def get_strategy_recommendation(self, task_id: str) -> List[Dict[str, Any]]:
        return self.feature_extractor.recommend_strategy(task_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "feature_extractor": self.feature_extractor.get_status(),
            "pattern_recognizer": self.pattern_recognizer.get_status(),
            "integration_history_count": len(self.integration_history)
        }