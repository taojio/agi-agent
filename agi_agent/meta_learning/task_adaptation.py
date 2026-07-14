import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class AdaptationStrategy(Enum):
    DIRECT_TRANSFER = "direct_transfer"
    FINE_TUNE = "fine_tune"
    FEATURE_EXTRACTION = "feature_extraction"
    METRIC_LEARNING = "metric_learning"
    META_ADAPTATION = "meta_adaptation"
    HYBRID = "hybrid"


class TaskDescriptor:
    def __init__(self, task_id: str, task_type: str,
                 input_dim: int, output_dim: int,
                 num_samples: int = 0, difficulty: float = 0.5):
        self.task_id = task_id
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.characteristics: Dict[str, Any] = {}

    def add_characteristic(self, key: str, value: Any):
        self.characteristics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_samples": self.num_samples,
            "difficulty": self.difficulty,
            "characteristics": self.characteristics
        }


class TransferLearningConfig:
    def __init__(self):
        self.source_task_id: Optional[str] = None
        self.target_task_id: Optional[str] = None
        self.adaptation_strategy = AdaptationStrategy.FINE_TUNE
        self.trainable_layers: List[str] = []
        self.freeze_layers: List[str] = []
        self.learning_rate: float = 0.001
        self.num_epochs: int = 10
        self.batch_size: int = 32
        self.use_augmentation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "adaptation_strategy": self.adaptation_strategy.value,
            "trainable_layers": self.trainable_layers,
            "freeze_layers": self.freeze_layers,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_augmentation": self.use_augmentation
        }


class AdaptationResult:
    def __init__(self, task_id: str, strategy: AdaptationStrategy):
        self.task_id = task_id
        self.strategy = strategy
        self.initial_accuracy: float = 0.0
        self.final_accuracy: float = 0.0
        self.accuracy_improvement: float = 0.0
        self.adaptation_time_ms: float = 0.0
        self.num_samples_used: int = 0
        self.success: bool = False
        self.completed_at = np.random.randint(1000000)

    def calculate_improvement(self):
        self.accuracy_improvement = self.final_accuracy - self.initial_accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "accuracy_improvement": self.accuracy_improvement,
            "adaptation_time_ms": self.adaptation_time_ms,
            "num_samples_used": self.num_samples_used,
            "success": self.success,
            "completed_at": self.completed_at
        }


class TaskAdaptationEngine:
    def __init__(self):
        self.task_descriptors: Dict[str, TaskDescriptor] = {}
        self.adaptation_history: deque = deque(maxlen=200)
        self._adaptation_strategies: Dict[str, AdaptationStrategy] = {}

    def register_task(self, descriptor: TaskDescriptor):
        self.task_descriptors[descriptor.task_id] = descriptor

    def create_task_descriptor(self, task_id: str, task_type: str,
                               input_dim: int, output_dim: int,
                               num_samples: int = 0, difficulty: float = 0.5) -> TaskDescriptor:
        descriptor = TaskDescriptor(task_id, task_type, input_dim, output_dim, num_samples, difficulty)
        self.register_task(descriptor)
        return descriptor

    def select_adaptation_strategy(self, source_task_id: str,
                                   target_task_id: str) -> AdaptationStrategy:
        source = self.task_descriptors.get(source_task_id)
        target = self.task_descriptors.get(target_task_id)
        
        if source is None or target is None:
            return AdaptationStrategy.FINE_TUNE
        
        if source.task_type == target.task_type:
            if source.input_dim == target.input_dim and source.output_dim == target.output_dim:
                return AdaptationStrategy.DIRECT_TRANSFER
            return AdaptationStrategy.FINE_TUNE
        
        if target.num_samples < 10:
            return AdaptationStrategy.META_ADAPTATION
        
        if target.difficulty > 0.7:
            return AdaptationStrategy.HYBRID
        
        return AdaptationStrategy.FEATURE_EXTRACTION

    def adapt(self, source_task_id: str, target_task_id: str,
              config: TransferLearningConfig = None) -> AdaptationResult:
        if config is None:
            config = TransferLearningConfig()
        
        strategy = self.select_adaptation_strategy(source_task_id, target_task_id)
        result = AdaptationResult(target_task_id, strategy)
        
        source = self.task_descriptors.get(source_task_id)
        target = self.task_descriptors.get(target_task_id)
        
        if source and target:
            similarity = self._compute_task_similarity(source, target)
            result.initial_accuracy = 0.3 + similarity * 0.2
            result.final_accuracy = min(0.95, result.initial_accuracy + 0.3 * (1 + similarity))
            result.num_samples_used = target.num_samples
        else:
            result.initial_accuracy = 0.2
            result.final_accuracy = 0.5
        
        result.calculate_improvement()
        result.adaptation_time_ms = np.random.uniform(50, 500)
        result.success = result.accuracy_improvement > 0.1
        
        self.adaptation_history.append(result)
        return result

    def _compute_task_similarity(self, source: TaskDescriptor,
                                target: TaskDescriptor) -> float:
        type_sim = 1.0 if source.task_type == target.task_type else 0.5
        dim_sim = 1.0 if source.input_dim == target.input_dim else 0.7
        diff_sim = 1.0 - abs(source.difficulty - target.difficulty)
        
        return (type_sim * 0.4 + dim_sim * 0.3 + diff_sim * 0.3)

    def batch_adapt(self, tasks: List[Dict[str, Any]]) -> List[AdaptationResult]:
        results = []
        
        for task in tasks:
            source_id = task.get("source_task_id")
            target_id = task.get("target_task_id")
            
            if source_id and target_id:
                result = self.adapt(source_id, target_id)
                results.append(result)
        
        return results

    def optimize_adaptation(self, task_id: str) -> Dict[str, Any]:
        target = self.task_descriptors.get(task_id)
        if target is None:
            return {"error": "Task not found"}
        
        source_candidates = []
        for source_id, source in self.task_descriptors.items():
            if source_id == task_id:
                continue
            similarity = self._compute_task_similarity(source, target)
            source_candidates.append({"source_id": source_id, "similarity": similarity})
        
        source_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        
        if source_candidates:
            best_source = source_candidates[0]
            strategy = self.select_adaptation_strategy(best_source["source_id"], task_id)
            
            return {
                "task_id": task_id,
                "best_source_task": best_source["source_id"],
                "similarity": best_source["similarity"],
                "recommended_strategy": strategy.value,
                "expected_improvement": min(0.4, best_source["similarity"] * 0.5),
                "alternative_sources": source_candidates[1:3]
            }
        
        return {
            "task_id": task_id,
            "best_source_task": None,
            "recommended_strategy": AdaptationStrategy.META_ADAPTATION.value,
            "expected_improvement": 0.2
        }

    def get_adaptation_summary(self) -> Dict[str, Any]:
        results = list(self.adaptation_history)
        
        if not results:
            return {"total_adaptations": 0, "avg_improvement": 0.0}
        
        successful = [r for r in results if r.success]
        avg_improvement = np.mean([r.accuracy_improvement for r in results])
        
        strategy_counts = {}
        for r in results:
            key = r.strategy.value
            strategy_counts[key] = strategy_counts.get(key, 0) + 1
        
        return {
            "total_adaptations": len(results),
            "successful_adaptations": len(successful),
            "success_rate": len(successful) / len(results),
            "avg_improvement": float(avg_improvement),
            "strategy_distribution": strategy_counts,
            "tasks_registered": len(self.task_descriptors)
        }

    def get_task_descriptor(self, task_id: str) -> Optional[Dict[str, Any]]:
        descriptor = self.task_descriptors.get(task_id)
        return descriptor.to_dict() if descriptor else None

    def get_adaptation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in list(self.adaptation_history)[-limit:]]