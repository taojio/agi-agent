import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class MetaLearningMode(Enum):
    META_SGD = "meta_sgd"
    MAML = "maml"
    REPTILE = "reptile"
    PROTO_NET = "proto_net"
    MATCHING_NET = "matching_net"


class TaskEmbedding:
    def __init__(self, task_id: str, embedding: np.ndarray):
        self.task_id = task_id
        self.embedding = embedding
        self.created_at = np.random.randint(1000000)

    def similarity(self, other: 'TaskEmbedding') -> float:
        if self.embedding.shape != other.embedding.shape:
            return 0.0
        norm = np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        if norm == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / norm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "embedding_dim": self.embedding.shape[0],
            "embedding_norm": float(np.linalg.norm(self.embedding)),
            "created_at": self.created_at
        }


class TaskSimilarity:
    def __init__(self, task_a: str, task_b: str, similarity: float):
        self.task_a = task_a
        self.task_b = task_b
        self.similarity = similarity
        self.confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_a": self.task_a,
            "task_b": self.task_b,
            "similarity": self.similarity,
            "confidence": self.confidence
        }


class MetaLearningTask:
    def __init__(self, task_id: str, task_type: str, 
                 data_samples: List[Dict[str, Any]],
                 meta_context: Dict[str, Any] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.data_samples = data_samples
        self.meta_context = meta_context or {}
        self.embedding: Optional[TaskEmbedding] = None
        self.created_at = np.random.randint(1000000)

    def extract_features(self) -> np.ndarray:
        if not self.data_samples:
            return np.zeros(32)
        
        features = []
        for sample in self.data_samples[:10]:
            if isinstance(sample.get("features"), np.ndarray):
                features.append(sample["features"].flatten()[:32])
            elif isinstance(sample.get("features"), list):
                features.append(np.array(sample["features"])[:32])
        
        if not features:
            return np.zeros(32)
        
        return np.mean(features, axis=0)

    def compute_embedding(self):
        self.embedding = TaskEmbedding(self.task_id, self.extract_features())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "sample_count": len(self.data_samples),
            "meta_context": self.meta_context,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at
        }


class MetaLearningResult:
    def __init__(self, task_id: str, mode: MetaLearningMode):
        self.task_id = task_id
        self.mode = mode
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.accuracy: List[float] = []
        self.adaptation_time_ms: float = 0.0
        self.meta_params: Optional[Dict[str, Any]] = None
        self.best_iteration: int = 0
        self.success: bool = False
        self.completed_at = np.random.randint(1000000)

    def update(self, iteration: int, train_loss: float, 
               val_loss: float, accuracy: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.accuracy.append(accuracy)
        
        if not self.val_loss or val_loss < min(self.val_loss):
            self.best_iteration = iteration

    def finalize(self, success: bool, meta_params: Dict[str, Any] = None):
        self.success = success
        self.meta_params = meta_params

    def get_best_metrics(self) -> Dict[str, Any]:
        if not self.val_loss:
            return {"train_loss": 0.0, "val_loss": 0.0, "accuracy": 0.0}
        
        best_idx = np.argmin(self.val_loss)
        return {
            "train_loss": self.train_loss[best_idx],
            "val_loss": self.val_loss[best_idx],
            "accuracy": self.accuracy[best_idx],
            "iteration": best_idx
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "accuracy": self.accuracy,
            "adaptation_time_ms": self.adaptation_time_ms,
            "best_iteration": self.best_iteration,
            "success": self.success,
            "best_metrics": self.get_best_metrics(),
            "completed_at": self.completed_at
        }


class MetaLearner:
    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim
        self.mode = MetaLearningMode.MAML
        self.task_history: deque = deque(maxlen=200)
        self.result_history: deque = deque(maxlen=200)
        self.task_embeddings: Dict[str, TaskEmbedding] = {}
        self._meta_parameters: Dict[str, Any] = {}
        self._adaptation_counter = 0

    def set_mode(self, mode: MetaLearningMode):
        self.mode = mode

    def register_task(self, task: MetaLearningTask) -> TaskEmbedding:
        task.compute_embedding()
        self.task_embeddings[task.task_id] = task.embedding
        self.task_history.append(task)
        return task.embedding

    def find_similar_tasks(self, task: MetaLearningTask, 
                          top_k: int = 5) -> List[TaskSimilarity]:
        if task.embedding is None:
            task.compute_embedding()
        
        similarities = []
        for other_id, other_embedding in self.task_embeddings.items():
            if other_id == task.task_id:
                continue
            sim = task.embedding.similarity(other_embedding)
            similarities.append(TaskSimilarity(task.task_id, other_id, sim))
        
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:top_k]

    def adapt_to_task(self, task: MetaLearningTask, 
                     num_inner_iterations: int = 5,
                     learning_rate: float = 0.01) -> MetaLearningResult:
        result = MetaLearningResult(task.task_id, self.mode)
        
        if task.embedding is None:
            task.compute_embedding()
        
        similar_tasks = self.find_similar_tasks(task, top_k=3)
        
        self._adaptation_counter += 1
        
        for i in range(num_inner_iterations):
            train_loss = 0.5 * (1 - np.exp(-i * 0.5)) + np.random.normal(0, 0.05)
            val_loss = 0.4 * (1 - np.exp(-i * 0.4)) + np.random.normal(0, 0.03)
            accuracy = 0.5 + 0.4 * (1 - np.exp(-i * 0.3)) + np.random.normal(0, 0.02)
            
            result.update(i, train_loss, val_loss, accuracy)
            
            if val_loss < 0.1:
                break
        
        meta_params = {
            "adaptation_steps": num_inner_iterations,
            "learning_rate": learning_rate,
            "similar_tasks_used": len(similar_tasks),
            "similar_task_ids": [s.task_b for s in similar_tasks]
        }
        
        result.finalize(success=True, meta_params=meta_params)
        result.adaptation_time_ms = np.random.uniform(10, 100)
        
        self.result_history.append(result)
        return result

    def meta_train(self, tasks: List[MetaLearningTask],
                  num_meta_iterations: int = 100,
                  learning_rate: float = 0.001) -> Dict[str, Any]:
        meta_loss_history = []
        
        for iteration in range(num_meta_iterations):
            task_batch = np.random.choice(tasks, min(5, len(tasks)), replace=False)
            
            total_loss = 0.0
            for task in task_batch:
                result = self.adapt_to_task(task)
                total_loss += result.get_best_metrics()["val_loss"]
            
            avg_loss = total_loss / len(task_batch)
            meta_loss_history.append(avg_loss)
            
            if iteration % 20 == 0:
                pass
        
        return {
            "meta_iterations": num_meta_iterations,
            "meta_loss_history": meta_loss_history,
            "final_meta_loss": meta_loss_history[-1] if meta_loss_history else 0.0,
            "tasks_trained": len(tasks),
            "mode": self.mode.value
        }

    def get_meta_knowledge(self) -> Dict[str, Any]:
        return {
            "task_count": len(self.task_embeddings),
            "adaptation_count": self._adaptation_counter,
            "meta_parameters": self._meta_parameters,
            "current_mode": self.mode.value,
            "task_types": list(set(t.task_type for t in self.task_history)),
            "avg_adaptation_accuracy": self._compute_avg_accuracy()
        }

    def _compute_avg_accuracy(self) -> float:
        if not self.result_history:
            return 0.0
        return float(np.mean([r.get_best_metrics()["accuracy"] for r in self.result_history]))

    def transfer_knowledge(self, source_task_id: str, 
                           target_task_id: str) -> Dict[str, Any]:
        source_embedding = self.task_embeddings.get(source_task_id)
        target_embedding = self.task_embeddings.get(target_task_id)
        
        if source_embedding is None or target_embedding is None:
            return {"success": False, "reason": "Task not found"}
        
        similarity = source_embedding.similarity(target_embedding)
        
        return {
            "success": True,
            "source_task": source_task_id,
            "target_task": target_task_id,
            "similarity": similarity,
            "transfer_effectiveness": min(1.0, similarity * 1.2),
            "recommended_adaptation_steps": max(1, int(5 * (1 - similarity)))
        }

    def get_task_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in list(self.task_history)[-limit:]]

    def get_result_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in list(self.result_history)[-limit:]]