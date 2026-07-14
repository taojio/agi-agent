import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from .meta_learner import MetaLearner, MetaLearningTask, MetaLearningResult, MetaLearningMode, TaskEmbedding
from .strategy_optimizer import LearningStrategyOptimizer, LearningStrategy, OptimizationResult
from .task_adaptation import TaskAdaptationEngine, TaskDescriptor, AdaptationStrategy
from .meta_knowledge_base import MetaKnowledgeBase, MetaRule, KnowledgeTransfer


class MetaLearningOrchestrator:
    def __init__(self):
        self.meta_learner = MetaLearner()
        self.strategy_optimizer = LearningStrategyOptimizer()
        self.task_adaptation = TaskAdaptationEngine()
        self.meta_knowledge_base = MetaKnowledgeBase()
        self._learning_contexts: Dict[str, Dict[str, Any]] = {}
        self._orchestration_history: deque = deque(maxlen=200)

    def register_task(self, task_id: str, task_type: str,
                     data_samples: List[Dict[str, Any]],
                     meta_context: Dict[str, Any] = None) -> Dict[str, Any]:
        meta_task = MetaLearningTask(task_id, task_type, data_samples, meta_context)
        embedding = self.meta_learner.register_task(meta_task)

        descriptor = TaskDescriptor(
            task_id=task_id,
            task_type=task_type,
            input_dim=len(data_samples[0].get("features", [])) if data_samples else 0,
            sample_count=len(data_samples)
        )
        self.task_adaptation.register_task(descriptor)

        self._learning_contexts[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "meta_task": meta_task,
            "descriptor": descriptor,
            "embedding": embedding,
            "status": "registered",
            "history": []
        }

        return {
            "task_id": task_id,
            "status": "registered",
            "embedding_dim": embedding.embedding.shape[0],
            "sample_count": len(data_samples)
        }

    def adapt_to_task(self, task_id: str, num_inner_iterations: int = 5,
                     learning_rate: float = 0.01,
                     strategy: LearningStrategy = None) -> Dict[str, Any]:
        if task_id not in self._learning_contexts:
            return {"error": "Task not found"}

        context = self._learning_contexts[task_id]
        context["status"] = "adapting"

        meta_task = context["meta_task"]

        if strategy:
            self.strategy_optimizer.register_strategy(strategy, meta_task.task_type)

        adaptation_result = self.meta_learner.adapt_to_task(
            meta_task, num_inner_iterations, learning_rate
        )

        transfer_result = self.task_adaptation.adapt(
            context["descriptor"],
            strategy=AdaptationStrategy.DIRECT_TRANSFER
        )

        context["history"].append({
            "type": "adaptation",
            "result": adaptation_result.to_dict(),
            "transfer": transfer_result.to_dict() if transfer_result else {}
        })
        context["status"] = "adapted"

        self._orchestration_history.append({
            "task_id": task_id,
            "type": "adaptation",
            "result": adaptation_result.to_dict(),
            "timestamp": np.random.randint(1000000)
        })

        return {
            "task_id": task_id,
            "status": "completed",
            "adaptation_result": adaptation_result.to_dict(),
            "transfer_result": transfer_result.to_dict() if transfer_result else {},
            "meta_knowledge": self.meta_learner.get_meta_knowledge()
        }

    def optimize_strategy(self, task_type: str, num_trials: int = 20) -> Dict[str, Any]:
        recommendation = self.strategy_optimizer.get_strategy_recommendation(task_type)

        strategy = LearningStrategy(recommendation["recommended_strategy"])

        opt_result = self.strategy_optimizer.optimize_hyperparameters(
            strategy, task_type, num_trials
        )

        self._orchestration_history.append({
            "task_type": task_type,
            "type": "strategy_optimization",
            "result": opt_result.to_dict(),
            "recommendation": recommendation,
            "timestamp": np.random.randint(1000000)
        })

        return {
            "task_type": task_type,
            "recommended_strategy": recommendation,
            "optimization_result": opt_result.to_dict(),
            "performance_summary": self.strategy_optimizer.get_performance_summary()
        }

    def transfer_knowledge(self, source_task_id: str, target_task_id: str) -> Dict[str, Any]:
        if source_task_id not in self._learning_contexts or target_task_id not in self._learning_contexts:
            return {"error": "Task not found"}

        transfer_result = self.meta_learner.transfer_knowledge(source_task_id, target_task_id)

        if transfer_result["success"]:
            source_context = self._learning_contexts[source_task_id]
            target_context = self._learning_contexts[target_task_id]

            knowledge_transfer = KnowledgeTransfer(
                source_task_id=source_task_id,
                target_task_id=target_task_id,
                similarity=transfer_result["similarity"],
                effectiveness=transfer_result["transfer_effectiveness"],
                transferred_rules=[]
            )
            self.meta_knowledge_base.add_transfer(knowledge_transfer)

            self._orchestration_history.append({
                "type": "knowledge_transfer",
                "source": source_task_id,
                "target": target_task_id,
                "result": transfer_result,
                "timestamp": np.random.randint(1000000)
            })

        return transfer_result

    def add_meta_rule(self, rule_id: str, condition: str,
                     action: str, rule_type: str = "heuristic") -> Dict[str, Any]:
        from .meta_knowledge_base import RuleType
        rule_type_enum = RuleType(rule_type) if isinstance(rule_type, str) else rule_type
        rule = MetaRule(rule_id=rule_id, condition=condition, action=action, rule_type=rule_type_enum)
        self.meta_knowledge_base.add_rule(rule)

        return {"status": "added", "rule_id": rule_id}

    def apply_meta_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.meta_knowledge_base.apply_rules(context)

    def consolidate_knowledge(self) -> Dict[str, Any]:
        consolidation = self.meta_knowledge_base.consolidate_knowledge()

        self._orchestration_history.append({
            "type": "knowledge_consolidation",
            "result": consolidation.to_dict(),
            "timestamp": np.random.randint(1000000)
        })

        return consolidation.to_dict()

    def run_meta_training(self, tasks: List[MetaLearningTask],
                         num_meta_iterations: int = 100) -> Dict[str, Any]:
        for task in tasks:
            if task.task_id not in self._learning_contexts:
                self.register_task(task.task_id, task.task_type, task.data_samples, task.meta_context)

        result = self.meta_learner.meta_train(tasks, num_meta_iterations)

        self._orchestration_history.append({
            "type": "meta_training",
            "result": result,
            "timestamp": np.random.randint(1000000)
        })

        return {
            "meta_training": result,
            "meta_knowledge": self.meta_learner.get_meta_knowledge(),
            "strategy_summary": self.strategy_optimizer.get_performance_summary()
        }

    def set_meta_learning_mode(self, mode: MetaLearningMode):
        self.meta_learner.set_mode(mode)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        if task_id not in self._learning_contexts:
            return None

        context = self._learning_contexts[task_id]
        return {
            "task_id": task_id,
            "task_type": context["task_type"],
            "status": context["status"],
            "history_count": len(context["history"]),
            "has_embedding": context["embedding"] is not None
        }

    def get_overview(self) -> Dict[str, Any]:
        return {
            "meta_learner": self.meta_learner.get_meta_knowledge(),
            "strategy_optimizer": self.strategy_optimizer.get_performance_summary(),
            "task_adaptation": self.task_adaptation.get_adaptation_summary() if hasattr(self.task_adaptation, 'get_adaptation_summary') else {},
            "meta_knowledge_base": self.meta_knowledge_base.get_summary(),
            "registered_tasks": len(self._learning_contexts),
            "total_orchestrations": len(self._orchestration_history)
        }

    def get_recent_orchestrations(self, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(self._orchestration_history)[-limit:]
        return recent[::-1]

    def get_strategy_recommendation(self, task_type: str,
                                   task_complexity: float = 0.5) -> Dict[str, Any]:
        return self.strategy_optimizer.get_strategy_recommendation(task_type, task_complexity)
