import numpy as np
from collections import deque
from enum import Enum


class LearningStrategy(Enum):
    DECISION_TREE = "decision_tree"
    SNN_LEARNING = "snn_learning"
    SKILL固化 = "skill固化"
    RULE_INDUCTION = "rule_induction"
    ANALOGICAL_TRANSFER = "analogical_transfer"


class StrategyOptimizer:
    def __init__(self):
        self.strategy_performance = {}
        self.tuning_parameters = {
            "fast_slow_threshold": 0.85,
            "hypothesis_count": 5,
            "simulation_depth": 10,
            "reflection_interval": 50
        }
        self.parameter_history = deque(maxlen=100)

    def update_strategy_performance(self, strategy_type, success, context):
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = {"success": 0, "total": 0, "contexts": []}
        
        self.strategy_performance[strategy_type]["total"] += 1
        if success:
            self.strategy_performance[strategy_type]["success"] += 1
        
        self.strategy_performance[strategy_type]["contexts"].append(context)
        if len(self.strategy_performance[strategy_type]["contexts"]) > 50:
            self.strategy_performance[strategy_type]["contexts"] = self.strategy_performance[strategy_type]["contexts"][-50:]

    def get_best_strategy(self, task_type):
        best_strategy = LearningStrategy.RULE_INDUCTION
        best_score = 0.0

        for strategy_type in LearningStrategy:
            perf = self.strategy_performance.get(strategy_type.value, {"success": 0, "total": 0})
            if perf["total"] > 0:
                score = perf["success"] / perf["total"]
                if task_type == "sequential":
                    score *= 1.2
                elif task_type == "logical":
                    score *= 1.1
                elif task_type == "pattern":
                    score *= 1.15
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_type

        return best_strategy, best_score

    def tune_parameter(self, param_name, value):
        if param_name in self.tuning_parameters:
            old_value = self.tuning_parameters[param_name]
            self.tuning_parameters[param_name] = value
            
            self.parameter_history.append({
                "parameter": param_name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": np.random.randint(1000000)
            })

    def get_parameters(self):
        return self.tuning_parameters

    def get_strategy_stats(self):
        stats = {}
        for strategy_type in LearningStrategy:
            perf = self.strategy_performance.get(strategy_type.value, {"success": 0, "total": 0})
            stats[strategy_type.value] = {
                "success_rate": perf["success"] / perf["total"] if perf["total"] > 0 else 0.5,
                "usage_count": perf["total"]
            }
        return stats


class MetaLearningEngine:
    def __init__(self):
        self.strategy_optimizer = StrategyOptimizer()
        self.learning_history = deque(maxlen=200)
        self.transfer_knowledge_base = {}
        self.evolution_rule_tuning = {
            "trigger_threshold": 0.5,
            "review_period": 100,
            "validation_standard": 0.8
        }

    def select_learning_strategy(self, task_type, context):
        strategy, score = self.strategy_optimizer.get_best_strategy(task_type)
        
        self.learning_history.append({
            "task_type": task_type,
            "strategy": strategy.value,
            "confidence": score,
            "context": context,
            "timestamp": np.random.randint(1000000)
        })
        
        return strategy, score

    def perform_meta_learning(self, learning_results):
        for result in learning_results:
            strategy_type = result.get("strategy", "unknown")
            success = result.get("success", False)
            context = result.get("context", {})
            
            self.strategy_optimizer.update_strategy_performance(strategy_type, success, context)

    def transfer_learning(self, source_task, target_task, knowledge):
        source_key = str(source_task)
        if source_key not in self.transfer_knowledge_base:
            self.transfer_knowledge_base[source_key] = []
        
        self.transfer_knowledge_base[source_key].append({
            "target_task": target_task,
            "knowledge": knowledge,
            "timestamp": np.random.randint(1000000)
        })

    def find_transferable_knowledge(self, target_task, threshold=0.7):
        matching_knowledge = []
        
        for source_task, knowledge_list in self.transfer_knowledge_base.items():
            similarity = self._compute_task_similarity(source_task, target_task)
            if similarity >= threshold:
                for knowledge in knowledge_list:
                    matching_knowledge.append({
                        "source_task": source_task,
                        "similarity": similarity,
                        "knowledge": knowledge["knowledge"]
                    })
        
        matching_knowledge.sort(key=lambda x: -x["similarity"])
        return matching_knowledge[:3]

    def _compute_task_similarity(self, task1, task2):
        t1 = str(task1).lower()
        t2 = str(task2).lower()
        
        common_words = set(t1.split()) & set(t2.split())
        total_words = set(t1.split()) | set(t2.split())
        
        if not total_words:
            return 0.0
        
        return len(common_words) / len(total_words)

    def optimize_evolution_rules(self, performance_data):
        avg_success = np.mean([p.get("success_rate", 0.5) for p in performance_data])
        
        if avg_success > 0.8:
            self.evolution_rule_tuning["trigger_threshold"] = min(0.7, self.evolution_rule_tuning["trigger_threshold"] + 0.1)
            self.evolution_rule_tuning["review_period"] = max(50, self.evolution_rule_tuning["review_period"] - 20)
        elif avg_success < 0.5:
            self.evolution_rule_tuning["trigger_threshold"] = max(0.3, self.evolution_rule_tuning["trigger_threshold"] - 0.1)
            self.evolution_rule_tuning["review_period"] = min(200, self.evolution_rule_tuning["review_period"] + 20)

    def get_meta_learning_stats(self):
        return {
            "learning_cycles": len(self.learning_history),
            "strategy_stats": self.strategy_optimizer.get_strategy_stats(),
            "tuning_parameters": self.strategy_optimizer.get_parameters(),
            "transfer_knowledge_count": sum(len(v) for v in self.transfer_knowledge_base.values()),
            "evolution_rules": self.evolution_rule_tuning
        }