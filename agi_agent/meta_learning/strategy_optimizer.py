import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LearningStrategy(Enum):
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    SELF_SUPERVISED = "self_supervised"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"


class StrategyPerformance:
    def __init__(self, strategy: LearningStrategy, task_type: str):
        self.strategy = strategy
        self.task_type = task_type
        self.accuracy: List[float] = []
        self.loss: List[float] = []
        self.train_time_ms: List[float] = []
        self.sample_efficiency: List[float] = []
        self.best_accuracy: float = 0.0
        self.best_loss: float = float('inf')

    def update(self, accuracy: float, loss: float, 
               train_time_ms: float, sample_efficiency: float):
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        self.train_time_ms.append(train_time_ms)
        self.sample_efficiency.append(sample_efficiency)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if loss < self.best_loss:
            self.best_loss = loss

    def get_summary(self) -> Dict[str, Any]:
        if not self.accuracy:
            return {}
        
        return {
            "strategy": self.strategy.value,
            "task_type": self.task_type,
            "avg_accuracy": float(np.mean(self.accuracy)),
            "avg_loss": float(np.mean(self.loss)),
            "avg_train_time_ms": float(np.mean(self.train_time_ms)),
            "avg_sample_efficiency": float(np.mean(self.sample_efficiency)),
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "sample_count": len(self.accuracy)
        }


class HyperparameterSpace:
    def __init__(self):
        self.parameters: Dict[str, Dict[str, Any]] = {}

    def add_parameter(self, name: str, param_type: str, 
                      min_value: float, max_value: float,
                      default: float = None):
        self.parameters[name] = {
            "name": name,
            "type": param_type,
            "min": min_value,
            "max": max_value,
            "default": default if default is not None else (min_value + max_value) / 2
        }

    def sample(self) -> Dict[str, float]:
        sample = {}
        for name, param in self.parameters.items():
            if param["type"] == "float":
                sample[name] = np.random.uniform(param["min"], param["max"])
            elif param["type"] == "int":
                sample[name] = int(np.random.randint(param["min"], param["max"] + 1))
            else:
                sample[name] = param["default"]
        return sample

    def get_default(self) -> Dict[str, float]:
        return {name: param["default"] for name, param in self.parameters.items()}

    def to_dict(self) -> Dict[str, Any]:
        return self.parameters


class OptimizationResult:
    def __init__(self):
        self.best_strategy: Optional[LearningStrategy] = None
        self.best_hyperparameters: Dict[str, float] = {}
        self.best_performance: float = 0.0
        self.trials: int = 0
        self.explored_strategies: List[str] = []
        self.optimization_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_strategy": self.best_strategy.value if self.best_strategy else None,
            "best_hyperparameters": self.best_hyperparameters,
            "best_performance": self.best_performance,
            "trials": self.trials,
            "explored_strategies": self.explored_strategies,
            "optimization_time_ms": self.optimization_time_ms
        }


class LearningStrategyOptimizer:
    def __init__(self):
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.hyperparameter_spaces: Dict[str, HyperparameterSpace] = {}
        self.optimization_history: deque = deque(maxlen=200)
        self._exploration_rate = 0.3

    def register_strategy(self, strategy: LearningStrategy,
                         task_type: str = "general"):
        key = f"{strategy.value}_{task_type}"
        if key not in self.strategy_performance:
            self.strategy_performance[key] = StrategyPerformance(strategy, task_type)

    def register_hyperparameter_space(self, strategy: LearningStrategy,
                                     space: HyperparameterSpace):
        self.hyperparameter_spaces[strategy.value] = space

    def record_performance(self, strategy: LearningStrategy, task_type: str,
                          accuracy: float, loss: float,
                          train_time_ms: float, sample_efficiency: float):
        key = f"{strategy.value}_{task_type}"
        if key not in self.strategy_performance:
            self.register_strategy(strategy, task_type)
        
        self.strategy_performance[key].update(accuracy, loss, train_time_ms, sample_efficiency)

    def select_best_strategy(self, task_type: str) -> Dict[str, Any]:
        candidates = {}
        for key, performance in self.strategy_performance.items():
            if key.endswith(f"_{task_type}") or key.endswith("_general"):
                summary = performance.get_summary()
                if summary:
                    score = summary["avg_accuracy"] * 0.6 + summary["avg_sample_efficiency"] * 0.4
                    candidates[key] = {"performance": summary, "score": score}
        
        if not candidates:
            return {"strategy": LearningStrategy.SUPERVISED.value, "reason": "No data available"}
        
        best_key = max(candidates, key=lambda k: candidates[k]["score"])
        best = candidates[best_key]
        
        return {
            "strategy": best_key.split("_")[0],
            "task_type": task_type,
            "score": best["score"],
            "performance": best["performance"]
        }

    def optimize_hyperparameters(self, strategy: LearningStrategy,
                                task_type: str,
                                num_trials: int = 20) -> OptimizationResult:
        result = OptimizationResult()
        result.optimization_time_ms = np.random.uniform(100, 500)
        
        space = self.hyperparameter_spaces.get(strategy.value)
        if space is None:
            space = HyperparameterSpace()
            space.add_parameter("learning_rate", "float", 0.0001, 0.1, 0.001)
            space.add_parameter("batch_size", "int", 8, 128, 32)
            space.add_parameter("epochs", "int", 10, 200, 50)
        
        best_score = -float('inf')
        best_params = {}
        
        for i in range(num_trials):
            if np.random.random() < self._exploration_rate:
                params = space.sample()
            else:
                params = self._exploit_params(space, best_params)
            
            score = self._evaluate_params(strategy, params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            result.trials += 1
        
        result.best_strategy = strategy
        result.best_hyperparameters = best_params
        result.best_performance = best_score
        result.explored_strategies = [strategy.value]
        
        self.optimization_history.append(result.to_dict())
        return result

    def _exploit_params(self, space: HyperparameterSpace,
                       best_params: Dict[str, float]) -> Dict[str, float]:
        if not best_params:
            return space.get_default()
        
        params = {}
        for name, param in space.parameters.items():
            if name in best_params:
                noise = np.random.normal(0, (param["max"] - param["min"]) * 0.1)
                params[name] = max(param["min"], min(param["max"], best_params[name] + noise))
                if param["type"] == "int":
                    params[name] = int(params[name])
            else:
                params[name] = param["default"]
        return params

    def _evaluate_params(self, strategy: LearningStrategy,
                        params: Dict[str, float]) -> float:
        base_score = 0.7
        lr_factor = 1.0 - abs(params.get("learning_rate", 0.001) - 0.01) * 10
        batch_factor = 1.0 - abs(params.get("batch_size", 32) - 32) / 128
        epoch_factor = 1.0 - abs(params.get("epochs", 50) - 50) / 200
        
        return base_score * lr_factor * batch_factor * epoch_factor + np.random.normal(0, 0.05)

    def tune_exploration_rate(self, performance_trend: float):
        if performance_trend < 0:
            self._exploration_rate = min(0.5, self._exploration_rate + 0.05)
        else:
            self._exploration_rate = max(0.1, self._exploration_rate - 0.02)

    def get_strategy_recommendation(self, task_type: str,
                                   task_complexity: float = 0.5) -> Dict[str, Any]:
        best_strategy = self.select_best_strategy(task_type)
        
        if task_complexity > 0.7:
            recommended_strategy = LearningStrategy.SELF_SUPERVISED
        elif task_complexity < 0.3:
            recommended_strategy = LearningStrategy.SUPERVISED
        else:
            recommended_strategy = LearningStrategy.FEW_SHOT
        
        return {
            "recommended_strategy": recommended_strategy.value,
            "best_history_strategy": best_strategy.get("strategy"),
            "task_type": task_type,
            "task_complexity": task_complexity,
            "confidence": min(1.0, best_strategy.get("score", 0.5) * 0.8 + 0.2)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        summaries = {}
        for key, performance in self.strategy_performance.items():
            summary = performance.get_summary()
            if summary:
                summaries[key] = summary
        
        return {
            "strategies_evaluated": len(summaries),
            "optimization_trials": sum(r["trials"] for r in self.optimization_history),
            "performance_by_strategy": summaries,
            "exploration_rate": self._exploration_rate
        }