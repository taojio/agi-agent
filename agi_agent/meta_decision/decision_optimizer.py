import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class OptimizationStrategy(Enum):
    GREEDY = "greedy"
    RANDOM_SEARCH = "random_search"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC = "genetic"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MULTI_CRITERIA = "multi_criteria"


class OptimizationResult:
    def __init__(self, strategy: OptimizationStrategy, decision_id: str):
        self.strategy = strategy
        self.decision_id = decision_id
        self.optimized_params: Dict[str, Any] = {}
        self.improvement: float = 0.0
        self.original_score: float = 0.0
        self.optimized_score: float = 0.0
        self.iterations: int = 0
        self.converged: bool = False
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "decision_id": self.decision_id,
            "optimized_params": self.optimized_params,
            "improvement": self.improvement,
            "original_score": self.original_score,
            "optimized_score": self.optimized_score,
            "iterations": self.iterations,
            "converged": self.converged,
            "timestamp": self.timestamp
        }


class DecisionStrategySelector:
    def __init__(self):
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.strategy_usage: Dict[str, int] = {}

    def select_strategy(self, decision_type: str, context: Dict[str, Any]) -> OptimizationStrategy:
        strategies = list(OptimizationStrategy)
        
        if decision_type in self.strategy_performance:
            perf = self.strategy_performance[decision_type]
            best_strategy = max(perf.keys(), key=lambda s: perf[s])
            
            if np.random.random() > 0.2:
                return OptimizationStrategy(best_strategy)
        
        if context.get("time_constraint", False):
            return OptimizationStrategy.GREEDY
        
        if context.get("complex", False):
            return OptimizationStrategy.MULTI_CRITERIA
        
        return OptimizationStrategy.RANDOM_SEARCH

    def update_strategy_performance(self, strategy: OptimizationStrategy,
                                    decision_type: str, score: float):
        if decision_type not in self.strategy_performance:
            self.strategy_performance[decision_type] = {}
        
        key = strategy.value
        if key not in self.strategy_performance[decision_type]:
            self.strategy_performance[decision_type][key] = []
        
        self.strategy_performance[decision_type][key].append(score)
        
        if len(self.strategy_performance[decision_type][key]) > 10:
            self.strategy_performance[decision_type][key] = \
                self.strategy_performance[decision_type][key][-10:]
        
        self.strategy_performance[decision_type][key] = float(np.mean(
            self.strategy_performance[decision_type][key]))
        
        self.strategy_usage[key] = self.strategy_usage.get(key, 0) + 1

    def get_strategy_ranking(self, decision_type: str) -> List[Dict[str, Any]]:
        if decision_type not in self.strategy_performance:
            return []
        
        perf = self.strategy_performance[decision_type]
        ranking = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        
        return [{"strategy": s, "avg_score": score} for s, score in ranking]


class DecisionOptimizer:
    def __init__(self):
        self.strategy_selector = DecisionStrategySelector()
        self.optimization_history: deque = deque(maxlen=200)
        self._evaluation_cache: Dict[str, float] = {}

    def optimize(self, decision_type: str, initial_params: Dict[str, Any],
                 objective_function: Callable[[Dict[str, Any]], float],
                 context: Dict[str, Any] = None) -> OptimizationResult:
        context = context or {}
        
        strategy = self.strategy_selector.select_strategy(decision_type, context)
        
        original_score = objective_function(initial_params)
        
        if strategy == OptimizationStrategy.GREEDY:
            result = self._greedy_optimize(initial_params, objective_function)
        elif strategy == OptimizationStrategy.RANDOM_SEARCH:
            result = self._random_search_optimize(initial_params, objective_function)
        elif strategy == OptimizationStrategy.SIMULATED_ANNEALING:
            result = self._simulated_annealing_optimize(initial_params, objective_function)
        elif strategy == OptimizationStrategy.MULTI_CRITERIA:
            result = self._multi_criteria_optimize(initial_params, objective_function)
        else:
            result = self._random_search_optimize(initial_params, objective_function)
        
        result.strategy = strategy
        result.original_score = original_score
        result.improvement = result.optimized_score - original_score
        
        self.strategy_selector.update_strategy_performance(
            strategy, decision_type, result.optimized_score)
        
        self.optimization_history.append(result)
        
        return result

    def _greedy_optimize(self, initial_params: Dict[str, Any],
                         objective_function: Callable[[Dict[str, Any]], float],
                         max_iterations: int = 50) -> OptimizationResult:
        result = OptimizationResult(OptimizationStrategy.GREEDY, "greedy_001")
        current_params = initial_params.copy()
        current_score = objective_function(current_params)
        
        for i in range(max_iterations):
            improved = False
            
            for key in current_params.keys():
                original_value = current_params[key]
                
                if isinstance(original_value, float):
                    for delta in [-0.1, 0.1, -0.05, 0.05]:
                        current_params[key] = original_value + delta
                        new_score = objective_function(current_params)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            break
                    
                    if not improved:
                        current_params[key] = original_value
            
            if not improved:
                result.converged = True
                break
            
            result.iterations = i + 1
        
        result.optimized_params = current_params
        result.optimized_score = current_score
        
        return result

    def _random_search_optimize(self, initial_params: Dict[str, Any],
                                objective_function: Callable[[Dict[str, Any]], float],
                                num_trials: int = 100) -> OptimizationResult:
        result = OptimizationResult(OptimizationStrategy.RANDOM_SEARCH, "rs_001")
        best_params = initial_params.copy()
        best_score = objective_function(initial_params)
        
        for i in range(num_trials):
            candidate_params = {}
            for key, value in initial_params.items():
                if isinstance(value, float):
                    candidate_params[key] = value + np.random.uniform(-0.5, 0.5)
                elif isinstance(value, int):
                    candidate_params[key] = value + np.random.randint(-5, 6)
                else:
                    candidate_params[key] = value
            
            score = objective_function(candidate_params)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
            
            result.iterations = i + 1
        
        result.optimized_params = best_params
        result.optimized_score = best_score
        result.converged = True
        
        return result

    def _simulated_annealing_optimize(self, initial_params: Dict[str, Any],
                                      objective_function: Callable[[Dict[str, Any]], float],
                                      max_iterations: int = 100) -> OptimizationResult:
        result = OptimizationResult(OptimizationStrategy.SIMULATED_ANNEALING, "sa_001")
        
        current_params = initial_params.copy()
        current_score = objective_function(current_params)
        best_params = current_params.copy()
        best_score = current_score
        
        temperature = 1.0
        cooling_rate = 0.95
        
        for i in range(max_iterations):
            candidate_params = {}
            for key, value in current_params.items():
                if isinstance(value, float):
                    candidate_params[key] = value + np.random.normal(0, 0.1)
                elif isinstance(value, int):
                    candidate_params[key] = value + np.random.randint(-2, 3)
                else:
                    candidate_params[key] = value
            
            candidate_score = objective_function(candidate_params)
            delta = candidate_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_params = candidate_params
                current_score = candidate_score
                
                if current_score > best_score:
                    best_params = current_params.copy()
                    best_score = current_score
            
            temperature *= cooling_rate
            result.iterations = i + 1
            
            if temperature < 0.01:
                result.converged = True
                break
        
        result.optimized_params = best_params
        result.optimized_score = best_score
        
        return result

    def _multi_criteria_optimize(self, initial_params: Dict[str, Any],
                                 objective_function: Callable[[Dict[str, Any]], float],
                                 num_objectives: int = 3) -> OptimizationResult:
        result = OptimizationResult(OptimizationStrategy.MULTI_CRITERIA, "mc_001")
        
        best_params = initial_params.copy()
        best_score = objective_function(initial_params)
        
        for i in range(50):
            weights = np.random.dirichlet(np.ones(num_objectives))
            
            def weighted_objective(params):
                base_score = objective_function(params)
                diversity = np.random.uniform(0.7, 1.3)
                robustness = np.random.uniform(0.8, 1.2)
                return weights[0] * base_score + weights[1] * diversity + weights[2] * robustness
            
            candidate_params = {}
            for key, value in initial_params.items():
                if isinstance(value, float):
                    candidate_params[key] = value + np.random.uniform(-0.3, 0.3)
                else:
                    candidate_params[key] = value
            
            score = weighted_objective(candidate_params)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
            
            result.iterations = i + 1
        
        result.optimized_params = best_params
        result.optimized_score = best_score
        result.converged = True
        
        return result

    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        recent = list(self.optimization_history)[-limit:]
        return [r.to_dict() for r in recent]

    def get_optimization_summary(self) -> Dict[str, Any]:
        if not self.optimization_history:
            return {"total_optimizations": 0, "avg_improvement": 0.0}
        
        results = list(self.optimization_history)
        avg_improvement = np.mean([r.improvement for r in results])
        avg_iterations = np.mean([r.iterations for r in results])
        convergence_rate = len([r for r in results if r.converged]) / len(results)
        
        return {
            "total_optimizations": len(results),
            "avg_improvement": float(avg_improvement),
            "avg_iterations": float(avg_iterations),
            "convergence_rate": convergence_rate,
            "strategies_used": self.strategy_selector.strategy_usage
        }