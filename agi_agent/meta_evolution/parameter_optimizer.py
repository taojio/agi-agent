import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple


class AdaptiveParameter:
    def __init__(self, name: str, value: float, min_value: float = 0.0,
                 max_value: float = 1.0, step: float = 0.01,
                 adaptive: bool = True):
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.adaptive = adaptive
        
        self.history: List[float] = [value]
        self.gradient: float = 0.0
        self.adjustment_count: int = 0
        self.success_count: int = 0

    def adjust(self, gradient: float, learning_rate: float = 0.1):
        if not self.adaptive:
            return
        
        self.gradient = gradient
        
        new_value = self.value + gradient * learning_rate * self.step
        
        new_value = max(self.min_value, min(self.max_value, new_value))
        
        self.value = new_value
        self.history.append(new_value)
        self.adjustment_count += 1

    def record_success(self):
        self.success_count += 1

    def get_stability(self) -> float:
        if len(self.history) < 3:
            return 0.5
        
        recent = self.history[-5:]
        variance = np.var(recent)
        return float(max(0.0, min(1.0, 1.0 - variance * 10)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "adaptive": self.adaptive,
            "gradient": self.gradient,
            "adjustment_count": self.adjustment_count,
            "success_count": self.success_count,
            "stability": self.get_stability(),
            "history_length": len(self.history)
        }


class ParameterSpace:
    def __init__(self):
        self.parameters: Dict[str, AdaptiveParameter] = {}
        self.constraints: List[Callable[[Dict[str, float]], bool]] = []

    def add_parameter(self, param: AdaptiveParameter):
        self.parameters[param.name] = param

    def create_parameter(self, name: str, value: float, min_value: float = 0.0,
                        max_value: float = 1.0, step: float = 0.01,
                        adaptive: bool = True) -> AdaptiveParameter:
        param = AdaptiveParameter(name, value, min_value, max_value, step, adaptive)
        self.add_parameter(param)
        return param

    def get_parameter(self, name: str) -> Optional[AdaptiveParameter]:
        return self.parameters.get(name)

    def add_constraint(self, constraint: Callable[[Dict[str, float]], bool]):
        self.constraints.append(constraint)

    def validate(self, params: Dict[str, float]) -> bool:
        for constraint in self.constraints:
            if not constraint(params):
                return False
        return True

    def sample(self) -> Dict[str, float]:
        return {name: param.value + np.random.normal(0, param.step) 
                for name, param in self.parameters.items()}

    def get_values(self) -> Dict[str, float]:
        return {name: param.value for name, param in self.parameters.items()}

    def update_values(self, new_values: Dict[str, float]):
        for name, value in new_values.items():
            if name in self.parameters:
                param = self.parameters[name]
                param.value = max(param.min_value, min(param.max_value, value))
                param.history.append(param.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
            "constraint_count": len(self.constraints)
        }


class ParameterTuningResult:
    def __init__(self):
        self.success: bool = False
        self.best_params: Dict[str, float] = {}
        self.best_score: float = 0.0
        self.original_score: float = 0.0
        self.improvement: float = 0.0
        self.iterations: int = 0
        self.converged: bool = False
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "original_score": self.original_score,
            "improvement": self.improvement,
            "iterations": self.iterations,
            "converged": self.converged,
            "timestamp": self.timestamp
        }


class ParameterOptimizer:
    def __init__(self):
        self.spaces: Dict[str, ParameterSpace] = {}
        self.tuning_history: deque = deque(maxlen=200)

    def create_space(self, space_id: str) -> ParameterSpace:
        space = ParameterSpace()
        self.spaces[space_id] = space
        return space

    def get_space(self, space_id: str) -> Optional[ParameterSpace]:
        return self.spaces.get(space_id)

    def optimize(self, space_id: str, objective_function: Callable[[Dict[str, float]], float],
                max_iterations: int = 100, learning_rate: float = 0.1) -> ParameterTuningResult:
        
        space = self.get_space(space_id)
        if not space:
            result = ParameterTuningResult()
            result.success = False
            return result
        
        result = ParameterTuningResult()
        original_values = space.get_values()
        result.original_score = objective_function(original_values)
        best_score = result.original_score
        best_params = original_values.copy()
        
        for i in range(max_iterations):
            current_values = space.get_values()
            current_score = objective_function(current_values)
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_values.copy()
                for name, param in space.parameters.items():
                    param.record_success()
            
            gradients = self._estimate_gradients(space, objective_function)
            
            for name, param in space.parameters.items():
                if param.adaptive:
                    gradient = gradients.get(name, 0.0)
                    param.adjust(gradient, learning_rate)
            
            result.iterations = i + 1
            
            if i > 0 and abs(best_score - current_score) < 1e-6:
                result.converged = True
                break
        
        space.update_values(best_params)
        
        result.success = True
        result.best_params = best_params
        result.best_score = best_score
        result.improvement = best_score - result.original_score
        
        self.tuning_history.append(result)
        
        return result

    def _estimate_gradients(self, space: ParameterSpace,
                           objective_function: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        gradients = {}
        base_values = space.get_values()
        base_score = objective_function(base_values)
        
        for name, param in space.parameters.items():
            if not param.adaptive:
                continue
            
            original_value = param.value
            param.value = min(param.max_value, original_value + param.step)
            new_values = space.get_values()
            new_score = objective_function(new_values)
            
            gradient = (new_score - base_score) / param.step
            gradients[name] = gradient
            
            param.value = original_value
        
        return gradients

    def adaptive_tune(self, space_id: str, objective_function: Callable[[Dict[str, float]], float],
                     feedback_fn: Callable[[], float] = None) -> ParameterTuningResult:
        
        space = self.get_space(space_id)
        if not space:
            result = ParameterTuningResult()
            result.success = False
            return result
        
        result = ParameterTuningResult()
        original_values = space.get_values()
        result.original_score = objective_function(original_values)
        best_score = result.original_score
        
        for i in range(50):
            current_values = space.get_values()
            current_score = objective_function(current_values)
            
            if current_score > best_score:
                best_score = current_score
            
            for name, param in space.parameters.items():
                stability = param.get_stability()
                
                if feedback_fn:
                    feedback = feedback_fn()
                    if feedback > 0:
                        param.adjust(0.1, learning_rate=0.05 * (1 - stability))
                    else:
                        param.adjust(-0.1, learning_rate=0.05 * (1 - stability))
            
            result.iterations = i + 1
            
            if abs(best_score - current_score) < 1e-6:
                result.converged = True
                break
        
        result.success = True
        result.best_score = best_score
        result.improvement = best_score - result.original_score
        
        return result

    def batch_optimize(self, space_ids: List[str],
                      objective_function: Callable[[Dict[str, float]], float]) -> List[ParameterTuningResult]:
        results = []
        for space_id in space_ids:
            result = self.optimize(space_id, objective_function)
            results.append(result)
        return results

    def get_optimization_summary(self) -> Dict[str, Any]:
        if not self.tuning_history:
            return {"total_optimizations": 0}
        
        results = list(self.tuning_history)
        avg_improvement = np.mean([r.improvement for r in results])
        avg_iterations = np.mean([r.iterations for r in results])
        convergence_rate = len([r for r in results if r.converged]) / len(results)
        
        return {
            "total_optimizations": len(results),
            "avg_improvement": float(avg_improvement),
            "avg_iterations": float(avg_iterations),
            "convergence_rate": convergence_rate,
            "spaces": list(self.spaces.keys())
        }