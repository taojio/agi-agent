import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple
from .decision_monitor import DecisionMonitor, DecisionPhase, DecisionTrace
from .decision_optimizer import DecisionOptimizer
from .quality_analyzer import DecisionQualityAnalyzer, QualityScore


class MetaDecisionOrchestrator:
    def __init__(self):
        self.monitor = DecisionMonitor()
        self.optimizer = DecisionOptimizer()
        self.quality_analyzer = DecisionQualityAnalyzer()
        self._decision_contexts: Dict[str, Dict[str, Any]] = {}
        self._orchestration_history: deque = deque(maxlen=200)

    def start_decision(self, decision_id: str, goal: str,
                      decision_type: str = "default") -> Dict[str, Any]:
        trace = self.monitor.start_monitoring(decision_id, goal)
        
        self._decision_contexts[decision_id] = {
            "decision_id": decision_id,
            "goal": goal,
            "decision_type": decision_type,
            "trace": trace,
            "phase": DecisionPhase.INITIATION.value,
            "factors": [],
            "options": [],
            "timeline": [],
        }
        
        return {"decision_id": decision_id, "status": "started"}

    def record_phase(self, decision_id: str, phase: DecisionPhase,
                    details: Dict[str, Any] = None):
        if decision_id not in self._decision_contexts:
            return
        
        self.monitor.record_phase(decision_id, phase, details)
        self._decision_contexts[decision_id]["phase"] = phase.value
        
        if details:
            self._decision_contexts[decision_id]["timeline"].append({
                "phase": phase.value,
                "details": details,
                "timestamp": np.random.randint(1000000)
            })

    def add_factor(self, decision_id: str, factor: str, weight: float = 1.0):
        if decision_id not in self._decision_contexts:
            return
        
        self.monitor.add_factor(decision_id, factor)
        self._decision_contexts[decision_id]["factors"].append({
            "factor": factor,
            "weight": weight
        })

    def add_option(self, decision_id: str, option: str, score: float = 0.0):
        if decision_id not in self._decision_contexts:
            return
        
        self.monitor.add_option(decision_id, option)
        self._decision_contexts[decision_id]["options"].append({
            "option": option,
            "score": score
        })

    def add_metric(self, decision_id: str, name: str, value: float, unit: str = ""):
        if decision_id in self._decision_contexts:
            self.monitor.add_metric(decision_id, name, value, unit)

    def optimize_decision(self, decision_id: str, params: Dict[str, Any],
                         objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        if decision_id not in self._decision_contexts:
            return {"error": "Decision not found"}
        
        decision_type = self._decision_contexts[decision_id]["decision_type"]
        
        result = self.optimizer.optimize(
            decision_type=decision_type,
            initial_params=params,
            objective_function=objective_function,
            context={"complex": len(self._decision_contexts[decision_id]["factors"]) > 5}
        )
        
        self.add_metric(decision_id, "optimization_improvement", result.improvement)
        
        return result.to_dict()

    def analyze_quality(self, decision_id: str) -> Dict[str, Any]:
        if decision_id not in self._decision_contexts:
            return {"error": "Decision not found"}
        
        context = self._decision_contexts[decision_id]
        decision_data = {
            "decision_id": decision_id,
            "goal": context["goal"],
            "factors_considered": [f["factor"] for f in context["factors"]],
            "options_considered": [o["option"] for o in context["options"]],
            "confidence": context.get("confidence", 0.5),
            "duration_ms": context.get("duration_ms", 0),
        }
        
        quality_score = self.quality_analyzer.analyze_quality(decision_data)
        biases = self.quality_analyzer.detect_biases(decision_data)
        improvements = self.quality_analyzer.suggest_improvements(decision_data)
        
        return {
            "quality_score": quality_score.to_dict(),
            "biases": biases,
            "improvement_suggestions": improvements
        }

    def complete_decision(self, decision_id: str, outcome: str,
                         confidence: float = 0.5) -> Dict[str, Any]:
        if decision_id not in self._decision_contexts:
            return {"error": "Decision not found"}
        
        context = self._decision_contexts[decision_id]
        
        quality_data = {
            "decision_id": decision_id,
            "goal": context["goal"],
            "factors_considered": [f["factor"] for f in context["factors"]],
            "options_considered": [o["option"] for o in context["options"]],
            "confidence": confidence,
            "outcome": outcome,
        }
        
        quality_score = self.quality_analyzer.analyze_quality(quality_data)
        
        performance = self.monitor.complete_decision(
            decision_id=decision_id,
            outcome=outcome,
            quality_score=quality_score.overall_score,
            confidence=confidence
        )
        
        self._orchestration_history.append({
            "decision_id": decision_id,
            "outcome": outcome,
            "quality_score": quality_score.overall_score,
            "confidence": confidence,
            "timestamp": np.random.randint(1000000)
        })
        
        del self._decision_contexts[decision_id]
        
        return {
            "status": "completed",
            "outcome": outcome,
            "quality_score": quality_score.to_dict(),
            "performance": performance.to_dict() if performance else {}
        }

    def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        if decision_id not in self._decision_contexts:
            trace = self.monitor.get_decision_trace(decision_id)
            if trace:
                return {"status": "completed", "trace": trace}
            return None
        
        context = self._decision_contexts[decision_id]
        return {
            "decision_id": decision_id,
            "goal": context["goal"],
            "phase": context["phase"],
            "factors": context["factors"],
            "options": context["options"],
            "status": "active"
        }

    def get_overview(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor.get_performance_summary(),
            "optimizer": self.optimizer.get_optimization_summary(),
            "quality_analyzer": self.quality_analyzer.get_quality_summary(),
            "active_decisions": len(self._decision_contexts),
            "total_orchestrations": len(self._orchestration_history)
        }

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        return self.monitor.detect_anomalies()

    def get_decision_patterns(self) -> Dict[str, Any]:
        return self.monitor.get_decision_patterns()

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(self._orchestration_history)[-limit:]
        return recent[::-1]

    def optimize_strategy(self, decision_type: str) -> Dict[str, Any]:
        quality_summary = self.quality_analyzer.get_quality_summary()
        overall_quality = quality_summary.get("overall_quality", 0.5)
        
        optimized_params = {}
        
        if overall_quality < 0.5:
            optimized_params["decision_temperature"] = min(2.0, 1.0 + (0.5 - overall_quality))
            optimized_params["exploration_rate"] = min(0.5, 0.1 + (0.5 - overall_quality) * 0.5)
        else:
            optimized_params["decision_temperature"] = max(0.5, 1.0 - (overall_quality - 0.5))
            optimized_params["exploration_rate"] = max(0.05, 0.1 - (overall_quality - 0.5) * 0.1)
        
        return {
            "success": True,
            "decision_type": decision_type,
            "current_quality": overall_quality,
            "optimized_params": optimized_params,
            "improvement": abs(overall_quality - 0.5)
        }