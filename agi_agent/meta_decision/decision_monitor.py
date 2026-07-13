import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class DecisionPhase(Enum):
    INITIATION = "initiation"
    INFORMATION_GATHERING = "information_gathering"
    OPTION_GENERATION = "option_generation"
    EVALUATION = "evaluation"
    SELECTION = "selection"
    EXECUTION = "execution"
    REVIEW = "review"


class DecisionMetric:
    def __init__(self, name: str, value: float, unit: str = ""):
        self.name = name
        self.value = value
        self.unit = unit
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp
        }


class DecisionPerformance:
    def __init__(self, decision_id: str):
        self.decision_id = decision_id
        self.duration_ms: float = 0.0
        self.quality_score: float = 0.0
        self.confidence: float = 0.0
        self.outcome: str = "unknown"
        self.outcome_score: float = 0.0
        self.resource_usage: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "duration_ms": self.duration_ms,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "outcome_score": self.outcome_score,
            "resource_usage": self.resource_usage
        }


class DecisionTrace:
    def __init__(self, decision_id: str, goal: str):
        self.decision_id = decision_id
        self.goal = goal
        self.phases: List[Dict[str, Any]] = []
        self.metrics: List[DecisionMetric] = []
        self.options_considered: List[str] = []
        self.factors_considered: List[str] = []
        self.start_time = np.random.randint(1000000)
        self.end_time: Optional[int] = None
        self.outcome: str = "pending"

    def record_phase(self, phase: DecisionPhase, details: Dict[str, Any] = None):
        self.phases.append({
            "phase": phase.value,
            "timestamp": np.random.randint(1000000),
            "details": details or {}
        })

    def add_metric(self, name: str, value: float, unit: str = ""):
        self.metrics.append(DecisionMetric(name, value, unit))

    def add_option(self, option: str):
        self.options_considered.append(option)

    def add_factor(self, factor: str):
        self.factors_considered.append(factor)

    def complete(self, outcome: str):
        self.outcome = outcome
        self.end_time = np.random.randint(1000000)

    def get_duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "goal": self.goal,
            "phases": self.phases,
            "metrics": [m.to_dict() for m in self.metrics],
            "options_considered": self.options_considered,
            "factors_considered": self.factors_considered,
            "duration": self.get_duration(),
            "outcome": self.outcome,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class DecisionMonitor:
    def __init__(self):
        self.traces: Dict[str, DecisionTrace] = {}
        self.performance_history: deque = deque(maxlen=200)
        self._active_decisions: Dict[str, DecisionTrace] = {}
        self._monitoring_enabled = True

    def start_monitoring(self, decision_id: str, goal: str) -> DecisionTrace:
        if not self._monitoring_enabled:
            return DecisionTrace(decision_id, goal)
        
        trace = DecisionTrace(decision_id, goal)
        self.traces[decision_id] = trace
        self._active_decisions[decision_id] = trace
        trace.record_phase(DecisionPhase.INITIATION)
        
        return trace

    def record_phase(self, decision_id: str, phase: DecisionPhase, details: Dict[str, Any] = None):
        if decision_id in self._active_decisions:
            self._active_decisions[decision_id].record_phase(phase, details)

    def add_metric(self, decision_id: str, name: str, value: float, unit: str = ""):
        if decision_id in self._active_decisions:
            self._active_decisions[decision_id].add_metric(name, value, unit)

    def add_option(self, decision_id: str, option: str):
        if decision_id in self._active_decisions:
            self._active_decisions[decision_id].add_option(option)

    def add_factor(self, decision_id: str, factor: str):
        if decision_id in self._active_decisions:
            self._active_decisions[decision_id].add_factor(factor)

    def complete_decision(self, decision_id: str, outcome: str,
                         quality_score: float = 0.0, confidence: float = 0.0):
        if decision_id in self._active_decisions:
            trace = self._active_decisions.pop(decision_id)
            trace.complete(outcome)
            
            performance = DecisionPerformance(decision_id)
            performance.duration_ms = trace.get_duration()
            performance.quality_score = quality_score
            performance.confidence = confidence
            performance.outcome = outcome
            
            self.performance_history.append(performance)
            
            return performance
        
        return None

    def get_decision_trace(self, decision_id: str) -> Optional[Dict[str, Any]]:
        trace = self.traces.get(decision_id)
        return trace.to_dict() if trace else None

    def get_active_decisions(self) -> List[Dict[str, Any]]:
        return [trace.to_dict() for trace in self._active_decisions.values()]

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {"total_decisions": 0, "avg_quality": 0.0}
        
        performances = list(self.performance_history)
        avg_quality = np.mean([p.quality_score for p in performances])
        avg_duration = np.mean([p.duration_ms for p in performances])
        avg_confidence = np.mean([p.confidence for p in performances])
        
        success_rate = len([p for p in performances if p.outcome == "success"]) / len(performances)
        
        return {
            "total_decisions": len(performances),
            "active_decisions": len(self._active_decisions),
            "avg_quality_score": float(avg_quality),
            "avg_confidence": float(avg_confidence),
            "avg_duration_ms": float(avg_duration),
            "success_rate": success_rate,
            "monitoring_enabled": self._monitoring_enabled
        }

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        anomalies = []
        
        for performance in self.performance_history:
            if performance.duration_ms > 1000:
                anomalies.append({
                    "type": "slow_decision",
                    "decision_id": performance.decision_id,
                    "duration_ms": performance.duration_ms,
                    "threshold": 1000
                })
            
            if performance.quality_score < 0.3:
                anomalies.append({
                    "type": "low_quality",
                    "decision_id": performance.decision_id,
                    "quality_score": performance.quality_score,
                    "threshold": 0.3
                })
            
            if performance.confidence < 0.2:
                anomalies.append({
                    "type": "low_confidence",
                    "decision_id": performance.decision_id,
                    "confidence": performance.confidence,
                    "threshold": 0.2
                })
        
        return anomalies[:10]

    def enable_monitoring(self):
        self._monitoring_enabled = True

    def disable_monitoring(self):
        self._monitoring_enabled = False

    def get_decision_patterns(self, window_size: int = 50) -> Dict[str, Any]:
        if len(self.performance_history) < window_size:
            return {"insufficient_data": True}
        
        recent = list(self.performance_history)[-window_size:]
        
        outcomes = [p.outcome for p in recent]
        qualities = [p.quality_score for p in recent]
        durations = [p.duration_ms for p in recent]
        
        return {
            "outcome_distribution": {
                "success": outcomes.count("success"),
                "failure": outcomes.count("failure"),
                "partial": outcomes.count("partial")
            },
            "quality_trend": float(np.polyfit(np.arange(len(qualities)), qualities, 1)[0]),
            "duration_trend": float(np.polyfit(np.arange(len(durations)), durations, 1)[0]),
            "avg_recent_quality": float(np.mean(qualities)),
            "avg_recent_duration": float(np.mean(durations))
        }