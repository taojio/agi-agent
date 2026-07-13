import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class RegulationMode(Enum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"


class CognitiveState(Enum):
    NORMAL = "normal"
    OVERLOAD = "overload"
    UNDERUTILIZED = "underutilized"
    LEARNING = "learning"
    ADAPTING = "adapting"
    CRITICAL = "critical"


class CognitiveMetric:
    def __init__(self, name: str, value: float, 
                 min_threshold: float = 0.0, max_threshold: float = 1.0,
                 warning_low: float = 0.2, warning_high: float = 0.8):
        self.name = name
        self.value = value
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.warning_low = warning_low
        self.warning_high = warning_high
        self.history: deque = deque(maxlen=100)
        self.trend: float = 0.0

    def update(self, value: float):
        self.history.append(self.value)
        self.value = value
        
        if len(self.history) >= 5:
            recent = list(self.history)[-5:]
            x = np.arange(len(recent))
            y = np.array(recent)
            self.trend = np.polyfit(x, y, 1)[0]

    def is_healthy(self) -> bool:
        return self.warning_low <= self.value <= self.warning_high

    def get_status(self) -> str:
        if self.value < self.min_threshold:
            return "critical_low"
        elif self.value < self.warning_low:
            return "warning_low"
        elif self.value > self.max_threshold:
            return "critical_high"
        elif self.value > self.warning_high:
            return "warning_high"
        return "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "trend": self.trend,
            "status": self.get_status(),
            "is_healthy": self.is_healthy(),
            "thresholds": {
                "min": self.min_threshold,
                "max": self.max_threshold,
                "warning_low": self.warning_low,
                "warning_high": self.warning_high
            }
        }


class CognitiveRegulator:
    def __init__(self):
        self.metrics: Dict[str, CognitiveMetric] = {}
        self.regulations: List[Dict[str, Any]] = []
        self.current_state = CognitiveState.NORMAL
        self.mode = RegulationMode.REACTIVE
        self._regulator_history: deque = deque(maxlen=200)

        self._init_default_metrics()

    def _init_default_metrics(self):
        self.add_metric(CognitiveMetric(
            "processing_load", 0.5, 0.0, 1.0, 0.3, 0.7
        ))
        self.add_metric(CognitiveMetric(
            "confidence", 0.5, 0.0, 1.0, 0.3, 0.85
        ))
        self.add_metric(CognitiveMetric(
            "attention_focus", 0.7, 0.0, 1.0, 0.4, 0.8
        ))
        self.add_metric(CognitiveMetric(
            "learning_progress", 0.5, 0.0, 1.0, 0.2, 0.8
        ))
        self.add_metric(CognitiveMetric(
            "decision_quality", 0.6, 0.0, 1.0, 0.3, 0.8
        ))
        self.add_metric(CognitiveMetric(
            "memory_efficiency", 0.7, 0.0, 1.0, 0.4, 0.85
        ))

    def add_metric(self, metric: CognitiveMetric):
        self.metrics[metric.name] = metric

    def update_metric(self, name: str, value: float):
        if name in self.metrics:
            self.metrics[name].update(value)
        else:
            self.add_metric(CognitiveMetric(name, value))

    def update_metrics(self, metrics_dict: Dict[str, float]):
        for name, value in metrics_dict.items():
            self.update_metric(name, value)

    def assess_state(self) -> CognitiveState:
        healthy_count = sum(1 for m in self.metrics.values() if m.is_healthy())
        total_count = len(self.metrics)
        
        if healthy_count == 0:
            return CognitiveState.CRITICAL
        elif healthy_count / total_count < 0.3:
            return CognitiveState.OVERLOAD
        elif healthy_count / total_count < 0.5:
            return CognitiveState.ADAPTING
        
        processing = self.metrics.get("processing_load")
        if processing and processing.value > 0.9:
            return CognitiveState.OVERLOAD
        elif processing and processing.value < 0.1:
            return CognitiveState.UNDERUTILIZED
        
        learning = self.metrics.get("learning_progress")
        if learning and learning.trend > 0.01:
            return CognitiveState.LEARNING
        
        return CognitiveState.NORMAL

    def regulate(self) -> List[Dict[str, Any]]:
        self.current_state = self.assess_state()
        
        regulations = []
        
        if self.current_state == CognitiveState.OVERLOAD:
            regulations.extend(self._handle_overload())
        elif self.current_state == CognitiveState.UNDERUTILIZED:
            regulations.extend(self._handle_underutilized())
        elif self.current_state == CognitiveState.LEARNING:
            regulations.extend(self._support_learning())
        elif self.current_state == CognitiveState.ADAPTING:
            regulations.extend(self._support_adaptation())
        elif self.current_state == CognitiveState.CRITICAL:
            regulations.extend(self._handle_critical())
        
        self.regulations.extend(regulations)
        self._regulator_history.append({
            "timestamp": np.random.randint(1000000),
            "state": self.current_state.value,
            "regulations": len(regulations),
            "metrics": {k: m.value for k, m in self.metrics.items()}
        })
        
        return regulations

    def _handle_overload(self) -> List[Dict[str, Any]]:
        return [
            {"action": "reduce_parallel_tasks", "target": "task_engine", "priority": "high"},
            {"action": "increase_memory_allocation", "target": "memory", "priority": "medium"},
            {"action": "simplify_reasoning", "target": "cognition", "priority": "high"},
            {"action": "postpone_non_critical", "target": "decision", "priority": "medium"}
        ]

    def _handle_underutilized(self) -> List[Dict[str, Any]]:
        return [
            {"action": "increase_parallel_tasks", "target": "task_engine", "priority": "medium"},
            {"action": "deepen_reasoning", "target": "cognition", "priority": "low"},
            {"action": "initiate_exploration", "target": "exploration", "priority": "medium"}
        ]

    def _support_learning(self) -> List[Dict[str, Any]]:
        return [
            {"action": "allocate_learning_resources", "target": "learning", "priority": "high"},
            {"action": "create_memory_connections", "target": "memory", "priority": "medium"},
            {"action": "reduce_interruptions", "target": "task_engine", "priority": "medium"}
        ]

    def _support_adaptation(self) -> List[Dict[str, Any]]:
        return [
            {"action": "analyze_performance_gaps", "target": "self_improvement", "priority": "high"},
            {"action": "initiate_strategy_adjustment", "target": "strategy", "priority": "medium"},
            {"action": "collect_additional_data", "target": "perception", "priority": "medium"}
        ]

    def _handle_critical(self) -> List[Dict[str, Any]]:
        return [
            {"action": "emergency_shutdown_non_critical", "target": "system", "priority": "critical"},
            {"action": "alert_operator", "target": "communication", "priority": "critical"},
            {"action": "initiate_safety_protocols", "target": "security", "priority": "critical"}
        ]

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "regulation_mode": self.mode.value,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "recent_regulations": len(self.regulations),
            "healthy_metrics": sum(1 for m in self.metrics.values() if m.is_healthy()),
            "total_metrics": len(self.metrics)
        }

    def get_regulation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._regulator_history)[-limit:]

    def set_mode(self, mode: RegulationMode):
        self.mode = mode

    def predict_future_state(self, steps_ahead: int = 5) -> Dict[str, Any]:
        predictions = {}
        
        for name, metric in self.metrics.items():
            if metric.trend != 0:
                predicted = metric.value + metric.trend * steps_ahead
                predictions[name] = {
                    "current": metric.value,
                    "predicted": max(metric.min_threshold, min(metric.max_threshold, predicted)),
                    "trend": metric.trend,
                    "will_be_healthy": metric.warning_low <= predicted <= metric.warning_high
                }
        
        return predictions