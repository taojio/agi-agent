"""
circuit_breaker.py - 行动熔断机制

执行过程中检测到异常（如偏离目标、风险超标、资源耗尽），立即暂停任务，
启动熔断保护，等待人工介入或自动纠错。
"""
import time
import numpy as np
from collections import deque
from enum import Enum


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TriggerCondition(Enum):
    DEVIATION = "deviation"
    RISK_EXCEEDED = "risk_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"


class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, success_threshold=3):
        self.state = BreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_attempt_time = 0
        
        self.trigger_history = deque(maxlen=100)
        self.actions_blocked = deque(maxlen=50)
        
        self.monitoring_metrics = {}

    def _should_open(self):
        return self.failure_count >= self.failure_threshold

    def _should_close(self):
        return self.success_count >= self.success_threshold

    def _should_attempt(self):
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            elapsed = time.time() - self.last_failure_time
            return elapsed >= self.recovery_timeout
        return True

    def attempt_action(self, action_id, action_description):
        if not self._should_attempt():
            self.actions_blocked.append({
                "timestamp": time.time(),
                "action_id": action_id,
                "action_description": action_description,
                "reason": "circuit_breaker_open",
                "state": self.state.value
            })
            return {"allowed": False, "reason": "Circuit breaker is open", "state": self.state.value}

        if self.state == BreakerState.OPEN:
            self.state = BreakerState.HALF_OPEN
            self.success_count = 0

        self.last_attempt_time = time.time()
        return {"allowed": True, "state": self.state.value}

    def record_success(self, action_id):
        if self.state == BreakerState.HALF_OPEN:
            self.success_count += 1
            if self._should_close():
                self.state = BreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                
                self.trigger_history.append({
                    "timestamp": time.time(),
                    "event": "breaker_closed",
                    "action_id": action_id,
                    "state": self.state.value
                })

    def record_failure(self, action_id=None, reason=None, trigger_condition=None):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self._should_open():
            self.state = BreakerState.OPEN
            self.success_count = 0
            
            self.trigger_history.append({
                "timestamp": self.last_failure_time,
                "event": "breaker_opened",
                "action_id": action_id,
                "reason": reason,
                "trigger_condition": trigger_condition.value,
                "state": self.state.value,
                "failure_count": self.failure_count
            })

    def trigger_breaker(self, action_id, reason, trigger_condition):
        self.state = BreakerState.OPEN
        self.failure_count = self.failure_threshold
        self.last_failure_time = time.time()
        
        self.trigger_history.append({
            "timestamp": self.last_failure_time,
            "event": "breaker_manually_triggered",
            "action_id": action_id,
            "reason": reason,
            "trigger_condition": trigger_condition.value,
            "state": self.state.value
        })

    def reset(self):
        self.state = BreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        self.trigger_history.append({
            "timestamp": time.time(),
            "event": "breaker_reset",
            "state": self.state.value
        })

    def set_monitoring_metric(self, metric_name, value):
        self.monitoring_metrics[metric_name] = value

    def check_monitoring_metrics(self):
        triggers = []
        
        if self.monitoring_metrics.get("deviation", 0) > 0.5:
            triggers.append({"type": TriggerCondition.DEVIATION, "value": self.monitoring_metrics["deviation"]})
        
        if self.monitoring_metrics.get("risk_score", 0) > 0.8:
            triggers.append({"type": TriggerCondition.RISK_EXCEEDED, "value": self.monitoring_metrics["risk_score"]})
        
        if self.monitoring_metrics.get("resource_usage", 0) > 0.95:
            triggers.append({"type": TriggerCondition.RESOURCE_EXHAUSTED, "value": self.monitoring_metrics["resource_usage"]})
        
        if self.monitoring_metrics.get("error_rate", 0) > 0.5:
            triggers.append({"type": TriggerCondition.ERROR_RATE, "value": self.monitoring_metrics["error_rate"]})
        
        return triggers

    def get_state(self):
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
            "last_attempt_time": self.last_attempt_time
        }

    def is_tripped(self):
        return self.state == BreakerState.OPEN

    def get_trigger_history(self, limit=20):
        return list(self.trigger_history)[-limit:]

    def get_blocked_actions(self, limit=20):
        return list(self.actions_blocked)[-limit:]

    def get_stats(self):
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_triggers": len(self.trigger_history),
            "total_blocked": len(self.actions_blocked),
            "monitoring_metrics": self.monitoring_metrics
        }