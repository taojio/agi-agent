import time
import psutil
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class AlertLevel(Enum):
    NORMAL = "normal"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class AlertThreshold:
    metric_name: str
    yellow_threshold: Optional[float] = None
    orange_threshold: Optional[float] = None
    red_threshold: Optional[float] = None
    higher_is_worse: bool = True
    description: str = ""

    def check(self, value: float) -> AlertLevel:
        if self.higher_is_worse:
            if self.red_threshold is not None and value >= self.red_threshold:
                return AlertLevel.RED
            if self.orange_threshold is not None and value >= self.orange_threshold:
                return AlertLevel.ORANGE
            if self.yellow_threshold is not None and value >= self.yellow_threshold:
                return AlertLevel.YELLOW
        else:
            if self.red_threshold is not None and value <= self.red_threshold:
                return AlertLevel.RED
            if self.orange_threshold is not None and value <= self.orange_threshold:
                return AlertLevel.ORANGE
            if self.yellow_threshold is not None and value <= self.yellow_threshold:
                return AlertLevel.YELLOW
        return AlertLevel.NORMAL


@dataclass
class AlertRecord:
    alert_id: str
    metric_name: str
    level: AlertLevel
    value: float
    threshold: float
    step: int
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_step: Optional[int] = None
    description: str = ""


@dataclass
class SystemMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_gb: float = 0.0
    disk_percent: float = 0.0
    step_latency_ms: float = 0.0


class TrainingMonitor:
    def __init__(self):
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, AlertRecord] = {}
        self.alert_counter = 0

        self.system_metrics_history: deque = deque(maxlen=500)
        self.performance_metrics_history: deque = deque(maxlen=1000)

        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        self.intervention_engine: Optional['InterventionEngine'] = None

        self._init_default_thresholds()

    def _init_default_thresholds(self):
        self.thresholds["free_energy"] = AlertThreshold(
            metric_name="free_energy",
            yellow_threshold=0.6,
            orange_threshold=0.7,
            red_threshold=0.8,
            higher_is_worse=True,
            description="自由能过高预警"
        )
        self.thresholds["confidence"] = AlertThreshold(
            metric_name="confidence",
            yellow_threshold=0.4,
            orange_threshold=0.3,
            red_threshold=0.2,
            higher_is_worse=False,
            description="置信度过低预警"
        )
        self.thresholds["latency_ms"] = AlertThreshold(
            metric_name="latency_ms",
            yellow_threshold=1000,
            orange_threshold=1500,
            red_threshold=2000,
            higher_is_worse=True,
            description="单步延迟过高预警"
        )
        self.thresholds["memory_percent"] = AlertThreshold(
            metric_name="memory_percent",
            yellow_threshold=80,
            orange_threshold=85,
            red_threshold=90,
            higher_is_worse=True,
            description="内存使用率过高预警"
        )
        self.thresholds["consecutive_failures"] = AlertThreshold(
            metric_name="consecutive_failures",
            yellow_threshold=10,
            orange_threshold=30,
            red_threshold=50,
            higher_is_worse=True,
            description="连续失败步数预警"
        )
        self.thresholds["stability_score"] = AlertThreshold(
            metric_name="stability_score",
            yellow_threshold=0.5,
            orange_threshold=0.35,
            red_threshold=0.2,
            higher_is_worse=False,
            description="系统稳定性预警"
        )

    def register_alert_callback(self, level: AlertLevel, callback: Callable):
        self.alert_callbacks[level].append(callback)

    def set_intervention_engine(self, engine: 'InterventionEngine'):
        self.intervention_engine = engine

    def check_metrics(self, metrics: Dict[str, float], step: int) -> List[AlertRecord]:
        new_alerts = []

        for metric_name, threshold in self.thresholds.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]
            alert_level = threshold.check(value)

            alert_key = f"{metric_name}_{alert_level.value}"

            if alert_level == AlertLevel.NORMAL:
                if alert_key in self.active_alerts:
                    self._resolve_alert(alert_key, step)
                continue

            if alert_key in self.active_alerts:
                continue

            alert = AlertRecord(
                alert_id=f"alert_{self.alert_counter}",
                metric_name=metric_name,
                level=alert_level,
                value=value,
                threshold=getattr(threshold, f"{alert_level.value}_threshold", 0) or 0,
                step=step,
                description=threshold.description
            )

            self.alert_counter += 1
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            new_alerts.append(alert)

            self._trigger_alert_callbacks(alert)

            if self.intervention_engine:
                self.intervention_engine.handle_alert(alert, metrics, step)

        return new_alerts

    def _resolve_alert(self, alert_key: str, step: int):
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolution_step = step
            del self.active_alerts[alert_key]

    def _trigger_alert_callbacks(self, alert: AlertRecord):
        for callback in self.alert_callbacks.get(alert.level, []):
            try:
                callback(alert)
            except Exception:
                pass

    def collect_system_metrics(self) -> SystemMetrics:
        metrics = SystemMetrics()

        try:
            metrics.cpu_percent = float(psutil.cpu_percent(interval=0.1))
            mem = psutil.virtual_memory()
            metrics.memory_percent = float(mem.percent)
            metrics.memory_gb = float(mem.used / (1024 ** 3))
            disk = psutil.disk_usage('/')
            metrics.disk_percent = float(disk.percent)
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                metrics.gpu_utilization = float(torch.cuda.utilization() / 100.0)
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                metrics.gpu_memory_gb = float(gpu_mem)
        except Exception:
            pass

        self.system_metrics_history.append({
            "timestamp": time.time(),
            **metrics.__dict__
        })

        return metrics

    def record_step_metrics(self, metrics: Dict[str, float], step: int):
        record = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        self.performance_metrics_history.append(record)
        self.check_metrics(metrics, step)

    def get_current_status(self) -> Dict[str, Any]:
        overall_level = AlertLevel.NORMAL
        for alert in self.active_alerts.values():
            if self._alert_level_priority(alert.level) > self._alert_level_priority(overall_level):
                overall_level = alert.level

        return {
            "overall_status": overall_level.value,
            "active_alerts_count": len(self.active_alerts),
            "active_alerts": [
                {
                    "id": a.alert_id,
                    "metric": a.metric_name,
                    "level": a.level.value,
                    "value": a.value,
                    "step": a.step
                }
                for a in self.active_alerts.values()
            ],
            "total_alerts": len(self.alert_history),
            "system_metrics": self._get_latest_system_metrics()
        }

    def _alert_level_priority(self, level: AlertLevel) -> int:
        priorities = {
            AlertLevel.NORMAL: 0,
            AlertLevel.YELLOW: 1,
            AlertLevel.ORANGE: 2,
            AlertLevel.RED: 3
        }
        return priorities.get(level, 0)

    def _get_latest_system_metrics(self) -> Dict[str, Any]:
        if not self.system_metrics_history:
            return {}
        latest = self.system_metrics_history[-1]
        return {k: v for k, v in latest.items() if k != "timestamp"}

    def get_alert_summary(self, window_steps: int = 1000) -> Dict[str, Any]:
        alerts = list(self.alert_history)[-100:]
        level_counts = {level.value: 0 for level in AlertLevel}
        metric_counts = {}

        for alert in alerts:
            level_counts[alert.level.value] += 1
            metric_counts[alert.metric_name] = metric_counts.get(alert.metric_name, 0) + 1

        return {
            "total_alerts": len(alerts),
            "level_distribution": level_counts,
            "metric_distribution": metric_counts,
            "active_count": len(self.active_alerts)
        }

    def update_threshold(self, metric_name: str, **kwargs):
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = AlertThreshold(metric_name=metric_name)

        threshold = self.thresholds[metric_name]
        for key, value in kwargs.items():
            if hasattr(threshold, key):
                setattr(threshold, key, value)


class InterventionEngine:
    def __init__(self):
        self.intervention_rules: Dict[AlertLevel, List[Dict[str, Any]]] = {
            level: [] for level in AlertLevel
        }
        self.intervention_history: deque = deque(maxlen=500)
        self.intervention_counter = 0

        self._init_default_rules()

    def _init_default_rules(self):
        self.intervention_rules[AlertLevel.YELLOW].append({
            "id": "yellow_log_only",
            "description": "黄色预警仅记录日志",
            "action": "log",
            "auto_execute": True
        })

        self.intervention_rules[AlertLevel.ORANGE].append({
            "id": "orange_adjust_lr",
            "description": "橙色预警自动调整学习率",
            "action": "adjust_learning_rate",
            "factor": 0.8,
            "auto_execute": True
        })
        self.intervention_rules[AlertLevel.ORANGE].append({
            "id": "orange_reduce_batch",
            "description": "橙色预警降低处理强度",
            "action": "reduce_processing",
            "auto_execute": False
        })

        self.intervention_rules[AlertLevel.RED].append({
            "id": "red_pause_training",
            "description": "红色预警暂停训练",
            "action": "pause_training",
            "auto_execute": True
        })
        self.intervention_rules[AlertLevel.RED].append({
            "id": "red_rollback",
            "description": "红色预警回滚到上一稳定版本",
            "action": "rollback_checkpoint",
            "auto_execute": False
        })
        self.intervention_rules[AlertLevel.RED].append({
            "id": "red_emergency_stop",
            "description": "紧急停止",
            "action": "emergency_stop",
            "auto_execute": False
        })

    def handle_alert(self, alert: AlertRecord, metrics: Dict[str, float], step: int) -> List[Dict[str, Any]]:
        executed = []

        for rule in self.intervention_rules.get(alert.level, []):
            if rule.get("auto_execute", False):
                result = self._execute_intervention(rule, alert, metrics, step)
                executed.append(result)

        return executed

    def _execute_intervention(self, rule: Dict[str, Any], alert: AlertRecord,
                              metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        self.intervention_counter += 1
        record = {
            "intervention_id": f"intervention_{self.intervention_counter}",
            "rule_id": rule.get("id"),
            "action": rule.get("action"),
            "alert_id": alert.alert_id,
            "alert_level": alert.level.value,
            "step": step,
            "timestamp": time.time(),
            "status": "executed",
            "details": {}
        }

        action = rule.get("action")
        if action == "log":
            record["details"]["message"] = f"Alert logged: {alert.description}"
        elif action == "adjust_learning_rate":
            record["details"]["factor"] = rule.get("factor", 0.8)
            record["details"]["description"] = "Learning rate adjustment requested"
        elif action == "pause_training":
            record["details"]["description"] = "Training pause requested"
        elif action == "rollback_checkpoint":
            record["details"]["description"] = "Checkpoint rollback requested"
        elif action == "emergency_stop":
            record["details"]["description"] = "Emergency stop requested"

        self.intervention_history.append(record)
        return record

    def get_intervention_summary(self) -> Dict[str, Any]:
        action_counts = {}
        for intervention in self.intervention_history:
            action = intervention.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_interventions": len(self.intervention_history),
            "action_distribution": action_counts,
            "active_rules": sum(len(rules) for rules in self.intervention_rules.values())
        }
