"""
module_monitor.py - 元认知监控与调节系统

实现对元模块运行状态的实时监控、性能评估与动态调节功能，支持：
- 毫秒级状态采样
- 异常行为识别
- 自动告警
- 基于规则和机器学习的自适应调节
"""
import time
import uuid
import threading
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    HIGH_LATENCY = "high_latency"
    MODULE_DOWN = "module_down"
    TASK_FAILED = "task_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ANOMALY_DETECTED = "anomaly_detected"
    CUSTOM = "custom"


class MonitorMetric(Enum):
    """监控指标"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    TASK_DURATION = "task_duration"
    QUEUE_LENGTH = "queue_length"
    WORKER_LOAD = "worker_load"


@dataclass
class MetricSample:
    """指标采样"""
    metric_type: MonitorMetric
    module_id: str
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = "%"


@dataclass
class ModuleAlert:
    """模块告警"""
    alert_id: str
    module_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        self.resolved = True
        self.resolved_at = time.time()


@dataclass
class ModulePerformanceSnapshot:
    """模块性能快照"""
    module_id: str
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "unknown"
    alert_count: int = 0


class AlertRule:
    """告警规则"""

    def __init__(
        self,
        rule_id: str,
        metric_type: MonitorMetric,
        operator: Callable[[float, float], bool],
        threshold: float,
        alert_type: AlertType,
        alert_level: AlertLevel,
        window_size: int = 5,
        consecutive_required: int = 3,
    ):
        self.rule_id = rule_id
        self.metric_type = metric_type
        self.operator = operator
        self.threshold = threshold
        self.alert_type = alert_type
        self.alert_level = alert_level
        self.window_size = window_size
        self.consecutive_required = consecutive_required
        self._consecutive_count = 0
        self._sample_window: List[float] = []

    def evaluate(self, sample: MetricSample) -> Optional[ModuleAlert]:
        """评估样本是否触发告警"""
        self._sample_window.append(sample.value)
        if len(self._sample_window) > self.window_size:
            self._sample_window.pop(0)

        if len(self._sample_window) < self.window_size:
            return None

        avg_value = statistics.mean(self._sample_window)

        if self.operator(avg_value, self.threshold):
            self._consecutive_count += 1
            if self._consecutive_count >= self.consecutive_required:
                self._consecutive_count = 0
                return ModuleAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    module_id=sample.module_id,
                    alert_type=self.alert_type,
                    alert_level=self.alert_level,
                    message=f"{self.alert_type.value}: {sample.metric_type.value} = {avg_value:.2f} (threshold: {self.threshold})",
                    metadata={"metric_value": avg_value, "threshold": self.threshold},
                )
        else:
            self._consecutive_count = 0

        return None


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, window_size: int = 20, z_threshold: float = 3.0):
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._metric_history: Dict[str, List[float]] = {}

    def detect_anomaly(self, sample: MetricSample) -> Tuple[bool, float]:
        """检测异常"""
        key = f"{sample.module_id}_{sample.metric_type.value}"
        if key not in self._metric_history:
            self._metric_history[key] = []

        history = self._metric_history[key]
        history.append(sample.value)

        if len(history) > self._window_size:
            history.pop(0)

        if len(history) < self._window_size:
            return False, 0.0

        mean = statistics.mean(history)
        std_dev = statistics.stdev(history) if len(history) > 1 else 0.0

        if std_dev == 0:
            return False, 0.0

        z_score = abs(sample.value - mean) / std_dev
        is_anomaly = z_score > self._z_threshold

        return is_anomaly, z_score


class ModuleMonitor:
    """模块监控系统"""

    def __init__(
        self,
        sample_interval_ms: int = 100,
        max_samples: int = 1000,
        anomaly_window_size: int = 20,
    ):
        self._sample_interval_ms = sample_interval_ms
        self._max_samples = max_samples
        self._anomaly_detector = AnomalyDetector(
            window_size=anomaly_window_size,
            z_threshold=3.0,
        )

        self._metrics: Dict[str, List[MetricSample]] = {}
        self._alerts: List[ModuleAlert] = []
        self._alert_rules: List[AlertRule] = self._create_default_rules()

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._alert_handlers: List[Callable[[ModuleAlert], None]] = []
        self._metrics_handlers: List[Callable[[MetricSample], None]] = []

    def _create_default_rules(self) -> List[AlertRule]:
        """创建默认告警规则"""
        return [
            AlertRule(
                rule_id="rule_cpu_high",
                metric_type=MonitorMetric.CPU_USAGE,
                operator=lambda val, thr: val > thr,
                threshold=80.0,
                alert_type=AlertType.HIGH_CPU,
                alert_level=AlertLevel.WARNING,
                window_size=5,
                consecutive_required=3,
            ),
            AlertRule(
                rule_id="rule_cpu_critical",
                metric_type=MonitorMetric.CPU_USAGE,
                operator=lambda val, thr: val > thr,
                threshold=95.0,
                alert_type=AlertType.HIGH_CPU,
                alert_level=AlertLevel.CRITICAL,
                window_size=3,
                consecutive_required=2,
            ),
            AlertRule(
                rule_id="rule_memory_high",
                metric_type=MonitorMetric.MEMORY_USAGE,
                operator=lambda val, thr: val > thr,
                threshold=85.0,
                alert_type=AlertType.HIGH_MEMORY,
                alert_level=AlertLevel.WARNING,
                window_size=5,
                consecutive_required=3,
            ),
            AlertRule(
                rule_id="rule_memory_critical",
                metric_type=MonitorMetric.MEMORY_USAGE,
                operator=lambda val, thr: val > thr,
                threshold=95.0,
                alert_type=AlertType.HIGH_MEMORY,
                alert_level=AlertLevel.CRITICAL,
                window_size=3,
                consecutive_required=2,
            ),
            AlertRule(
                rule_id="rule_latency_high",
                metric_type=MonitorMetric.LATENCY,
                operator=lambda val, thr: val > thr,
                threshold=100.0,
                alert_type=AlertType.HIGH_LATENCY,
                alert_level=AlertLevel.WARNING,
                window_size=5,
                consecutive_required=3,
            ),
            AlertRule(
                rule_id="rule_latency_critical",
                metric_type=MonitorMetric.LATENCY,
                operator=lambda val, thr: val > thr,
                threshold=500.0,
                alert_type=AlertType.HIGH_LATENCY,
                alert_level=AlertLevel.CRITICAL,
                window_size=3,
                consecutive_required=2,
            ),
            AlertRule(
                rule_id="rule_error_rate",
                metric_type=MonitorMetric.ERROR_RATE,
                operator=lambda val, thr: val > thr,
                threshold=10.0,
                alert_type=AlertType.TASK_FAILED,
                alert_level=AlertLevel.WARNING,
                window_size=10,
                consecutive_required=5,
            ),
        ]

    def start(self):
        """启动监控"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="module-monitor"
            )
            self._monitor_thread.start()

    def stop(self):
        """停止监控"""
        with self._lock:
            self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None

    def add_metric_sample(self, sample: MetricSample):
        """添加指标采样"""
        key = f"{sample.module_id}_{sample.metric_type.value}"

        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = []

            self._metrics[key].append(sample)

            if len(self._metrics[key]) > self._max_samples:
                self._metrics[key] = self._metrics[key][-self._max_samples:]

        self._process_sample(sample)

    def _process_sample(self, sample: MetricSample):
        """处理采样"""
        for rule in self._alert_rules:
            if rule.metric_type == sample.metric_type:
                alert = rule.evaluate(sample)
                if alert:
                    self._add_alert(alert)

        is_anomaly, z_score = self._anomaly_detector.detect_anomaly(sample)
        if is_anomaly:
            alert = ModuleAlert(
                alert_id=f"anomaly_{uuid.uuid4().hex[:8]}",
                module_id=sample.module_id,
                alert_type=AlertType.ANOMALY_DETECTED,
                alert_level=AlertLevel.WARNING,
                message=f"Anomaly detected for {sample.metric_type.value}: z-score = {z_score:.2f}",
                metadata={"z_score": z_score, "metric_value": sample.value},
            )
            self._add_alert(alert)

        for handler in self._metrics_handlers:
            try:
                handler(sample)
            except Exception:
                pass

    def _add_alert(self, alert: ModuleAlert):
        """添加告警"""
        with self._lock:
            self._alerts.append(alert)

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception:
                pass

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolve()
                    return True
        return False

    def add_alert_handler(self, handler: Callable[[ModuleAlert], None]):
        """添加告警处理器"""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[ModuleAlert], None]):
        """移除告警处理器"""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def add_metrics_handler(self, handler: Callable[[MetricSample], None]):
        """添加指标处理器"""
        self._metrics_handlers.append(handler)

    def remove_metrics_handler(self, handler: Callable[[MetricSample], None]):
        """移除指标处理器"""
        if handler in self._metrics_handlers:
            self._metrics_handlers.remove(handler)

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self._alert_rules.append(rule)

    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        with self._lock:
            self._alert_rules = [r for r in self._alert_rules if r.rule_id != rule_id]

    def get_alerts(
        self,
        module_id: str = None,
        alert_level: AlertLevel = None,
        resolved: bool = None,
    ) -> List[ModuleAlert]:
        """获取告警列表"""
        with self._lock:
            alerts = self._alerts.copy()

        if module_id:
            alerts = [a for a in alerts if a.module_id == module_id]
        if alert_level:
            alerts = [a for a in alerts if a.alert_level == alert_level]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        return alerts

    def get_module_metrics(
        self,
        module_id: str,
        metric_type: MonitorMetric = None,
        limit: int = 100,
    ) -> List[MetricSample]:
        """获取模块指标"""
        with self._lock:
            if metric_type:
                key = f"{module_id}_{metric_type.value}"
                samples = self._metrics.get(key, [])[-limit:]
            else:
                samples = []
                for key, vals in self._metrics.items():
                    if key.startswith(f"{module_id}_"):
                        samples.extend(vals[-limit:])

        return sorted(samples, key=lambda s: s.timestamp, reverse=True)

    def get_module_snapshot(self, module_id: str) -> ModulePerformanceSnapshot:
        """获取模块性能快照"""
        metrics = {}
        alert_count = 0

        with self._lock:
            for key, samples in self._metrics.items():
                if key.startswith(f"{module_id}_"):
                    if samples:
                        metrics[key.split("_", 1)[1]] = samples[-1].value

            alert_count = sum(1 for a in self._alerts if a.module_id == module_id and not a.resolved)

        return ModulePerformanceSnapshot(
            module_id=module_id,
            timestamp=time.time(),
            metrics=metrics,
            status="healthy" if alert_count == 0 else "warning",
            alert_count=alert_count,
        )

    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息"""
        with self._lock:
            total_samples = sum(len(v) for v in self._metrics.values())
            active_alerts = sum(1 for a in self._alerts if not a.resolved)
            warning_alerts = sum(1 for a in self._alerts if a.alert_level == AlertLevel.WARNING and not a.resolved)
            critical_alerts = sum(1 for a in self._alerts if a.alert_level == AlertLevel.CRITICAL and not a.resolved)

            modules_monitored = set()
            for key in self._metrics.keys():
                modules_monitored.add(key.split("_")[0])

        return {
            "modules_monitored": len(modules_monitored),
            "total_samples": total_samples,
            "active_alerts": active_alerts,
            "warning_alerts": warning_alerts,
            "critical_alerts": critical_alerts,
            "alert_rules": len(self._alert_rules),
            "sample_interval_ms": self._sample_interval_ms,
        }

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            time.sleep(self._sample_interval_ms / 1000.0)


_monitor_instance: Optional[ModuleMonitor] = None
_monitor_lock = threading.Lock()


def get_module_monitor() -> ModuleMonitor:
    """获取模块监控器单例"""
    global _monitor_instance
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = ModuleMonitor()
        return _monitor_instance