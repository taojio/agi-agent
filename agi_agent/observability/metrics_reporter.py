"""
observability/metrics_reporter.py - 指标实时上报 (T018)

统计推理耗时、工具调用频次、内存/显存占用、任务成功率。
prometheus_client 可选导入，降级用纯字典维护指标。
"""
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.observability")

# 可选依赖：prometheus_client
try:  # pragma: no cover - 环境相关
    from prometheus_client import (  # type: ignore
        Counter as PromCounter,
        Gauge as PromGauge,
        Histogram as PromHistogram,
    )

    _HAS_PROM = True
except Exception:  # noqa: BLE001
    PromCounter = PromGauge = PromHistogram = None  # type: ignore
    _HAS_PROM = False

# psutil 可用
try:  # pragma: no cover - 环境相关
    import psutil

    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


@dataclass
class MetricsConfig:
    """指标上报配置"""
    use_prometheus: bool = True
    namespace: str = "agi_agent"
    default_buckets: Tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    collect_system: bool = True


class MetricsReporter(BaseModule):
    """指标实时上报器 (T018)

    提供 record / counter / gauge / histogram / snapshot / format_prometheus
    方法。优先复用 prometheus_client；不可用时降级为内存字典统计。
    """

    name = "metrics_reporter"
    version = "1.0.0"
    description = "指标实时上报 (T018)"

    def __init__(self, config: Optional[MetricsConfig] = None):
        super().__init__()
        self._cfg = config or MetricsConfig()
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._records: List[Dict[str, Any]] = []
        self._prom_counters: Dict[str, Any] = {}
        self._prom_gauges: Dict[str, Any] = {}
        self._prom_histograms: Dict[str, Any] = {}
        self._use_prom = self._cfg.use_prometheus and _HAS_PROM

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._use_prom:
            logger.info("MetricsReporter 使用 prometheus_client 后端")
        else:
            logger.info("MetricsReporter 使用内存字典后端")

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    @property
    def backend(self) -> str:
        return "prometheus" if self._use_prom else "dict"

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录一条原始指标

        Args:
            metric_name: 指标名
            value: 指标值
            labels: 标签
        """
        labels = labels or {}
        with self._lock:
            self._records.append(
                {
                    "name": metric_name,
                    "value": float(value),
                    "labels": dict(labels),
                    "timestamp": time.time(),
                }
            )
            if len(self._records) > 100000:
                self._records = self._records[-50000:]

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0) -> float:
        """计数器自增

        Args:
            name: 指标名
            labels: 标签
            amount: 自增量

        Returns:
            float: 自增后的当前值
        """
        labels = labels or {}
        key = self._label_key(labels)
        with self._lock:
            self._counters[name][key] += float(amount)
            val = self._counters[name][key]
        if self._use_prom:
            try:
                pc = self._prom_counters.get(name)
                if pc is None:
                    pc = PromCounter(self._safe_name(name), f"counter {name}", list(labels.keys()) if labels else [])
                    self._prom_counters[name] = pc
                pc.labels(**labels).inc(amount) if labels else pc.inc(amount)
            except Exception as e:  # noqa: BLE001
                logger.debug("prometheus counter 更新失败: %s", e)
        return val

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """设置仪表盘值

        Args:
            name: 指标名
            value: 当前值
            labels: 标签
        """
        labels = labels or {}
        key = self._label_key(labels)
        with self._lock:
            self._gauges[name][key] = float(value)
        if self._use_prom:
            try:
                pg = self._prom_gauges.get(name)
                if pg is None:
                    pg = PromGauge(self._safe_name(name), f"gauge {name}", list(labels.keys()) if labels else [])
                    self._prom_gauges[name] = pg
                pg.labels(**labels).set(value) if labels else pg.set(value)
            except Exception as e:  # noqa: BLE001
                logger.debug("prometheus gauge 更新失败: %s", e)

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """直方图观测

        Args:
            name: 指标名
            value: 观测值
            labels: 标签
        """
        with self._lock:
            self._histograms[name].append(float(value))
            if len(self._histograms[name]) > 100000:
                self._histograms[name] = self._histograms[name][-50000:]
        if self._use_prom:
            try:
                ph = self._prom_histograms.get(name)
                if ph is None:
                    ph = PromHistogram(
                        self._safe_name(name),
                        f"histogram {name}",
                        list(labels.keys()) if labels else [],
                        buckets=self._cfg.default_buckets,
                    )
                    self._prom_histograms[name] = ph
                ph.labels(**labels).observe(value) if labels else ph.observe(value)
            except Exception as e:  # noqa: BLE001
                logger.debug("prometheus histogram 更新失败: %s", e)

    def snapshot(self) -> Dict[str, Any]:
        """获取当前所有指标的快照

        Returns:
            dict: 指标快照
        """
        with self._lock:
            counters = {k: dict(v) for k, v in self._counters.items()}
            gauges = {k: dict(v) for k, v in self._gauges.items()}
            histograms = {
                k: {
                    "count": len(v),
                    "sum": float(sum(v)),
                    "avg": float(sum(v) / len(v)) if v else 0.0,
                    "min": float(min(v)) if v else 0.0,
                    "max": float(max(v)) if v else 0.0,
                }
                for k, v in self._histograms.items()
            }
        snap = {
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
            "system": self._system_metrics(),
        }
        return snap

    def format_prometheus(self) -> str:
        """格式化指标为 Prometheus 文本格式

        Returns:
            str: Prometheus exposition 文本
        """
        if self._use_prom:
            try:
                from prometheus_client import generate_latest  # type: ignore

                return generate_latest().decode("utf-8")
            except Exception as e:  # noqa: BLE001
                logger.debug("generate_latest 失败，使用字典回退: %s", e)
        snap = self.snapshot()
        lines: List[str] = []
        for name, label_map in snap["counters"].items():
            metric = self._safe_name(name)
            lines.append(f"# HELP {metric} counter")
            lines.append(f"# TYPE {metric} counter")
            for lkey, val in label_map.items():
                lines.append(f'{metric}{{{lkey}}} {val}' if lkey else f"{metric} {val}")
        for name, label_map in snap["gauges"].items():
            metric = self._safe_name(name)
            lines.append(f"# HELP {metric} gauge")
            lines.append(f"# TYPE {metric} gauge")
            for lkey, val in label_map.items():
                lines.append(f'{metric}{{{lkey}}} {val}' if lkey else f"{metric} {val}")
        for name, stats in snap["histograms"].items():
            metric = self._safe_name(name)
            lines.append(f"# HELP {metric} histogram")
            lines.append(f"# TYPE {metric} histogram")
            lines.append(f'{metric}_count {stats["count"]}')
            lines.append(f'{metric}_sum {stats["sum"]}')
            lines.append(f'{metric}_avg {stats["avg"]}')
        sysm = snap.get("system", {})
        if sysm:
            lines.append("# HELP agi_agent_system_info system metrics")
            lines.append("# TYPE agi_agent_system_info gauge")
            for k, v in sysm.items():
                lines.append(f'agi_agent_system_info{{kind="{k}"}} {v}')
        return "\n".join(lines) + ("\n" if lines else "")

    # ====== 内部 ======
    def _system_metrics(self) -> Dict[str, float]:
        if not self._cfg.collect_system or not _HAS_PSUTIL:
            return {}
        try:
            mem = psutil.virtual_memory()
            return {
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "memory_percent": float(mem.percent),
                "memory_used_bytes": float(mem.used),
            }
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _label_key(labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)
