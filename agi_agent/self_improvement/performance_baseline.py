"""
self_improvement/performance_baseline.py - 性能基准系统

提供4类性能基准：
- 核心性能基准：自由能、收敛速度、预测准确率、决策质量
- 模块性能基准：延迟、吞吐量、内存占用、错误率
- 集成性能基准：端到端延迟、协同效率、数据一致性
- 稳定性基准：故障频率、平均恢复时间、资源泄漏率
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class BaselineType(Enum):
    """基准类型"""
    CORE = "core"                  # 核心性能
    MODULE = "module"              # 模块性能
    INTEGRATION = "integration"    # 集成性能
    STABILITY = "stability"        # 稳定性


class TrendDirection(Enum):
    """趋势方向"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass
class MetricSample:
    """指标样本"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    baseline_type: BaselineType
    description: str
    unit: str
    target: Optional[float] = None       # 目标值
    warning_threshold: Optional[float] = None  # 告警阈值
    critical_threshold: Optional[float] = None  # 严重阈值
    higher_is_better: bool = True        # 值越高越好
    samples: deque = field(default_factory=lambda: deque(maxlen=500))
    baseline_value: Optional[float] = None

    def add_sample(self, value: float, **metadata) -> None:
        self.samples.append(MetricSample(
            timestamp=time.time(),
            value=float(value),
            metadata=metadata,
        ))
        # 自动设置基线（前10个样本的均值）
        if self.baseline_value is None and len(self.samples) >= 10:
            self.baseline_value = float(np.mean([s.value for s in list(self.samples)[:10]]))

    @property
    def latest(self) -> Optional[float]:
        return self.samples[-1].value if self.samples else None

    @property
    def average(self) -> Optional[float]:
        if not self.samples:
            return None
        return float(np.mean([s.value for s in self.samples]))

    @property
    def percentile_95(self) -> Optional[float]:
        if not self.samples:
            return None
        return float(np.percentile([s.value for s in self.samples], 95))

    @property
    def trend(self) -> TrendDirection:
        if len(self.samples) < 5:
            return TrendDirection.STABLE
        values = [s.value for s in list(self.samples)[-20:]]
        if len(values) < 5:
            return TrendDirection.STABLE
        # 线性回归斜率
        try:
            x_arr = np.arange(len(values))
            y_arr = np.array(values)
            slope = float(np.polyfit(x_arr, y_arr, 1)[0])
            std = float(np.std(y_arr))
            mean = float(np.mean(y_arr))
            # 计算 R² 判断线性拟合度（高 R² 表示真趋势，低 R² 表示震荡）
            if std > 0:
                predicted = slope * x_arr + (mean - slope * (len(values) - 1) / 2)
                ss_res = float(np.sum((y_arr - predicted) ** 2))
                ss_tot = float(np.sum((y_arr - mean) ** 2))
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            else:
                r_squared = 1.0
            # 震荡判断：R² 低且变异系数高才算震荡
            cv = std / abs(mean) if mean != 0 else 0.0
            if r_squared < 0.5 and cv > 0.3:
                return TrendDirection.VOLATILE
            if abs(slope) < 1e-6:
                return TrendDirection.STABLE
            if self.higher_is_better:
                return TrendDirection.IMPROVING if slope > 0 else TrendDirection.DECLINING
            else:
                return TrendDirection.IMPROVING if slope < 0 else TrendDirection.DECLINING
        except (ValueError, np.linalg.LinAlgError):
            return TrendDirection.STABLE

    def is_warning(self) -> bool:
        if self.warning_threshold is None or self.latest is None:
            return False
        if self.higher_is_better:
            return self.latest < self.warning_threshold
        return self.latest > self.warning_threshold

    def is_critical(self) -> bool:
        if self.critical_threshold is None or self.latest is None:
            return False
        if self.higher_is_better:
            return self.latest < self.critical_threshold
        return self.latest > self.critical_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "baseline_type": self.baseline_type.value,
            "description": self.description,
            "unit": self.unit,
            "target": self.target,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "higher_is_better": self.higher_is_better,
            "latest": self.latest,
            "average": self.average,
            "p95": self.percentile_95,
            "trend": self.trend.value,
            "is_warning": self.is_warning(),
            "is_critical": self.is_critical(),
            "baseline_value": self.baseline_value,
            "sample_count": len(self.samples),
        }


class PerformanceBaseline:
    """性能基准系统

    管理4类性能指标，提供基线对比、退化检测、趋势分析。

    Attributes:
        metrics: 指标库 {metric_name: PerformanceMetric}
    """

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self._init_default_metrics()
        self._baseline_snapshots: deque = deque(maxlen=20)

    def _init_default_metrics(self) -> None:
        """初始化默认性能指标"""
        # 核心性能基准
        defaults = [
            ("free_energy", BaselineType.CORE, "自由能", "[0-1]",
             0.3, 0.6, 0.8, False),
            ("convergence_speed", BaselineType.CORE, "收敛速度", "steps",
             100, 200, 500, False),
            ("prediction_accuracy", BaselineType.CORE, "预测准确率", "%",
             0.85, 0.7, 0.5, True),
            ("decision_quality", BaselineType.CORE, "决策质量", "[0-1]",
             0.8, 0.6, 0.4, True),
            # 模块性能基准
            ("module_latency", BaselineType.MODULE, "模块延迟", "ms",
             50, 100, 500, False),
            ("module_throughput", BaselineType.MODULE, "吞吐量", "req/s",
             1000, 500, 100, True),
            ("memory_usage", BaselineType.MODULE, "内存占用", "%",
             70, 85, 95, False),
            ("error_rate", BaselineType.MODULE, "错误率", "%",
             1, 5, 10, False),
            # 集成性能基准
            ("end_to_end_latency", BaselineType.INTEGRATION, "端到端延迟", "ms",
             200, 500, 1000, False),
            ("coordination_efficiency", BaselineType.INTEGRATION, "协同效率", "[0-1]",
             0.8, 0.6, 0.4, True),
            ("data_consistency", BaselineType.INTEGRATION, "数据一致性", "%",
             99, 95, 90, True),
            # 稳定性基准
            ("failure_frequency", BaselineType.STABILITY, "故障频率", "per_hour",
             0.1, 0.5, 1.0, False),
            ("mttr", BaselineType.STABILITY, "平均恢复时间", "seconds",
             60, 300, 600, False),
            ("resource_leak_rate", BaselineType.STABILITY, "资源泄漏率", "per_hour",
             0.01, 0.1, 0.5, False),
        ]
        for name, btype, desc, unit, target, warn, crit, higher_better in defaults:
            self.metrics[name] = PerformanceMetric(
                name=name,
                baseline_type=btype,
                description=desc,
                unit=unit,
                target=target,
                warning_threshold=warn,
                critical_threshold=crit,
                higher_is_better=higher_better,
            )

    def register_metric(self, name: str, baseline_type: BaselineType,
                         description: str, unit: str = "",
                         target: Optional[float] = None,
                         warning_threshold: Optional[float] = None,
                         critical_threshold: Optional[float] = None,
                         higher_is_better: bool = True) -> None:
        """注册自定义指标"""
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(
                name=name,
                baseline_type=baseline_type,
                description=description,
                unit=unit,
                target=target,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                higher_is_better=higher_is_better,
            )

    def record(self, metric_name: str, value: float, **metadata) -> None:
        """记录指标值"""
        if metric_name not in self.metrics:
            # 自动注册
            self.register_metric(metric_name, BaselineType.MODULE,
                                 f"Auto-registered: {metric_name}")
        self.metrics[metric_name].add_sample(value, **metadata)

    def record_batch(self, metrics: Dict[str, float]) -> None:
        """批量记录指标"""
        for name, value in metrics.items():
            self.record(name, value)

    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        return self.metrics.get(name)

    def get_metrics_by_type(self, baseline_type: BaselineType) -> List[PerformanceMetric]:
        return [m for m in self.metrics.values() if m.baseline_type == baseline_type]

    def capture_baseline(self, label: str = "") -> Dict[str, Any]:
        """捕获当前基线快照"""
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "metrics": {name: m.average for name, m in self.metrics.items()
                        if m.samples},
        }
        self._baseline_snapshots.append(snapshot)
        # 更新各指标的基线值
        for name, metric in self.metrics.items():
            if metric.samples and name in snapshot["metrics"]:
                metric.baseline_value = snapshot["metrics"][name]
        return snapshot

    def compare_with_baseline(self, baseline_snapshot: Optional[Dict[str, Any]] = None
                               ) -> Dict[str, Any]:
        """与基线对比"""
        if baseline_snapshot is None:
            if not self._baseline_snapshots:
                return {"status": "no_baseline"}
            baseline_snapshot = self._baseline_snapshots[-1]

        baseline_metrics = baseline_snapshot.get("metrics", {})
        improvements = []
        regressions = []
        unchanged = []

        for name, baseline_value in baseline_metrics.items():
            if name not in self.metrics:
                continue
            current = self.metrics[name].average
            if current is None:
                continue
            diff = current - baseline_value
            relative = diff / baseline_value if baseline_value != 0 else 0.0

            entry = {
                "metric": name,
                "baseline": float(baseline_value),
                "current": float(current),
                "diff": float(diff),
                "relative_change": float(relative),
            }
            # 判断改善/退化
            if self.metrics[name].higher_is_better:
                if diff > 0.01 * abs(baseline_value):
                    improvements.append(entry)
                elif diff < -0.01 * abs(baseline_value):
                    regressions.append(entry)
                else:
                    unchanged.append(entry)
            else:
                if diff < -0.01 * abs(baseline_value):
                    improvements.append(entry)
                elif diff > 0.01 * abs(baseline_value):
                    regressions.append(entry)
                else:
                    unchanged.append(entry)

        return {
            "baseline_label": baseline_snapshot.get("label", ""),
            "baseline_timestamp": baseline_snapshot.get("timestamp"),
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": unchanged,
            "improvement_count": len(improvements),
            "regression_count": len(regressions),
        }

    def detect_regression(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """检测性能退化"""
        regressions = []
        for name, metric in self.metrics.items():
            if metric.baseline_value is None or metric.average is None:
                continue
            diff = metric.average - metric.baseline_value
            relative = abs(diff) / max(abs(metric.baseline_value), 1e-6)
            if relative >= threshold:
                is_regression = (diff < 0 and metric.higher_is_better) or \
                                (diff > 0 and not metric.higher_is_better)
                if is_regression:
                    regressions.append({
                        "metric": name,
                        "baseline": float(metric.baseline_value),
                        "current": float(metric.average),
                        "regression_pct": float(relative * 100),
                        "trend": metric.trend.value,
                        "is_warning": metric.is_warning(),
                        "is_critical": metric.is_critical(),
                    })
        regressions.sort(key=lambda x: -x["regression_pct"])
        return regressions

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        warnings = []
        criticals = []
        for name, metric in self.metrics.items():
            if not metric.samples:
                continue
            if metric.is_critical():
                criticals.append(name)
            elif metric.is_warning():
                warnings.append(name)

        return {
            "total_metrics": len(self.metrics),
            "active_metrics": sum(1 for m in self.metrics.values() if m.samples),
            "warnings": warnings,
            "criticals": criticals,
            "warning_count": len(warnings),
            "critical_count": len(criticals),
            "overall_status": "critical" if criticals else \
                              "warning" if warnings else "healthy",
            "baselines_captured": len(self._baseline_snapshots),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取完整仪表盘数据"""
        return {
            "core_metrics": [m.to_dict() for m in self.get_metrics_by_type(BaselineType.CORE)],
            "module_metrics": [m.to_dict() for m in self.get_metrics_by_type(BaselineType.MODULE)],
            "integration_metrics": [m.to_dict() for m in self.get_metrics_by_type(BaselineType.INTEGRATION)],
            "stability_metrics": [m.to_dict() for m in self.get_metrics_by_type(BaselineType.STABILITY)],
            "health": self.get_health_summary(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_metrics": len(self.metrics),
            "total_samples": sum(len(m.samples) for m in self.metrics.values()),
            "baselines_captured": len(self._baseline_snapshots),
            "metrics_by_type": {
                btype.value: sum(1 for m in self.metrics.values()
                                  if m.baseline_type == btype)
                for btype in BaselineType
            },
        }
