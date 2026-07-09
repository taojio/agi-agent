"""
self_improvement/regression_validator.py - 回归验证框架

对比基线指标、检测性能退化、验证改进效果。
防止自我改进过程中引入退化。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class RegressionSeverity(Enum):
    """退化严重度"""
    NONE = 0
    MINOR = 1       # <5% 退化
    MODERATE = 2    # 5-15% 退化
    MAJOR = 3       # 15-30% 退化
    CRITICAL = 4    # >30% 退化


class ValidationStatus(Enum):
    """验证状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RegressionItem:
    """退化项"""
    metric_name: str
    baseline_value: float
    current_value: float
    relative_change: float     # 相对变化（百分比）
    absolute_change: float
    severity: RegressionSeverity
    is_regression: bool        # 是否为退化（方向相反）
    higher_is_better: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": float(self.baseline_value),
            "current_value": float(self.current_value),
            "relative_change_pct": float(self.relative_change * 100),
            "absolute_change": float(self.absolute_change),
            "severity": self.severity.name,
            "is_regression": self.is_regression,
            "higher_is_better": self.higher_is_better,
        }


@dataclass
class RegressionReport:
    """退化报告"""
    report_id: str
    timestamp: float
    baseline_label: str
    regressions: List[RegressionItem] = field(default_factory=list)
    improvements: List[RegressionItem] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    overall_status: str = "passed"  # passed / failed / warning
    severity: RegressionSeverity = RegressionSeverity.NONE
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "baseline_label": self.baseline_label,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "unchanged": self.unchanged,
            "overall_status": self.overall_status,
            "severity": self.severity.name,
            "summary": self.summary,
        }


@dataclass
class ValidationResult:
    """改进验证结果"""
    validation_id: str
    improvement_id: str
    status: ValidationStatus
    started_at: float
    completed_at: Optional[float] = None
    report: Optional[RegressionReport] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def duration(self) -> float:
        return (self.completed_at or time.time()) - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "improvement_id": self.improvement_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "report": self.report.to_dict() if self.report else None,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "notes": self.notes,
        }


class RegressionValidator:
    """回归验证框架

    执行完整的回归测试套件，对比基线，检测退化。
    改进提案必须通过回归验证才能部署。

    Attributes:
        baselines: 已保存的基线 {label: metrics_dict}
        test_suites: 测试套件
        validations: 验证历史
    """

    def __init__(self, regression_threshold: float = 0.05,
                 critical_threshold: float = 0.15):
        self.regression_threshold = regression_threshold  # 5%
        self.critical_threshold = critical_threshold      # 15%
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.test_suites: Dict[str, Callable] = {}
        self.validations: deque = deque(maxlen=100)
        self._validation_counter = 0
        self._report_counter = 0

        # 退化容忍度（按指标）
        self.tolerance_overrides: Dict[str, float] = {}
        # 改进必须达到的最低收益
        self.min_improvement_gain = 0.02

    def save_baseline(self, label: str, metrics: Dict[str, float]) -> None:
        """保存基线"""
        self.baselines[label] = dict(metrics)

    def get_baseline(self, label: str) -> Optional[Dict[str, float]]:
        return self.baselines.get(label)

    def register_test_suite(self, name: str, runner: Callable[[], Dict[str, float]]) -> None:
        """注册测试套件"""
        self.test_suites[name] = runner

    def set_tolerance(self, metric_name: str, tolerance: float) -> None:
        """设置特定指标的退化容忍度"""
        self.tolerance_overrides[metric_name] = tolerance

    def detect_regression(self, baseline: Dict[str, float],
                           current: Dict[str, float],
                           higher_is_better: Optional[Dict[str, bool]] = None
                           ) -> RegressionReport:
        """检测退化

        Args:
            baseline: 基线指标
            current: 当前指标
            higher_is_better: 各指标是否值越高越好

        Returns:
            RegressionReport: 退化报告
        """
        self._report_counter += 1
        report = RegressionReport(
            report_id=f"report_{self._report_counter}",
            timestamp=time.time(),
            baseline_label="custom",
        )

        higher_better_map = higher_is_better or {}
        max_severity = RegressionSeverity.NONE
        has_regression = False

        for metric_name, baseline_value in baseline.items():
            if metric_name not in current:
                continue
            current_value = current[metric_name]
            higher_better = higher_better_map.get(metric_name, True)
            tolerance = self.tolerance_overrides.get(metric_name, self.regression_threshold)

            abs_change = current_value - baseline_value
            rel_change = abs_change / abs(baseline_value) if baseline_value != 0 else 0.0

            # 判断是否为退化
            is_regression = (abs_change < 0 and higher_better) or \
                            (abs_change > 0 and not higher_better)

            # 严重度
            magnitude = abs(rel_change)
            if magnitude < tolerance:
                severity = RegressionSeverity.NONE
            elif magnitude < 0.10:
                severity = RegressionSeverity.MINOR
            elif magnitude < self.critical_threshold:  # < 0.15
                severity = RegressionSeverity.MODERATE
            elif magnitude < 0.30:
                severity = RegressionSeverity.MAJOR
            else:
                severity = RegressionSeverity.CRITICAL

            item = RegressionItem(
                metric_name=metric_name,
                baseline_value=float(baseline_value),
                current_value=float(current_value),
                relative_change=float(rel_change),
                absolute_change=float(abs_change),
                severity=severity,
                is_regression=is_regression,
                higher_is_better=higher_better,
            )

            if severity == RegressionSeverity.NONE:
                report.unchanged.append(metric_name)
            elif is_regression:
                report.regressions.append(item)
                has_regression = True
                if severity.value > max_severity.value:
                    max_severity = severity
            else:
                report.improvements.append(item)

        # 整体状态
        if max_severity.value >= RegressionSeverity.CRITICAL.value:
            report.overall_status = "failed"
        elif max_severity.value >= RegressionSeverity.MODERATE.value:
            report.overall_status = "failed"
        elif max_severity.value >= RegressionSeverity.MINOR.value:
            report.overall_status = "warning"
        else:
            report.overall_status = "passed"

        report.severity = max_severity
        report.summary = (f"Regressions: {len(report.regressions)}, "
                         f"Improvements: {len(report.improvements)}, "
                         f"Unchanged: {len(report.unchanged)}, "
                         f"Status: {report.overall_status}")

        return report

    def validate_improvement(self, improvement_id: str,
                               metrics_before: Dict[str, float],
                               metrics_after: Dict[str, float],
                               higher_is_better: Optional[Dict[str, bool]] = None,
                               notes: str = "") -> ValidationResult:
        """验证改进效果

        Args:
            improvement_id: 改进 ID
            metrics_before: 改进前指标
            metrics_after: 改进后指标
            higher_is_better: 各指标方向
            notes: 备注

        Returns:
            ValidationResult: 验证结果
        """
        self._validation_counter += 1
        validation = ValidationResult(
            validation_id=f"val_{self._validation_counter}",
            improvement_id=improvement_id,
            status=ValidationStatus.RUNNING,
            started_at=time.time(),
            metrics_before=dict(metrics_before),
            metrics_after=dict(metrics_after),
            notes=notes,
        )

        # 检测退化
        report = self.detect_regression(metrics_before, metrics_after, higher_is_better)
        validation.report = report

        # 判断是否通过
        # CRITICAL 退化直接失败
        if report.severity == RegressionSeverity.CRITICAL:
            validation.status = ValidationStatus.FAILED
            validation.notes = (f"Critical regression detected: {len(report.regressions)} metrics "
                               f"regressed (severity: {report.severity.name})")
        elif report.severity == RegressionSeverity.MAJOR:
            # MAJOR 退化：检查改进是否能显著抵消
            improvement_magnitudes = [abs(i.relative_change) for i in report.improvements]
            regression_magnitudes = [abs(r.relative_change) for r in report.regressions]
            total_improvement = sum(improvement_magnitudes)
            total_regression = sum(regression_magnitudes)
            # MAJOR 退化需要改进 2 倍以上才能通过
            if total_improvement >= total_regression * 2 + self.min_improvement_gain:
                validation.status = ValidationStatus.PASSED
                validation.notes = (f"Passed despite MAJOR regression: improvements ({total_improvement:.3f}) "
                                   f"significantly outweigh regressions ({total_regression:.3f})")
            else:
                validation.status = ValidationStatus.FAILED
                validation.notes = (f"MAJOR regression not sufficiently compensated: "
                                   f"improvements {total_improvement:.3f} vs "
                                   f"regressions {total_regression:.3f}")
        elif report.overall_status == "failed":
            # MODERATE 退化：检查改进是否抵消
            improvement_magnitudes = [abs(i.relative_change) for i in report.improvements]
            regression_magnitudes = [abs(r.relative_change) for r in report.regressions]
            total_improvement = sum(improvement_magnitudes)
            total_regression = sum(regression_magnitudes)
            if total_improvement - total_regression >= self.min_improvement_gain:
                validation.status = ValidationStatus.PASSED
                validation.notes = (f"Passed with regressions: improvements ({total_improvement:.3f}) "
                                   f"outweigh regressions ({total_regression:.3f})")
            else:
                validation.status = ValidationStatus.FAILED
                validation.notes = (f"Insufficient improvement: {total_improvement:.3f} vs "
                                   f"regression {total_regression:.3f}")
        elif report.overall_status == "warning":
            # MINOR 退化：检查改进是否抵消
            improvement_magnitudes = [abs(i.relative_change) for i in report.improvements]
            regression_magnitudes = [abs(r.relative_change) for r in report.regressions]
            total_improvement = sum(improvement_magnitudes)
            total_regression = sum(regression_magnitudes)
            if total_improvement - total_regression >= self.min_improvement_gain:
                validation.status = ValidationStatus.PASSED
                validation.notes = (f"Passed with warnings: improvements ({total_improvement:.3f}) "
                                   f"outweigh regressions ({total_regression:.3f})")
            else:
                validation.status = ValidationStatus.FAILED
                validation.notes = (f"Insufficient improvement: {total_improvement:.3f} vs "
                                   f"regression {total_regression:.3f}")
        else:
            validation.status = ValidationStatus.PASSED
            if report.improvements:
                validation.notes = (f"Passed: {len(report.improvements)} metrics improved, "
                                   f"no regressions")
            else:
                validation.notes = "Passed: no changes detected"

        validation.completed_at = time.time()
        self.validations.append(validation)
        return validation

    def run_full_regression(self, test_suite_name: str = "default",
                              baseline_label: Optional[str] = None) -> RegressionReport:
        """运行完整回归测试"""
        if test_suite_name not in self.test_suites:
            return RegressionReport(
                report_id=f"report_{self._report_counter + 1}",
                timestamp=time.time(),
                baseline_label=baseline_label or "unknown",
                summary=f"Test suite '{test_suite_name}' not found",
                overall_status="failed",
            )

        current_metrics = self.test_suites[test_suite_name]()

        if baseline_label and baseline_label in self.baselines:
            baseline = self.baselines[baseline_label]
        elif self.baselines:
            # 使用最新基线
            baseline_label = list(self.baselines.keys())[-1]
            baseline = self.baselines[baseline_label]
        else:
            return RegressionReport(
                report_id=f"report_{self._report_counter + 1}",
                timestamp=time.time(),
                baseline_label="none",
                summary="No baseline available",
                overall_status="failed",
            )

        report = self.detect_regression(baseline, current_metrics)
        report.baseline_label = baseline_label
        return report

    def is_improvement_validated(self, improvement_id: str) -> bool:
        """检查改进是否已通过验证"""
        for v in self.validations:
            if v.improvement_id == improvement_id:
                return v.status == ValidationStatus.PASSED
        return False

    def get_validation_history(self, limit: int = 20) -> List[ValidationResult]:
        return list(self.validations)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        validation_list = list(self.validations)
        passed = sum(1 for v in validation_list if v.status == ValidationStatus.PASSED)
        failed = sum(1 for v in validation_list if v.status == ValidationStatus.FAILED)
        return {
            "total_baselines": len(self.baselines),
            "total_test_suites": len(self.test_suites),
            "total_validations": len(validation_list),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(validation_list) if validation_list else 0.0,
            "regression_threshold": self.regression_threshold,
            "critical_threshold": self.critical_threshold,
        }
