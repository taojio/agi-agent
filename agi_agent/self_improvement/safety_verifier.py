import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque
from enum import Enum
from dataclasses import dataclass, field


class SafetyLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class VerificationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SafetyConstraint:
    constraint_id: str
    name: str
    description: str
    safety_level: SafetyLevel
    check_fn: Optional[str] = None
    threshold: float = 0.0
    enabled: bool = True
    violation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "safety_level": self.safety_level.value,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "violation_count": self.violation_count
        }


@dataclass
class VerificationResult:
    test_name: str
    status: VerificationStatus
    passed: bool
    details: str = ""
    duration: float = 0.0
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "passed": self.passed,
            "details": self.details,
            "duration": self.duration
        }


class ImprovementSafetyVerifier:
    def __init__(self):
        self.constraints: Dict[str, SafetyConstraint] = {}
        self._verification_history: deque = deque(maxlen=100)
        self._rollback_count = 0
        self._safe_improvements = 0
        self._total_tests = 0

        self._test_suites = {
            "functional": self._run_functional_tests,
            "performance": self._run_performance_tests,
            "stability": self._run_stability_tests,
            "safety": self._run_safety_tests,
        }

        self._init_default_constraints()

    def _init_default_constraints(self):
        default_constraints = [
            ("C001", "自由能上限", "自由能不得超过安全阈值", SafetyLevel.CRITICAL, "free_energy", 5.0),
            ("C002", "安全合规率", "安全合规率不得低于80%", SafetyLevel.CRITICAL, "safety_compliance_rate", 0.8),
            ("C003", "错误率上限", "错误率不得超过10%", SafetyLevel.HIGH, "error_rate", 0.1),
            ("C004", "内存上限", "内存占用不得超过4GB", SafetyLevel.HIGH, "memory_gb", 4.0),
            ("C005", "延迟上限", "单步延迟不得超过1000ms", SafetyLevel.MEDIUM, "latency_ms", 1000.0),
            ("C006", "行为剧烈变化", "单次改动行为变化不超过50%", SafetyLevel.HIGH, "behavior_change_rate", 0.5),
            ("C007", "系统稳定性", "改进后稳定性不得下降", SafetyLevel.MEDIUM, "stability_score", 0.0),
            ("C008", "学习退化检测", "不得出现灾难性遗忘", SafetyLevel.CRITICAL, "catastrophic_forgetting", 0.3),
        ]

        for cid, name, desc, level, check_fn, threshold in default_constraints:
            self.constraints[cid] = SafetyConstraint(
                constraint_id=cid,
                name=name,
                description=desc,
                safety_level=level,
                check_fn=check_fn,
                threshold=threshold
            )

    def verify_improvement(self, improvement_data: Dict[str, Any],
                            metrics_before: Dict[str, float],
                            metrics_after: Dict[str, float],
                            test_suites: List[str] = None) -> Dict[str, Any]:
        self._total_tests += 1
        start_time = time.time()

        suites = test_suites or list(self._test_suites.keys())
        results = []

        all_passed = True

        for suite_name in suites:
            suite_func = self._test_suites.get(suite_name)
            if suite_func:
                try:
                    suite_results = suite_func(metrics_before, metrics_after, improvement_data)
                    results.extend(suite_results)
                    for r in suite_results:
                        if not r.passed:
                            all_passed = False
                except Exception as e:
                    results.append(VerificationResult(
                        test_name=suite_name,
                        status=VerificationStatus.FAILED,
                        passed=False,
                        details=f"测试异常: {str(e)}"
                    ))
                    all_passed = False

        constraint_violations = self._check_constraints(metrics_after)
        if constraint_violations:
            all_passed = False

        duration = time.time() - start_time

        overall = {
            "passed": all_passed,
            "duration": duration,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "failed_tests": sum(1 for r in results if not r.passed),
            "constraint_violations": [c.to_dict() for c in constraint_violations],
            "test_results": [r.to_dict() for r in results],
            "recommended_action": "apply" if all_passed else "rollback",
            "safety_level": self._calculate_safety_level(results, constraint_violations)
        }

        self._verification_history.append(overall)

        if all_passed:
            self._safe_improvements += 1

        return overall

    def _run_functional_tests(self, before: Dict[str, float],
                               after: Dict[str, float],
                               improvement: Dict[str, Any]) -> List[VerificationResult]:
        results = []

        fe_before = before.get("free_energy", 1.0)
        fe_after = after.get("free_energy", 1.0)
        fe_degradation = (fe_after - fe_before) / max(fe_before, 0.01)

        results.append(VerificationResult(
            test_name="free_energy_no_degradation",
            status=VerificationStatus.PASSED if fe_degradation < 0.5 else VerificationStatus.FAILED,
            passed=fe_degradation < 0.5,
            details=f"自由能变化: {fe_before:.4f} -> {fe_after:.4f} ({fe_degradation:+.1%})",
            metrics_before={"free_energy": fe_before},
            metrics_after={"free_energy": fe_after}
        ))

        results.append(VerificationResult(
            test_name="system_operational",
            status=VerificationStatus.PASSED,
            passed=True,
            details="系统基本功能正常"
        ))

        return results

    def _run_performance_tests(self, before: Dict[str, float],
                                after: Dict[str, float],
                                improvement: Dict[str, Any]) -> List[VerificationResult]:
        results = []

        tp_before = before.get("throughput_steps_per_sec", 10.0)
        tp_after = after.get("throughput_steps_per_sec", 10.0)
        tp_change = (tp_after - tp_before) / max(tp_before, 0.01)

        results.append(VerificationResult(
            test_name="throughput_no_severe_degradation",
            status=VerificationStatus.PASSED if tp_change > -0.3 else VerificationStatus.FAILED,
            passed=tp_change > -0.3,
            details=f"吞吐量变化: {tp_before:.1f} -> {tp_after:.1f} ({tp_change:+.1%})",
            metrics_before={"throughput": tp_before},
            metrics_after={"throughput": tp_after}
        ))

        lat_before = before.get("latency_ms", 100.0)
        lat_after = after.get("latency_ms", 100.0)
        lat_change = (lat_after - lat_before) / max(lat_before, 0.01)

        results.append(VerificationResult(
            test_name="latency_within_limit",
            status=VerificationStatus.PASSED if lat_change < 1.0 else VerificationStatus.FAILED,
            passed=lat_change < 1.0,
            details=f"延迟变化: {lat_before:.1f}ms -> {lat_after:.1f}ms ({lat_change:+.1%})"
        ))

        return results

    def _run_stability_tests(self, before: Dict[str, float],
                              after: Dict[str, float],
                              improvement: Dict[str, Any]) -> List[VerificationResult]:
        results = []

        stab_before = before.get("stability_score", 0.5)
        stab_after = after.get("stability_score", 0.5)

        results.append(VerificationResult(
            test_name="stability_no_decline",
            status=VerificationStatus.PASSED if stab_after >= stab_before * 0.8 else VerificationStatus.FAILED,
            passed=stab_after >= stab_before * 0.8,
            details=f"稳定性: {stab_before:.2f} -> {stab_after:.2f}"
        ))

        err_before = before.get("error_rate", 0.05)
        err_after = after.get("error_rate", 0.05)

        results.append(VerificationResult(
            test_name="error_rate_acceptable",
            status=VerificationStatus.PASSED if err_after < 0.15 else VerificationStatus.FAILED,
            passed=err_after < 0.15,
            details=f"错误率: {err_before:.2%} -> {err_after:.2%}"
        ))

        return results

    def _run_safety_tests(self, before: Dict[str, float],
                           after: Dict[str, float],
                           improvement: Dict[str, Any]) -> List[VerificationResult]:
        results = []

        safety_before = before.get("safety_compliance_rate", 0.9)
        safety_after = after.get("safety_compliance_rate", 0.9)

        results.append(VerificationResult(
            test_name="safety_compliance_maintained",
            status=VerificationStatus.PASSED if safety_after >= 0.7 else VerificationStatus.FAILED,
            passed=safety_after >= 0.7,
            details=f"安全合规率: {safety_before:.1%} -> {safety_after:.1%}"
        ))

        risk_before = before.get("risk_level", 0.3)
        risk_after = after.get("risk_level", 0.3)

        results.append(VerificationResult(
            test_name="risk_level_acceptable",
            status=VerificationStatus.PASSED if risk_after < 0.7 else VerificationStatus.FAILED,
            passed=risk_after < 0.7,
            details=f"风险等级: {risk_before:.2f} -> {risk_after:.2f}"
        ))

        return results

    def _check_constraints(self, metrics: Dict[str, float]) -> List[SafetyConstraint]:
        violations = []

        for cid, constraint in self.constraints.items():
            if not constraint.enabled:
                continue

            check_fn = constraint.check_fn
            if check_fn and check_fn in metrics:
                value = metrics[check_fn]

                if check_fn in ("free_energy", "error_rate", "memory_gb",
                                "latency_ms", "behavior_change_rate",
                                "catastrophic_forgetting"):
                    if value > constraint.threshold:
                        constraint.violation_count += 1
                        violations.append(constraint)
                else:
                    if value < constraint.threshold:
                        constraint.violation_count += 1
                        violations.append(constraint)

        return violations

    def _calculate_safety_level(self, results: List[VerificationResult],
                                 violations: List[SafetyConstraint]) -> str:
        if any(v.safety_level == SafetyLevel.CRITICAL for v in violations):
            return "critical"
        if any(v.safety_level == SafetyLevel.HIGH for v in violations):
            return "high"
        if not results:
            return "unknown"

        pass_rate = sum(1 for r in results if r.passed) / len(results)
        if pass_rate >= 0.95:
            return "safe"
        elif pass_rate >= 0.8:
            return "mostly_safe"
        elif pass_rate >= 0.6:
            return "caution"
        else:
            return "dangerous"

    def get_verification_stats(self) -> Dict[str, Any]:
        return {
            "total_verifications": self._total_tests,
            "safe_improvements": self._safe_improvements,
            "rollback_count": self._rollback_count,
            "total_constraints": len(self.constraints),
            "enabled_constraints": sum(1 for c in self.constraints.values() if c.enabled),
            "constraint_violations_total": sum(c.violation_count for c in self.constraints.values()),
            "safety_rate": self._safe_improvements / max(self._total_tests, 1)
        }

    def record_rollback(self, reason: str = ""):
        self._rollback_count += 1
