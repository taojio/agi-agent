"""
self_improvement/feedback_loop.py - 反馈闭环系统

完整的反馈闭环：监控 → 诊断 → 改进 → 验证 → 部署
将性能监控、问题诊断、改进生成、效果验证串联为自动循环。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class FeedbackPhase(Enum):
    """反馈阶段"""
    MONITORING = "monitoring"      # 监控
    DIAGNOSIS = "diagnosis"        # 诊断
    IMPROVEMENT = "improvement"    # 改进生成
    VALIDATION = "validation"      # 验证
    DEPLOYMENT = "deployment"      # 部署
    COMPLETED = "completed"        # 完成
    FAILED = "failed"              # 失败


class IssueSeverity(Enum):
    """问题严重度"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class IssueType(Enum):
    """问题类型"""
    PERFORMANCE = "performance"          # 性能问题
    ACCURACY = "accuracy"                # 准确率问题
    RESOURCE = "resource"                # 资源问题
    STABILITY = "stability"              # 稳定性问题
    LOGIC = "logic"                      # 逻辑问题
    CONFIGURATION = "configuration"      # 配置问题
    UNKNOWN = "unknown"


@dataclass
class Issue:
    """检测到的问题"""
    issue_id: str
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    detected_at: float = field(default_factory=time.time)
    affected_metrics: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "open"  # open / diagnosing / fixing / resolved / ignored
    proposed_fixes: List[Dict[str, Any]] = field(default_factory=list)
    applied_fix: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.name,
            "description": self.description,
            "detected_at": self.detected_at,
            "affected_metrics": self.affected_metrics,
            "evidence": self.evidence,
            "status": self.status,
            "proposed_fixes": self.proposed_fixes,
            "applied_fix": self.applied_fix,
        }


@dataclass
class ImprovementProposal:
    """改进提案"""
    proposal_id: str
    issue_id: str
    description: str
    action: str
    expected_impact: float          # 预期改善（百分比）
    risk_level: str = "low"         # low / medium / high
    priority: float = 0.5           # 0-1
    created_at: float = field(default_factory=time.time)
    status: str = "proposed"        # proposed / testing / applied / validated / rejected
    validation_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "issue_id": self.issue_id,
            "description": self.description,
            "action": self.action,
            "expected_impact": float(self.expected_impact),
            "risk_level": self.risk_level,
            "priority": float(self.priority),
            "created_at": self.created_at,
            "status": self.status,
            "validation_result": self.validation_result,
        }


@dataclass
class FeedbackCycle:
    """反馈循环记录"""
    cycle_id: str
    started_at: float
    completed_at: Optional[float] = None
    phases: Dict[FeedbackPhase, float] = field(default_factory=dict)
    issues_detected: List[str] = field(default_factory=list)
    proposals_generated: List[str] = field(default_factory=list)
    improvements_applied: List[str] = field(default_factory=list)
    success: bool = False
    summary: str = ""

    @property
    def duration(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "phases": {p.value: t for p, t in self.phases.items()},
            "issues_detected": self.issues_detected,
            "proposals_generated": self.proposals_generated,
            "improvements_applied": self.improvements_applied,
            "success": self.success,
            "summary": self.summary,
        }


class FeedbackLoop:
    """反馈闭环系统

    自动化反馈闭环：定期收集指标 → 检测问题 → 诊断根因 → 生成改进 → 验证 → 部署

    Attributes:
        issues: 已检测到的问题
        proposals: 改进提案
        cycles: 反馈循环历史
    """

    def __init__(self, performance_baseline=None, reflection_engine=None):
        self.issues: Dict[str, Issue] = {}
        self.proposals: Dict[str, ImprovementProposal] = {}
        self.cycles: deque = deque(maxlen=100)
        self.performance_baseline = performance_baseline
        self.reflection_engine = reflection_engine

        self._issue_counter = 0
        self._proposal_counter = 0
        self._cycle_counter = 0

        # 改进动作处理器
        self._action_handlers: Dict[str, Callable] = {}
        # 问题检测器
        self._issue_detectors: List[Callable] = [
            self._detect_performance_issues,
            self._detect_accuracy_issues,
            self._detect_resource_issues,
            self._detect_stability_issues,
        ]

    def register_action_handler(self, action_type: str,
                                  handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """注册改进动作处理器"""
        self._action_handlers[action_type] = handler

    def run_cycle(self, current_metrics: Optional[Dict[str, float]] = None) -> FeedbackCycle:
        """执行一轮反馈循环

        Args:
            current_metrics: 当前性能指标（如未提供则从 baseline 获取）

        Returns:
            FeedbackCycle: 循环记录
        """
        self._cycle_counter += 1
        cycle = FeedbackCycle(
            cycle_id=f"cycle_{self._cycle_counter}",
            started_at=time.time(),
        )
        cycle.phases[FeedbackPhase.MONITORING] = time.time()

        # 阶段1: 监控
        metrics = current_metrics or self._collect_metrics()
        cycle.phases[FeedbackPhase.DIAGNOSIS] = time.time()

        # 阶段2: 诊断
        new_issues = self._diagnose(metrics)
        cycle.issues_detected = [i.issue_id for i in new_issues]

        cycle.phases[FeedbackPhase.IMPROVEMENT] = time.time()

        # 阶段3: 改进生成
        new_proposals = self._generate_proposals(new_issues)
        cycle.proposals_generated = [p.proposal_id for p in new_proposals]

        cycle.phases[FeedbackPhase.VALIDATION] = time.time()

        # 阶段4: 验证（模拟）
        validated = self._validate_proposals(new_proposals)

        cycle.phases[FeedbackPhase.DEPLOYMENT] = time.time()

        # 阶段5: 部署（应用改进）
        applied = self._deploy_proposals(validated)
        cycle.improvements_applied = [p.proposal_id for p in applied]

        cycle.phases[FeedbackPhase.COMPLETED] = time.time()
        cycle.completed_at = time.time()
        cycle.success = len(applied) > 0
        cycle.summary = (f"Detected {len(new_issues)} issues, "
                        f"generated {len(new_proposals)} proposals, "
                        f"applied {len(applied)} improvements")

        self.cycles.append(cycle)
        return cycle

    def _collect_metrics(self) -> Dict[str, float]:
        """收集当前指标"""
        if self.performance_baseline is None:
            return {}
        metrics = {}
        for name, metric in self.performance_baseline.metrics.items():
            if metric.latest is not None:
                metrics[name] = metric.latest
        return metrics

    def _diagnose(self, metrics: Dict[str, float]) -> List[Issue]:
        """诊断：运行所有问题检测器"""
        new_issues = []
        for detector in self._issue_detectors:
            try:
                issues = detector(metrics)
                new_issues.extend(issues)
            except Exception:
                pass
        # 注册新问题
        for issue in new_issues:
            self.issues[issue.issue_id] = issue
        return new_issues

    def _detect_performance_issues(self, metrics: Dict[str, float]) -> List[Issue]:
        """检测性能问题"""
        issues = []
        # 高延迟
        latency = metrics.get("module_latency") or metrics.get("end_to_end_latency")
        if latency is not None and latency > 100:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.PERFORMANCE,
                severity=IssueSeverity.WARNING if latency < 500 else IssueSeverity.ERROR,
                description=f"High latency detected: {latency}ms",
                affected_metrics=["module_latency"] if "module_latency" in metrics else ["end_to_end_latency"],
                evidence={"latency": latency, "threshold": 100},
            ))
        # 低吞吐量
        throughput = metrics.get("module_throughput")
        if throughput is not None and throughput < 500:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.PERFORMANCE,
                severity=IssueSeverity.WARNING,
                description=f"Low throughput: {throughput} req/s",
                affected_metrics=["module_throughput"],
                evidence={"throughput": throughput, "threshold": 500},
            ))
        return issues

    def _detect_accuracy_issues(self, metrics: Dict[str, float]) -> List[Issue]:
        """检测准确率问题"""
        issues = []
        accuracy = metrics.get("prediction_accuracy")
        if accuracy is not None and accuracy < 0.7:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.ACCURACY,
                severity=IssueSeverity.ERROR if accuracy < 0.5 else IssueSeverity.WARNING,
                description=f"Low prediction accuracy: {accuracy:.2%}",
                affected_metrics=["prediction_accuracy"],
                evidence={"accuracy": accuracy, "threshold": 0.7},
            ))
        decision_quality = metrics.get("decision_quality")
        if decision_quality is not None and decision_quality < 0.6:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.ACCURACY,
                severity=IssueSeverity.WARNING,
                description=f"Low decision quality: {decision_quality:.2f}",
                affected_metrics=["decision_quality"],
                evidence={"decision_quality": decision_quality, "threshold": 0.6},
            ))
        return issues

    def _detect_resource_issues(self, metrics: Dict[str, float]) -> List[Issue]:
        """检测资源问题"""
        issues = []
        memory = metrics.get("memory_usage")
        if memory is not None and memory > 85:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.RESOURCE,
                severity=IssueSeverity.CRITICAL if memory > 95 else IssueSeverity.WARNING,
                description=f"High memory usage: {memory}%",
                affected_metrics=["memory_usage"],
                evidence={"memory_usage": memory, "threshold": 85},
            ))
        error_rate = metrics.get("error_rate")
        if error_rate is not None and error_rate > 5:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.RESOURCE,
                severity=IssueSeverity.ERROR,
                description=f"High error rate: {error_rate}%",
                affected_metrics=["error_rate"],
                evidence={"error_rate": error_rate, "threshold": 5},
            ))
        return issues

    def _detect_stability_issues(self, metrics: Dict[str, float]) -> List[Issue]:
        """检测稳定性问题"""
        issues = []
        failure_freq = metrics.get("failure_frequency")
        if failure_freq is not None and failure_freq > 0.5:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.STABILITY,
                severity=IssueSeverity.CRITICAL,
                description=f"High failure frequency: {failure_freq}/hour",
                affected_metrics=["failure_frequency"],
                evidence={"failure_frequency": failure_freq, "threshold": 0.5},
            ))
        mttr = metrics.get("mttr")
        if mttr is not None and mttr > 300:
            self._issue_counter += 1
            issues.append(Issue(
                issue_id=f"issue_{self._issue_counter}",
                issue_type=IssueType.STABILITY,
                severity=IssueSeverity.WARNING,
                description=f"Long recovery time: {mttr}s",
                affected_metrics=["mttr"],
                evidence={"mttr": mttr, "threshold": 300},
            ))
        return issues

    def _generate_proposals(self, issues: List[Issue]) -> List[ImprovementProposal]:
        """为问题生成改进提案"""
        proposals = []
        for issue in issues:
            fixes = self._propose_fixes(issue)
            for fix in fixes:
                self._proposal_counter += 1
                proposal = ImprovementProposal(
                    proposal_id=f"prop_{self._proposal_counter}",
                    issue_id=issue.issue_id,
                    description=fix["description"],
                    action=fix["action"],
                    expected_impact=fix.get("expected_impact", 0.1),
                    risk_level=fix.get("risk_level", "low"),
                    priority=fix.get("priority", issue.severity.value / 4.0),
                )
                self.proposals[proposal.proposal_id] = proposal
                issue.proposed_fixes.append(fix)
                proposals.append(proposal)
        return proposals

    def _propose_fixes(self, issue: Issue) -> List[Dict[str, Any]]:
        """为单个问题提出修复方案"""
        fixes = []
        if issue.issue_type == IssueType.PERFORMANCE:
            if "latency" in issue.description.lower():
                fixes.append({
                    "description": "Optimize hot path code",
                    "action": "optimize_performance",
                    "expected_impact": 0.3,
                    "risk_level": "medium",
                    "priority": 0.8,
                })
                fixes.append({
                    "description": "Cache frequent computations",
                    "action": "add_caching",
                    "expected_impact": 0.2,
                    "risk_level": "low",
                    "priority": 0.6,
                })
            elif "throughput" in issue.description.lower():
                fixes.append({
                    "description": "Parallelize request processing",
                    "action": "parallelize",
                    "expected_impact": 0.4,
                    "risk_level": "medium",
                    "priority": 0.7,
                })

        elif issue.issue_type == IssueType.ACCURACY:
            fixes.append({
                "description": "Retrain model with recent data",
                "action": "retrain",
                "expected_impact": 0.25,
                "risk_level": "medium",
                "priority": 0.9,
            })
            fixes.append({
                "description": "Adjust decision thresholds",
                "action": "tune_thresholds",
                "expected_impact": 0.15,
                "risk_level": "low",
                "priority": 0.7,
            })

        elif issue.issue_type == IssueType.RESOURCE:
            if "memory" in issue.description.lower():
                fixes.append({
                    "description": "Release unused resources",
                    "action": "release_resources",
                    "expected_impact": 0.2,
                    "risk_level": "low",
                    "priority": 0.8,
                })
            elif "error" in issue.description.lower():
                fixes.append({
                    "description": "Add error handling and retry logic",
                    "action": "improve_error_handling",
                    "expected_impact": 0.3,
                    "risk_level": "low",
                    "priority": 0.9,
                })

        elif issue.issue_type == IssueType.STABILITY:
            fixes.append({
                "description": "Add circuit breaker",
                "action": "add_circuit_breaker",
                "expected_impact": 0.4,
                "risk_level": "medium",
                "priority": 0.95,
            })

        if not fixes:
            fixes.append({
                "description": "Investigate root cause",
                "action": "investigate",
                "expected_impact": 0.0,
                "risk_level": "low",
                "priority": 0.5,
            })

        return fixes

    def _validate_proposals(self, proposals: List[ImprovementProposal]
                              ) -> List[ImprovementProposal]:
        """验证提案（模拟）"""
        validated = []
        for proposal in proposals:
            # 风险评估
            risk_score = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(proposal.risk_level, 0.5)
            # 高风险低收益的提案被拒绝
            if risk_score > 0.7 and proposal.expected_impact < 0.3:
                proposal.status = "rejected"
                proposal.validation_result = {"reason": "High risk, low impact"}
                continue
            proposal.status = "testing"
            proposal.validation_result = {
                "passed": True,
                "estimated_impact": proposal.expected_impact,
                "risk_score": risk_score,
            }
            validated.append(proposal)
        return validated

    def _deploy_proposals(self, proposals: List[ImprovementProposal]
                            ) -> List[ImprovementProposal]:
        """部署提案"""
        deployed = []
        for proposal in proposals:
            action_type = proposal.action
            if action_type in self._action_handlers:
                try:
                    result = self._action_handlers[action_type]({
                        "proposal": proposal.to_dict(),
                        "issue": self.issues.get(proposal.issue_id),
                    })
                    proposal.status = "applied"
                    proposal.validation_result = proposal.validation_result or {}
                    proposal.validation_result.update({"deployment": result})
                    deployed.append(proposal)
                    # 更新关联问题状态
                    if proposal.issue_id in self.issues:
                        self.issues[proposal.issue_id].status = "resolved"
                        self.issues[proposal.issue_id].applied_fix = proposal.to_dict()
                except Exception as e:
                    proposal.status = "rejected"
                    proposal.validation_result = {"error": str(e)}
            else:
                # 没有处理器，标记为已应用（模拟）
                proposal.status = "applied"
                deployed.append(proposal)
                if proposal.issue_id in self.issues:
                    self.issues[proposal.issue_id].status = "resolved"
                    self.issues[proposal.issue_id].applied_fix = proposal.to_dict()
        return deployed

    def get_open_issues(self) -> List[Issue]:
        return [i for i in self.issues.values() if i.status == "open"]

    def get_pending_proposals(self) -> List[ImprovementProposal]:
        return [p for p in self.proposals.values()
                if p.status in ("proposed", "testing")]

    def get_cycle_history(self, limit: int = 20) -> List[FeedbackCycle]:
        return list(self.cycles)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cycles": len(self.cycles),
            "total_issues": len(self.issues),
            "open_issues": len(self.get_open_issues()),
            "total_proposals": len(self.proposals),
            "pending_proposals": len(self.get_pending_proposals()),
            "resolved_issues": sum(1 for i in self.issues.values() if i.status == "resolved"),
            "success_rate": (sum(1 for c in self.cycles if c.success) / len(self.cycles)
                            if self.cycles else 0.0),
            "avg_cycle_duration": (float(np.mean([c.duration for c in self.cycles]))
                                    if self.cycles else 0.0),
        }
