import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from enum import Enum
from dataclasses import dataclass, field


class DiagnosticSeverity(Enum):
    CRITICAL = 0
    WARNING = 1
    INFO = 2
    IMPROVEMENT_OPPORTUNITY = 3


class DiagnosticCategory(Enum):
    PERFORMANCE = "performance"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"
    ARCHITECTURE = "architecture"
    LEARNING = "learning"
    SAFETY = "safety"


@dataclass
class DiagnosticFinding:
    finding_id: str
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    title: str
    description: str
    affected_components: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    discovered_at: float = 0.0
    status: str = "open"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "severity": self.severity.name.lower(),
            "title": self.title,
            "description": self.description,
            "affected_components": self.affected_components,
            "suggested_actions": self.suggested_actions,
            "confidence": self.confidence,
            "status": self.status
        }


class SelfDiagnosticEngine:
    def __init__(self):
        self.findings: List[DiagnosticFinding] = []
        self._finding_counter = 0
        self._diagnostic_history: deque = deque(maxlen=100)
        self._component_status: Dict[str, Dict[str, Any]] = {}

        self._diagnostic_checks = {
            "performance": self._check_performance,
            "stability": self._check_stability,
            "efficiency": self._check_efficiency,
            "architecture": self._check_architecture,
            "learning": self._check_learning,
            "safety": self._check_safety,
        }

        self._last_diagnostic_time = 0

    def run_diagnostics(self, system_state: Dict[str, Any],
                        metrics: Dict[str, Any] = None) -> List[DiagnosticFinding]:
        self._last_diagnostic_time = time.time()
        new_findings = []

        for check_name, check_func in self._diagnostic_checks.items():
            try:
                findings = check_func(system_state, metrics or {})
                new_findings.extend(findings)
            except Exception:
                pass

        self.findings.extend(new_findings)

        self._diagnostic_history.append({
            "timestamp": time.time(),
            "findings_count": len(new_findings),
            "findings": [f.to_dict() for f in new_findings]
        })

        return new_findings

    def _check_performance(self, system_state: Dict[str, Any],
                            metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        fe = metrics.get("free_energy", 1.0)
        if fe > 0.8:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.PERFORMANCE,
                severity=DiagnosticSeverity.WARNING,
                title="自由能过高",
                description=f"当前自由能为 {fe:.4f}，超过健康阈值 0.8",
                affected_components=["perception", "cognitive"],
                suggested_actions=[
                    "增加学习率加速收敛",
                    "扩展网络结构增强表征能力",
                    "检查输入数据质量"
                ],
                confidence=min(1.0, fe / 1.5),
                discovered_at=time.time()
            ))

        conf = metrics.get("confidence", 0.5)
        if conf < 0.4:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.PERFORMANCE,
                severity=DiagnosticSeverity.INFO,
                title="认知置信度偏低",
                description=f"当前置信度 {conf:.2f}，低于正常水平",
                affected_components=["cognitive"],
                suggested_actions=["增加训练步数", "扩充知识图谱"],
                confidence=0.7,
                discovered_at=time.time()
            ))

        return findings

    def _check_stability(self, system_state: Dict[str, Any],
                          metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > 0.05:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.STABILITY,
                severity=DiagnosticSeverity.CRITICAL,
                title="错误率超标",
                description=f"错误率 {error_rate:.2%} 超过阈值 5%",
                affected_components=["execution"],
                suggested_actions=["降低探索率", "回滚到稳定版本", "增加安全约束"],
                confidence=0.9,
                discovered_at=time.time()
            ))

        return findings

    def _check_efficiency(self, system_state: Dict[str, Any],
                           metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        throughput = metrics.get("throughput_steps_per_sec", 10.0)
        if throughput < 20.0:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.EFFICIENCY,
                severity=DiagnosticSeverity.IMPROVEMENT_OPPORTUNITY,
                title="处理速度有提升空间",
                description=f"当前吞吐量 {throughput:.1f} 步/秒",
                affected_components=["perception", "cognitive", "execution"],
                suggested_actions=["启用批处理", "优化网络结构", "考虑硬件加速"],
                confidence=0.6,
                discovered_at=time.time()
            ))

        return findings

    def _check_architecture(self, system_state: Dict[str, Any],
                             metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        fe_trend = metrics.get("free_energy_trend", "stable")
        if fe_trend == "stable" and metrics.get("free_energy", 1.0) > 0.3:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.ARCHITECTURE,
                severity=DiagnosticSeverity.IMPROVEMENT_OPPORTUNITY,
                title="架构可能进入瓶颈",
                description="自由能长期稳定但未达目标，建议扩展结构",
                affected_components=["perception", "evolution"],
                suggested_actions=["触发布尔进化", "增加隐藏层维度", "增加模块数量"],
                confidence=0.5,
                discovered_at=time.time()
            ))

        return findings

    def _check_learning(self, system_state: Dict[str, Any],
                         metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        learning_eff = metrics.get("learning_rate_efficiency", 0.5)
        if learning_eff < 0.4:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.LEARNING,
                severity=DiagnosticSeverity.WARNING,
                title="学习效率低下",
                description="学习率与收敛速度不匹配",
                affected_components=["meta_learning"],
                suggested_actions=["调整学习率池", "启用元学习", "增加学习率衰减"],
                confidence=0.7,
                discovered_at=time.time()
            ))

        return findings

    def _check_safety(self, system_state: Dict[str, Any],
                       metrics: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings = []

        safety_score = metrics.get("safety_compliance_rate", 0.9)
        if safety_score < 0.8:
            self._finding_counter += 1
            findings.append(DiagnosticFinding(
                finding_id=f"diag_{self._finding_counter}",
                category=DiagnosticCategory.SAFETY,
                severity=DiagnosticSeverity.CRITICAL,
                title="安全合规率低",
                description=f"安全合规率仅 {safety_score:.1%}",
                affected_components=["safety_monitor", "compliance_checker"],
                suggested_actions=["加强安全约束", "审查风险行为", "启用紧急停止机制"],
                confidence=0.95,
                discovered_at=time.time()
            ))

        return findings

    def get_findings_by_severity(self, severity: DiagnosticSeverity) -> List[DiagnosticFinding]:
        return [f for f in self.findings if f.severity == severity and f.status == "open"]

    def get_findings_by_category(self, category: DiagnosticCategory) -> List[DiagnosticFinding]:
        return [f for f in self.findings if f.category == category]

    def resolve_finding(self, finding_id: str, resolution: str = "") -> bool:
        for f in self.findings:
            if f.finding_id == finding_id:
                f.status = "resolved"
                return True
        return False

    def get_diagnostic_summary(self) -> Dict[str, Any]:
        open_findings = [f for f in self.findings if f.status == "open"]

        severity_counts = {}
        category_counts = {}

        for f in open_findings:
            sev = f.severity.name.lower()
            cat = f.category.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_findings": len(self.findings),
            "open_findings": len(open_findings),
            "resolved_findings": len(self.findings) - len(open_findings),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "last_diagnostic": self._last_diagnostic_time,
            "total_diagnostics_run": len(self._diagnostic_history)
        }
