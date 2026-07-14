import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque
from enum import Enum
from dataclasses import dataclass, field


class ImprovementType(Enum):
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    ALGORITHM = "algorithm"
    KNOWLEDGE = "knowledge"
    BEHAVIOR = "behavior"


class ImprovementStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VERIFIED = "verified"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class ImprovementProposal:
    proposal_id: str
    improvement_type: ImprovementType
    target_component: str
    description: str
    change_details: Dict[str, Any] = field(default_factory=dict)
    expected_benefit: float = 0.0
    estimated_risk: float = 0.0
    status: ImprovementStatus = ImprovementStatus.PROPOSED
    created_at: float = 0.0
    test_results: Optional[Dict[str, Any]] = None
    applied_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "type": self.improvement_type.value,
            "target": self.target_component,
            "description": self.description,
            "expected_benefit": self.expected_benefit,
            "estimated_risk": self.estimated_risk,
            "status": self.status.value,
            "created_at": self.created_at
        }


class RecursiveSelfImprover:
    def __init__(self, max_concurrent_improvements: int = 3):
        self.max_concurrent_improvements = max_concurrent_improvements
        self.proposals: List[ImprovementProposal] = []
        self._proposal_counter = 0

        self._improvement_history: deque = deque(maxlen=200)
        self._baseline_performance: Dict[str, float] = {}
        self._current_performance: Dict[str, float] = {}

        self._generations = 0
        self._total_improvements = 0
        self._successful_improvements = 0
        self._failed_improvements = 0
        self._rolled_back = 0

        self._improvement_strategies = {
            ImprovementType.HYPERPARAMETER: self._propose_hyperparameter_changes,
            ImprovementType.ARCHITECTURE: self._propose_architecture_changes,
            ImprovementType.ALGORITHM: self._propose_algorithm_changes,
            ImprovementType.KNOWLEDGE: self._propose_knowledge_changes,
            ImprovementType.BEHAVIOR: self._propose_behavior_changes,
        }

        self._apply_callbacks: Dict[str, Callable] = {}
        self._revert_callbacks: Dict[str, Callable] = {}

    def generate_proposals(self, diagnostic_findings: List[Any],
                            performance_metrics: Dict[str, Any],
                            max_proposals: int = 5) -> List[ImprovementProposal]:
        proposals = []

        for imp_type, strategy in self._improvement_strategies.items():
            try:
                new_proposals = strategy(diagnostic_findings, performance_metrics)
                proposals.extend(new_proposals)
            except Exception:
                pass

        proposals.sort(
            key=lambda p: (p.expected_benefit - p.estimated_risk),
            reverse=True
        )

        selected = proposals[:max_proposals]
        self.proposals.extend(selected)

        return selected

    def _propose_hyperparameter_changes(self, findings: List[Any],
                                          metrics: Dict[str, Any]) -> List[ImprovementProposal]:
        proposals = []

        learning_eff = metrics.get("learning_rate_efficiency", 0.5)
        if learning_eff < 0.5:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.HYPERPARAMETER,
                target_component="meta_learning",
                description="调整学习率池，增加更细粒度的选项",
                change_details={
                    "action": "expand_learning_rate_pool",
                    "new_lrs": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
                },
                expected_benefit=0.15,
                estimated_risk=0.1,
                created_at=time.time()
            ))

        fe = metrics.get("free_energy", 1.0)
        if fe > 0.5:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.HYPERPARAMETER,
                target_component="perception",
                description="增加感知层学习率以加速收敛",
                change_details={"action": "increase_lr", "component": "perception", "factor": 1.5},
                expected_benefit=0.1,
                estimated_risk=0.2,
                created_at=time.time()
            ))

        return proposals

    def _propose_architecture_changes(self, findings: List[Any],
                                       metrics: Dict[str, Any]) -> List[ImprovementProposal]:
        proposals = []

        fe_stable = metrics.get("free_energy_trend", "stable") == "stable"
        fe_high = metrics.get("free_energy", 1.0) > 0.3

        if fe_stable and fe_high:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.ARCHITECTURE,
                target_component="perception",
                description="扩展感知网络隐藏维度",
                change_details={"action": "grow_hidden_dim", "delta": 8},
                expected_benefit=0.2,
                estimated_risk=0.3,
                created_at=time.time()
            ))

            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.ARCHITECTURE,
                target_component="evolution",
                description="触发布尔进化优化结构",
                change_details={"action": "trigger_neat_evolution"},
                expected_benefit=0.25,
                estimated_risk=0.4,
                created_at=time.time()
            ))

        return proposals

    def _propose_algorithm_changes(self, findings: List[Any],
                                    metrics: Dict[str, Any]) -> List[ImprovementProposal]:
        proposals = []

        prediction_acc = metrics.get("prediction_accuracy", 0.5)
        if prediction_acc < 0.6:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.ALGORITHM,
                target_component="cognitive",
                description="增加预测编码层数",
                change_details={"action": "add_pc_layer"},
                expected_benefit=0.18,
                estimated_risk=0.25,
                created_at=time.time()
            ))

        return proposals

    def _propose_knowledge_changes(self, findings: List[Any],
                                    metrics: Dict[str, Any]) -> List[ImprovementProposal]:
        proposals = []

        conf = metrics.get("confidence", 0.5)
        if conf < 0.5:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.KNOWLEDGE,
                target_component="knowledge_graph",
                description="增强知识图谱，提升认知置信度",
                change_details={"action": "knowledge_consolidation"},
                expected_benefit=0.12,
                estimated_risk=0.05,
                created_at=time.time()
            ))

        return proposals

    def _propose_behavior_changes(self, findings: List[Any],
                                   metrics: Dict[str, Any]) -> List[ImprovementProposal]:
        proposals = []

        error_rate = metrics.get("error_rate", 0.05)
        if error_rate > 0.03:
            self._proposal_counter += 1
            proposals.append(ImprovementProposal(
                proposal_id=f"imp_{self._proposal_counter}",
                improvement_type=ImprovementType.BEHAVIOR,
                target_component="execution",
                description="降低探索率以减少错误",
                change_details={"action": "reduce_exploration", "factor": 0.8},
                expected_benefit=0.08,
                estimated_risk=0.15,
                created_at=time.time()
            ))

        return proposals

    def apply_improvement(self, proposal_id: str,
                           apply_callback: Callable = None) -> bool:
        proposal = self._find_proposal(proposal_id)
        if not proposal:
            return False

        proposal.status = ImprovementStatus.TESTING

        try:
            if apply_callback:
                result = apply_callback(proposal.change_details)
            else:
                result = self._auto_apply(proposal)

            if result:
                proposal.status = ImprovementStatus.APPLIED
                proposal.applied_at = time.time()
                self._total_improvements += 1
                self._successful_improvements += 1
                self._generations += 1

                self._improvement_history.append({
                    "proposal_id": proposal_id,
                    "type": proposal.improvement_type.value,
                    "result": "applied",
                    "timestamp": time.time()
                })

                return True
            else:
                proposal.status = ImprovementStatus.REJECTED
                self._failed_improvements += 1
                return False

        except Exception as e:
            proposal.status = ImprovementStatus.REJECTED
            self._failed_improvements += 1
            return False

    def _auto_apply(self, proposal: ImprovementProposal) -> bool:
        return True

    def _find_proposal(self, proposal_id: str) -> Optional[ImprovementProposal]:
        for p in self.proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    def set_baseline(self, baseline_metrics: Dict[str, float]):
        self._baseline_performance = dict(baseline_metrics)

    def record_improvement_result(self, proposal_id: str,
                                   new_metrics: Dict[str, float],
                                   success: bool):
        if success:
            self._current_performance = dict(new_metrics)
        else:
            self._rolled_back += 1

    def get_improvement_stats(self) -> Dict[str, Any]:
        return {
            "generations": self._generations,
            "total_proposed": len(self.proposals),
            "total_applied": self._total_improvements,
            "successful": self._successful_improvements,
            "failed": self._failed_improvements,
            "rolled_back": self._rolled_back,
            "success_rate": self._successful_improvements / max(self._total_improvements, 1),
            "by_type": self._count_by_type()
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for p in self.proposals:
            t = p.improvement_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts
