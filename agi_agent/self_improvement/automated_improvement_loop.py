import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .self_improver import ImprovementProposal, ImprovementType, ImprovementStatus
from .bootstrapped_improver import BootstrappedSelfImprover
from .tiered_modification import TieredModificationRequest


class IterationPhase(Enum):
    IDLE = "idle"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    VERIFICATION = "verification"
    TESTING = "testing"
    APPLICATION = "application"
    EVALUATION = "evaluation"
    ROLLBACK = "rollback"
    COMPLETED = "completed"


class ImprovementOutcome(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ROLLED_BACK = "rolled_back"


@dataclass
class IterationRecord:
    iteration_id: str
    phase: IterationPhase
    start_time: float
    end_time: Optional[float] = None
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None
    proposals: List[ImprovementProposal] = field(default_factory=list)
    applied_proposals: List[str] = field(default_factory=list)
    outcome: Optional[ImprovementOutcome] = None
    rollback_reason: Optional[str] = None
    improvement_score: float = 0.0


@dataclass
class PerformanceThreshold:
    metric_name: str
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    warning_low: float = 0.3
    warning_high: float = 0.7
    critical_low: float = 0.1
    critical_high: float = 0.9

    def check_status(self, value: float) -> str:
        if value <= self.critical_low or value >= self.critical_high:
            return "critical"
        elif value <= self.warning_low or value >= self.warning_high:
            return "warning"
        return "healthy"


class AutomatedSelfImprovementLoop:
    def __init__(self, agent_ref=None, bootstrap_improver: BootstrappedSelfImprover = None):
        self.agent_ref = agent_ref
        self.bootstrap_improver = bootstrap_improver

        self.current_phase = IterationPhase.IDLE
        self.iteration_history: deque = deque(maxlen=100)
        self._current_iteration: Optional[IterationRecord] = None
        self._iteration_counter = 0

        self._performance_thresholds = {
            "free_energy": PerformanceThreshold(
                "free_energy", min_threshold=0.0, max_threshold=1.0,
                warning_high=0.6, critical_high=0.8
            ),
            "confidence": PerformanceThreshold(
                "confidence", min_threshold=0.0, max_threshold=1.0,
                warning_low=0.4, critical_low=0.2
            ),
            "error_rate": PerformanceThreshold(
                "error_rate", min_threshold=0.0, max_threshold=1.0,
                warning_high=0.05, critical_high=0.1
            ),
            "stability_score": PerformanceThreshold(
                "stability_score", min_threshold=0.0, max_threshold=1.0,
                warning_low=0.5, critical_low=0.3
            ),
            "throughput_steps_per_sec": PerformanceThreshold(
                "throughput_steps_per_sec", min_threshold=0.0, max_threshold=1000.0,
                warning_low=10.0, critical_low=5.0
            ),
            "learning_rate_efficiency": PerformanceThreshold(
                "learning_rate_efficiency", min_threshold=0.0, max_threshold=1.0,
                warning_low=0.4, critical_low=0.2
            )
        }

        self._improvement_callbacks: Dict[str, Callable] = {}
        self._rollback_callbacks: Dict[str, Callable] = {}
        self._test_callbacks: Dict[str, Callable] = {}

        self._metrics_buffer: deque = deque(maxlen=50)
        self._improvement_interval = 100
        self._evaluation_window = 20
        self._min_improvement_required = 0.05
        self._max_concurrent_improvements = 3

        self._enabled = True
        self._paused = False

    def set_agent_ref(self, agent_ref):
        self.agent_ref = agent_ref
        if self.bootstrap_improver and not self.bootstrap_improver._initialized:
            self.bootstrap_improver.initialize(agent_ref)

    def register_callback(self, event_type: str, callback: Callable):
        if event_type == "apply":
            self._improvement_callbacks[callback.__name__] = callback
        elif event_type == "rollback":
            self._rollback_callbacks[callback.__name__] = callback
        elif event_type == "test":
            self._test_callbacks[callback.__name__] = callback

    def should_trigger_improvement(self, current_step: int) -> bool:
        if not self._enabled or self._paused:
            return False

        if self.current_phase != IterationPhase.IDLE:
            return False

        if current_step % self._improvement_interval != 0:
            return False

        if len(self._metrics_buffer) < self._evaluation_window:
            return False

        return self._detect_performance_degradation()

    def _detect_performance_degradation(self) -> bool:
        if not self._metrics_buffer:
            return False

        recent_metrics = list(self._metrics_buffer)[-self._evaluation_window:]

        for metric_name, threshold in self._performance_thresholds.items():
            values = [m.get(metric_name, 0.5) for m in recent_metrics if m.get(metric_name) is not None]
            if not values:
                continue

            avg_value = np.mean(values)
            status = threshold.check_status(avg_value)
            if status == "critical":
                return True

        return False

    def start_iteration(self, step: int) -> bool:
        if self.current_phase != IterationPhase.IDLE:
            return False

        self._iteration_counter += 1
        iteration_id = f"iter_{self._iteration_counter}_{step}"

        self._current_iteration = IterationRecord(
            iteration_id=iteration_id,
            phase=IterationPhase.DIAGNOSIS,
            start_time=time.time(),
            metrics_before=self._collect_current_metrics()
        )

        self.current_phase = IterationPhase.DIAGNOSIS
        return True

    def run_full_iteration(self, step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        if not self.start_iteration(step):
            return {"status": "skipped", "reason": "Already in iteration"}

        self._metrics_buffer.append(metrics)

        try:
            diagnostics = self._run_diagnosis()
            proposals = self._generate_proposals(diagnostics)
            verified = self._verify_proposals(proposals)
            tested = self._test_proposals(verified)
            applied = self._apply_proposals(tested)
            evaluation = self._evaluate_results(applied)

            self.current_phase = IterationPhase.COMPLETED
            self._current_iteration.phase = IterationPhase.COMPLETED
            self._current_iteration.end_time = time.time()
            self._current_iteration.outcome = evaluation.get("outcome")
            self._current_iteration.improvement_score = evaluation.get("improvement_score", 0.0)

            self.iteration_history.append(self._current_iteration)

            return {
                "status": "completed",
                "iteration_id": self._current_iteration.iteration_id,
                "outcome": evaluation.get("outcome", "unknown").value if evaluation.get("outcome") else "unknown",
                "improvement_score": evaluation.get("improvement_score", 0.0),
                "proposals_count": len(proposals),
                "applied_count": len(applied),
                "diagnostics": diagnostics,
                "evaluation": evaluation
            }

        except Exception as e:
            self._handle_iteration_failure(str(e))
            return {"status": "failed", "error": str(e)}

    def _run_diagnosis(self) -> Dict[str, Any]:
        if self.current_phase != IterationPhase.DIAGNOSIS:
            return {}

        self.current_phase = IterationPhase.PROPOSAL

        diagnostics = {
            "issues": [],
            "metrics": {},
            "threshold_violations": []
        }

        if self._current_iteration and self._current_iteration.metrics_before:
            for metric_name, threshold in self._performance_thresholds.items():
                value = self._current_iteration.metrics_before.get(metric_name)
                if value is None:
                    continue

                diagnostics["metrics"][metric_name] = value
                status = threshold.check_status(value)
                if status != "healthy":
                    diagnostics["threshold_violations"].append({
                        "metric": metric_name,
                        "value": value,
                        "status": status,
                        "threshold": {
                            "warning_low": threshold.warning_low,
                            "warning_high": threshold.warning_high,
                            "critical_low": threshold.critical_low,
                            "critical_high": threshold.critical_high
                        }
                    })

        if self.agent_ref and hasattr(self.agent_ref, 'self_diagnostic'):
            diagnostic_result = self.agent_ref.self_diagnostic.run_diagnostics(
                system_state={"step": self.agent_ref.train_step},
                metrics=self._current_iteration.metrics_before or {}
            )
            diagnostics["issues"].extend(diagnostic_result.get("issues", []))

        return diagnostics

    def _generate_proposals(self, diagnostics: Dict[str, Any]) -> List[ImprovementProposal]:
        if self.current_phase != IterationPhase.PROPOSAL:
            return []

        self.current_phase = IterationPhase.VERIFICATION

        proposals = []

        if self.bootstrap_improver:
            tier1_proposals = self.bootstrap_improver.propose_tier1_improvements(
                performance_metrics=diagnostics.get("metrics", {}),
                diagnostic_findings=diagnostics.get("issues", [])
            )
            for prop in tier1_proposals[:self._max_concurrent_improvements]:
                proposals.append(self._convert_to_proposal(prop))

        if self.agent_ref and hasattr(self.agent_ref, 'self_improver'):
            additional_proposals = self.agent_ref.self_improver.generate_proposals(
                diagnostic_findings=diagnostics.get("issues", []),
                performance_metrics=diagnostics.get("metrics", {}),
                max_proposals=self._max_concurrent_improvements - len(proposals)
            )
            proposals.extend(additional_proposals)

        proposals.sort(key=lambda p: (p.expected_benefit - p.estimated_risk), reverse=True)

        if self._current_iteration:
            self._current_iteration.proposals = proposals

        return proposals[:self._max_concurrent_improvements]

    def _convert_to_proposal(self, request: TieredModificationRequest) -> ImprovementProposal:
        improvement_type = ImprovementType.HYPERPARAMETER
        if request.action == "add_rule":
            improvement_type = ImprovementType.BEHAVIOR

        return ImprovementProposal(
            proposal_id=f"imp_{request.request_id}",
            improvement_type=improvement_type,
            target_component=request.target,
            description=f"{request.action} on {request.target}",
            change_details=request.details,
            expected_benefit=0.15,
            estimated_risk=0.1,
            created_at=request.created_at
        )

    def _verify_proposals(self, proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
        if self.current_phase != IterationPhase.VERIFICATION:
            return []

        self.current_phase = IterationPhase.TESTING

        verified = []

        for proposal in proposals:
            if self.bootstrap_improver:
                try:
                    request = self._proposal_to_request(proposal)
                    if request:
                        verification = self.bootstrap_improver.verify_and_apply(request)
                        if verification.get("verified", False):
                            proposal.status = ImprovementStatus.VERIFIED
                            verified.append(proposal)
                except Exception:
                    pass
            else:
                proposal.status = ImprovementStatus.VERIFIED
                verified.append(proposal)

        return verified

    def _proposal_to_request(self, proposal: ImprovementProposal) -> Optional[TieredModificationRequest]:
        if self.bootstrap_improver and hasattr(self.bootstrap_improver.tiered_modifier, 'request_param_change'):
            if proposal.improvement_type == ImprovementType.HYPERPARAMETER:
                action = proposal.change_details.get("action")
                if action == "increase_lr" or action == "expand_learning_rate_pool":
                    target = f"{proposal.target_component}.learning_rate"
                    new_value = proposal.change_details.get("factor", 1.5)
                    return self.bootstrap_improver.tiered_modifier.request_param_change(target, new_value)
        return None

    def _test_proposals(self, proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
        if self.current_phase != IterationPhase.TESTING:
            return []

        self.current_phase = IterationPhase.APPLICATION

        tested = []

        for proposal in proposals:
            if proposal.improvement_type in self._test_callbacks:
                try:
                    test_result = self._test_callbacks[proposal.improvement_type.value](proposal)
                    if test_result.get("success", True):
                        tested.append(proposal)
                except Exception:
                    pass
            else:
                tested.append(proposal)

        return tested

    def _apply_proposals(self, proposals: List[ImprovementProposal]) -> List[str]:
        if self.current_phase != IterationPhase.APPLICATION:
            return []

        self.current_phase = IterationPhase.EVALUATION

        applied_ids = []

        for proposal in proposals:
            applied = False

            if self.bootstrap_improver:
                request = self._proposal_to_request(proposal)
                if request:
                    result = self.bootstrap_improver.verify_and_apply(request)
                    applied = result.get("applied", False)

            if not applied and self.agent_ref and hasattr(self.agent_ref, 'self_improver'):
                applied = self.agent_ref.self_improver.apply_improvement(proposal.proposal_id)

            if applied:
                proposal.status = ImprovementStatus.APPLIED
                applied_ids.append(proposal.proposal_id)

        if self._current_iteration:
            self._current_iteration.applied_proposals = applied_ids

        return applied_ids

    def _evaluate_results(self, applied_ids: List[str]) -> Dict[str, Any]:
        if self.current_phase != IterationPhase.EVALUATION:
            return {}

        metrics_after = self._collect_current_metrics()

        if self._current_iteration:
            self._current_iteration.metrics_after = metrics_after

        improvement_score = 0.0
        improvement_count = 0

        if self._current_iteration and self._current_iteration.metrics_before:
            for metric_name in ["confidence", "stability_score", "learning_rate_efficiency"]:
                before = self._current_iteration.metrics_before.get(metric_name, 0.5)
                after = metrics_after.get(metric_name, 0.5)

                if metric_name == "free_energy":
                    improvement = max(0, before - after) / max(before, 0.01)
                else:
                    improvement = max(0, after - before) / max(1 - before, 0.01)

                improvement_score += improvement
                improvement_count += 1

        if improvement_count > 0:
            improvement_score /= improvement_count

        needs_rollback = improvement_score < self._min_improvement_required
        if needs_rollback and applied_ids:
            self._trigger_rollback(applied_ids)
            return {
                "outcome": ImprovementOutcome.ROLLED_BACK,
                "improvement_score": improvement_score,
                "needs_rollback": True,
                "metrics_after": metrics_after
            }

        if improvement_score >= self._min_improvement_required:
            outcome = ImprovementOutcome.SUCCESS
        elif improvement_score > 0:
            outcome = ImprovementOutcome.PARTIAL_SUCCESS
        else:
            outcome = ImprovementOutcome.FAILURE

        return {
            "outcome": outcome,
            "improvement_score": improvement_score,
            "needs_rollback": False,
            "metrics_after": metrics_after,
            "applied_count": len(applied_ids)
        }

    def _trigger_rollback(self, applied_ids: List[str]):
        self.current_phase = IterationPhase.ROLLBACK

        if self._current_iteration:
            self._current_iteration.phase = IterationPhase.ROLLBACK
            self._current_iteration.rollback_reason = "Improvement did not meet threshold"

        for callback in self._rollback_callbacks.values():
            try:
                callback(applied_ids)
            except Exception:
                pass

        self.current_phase = IterationPhase.COMPLETED

    def _handle_iteration_failure(self, error: str):
        if self._current_iteration:
            self._current_iteration.end_time = time.time()
            self._current_iteration.outcome = ImprovementOutcome.FAILURE
            self._current_iteration.rollback_reason = error
            self.iteration_history.append(self._current_iteration)

        self.current_phase = IterationPhase.IDLE
        self._current_iteration = None

    def _collect_current_metrics(self) -> Dict[str, float]:
        metrics = {}

        if self.agent_ref:
            if hasattr(self.agent_ref, 'train_step'):
                metrics["step"] = float(self.agent_ref.train_step)
            if hasattr(self.agent_ref, 'last_fe'):
                metrics["free_energy"] = float(self.agent_ref.last_fe)

            if hasattr(self.agent_ref, 'meta_cog'):
                try:
                    meta_metrics = self.agent_ref.meta_cog.get_all_metrics()
                    if isinstance(meta_metrics, dict):
                        cognitive = meta_metrics.get("cognitive", {})
                        metrics["confidence"] = float(cognitive.get("confidence", 0.5))
                        metrics["free_energy"] = float(cognitive.get("free_energy", 1.0))
                except Exception:
                    pass

            if hasattr(self.agent_ref, 'evaluator'):
                try:
                    eval_metrics = self.agent_ref.evaluator.get_recent_metrics()
                    if isinstance(eval_metrics, dict):
                        metrics.update(eval_metrics)
                except Exception:
                    pass

            if hasattr(self.agent_ref, 'homeostasis'):
                try:
                    homeo_state = self.agent_ref.homeostasis.get_state()
                    if isinstance(homeo_state, dict):
                        metrics["energy_level"] = float(homeo_state.get("energy_level", 0.5))
                except Exception:
                    pass

        metrics["error_rate"] = min(1.0, max(0.0, metrics.get("free_energy", 1.0) / 5.0))
        metrics["stability_score"] = max(0.0, 1.0 - metrics.get("free_energy", 1.0))

        return metrics

    def record_metrics(self, metrics: Dict[str, float]):
        self._metrics_buffer.append(metrics)

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase.value,
            "enabled": self._enabled,
            "paused": self._paused,
            "iteration_count": self._iteration_counter,
            "history_size": len(self.iteration_history),
            "metrics_buffer_size": len(self._metrics_buffer),
            "improvement_interval": self._improvement_interval,
            "current_iteration": {
                "id": self._current_iteration.iteration_id if self._current_iteration else None,
                "phase": self._current_iteration.phase.value if self._current_iteration else None,
                "start_time": self._current_iteration.start_time if self._current_iteration else None
            } if self._current_iteration else None,
            "thresholds": {k: {
                "warning_low": v.warning_low,
                "warning_high": v.warning_high,
                "critical_low": v.critical_low,
                "critical_high": v.critical_high
            } for k, v in self._performance_thresholds.items()}
        }

    def get_recent_iterations(self, count: int = 10) -> List[Dict[str, Any]]:
        records = list(self.iteration_history)[-count:]
        return [{
            "iteration_id": r.iteration_id,
            "phase": r.phase.value,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "duration": (r.end_time - r.start_time) if r.end_time else None,
            "proposals_count": len(r.proposals),
            "applied_count": len(r.applied_proposals),
            "outcome": r.outcome.value if r.outcome else None,
            "improvement_score": r.improvement_score,
            "rollback_reason": r.rollback_reason
        } for r in records]

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def set_improvement_interval(self, interval: int):
        self._improvement_interval = interval

    def set_min_improvement_threshold(self, threshold: float):
        self._min_improvement_required = threshold