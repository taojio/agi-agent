import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class TrainingPhase(Enum):
    COLD_START = "cold_start"
    SUPERVISED_SHAPING = "supervised_shaping"
    AUTONOMOUS_EXPLORATION = "autonomous_exploration"
    EVOLUTIONARY_OPTIMIZATION = "evolutionary_optimization"
    ALIGNMENT_CONSOLIDATION = "alignment_consolidation"


class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    CONVERGED = "converged"
    COMPLETED = "completed"


@dataclass
class PhaseConfig:
    phase: TrainingPhase
    name: str
    description: str
    min_steps: int
    max_steps: int
    target_metrics: Dict[str, float]
    enabled_features: List[str]
    disabled_features: List[str]
    learning_rate_range: tuple
    data_ratio: Dict[str, float]


@dataclass
class PhaseTransitionRule:
    rule_id: str
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    conditions: Dict[str, Any]
    min_stable_steps: int = 100
    description: str = ""

    def check(self, metrics: Dict[str, float], stable_steps: int) -> bool:
        if stable_steps < self.min_stable_steps:
            return False
        for metric_name, condition in self.conditions.items():
            if metric_name not in metrics:
                return False
            value = metrics[metric_name]
            if isinstance(condition, dict):
                if "min" in condition and value < condition["min"]:
                    return False
                if "max" in condition and value > condition["max"]:
                    return False
            elif isinstance(condition, (int, float)):
                if value < condition:
                    return False
        return True


@dataclass
class PhaseRecord:
    phase: TrainingPhase
    start_step: int
    end_step: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metrics_at_start: Dict[str, float] = field(default_factory=dict)
    metrics_at_end: Dict[str, float] = field(default_factory=dict)
    transition_reason: str = ""


class TrainingPhaseManager:
    def __init__(self):
        self.current_phase = TrainingPhase.COLD_START
        self.current_phase_start_step = 0
        self.current_phase_start_time = time.time()
        self.phase_history: List[PhaseRecord] = []
        self.phase_status: Dict[TrainingPhase, PhaseStatus] = {
            p: PhaseStatus.NOT_STARTED for p in TrainingPhase
        }
        self.phase_status[TrainingPhase.COLD_START] = PhaseStatus.IN_PROGRESS
        self.stable_counter = 0
        self.transition_rules: List[PhaseTransitionRule] = []
        self.phase_configs: Dict[TrainingPhase, PhaseConfig] = {}
        self.transition_callbacks: List[Callable] = []

        self._init_default_configs()
        self._init_default_transition_rules()

    def _init_default_configs(self):
        self.phase_configs[TrainingPhase.COLD_START] = PhaseConfig(
            phase=TrainingPhase.COLD_START,
            name="冷启动奠基期",
            description="建立智能体的基础感知与认知框架，确保系统稳定运行",
            min_steps=1000,
            max_steps=5000,
            target_metrics={
                "free_energy": 0.3,
                "confidence": 0.5,
                "stability_score": 0.7
            },
            enabled_features=[
                "perception_basic",
                "cognitive_basic",
                "homeostasis_basic",
                "safety_boundary"
            ],
            disabled_features=[
                "auto_evolution",
                "unsupervised_learning",
                "autonomous_decision",
                "snn_synaptic_plasticity",
                "memory_archiving"
            ],
            learning_rate_range=(1e-4, 1e-3),
            data_ratio={"basic": 0.8, "exploration": 0.2}
        )

        self.phase_configs[TrainingPhase.SUPERVISED_SHAPING] = PhaseConfig(
            phase=TrainingPhase.SUPERVISED_SHAPING,
            name="监督塑形期",
            description="通过结构化训练数据，塑造核心认知能力与行为模式",
            min_steps=5000,
            max_steps=50000,
            target_metrics={
                "kg_node_count": 500,
                "decision_accuracy": 0.7,
                "knowledge_retention": 0.6
            },
            enabled_features=[
                "multimodal_fusion",
                "knowledge_graph",
                "causal_reasoning",
                "behavior_shaping"
            ],
            disabled_features=[
                "auto_evolution",
                "architecture_mutation"
            ],
            learning_rate_range=(5e-4, 2e-3),
            data_ratio={"basic": 0.4, "domain": 0.3, "exploration": 0.2, "adversarial": 0.1}
        )

        self.phase_configs[TrainingPhase.AUTONOMOUS_EXPLORATION] = PhaseConfig(
            phase=TrainingPhase.AUTONOMOUS_EXPLORATION,
            name="自主探索期",
            description="赋予智能体自主探索与学习能力，扩展知识边界",
            min_steps=50000,
            max_steps=200000,
            target_metrics={
                "novelty_handling": 0.6,
                "autonomous_goals": 100,
                "skill_count": 20
            },
            enabled_features=[
                "active_exploration",
                "online_learning",
                "goal_self_setting",
                "meta_learning",
                "curiosity_driven"
            ],
            disabled_features=[
                "full_evolution"
            ],
            learning_rate_range=(1e-4, 1e-2),
            data_ratio={"basic": 0.2, "domain": 0.3, "exploration": 0.4, "adversarial": 0.1}
        )

        self.phase_configs[TrainingPhase.EVOLUTIONARY_OPTIMIZATION] = PhaseConfig(
            phase=TrainingPhase.EVOLUTIONARY_OPTIMIZATION,
            name="进化优化期",
            description="启动多级进化机制，优化架构与策略，实现能力跃升",
            min_steps=200000,
            max_steps=500000,
            target_metrics={
                "evolution_generations": 50,
                "performance_improvement": 0.3,
                "architecture_optimality": 0.8
            },
            enabled_features=[
                "micro_evolution",
                "meso_evolution",
                "macro_evolution",
                "meta_evolution",
                "neat_algorithm",
                "architecture_search"
            ],
            disabled_features=[],
            learning_rate_range=(1e-5, 5e-2),
            data_ratio={"basic": 0.1, "domain": 0.3, "exploration": 0.4, "adversarial": 0.2}
        )

        self.phase_configs[TrainingPhase.ALIGNMENT_CONSOLIDATION] = PhaseConfig(
            phase=TrainingPhase.ALIGNMENT_CONSOLIDATION,
            name="对齐巩固期",
            description="确保行为与价值体系对齐，巩固训练成果，准备部署",
            min_steps=500000,
            max_steps=1000000,
            target_metrics={
                "safety_compliance_rate": 0.99,
                "value_alignment": 0.9,
                "adversarial_robustness": 0.85
            },
            enabled_features=[
                "safety_reinforcement",
                "value_alignment",
                "adversarial_training",
                "regression_testing",
                "red_teaming"
            ],
            disabled_features=[
                "architecture_major_changes",
                "high_risk_evolution"
            ],
            learning_rate_range=(1e-5, 1e-4),
            data_ratio={"basic": 0.2, "domain": 0.3, "exploration": 0.1, "adversarial": 0.4}
        )

    def _init_default_transition_rules(self):
        self.transition_rules.append(PhaseTransitionRule(
            rule_id="cold_to_supervised",
            from_phase=TrainingPhase.COLD_START,
            to_phase=TrainingPhase.SUPERVISED_SHAPING,
            conditions={
                "free_energy": {"max": 0.3},
                "confidence": {"min": 0.5},
                "stability_score": {"min": 0.7}
            },
            min_stable_steps=200,
            description="基础能力稳定后进入监督塑形期"
        ))

        self.transition_rules.append(PhaseTransitionRule(
            rule_id="supervised_to_exploration",
            from_phase=TrainingPhase.SUPERVISED_SHAPING,
            to_phase=TrainingPhase.AUTONOMOUS_EXPLORATION,
            conditions={
                "decision_accuracy": {"min": 0.7},
                "kg_node_count": {"min": 500},
                "knowledge_retention": {"min": 0.6}
            },
            min_stable_steps=500,
            description="知识体系建立后进入自主探索期"
        ))

        self.transition_rules.append(PhaseTransitionRule(
            rule_id="exploration_to_evolution",
            from_phase=TrainingPhase.AUTONOMOUS_EXPLORATION,
            to_phase=TrainingPhase.EVOLUTIONARY_OPTIMIZATION,
            conditions={
                "novelty_handling": {"min": 0.6},
                "autonomous_goals": {"min": 100},
                "learning_efficiency": {"min": 0.5}
            },
            min_stable_steps=1000,
            description="自主能力成熟后启动进化优化"
        ))

        self.transition_rules.append(PhaseTransitionRule(
            rule_id="evolution_to_alignment",
            from_phase=TrainingPhase.EVOLUTIONARY_OPTIMIZATION,
            to_phase=TrainingPhase.ALIGNMENT_CONSOLIDATION,
            conditions={
                "performance_improvement": {"min": 0.3},
                "evolution_success_rate": {"min": 0.6},
                "stability_score": {"min": 0.8}
            },
            min_stable_steps=2000,
            description="进化优化达标后进入对齐巩固"
        ))

    def register_transition_callback(self, callback: Callable):
        self.transition_callbacks.append(callback)

    def get_current_config(self) -> Optional[PhaseConfig]:
        return self.phase_configs.get(self.current_phase)

    def check_phase_transition(self, current_step: int, metrics: Dict[str, float]) -> Optional[TrainingPhase]:
        phase_config = self.get_current_config()
        if phase_config is None:
            return None

        metrics_window = metrics.get("_window_size", 0)

        is_stable = self._check_stability(metrics)
        if is_stable:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        for rule in self.transition_rules:
            if rule.from_phase != self.current_phase:
                continue
            if rule.check(metrics, self.stable_counter):
                if current_step >= phase_config.min_steps:
                    return rule.to_phase

        if current_step >= phase_config.max_steps:
            phases = list(TrainingPhase)
            current_idx = phases.index(self.current_phase)
            if current_idx < len(phases) - 1:
                return phases[current_idx + 1]

        return None

    def _check_stability(self, metrics: Dict[str, float]) -> bool:
        if "free_energy_trend" in metrics:
            fe_std = metrics.get("free_energy_std", 1.0)
            if fe_std > 0.1:
                return False
        if "confidence_trend" in metrics:
            conf_std = metrics.get("confidence_std", 1.0)
            if conf_std > 0.15:
                return False
        return True

    def transition_to_phase(self, new_phase: TrainingPhase, current_step: int,
                            metrics: Dict[str, float], reason: str = "") -> bool:
        if new_phase == self.current_phase:
            return False

        old_phase = self.current_phase

        if self.phase_history and self.phase_history[-1].phase == old_phase:
            self.phase_history[-1].end_step = current_step
            self.phase_history[-1].end_time = time.time()
            self.phase_history[-1].metrics_at_end = metrics.copy()

        self.phase_status[old_phase] = PhaseStatus.COMPLETED
        self.phase_status[new_phase] = PhaseStatus.IN_PROGRESS

        self.current_phase = new_phase
        self.current_phase_start_step = current_step
        self.current_phase_start_time = time.time()
        self.stable_counter = 0

        self.phase_history.append(PhaseRecord(
            phase=new_phase,
            start_step=current_step,
            start_time=time.time(),
            metrics_at_start=metrics.copy(),
            transition_reason=reason
        ))

        for callback in self.transition_callbacks:
            try:
                callback(old_phase, new_phase, current_step, reason)
            except Exception:
                pass

        return True

    def get_phase_progress(self, current_step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        config = self.get_current_config()
        if config is None:
            return {}

        steps_in_phase = current_step - self.current_phase_start_step
        step_progress = min(1.0, steps_in_phase / config.max_steps)

        metric_progress = {}
        total_target_achieved = 0
        total_targets = 0

        for metric_name, target_value in config.target_metrics.items():
            current_value = metrics.get(metric_name, 0.0)
            if target_value > 0:
                progress = min(1.0, current_value / target_value)
            else:
                progress = 1.0 if current_value <= abs(target_value) else 0.0

            metric_progress[metric_name] = {
                "current": current_value,
                "target": target_value,
                "progress": progress
            }
            total_targets += 1
            if progress >= 1.0:
                total_target_achieved += 1

        target_progress = total_target_achieved / max(1, total_targets)

        overall_progress = 0.4 * step_progress + 0.6 * target_progress

        return {
            "phase": self.current_phase.value,
            "phase_name": config.name,
            "steps_in_phase": steps_in_phase,
            "step_progress": step_progress,
            "target_progress": target_progress,
            "overall_progress": overall_progress,
            "stable_steps": self.stable_counter,
            "metrics": metric_progress,
            "enabled_features": config.enabled_features,
            "disabled_features": config.disabled_features
        }

    def is_feature_enabled(self, feature_name: str) -> bool:
        config = self.get_current_config()
        if config is None:
            return True
        if feature_name in config.disabled_features:
            return False
        if config.enabled_features and feature_name not in config.enabled_features:
            return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        phases_ordered = list(TrainingPhase)
        completed_phases = sum(
            1 for p in phases_ordered
            if self.phase_status[p] == PhaseStatus.COMPLETED
        )
        total_phases = len(phases_ordered)

        return {
            "current_phase": self.current_phase.value,
            "current_phase_name": self.get_current_config().name if self.get_current_config() else "",
            "phase_start_step": self.current_phase_start_step,
            "stable_steps": self.stable_counter,
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "overall_phase_progress": completed_phases / total_phases,
            "phase_status": {p.value: s.value for p, s in self.phase_status.items()},
            "phase_history_count": len(self.phase_history)
        }
