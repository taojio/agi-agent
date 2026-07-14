import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .training_goals import TrainingGoalManager, GoalLayer
from .training_phases import TrainingPhase, TrainingPhaseManager, PhaseStatus
from .data_pipeline import DataPipeline, DataQualityLevel, TrainingData
from .architecture_optimizer import ArchitectureOptimizer, OptimizationDimension
from .training_params import TrainingParams, LearningRateScheduler, LRScheduleType
from .evaluation_system import TrainingEvaluator, ConvergenceDetector, ConvergenceState, MetricTier
from .training_monitor import TrainingMonitor, InterventionEngine, AlertLevel
from .checkpoint_manager import CheckpointManager, CheckpointType


class TrainingState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    CONVERGED = "converged"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TrainingStats:
    total_steps: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    phases_completed: int = 0
    goals_achieved: int = 0
    checkpoints_saved: int = 0
    alerts_total: int = 0
    interventions_total: int = 0
    architecture_changes: int = 0


class TrainingRegime:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state = TrainingState.NOT_STARTED
        self.stats = TrainingStats()

        self.goal_manager = TrainingGoalManager()
        self.phase_manager = TrainingPhaseManager()
        self.data_pipeline = DataPipeline(
            max_buffer_size=self.config.get("data_buffer_size", 10000)
        )
        self.architecture_optimizer = ArchitectureOptimizer(
            initial_hidden_dim=self.config.get("initial_hidden_dim", 64),
            min_hidden_dim=self.config.get("min_hidden_dim", 16),
            max_hidden_dim=self.config.get("max_hidden_dim", 512)
        )
        self.training_params = TrainingParams()
        self.lr_scheduler = LearningRateScheduler(self.training_params)
        self.evaluator = TrainingEvaluator()
        self.convergence_detector = ConvergenceDetector(
            window_size=self.config.get("convergence_window", 500)
        )
        self.monitor = TrainingMonitor()
        self.intervention_engine = InterventionEngine()
        self.checkpoint_manager = CheckpointManager(
            base_dir=self.config.get("checkpoint_dir", "./agent_checkpoints"),
            max_checkpoints=self.config.get("max_checkpoints", 10)
        )

        self.monitor.set_intervention_engine(self.intervention_engine)

        self._agent_ref = None
        self._step_callbacks: List[Callable] = []
        self._phase_transition_callbacks: List[Callable] = []
        self._checkpoint_callbacks: List[Callable] = []

        self._init_callbacks()

    def _init_callbacks(self):
        self.phase_manager.register_transition_callback(self._on_phase_transition)
        self.goal_manager.register_goal_callback(self._on_goal_achieved)
        self.checkpoint_manager.register_save_callback(self._on_checkpoint_saved)

    def set_agent_ref(self, agent):
        self._agent_ref = agent

    def register_step_callback(self, callback: Callable):
        self._step_callbacks.append(callback)

    def register_phase_transition_callback(self, callback: Callable):
        self._phase_transition_callbacks.append(callback)

    def register_checkpoint_callback(self, callback: Callable):
        self._checkpoint_callbacks.append(callback)

    def start_training(self, initial_metrics: Dict[str, float] = None) -> bool:
        if self.state == TrainingState.RUNNING:
            return False

        self.state = TrainingState.INITIALIZING
        self.stats.start_time = time.time()

        if initial_metrics:
            self.evaluator.batch_update(initial_metrics, step=0)

        self.state = TrainingState.RUNNING
        return True

    def step(self, step: int, step_metrics: Dict[str, float] = None,
             training_data: List[TrainingData] = None) -> Dict[str, Any]:
        if self.state not in (TrainingState.RUNNING, TrainingState.PAUSED):
            return {"error": "Training not running"}

        if self.state == TrainingState.PAUSED:
            return {"status": "paused", "step": step}

        self.stats.total_steps = step
        metrics = step_metrics or {}

        self.evaluator.batch_update(metrics, step)

        self.goal_manager.update_goal("free_energy", metrics.get("free_energy", 0.5), step)
        self.goal_manager.update_goal("confidence", metrics.get("confidence", 0.5), step)
        self.goal_manager.update_goal("decision_accuracy", metrics.get("decision_accuracy", 0.5), step)
        if "kg_node_count" in metrics:
            self.goal_manager.update_goal("kg_node_count", metrics["kg_node_count"], step)

        if training_data:
            for data in training_data:
                self.data_pipeline.add_data(data)

        current_lr = self.lr_scheduler.step(step, metrics)
        metrics["learning_rate"] = current_lr

        arch_optimizations = self.architecture_optimizer.check_and_optimize(step, metrics)
        if arch_optimizations:
            self.stats.architecture_changes += len(arch_optimizations)

        new_phase = self.phase_manager.check_phase_transition(step, metrics)
        if new_phase:
            self._handle_phase_transition(new_phase, step, metrics)
            self.stats.phases_completed += 1

        alerts = self.monitor.record_step_metrics(metrics, step)
        if alerts:
            self.stats.alerts_total += len(alerts)

        convergence_state = self.convergence_detector.update(
            metrics.get("free_energy", 0.5), step
        )

        should_save = self._should_save_checkpoint(step, metrics, convergence_state)
        if should_save:
            self._save_training_checkpoint(step, metrics)

        evaluation = self.evaluator.evaluate(step)

        step_result = {
            "step": step,
            "state": self.state.value,
            "phase": self.phase_manager.current_phase.value,
            "phase_name": self.phase_manager.get_current_config().name if self.phase_manager.get_current_config() else "",
            "overall_score": evaluation.overall_score,
            "convergence_state": convergence_state.value,
            "learning_rate": current_lr,
            "alerts": [a.alert_id for a in alerts],
            "architecture_changes": arch_optimizations,
            "goal_progress": self.goal_manager.get_overall_progress(),
            "phase_progress": self.phase_manager.get_phase_progress(step, metrics),
            "evaluation": {
                "tier_scores": evaluation.tier_scores,
                "trends": evaluation.trends
            }
        }

        for callback in self._step_callbacks:
            try:
                callback(step, step_result)
            except Exception:
                pass

        if convergence_state == ConvergenceState.CONVERGED and \
           self.phase_manager.current_phase == list(TrainingPhase)[-1]:
            self._complete_training(step, metrics)

        return step_result

    def _handle_phase_transition(self, new_phase: TrainingPhase, step: int, metrics: Dict[str, float]):
        reason = f"Transitioning to {new_phase.value} phase"
        self.phase_manager.transition_to_phase(new_phase, step, metrics, reason)

        phase_config = self.phase_manager.get_current_config()
        if phase_config:
            self.data_pipeline.set_data_ratios(phase_config.data_ratio)

            lr_min, lr_max = phase_config.learning_rate_range
            self.training_params.min_learning_rate = lr_min
            self.training_params.max_learning_rate = lr_max

            self._adjust_schedule_for_phase(new_phase)

        for callback in self._phase_transition_callbacks:
            try:
                callback(self.phase_manager.current_phase, new_phase, step)
            except Exception:
                pass

        self._save_training_checkpoint(step, metrics, CheckpointType.MILESTONE,
                                       description=f"Phase transition to {new_phase.value}")

    def _adjust_schedule_for_phase(self, phase: TrainingPhase):
        if phase == TrainingPhase.COLD_START:
            self.lr_scheduler.set_schedule_type(LRScheduleType.FIXED)
        elif phase == TrainingPhase.SUPERVISED_SHAPING:
            self.lr_scheduler.set_schedule_type(LRScheduleType.WARMUP)
        elif phase == TrainingPhase.AUTONOMOUS_EXPLORATION:
            self.lr_scheduler.set_schedule_type(LRScheduleType.ADAPTIVE)
        elif phase == TrainingPhase.EVOLUTIONARY_OPTIMIZATION:
            self.lr_scheduler.set_schedule_type(LRScheduleType.META_LEARNING)
        elif phase == TrainingPhase.ALIGNMENT_CONSOLIDATION:
            self.lr_scheduler.set_schedule_type(LRScheduleType.COSINE_ANNEALING)

    def _should_save_checkpoint(self, step: int, metrics: Dict[str, float],
                                 convergence_state: ConvergenceState) -> bool:
        save_interval = self.config.get("save_interval", 1000)
        if step % save_interval == 0 and step > 0:
            return True

        if convergence_state == ConvergenceState.CONVERGED:
            return True

        if metrics.get("performance_improvement", 0) > 0.1:
            return True

        return False

    def _save_training_checkpoint(self, step: int, metrics: Dict[str, float],
                                   checkpoint_type: CheckpointType = CheckpointType.PERIODIC,
                                   description: str = "") -> Optional[str]:
        agent_state = self._collect_agent_state(metrics)

        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            agent_state=agent_state,
            step=step,
            checkpoint_type=checkpoint_type,
            description=description,
            phase=self.phase_manager.current_phase.value,
            performance_score=metrics.get("overall_score", 0.0)
        )

        if checkpoint_id:
            self.stats.checkpoints_saved += 1

        return checkpoint_id

    def _collect_agent_state(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        state = {
            "models": {},
            "configs": {},
            "logs": {},
            "metadata": {},
            "state": {}
        }

        if self._agent_ref is not None:
            if hasattr(self._agent_ref, 'perception'):
                state["models"]["perception"] = self._agent_ref.perception
            if hasattr(self._agent_ref, 'execution'):
                state["models"]["execution"] = self._agent_ref.execution

            state["configs"]["architecture"] = {
                "hidden_dim": self.architecture_optimizer.state.hidden_dim,
                "num_layers": self.architecture_optimizer.state.num_layers,
                "connection_density": self.architecture_optimizer.state.connection_density
            }
            state["configs"]["hyperparams"] = self.training_params.to_dict()
            state["configs"]["training_phase"] = self.phase_manager.current_phase.value

            if hasattr(self._agent_ref, 'knowledge_graph'):
                try:
                    kg_state = self._agent_ref.knowledge_graph.get_summary()
                    state["state"]["knowledge_graph"] = kg_state
                except Exception:
                    pass

            if hasattr(self._agent_ref, 'homeostasis'):
                try:
                    homeo_state = self._agent_ref.homeostasis.get_state()
                    state["state"]["homeostasis"] = homeo_state
                except Exception:
                    pass

        state["logs"]["metrics_snapshot"] = metrics
        state["logs"]["training_stats"] = {
            "total_steps": self.stats.total_steps,
            "phases_completed": self.stats.phases_completed,
            "goals_achieved": self.stats.goals_achieved
        }

        state["metadata"]["training_goals"] = self.goal_manager.get_summary()
        state["metadata"]["evaluation"] = self.evaluator.get_evaluation_report()
        state["metadata"]["data_stats"] = self.data_pipeline.get_quality_stats()

        return state

    def _on_phase_transition(self, old_phase, new_phase, step, reason):
        self._save_training_checkpoint(
            step,
            {"phase_transition": reason},
            CheckpointType.MILESTONE,
            f"Transition: {old_phase.value} -> {new_phase.value}"
        )

    def _on_goal_achieved(self, goal):
        self.stats.goals_achieved += 1

    def _on_checkpoint_saved(self, info):
        for callback in self._checkpoint_callbacks:
            try:
                callback(info)
            except Exception:
                pass

    def pause_training(self):
        if self.state == TrainingState.RUNNING:
            self.state = TrainingState.PAUSED
            return True
        return False

    def resume_training(self):
        if self.state == TrainingState.PAUSED:
            self.state = TrainingState.RUNNING
            return True
        return False

    def _complete_training(self, step: int, metrics: Dict[str, float]):
        self.state = TrainingState.COMPLETED
        self.stats.end_time = time.time()

        self._save_training_checkpoint(
            step, metrics,
            CheckpointType.MILESTONE,
            description="Training completed"
        )

    def get_training_summary(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "stats": {
                "total_steps": self.stats.total_steps,
                "duration_seconds": time.time() - self.stats.start_time if self.stats.start_time else 0,
                "phases_completed": self.stats.phases_completed,
                "goals_achieved": self.stats.goals_achieved,
                "checkpoints_saved": self.stats.checkpoints_saved,
                "alerts_total": self.stats.alerts_total,
                "architecture_changes": self.stats.architecture_changes
            },
            "current_phase": self.phase_manager.get_summary(),
            "goals": self.goal_manager.get_summary(),
            "evaluation": self.evaluator.get_evaluation_report(),
            "monitor": self.monitor.get_current_status(),
            "checkpoints": self.checkpoint_manager.get_summary(),
            "data_pipeline": self.data_pipeline.get_quality_stats(),
            "architecture": self.architecture_optimizer.get_optimization_stats(),
            "learning_rate": self.lr_scheduler.get_lr_stats()
        }

    def load_checkpoint(self, checkpoint_id: str = None) -> bool:
        data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if data is None:
            return False

        info = data.get("info")
        if info:
            self.stats.total_steps = info.step

        configs = data.get("configs", {})
        if "hyperparams" in configs:
            self.training_params = TrainingParams.from_dict(configs["hyperparams"])
            self.lr_scheduler = LearningRateScheduler(self.training_params)

        if "architecture" in configs:
            arch = configs["architecture"]
            self.architecture_optimizer.state.hidden_dim = arch.get("hidden_dim", 64)
            self.architecture_optimizer.state.num_layers = arch.get("num_layers", 3)

        return True

    def get_current_phase_config(self):
        return self.phase_manager.get_current_config()

    def is_feature_enabled(self, feature_name: str) -> bool:
        return self.phase_manager.is_feature_enabled(feature_name)
