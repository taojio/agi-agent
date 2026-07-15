"""
decision/decision_feedback_loop.py - 决策质量-策略调整反馈闭环

实现决策执行效果数据采集、质量评估到策略调整的映射算法、闭环周期控制
支持决策策略自动调整周期≤1小时
"""
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque


class FeedbackTriggerType(Enum):
    MANUAL = "manual"
    PERIODIC = "periodic"
    THRESHOLD = "threshold"
    EVENT = "event"
    QUALITY_DEGRADATION = "quality_degradation"


class FeedbackStatus(Enum):
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    ADJUSTING = "adjusting"
    COMPLETED = "completed"
    FAILED = "failed"


class AdjustmentActionType(Enum):
    WEIGHT_UPDATE = "weight_update"
    STRATEGY_SWITCH = "strategy_switch"
    PARAMETER_TWEAK = "parameter_tweak"
    LEARNING_RATE_ADJUST = "learning_rate_adjust"
    EXPLORATION_RATE_ADJUST = "exploration_rate_adjust"


@dataclass
class ExecutionDataPoint:
    decision_id: str
    strategy_id: str
    timestamp: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    success: bool
    execution_time: float
    resource_usage: Dict[str, float]
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "success": self.success,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "metadata": self.metadata,
        }


@dataclass
class FeedbackEvent:
    event_id: str
    trigger_type: FeedbackTriggerType
    timestamp: float
    decision_ids: List[str]
    quality_metrics: Dict[str, float]
    adjustment_actions: List[Dict[str, Any]] = field(default_factory=list)
    status: FeedbackStatus = FeedbackStatus.COLLECTING
    processing_time_ms: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger_type": self.trigger_type.value,
            "timestamp": self.timestamp,
            "decision_ids": self.decision_ids,
            "quality_metrics": self.quality_metrics,
            "adjustment_actions": self.adjustment_actions,
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
        }


@dataclass
class QualityAdjustmentMapping:
    metric_name: str
    metric_threshold: float
    adjustment_target: str
    adjustment_delta: float
    adjustment_type: AdjustmentActionType
    condition: str = "below"

    def check_trigger(self, current_value: float) -> bool:
        if self.condition == "below":
            return current_value < self.metric_threshold
        elif self.condition == "above":
            return current_value > self.metric_threshold
        elif self.condition == "equal":
            return abs(current_value - self.metric_threshold) < 0.01
        return False


class DecisionFeedbackLoop:
    def __init__(self, adjustment_period: int = 3600, quality_threshold: float = 0.6):
        self._execution_data: deque = deque(maxlen=10000)
        self._feedback_events: deque = deque(maxlen=500)
        self._adjustment_period = adjustment_period
        self._quality_threshold = quality_threshold
        self._last_adjustment_time = 0.0
        self._quality_adjustment_mappings: List[QualityAdjustmentMapping] = self._initialize_mappings()
        self._data_collector: Dict[str, List[ExecutionDataPoint]] = {}
        self._adjustment_history: deque = deque(maxlen=200)
        self._loop_stats = {
            "total_executions": 0,
            "total_feedback_events": 0,
            "total_adjustments": 0,
            "avg_processing_time_ms": 0.0,
            "auto_adjustment_triggers": 0,
        }

    def _initialize_mappings(self) -> List[QualityAdjustmentMapping]:
        return [
            QualityAdjustmentMapping(
                metric_name="prediction_accuracy",
                metric_threshold=0.7,
                adjustment_target="reward_weight",
                adjustment_delta=0.05,
                adjustment_type=AdjustmentActionType.WEIGHT_UPDATE,
                condition="below",
            ),
            QualityAdjustmentMapping(
                metric_name="success_rate",
                metric_threshold=0.75,
                adjustment_target="success_weight",
                adjustment_delta=0.05,
                adjustment_type=AdjustmentActionType.WEIGHT_UPDATE,
                condition="below",
            ),
            QualityAdjustmentMapping(
                metric_name="timeliness",
                metric_threshold=0.6,
                adjustment_target="efficiency_weight",
                adjustment_delta=0.03,
                adjustment_type=AdjustmentActionType.WEIGHT_UPDATE,
                condition="below",
            ),
            QualityAdjustmentMapping(
                metric_name="cost_effectiveness",
                metric_threshold=1.0,
                adjustment_target="cost_weight",
                adjustment_delta=-0.05,
                adjustment_type=AdjustmentActionType.WEIGHT_UPDATE,
                condition="below",
            ),
            QualityAdjustmentMapping(
                metric_name="decision_confidence",
                metric_threshold=0.6,
                adjustment_target="exploration_rate",
                adjustment_delta=0.05,
                adjustment_type=AdjustmentActionType.EXPLORATION_RATE_ADJUST,
                condition="below",
            ),
            QualityAdjustmentMapping(
                metric_name="overall_quality",
                metric_threshold=0.6,
                adjustment_target="learning_rate",
                adjustment_delta=-0.0001,
                adjustment_type=AdjustmentActionType.LEARNING_RATE_ADJUST,
                condition="below",
            ),
        ]

    def collect_execution_data(self, data_point: ExecutionDataPoint):
        self._execution_data.append(data_point)
        self._loop_stats["total_executions"] += 1

        if data_point.strategy_id not in self._data_collector:
            self._data_collector[data_point.strategy_id] = []
        self._data_collector[data_point.strategy_id].append(data_point)
        if len(self._data_collector[data_point.strategy_id]) > 200:
            self._data_collector[data_point.strategy_id] = self._data_collector[data_point.strategy_id][-200:]

    def check_periodic_adjustment(self) -> bool:
        now = time.time()
        return now - self._last_adjustment_time >= self._adjustment_period

    def check_quality_degradation(self, strategy_id: str) -> bool:
        data_points = self._data_collector.get(strategy_id, [])
        if len(data_points) < 10:
            return False

        recent = data_points[-10:]
        success_rates = [1.0 if dp.success else 0.0 for dp in recent]
        avg_success_rate = np.mean(success_rates)

        execution_times = [dp.execution_time for dp in recent]
        avg_execution_time = np.mean(execution_times)

        return avg_success_rate < self._quality_threshold or avg_execution_time > 30.0

    def analyze_execution_data(self, strategy_id: str = None) -> Dict[str, Any]:
        if strategy_id:
            data_points = self._data_collector.get(strategy_id, [])
        else:
            data_points = list(self._execution_data)

        if not data_points:
            return {"error": "No execution data available"}

        analysis = {}
        recent = data_points[-50:]

        analysis["total_executions"] = len(data_points)
        analysis["recent_executions"] = len(recent)
        analysis["success_rate"] = float(np.mean([1.0 if dp.success else 0.0 for dp in recent]))

        execution_times = [dp.execution_time for dp in recent if dp.execution_time > 0]
        analysis["avg_execution_time"] = float(np.mean(execution_times)) if execution_times else 0.0
        analysis["min_execution_time"] = float(np.min(execution_times)) if execution_times else 0.0
        analysis["max_execution_time"] = float(np.max(execution_times)) if execution_times else 0.0

        if recent[0].expected_outcome and recent[0].actual_outcome:
            accuracies = []
            for dp in recent:
                if dp.expected_outcome and dp.actual_outcome:
                    common_keys = set(dp.expected_outcome.keys()) & set(dp.actual_outcome.keys())
                    if common_keys:
                        errors = []
                        for key in common_keys:
                            exp = dp.expected_outcome[key]
                            act = dp.actual_outcome[key]
                            if isinstance(exp, (int, float)) and isinstance(act, (int, float)) and abs(exp) > 1e-10:
                                errors.append(abs(act - exp) / abs(exp))
                        if errors:
                            accuracies.append(1.0 - np.mean(errors))
            analysis["prediction_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.5

        resource_usage = {}
        for dp in recent:
            for resource, value in dp.resource_usage.items():
                if resource not in resource_usage:
                    resource_usage[resource] = []
                resource_usage[resource].append(value)
        analysis["resource_usage"] = {
            resource: {"avg": float(np.mean(values)), "max": float(np.max(values))}
            for resource, values in resource_usage.items()
        }

        strategy_counts = {}
        for dp in data_points:
            strategy_counts[dp.strategy_id] = strategy_counts.get(dp.strategy_id, 0) + 1
        analysis["strategy_distribution"] = strategy_counts

        return analysis

    def map_quality_to_adjustments(self, quality_metrics: Dict[str, float],
                                    strategy_id: str) -> List[Dict[str, Any]]:
        adjustments = []

        for mapping in self._quality_adjustment_mappings:
            metric_value = quality_metrics.get(mapping.metric_name, 0.5)
            if mapping.check_trigger(metric_value):
                adjustment = {
                    "action_type": mapping.adjustment_type.value,
                    "target": mapping.adjustment_target,
                    "delta": mapping.adjustment_delta,
                    "metric_name": mapping.metric_name,
                    "current_value": metric_value,
                    "threshold": mapping.metric_threshold,
                    "strategy_id": strategy_id,
                }
                adjustments.append(adjustment)

        if not adjustments:
            overall_quality = quality_metrics.get("overall_quality", 0.5)
            if overall_quality > 0.8:
                adjustments.append({
                    "action_type": AdjustmentActionType.PARAMETER_TWEAK.value,
                    "target": "learning_rate",
                    "delta": 0.0001,
                    "reason": "Quality is high, increasing learning rate for faster adaptation",
                })

        return adjustments

    def process_feedback(self, trigger_type: FeedbackTriggerType,
                          decision_ids: List[str] = None,
                          strategy_id: str = None) -> FeedbackEvent:
        start_time = time.time()

        event_id = f"feedback_{int(time.time())}"
        event = FeedbackEvent(
            event_id=event_id,
            trigger_type=trigger_type,
            timestamp=time.time(),
            decision_ids=decision_ids or [],
            quality_metrics={},
            status=FeedbackStatus.COLLECTING,
        )

        event.status = FeedbackStatus.ANALYZING

        analysis = self.analyze_execution_data(strategy_id)
        quality_metrics = {
            "success_rate": analysis.get("success_rate", 0.5),
            "prediction_accuracy": analysis.get("prediction_accuracy", 0.5),
            "avg_execution_time": analysis.get("avg_execution_time", 0.0),
            "timeliness": 1.0 - min(1.0, analysis.get("avg_execution_time", 0.0) / 30.0),
            "overall_quality": self._calculate_overall_quality(analysis),
        }
        event.quality_metrics = quality_metrics

        adjustments = self.map_quality_to_adjustments(quality_metrics, strategy_id or "default")
        event.adjustment_actions = adjustments

        event.status = FeedbackStatus.COMPLETED
        event.processing_time_ms = (time.time() - start_time) * 1000

        self._feedback_events.append(event)
        self._loop_stats["total_feedback_events"] += 1
        self._loop_stats["total_adjustments"] += len(adjustments)
        self._loop_stats["avg_processing_time_ms"] = (
            self._loop_stats["avg_processing_time_ms"] *
            (self._loop_stats["total_feedback_events"] - 1) + event.processing_time_ms
        ) / self._loop_stats["total_feedback_events"]

        if trigger_type == FeedbackTriggerType.PERIODIC:
            self._last_adjustment_time = time.time()

        return event

    def _calculate_overall_quality(self, analysis: Dict[str, Any]) -> float:
        success_rate = analysis.get("success_rate", 0.5)
        accuracy = analysis.get("prediction_accuracy", 0.5)
        execution_time = analysis.get("avg_execution_time", 30.0)
        timeliness = 1.0 - min(1.0, execution_time / 30.0)

        return float(0.3 * success_rate + 0.3 * accuracy + 0.4 * timeliness)

    def execute_adjustments(self, adjustments: List[Dict[str, Any]],
                             strategy_adjustment_engine=None) -> List[Dict[str, Any]]:
        results = []

        for adjustment in adjustments:
            action_type = adjustment.get("action_type")
            target = adjustment.get("target")
            delta = adjustment.get("delta")
            strategy_id = adjustment.get("strategy_id", "default")

            result = {
                "action_type": action_type,
                "target": target,
                "delta": delta,
                "strategy_id": strategy_id,
                "success": False,
            }

            if strategy_adjustment_engine:
                if action_type == AdjustmentActionType.WEIGHT_UPDATE.value:
                    delta_weights = {target: delta}
                    request = strategy_adjustment_engine.create_adjustment_request(
                        strategy_id=strategy_id,
                        adjustment_type=self._map_to_adjustment_type(action_type),
                        adjustment_delta=delta_weights,
                    )
                    adj_result = strategy_adjustment_engine.adjust_strategy(request)
                    result["success"] = adj_result.success
                    result["adjustment_result"] = adj_result.to_dict()

                elif action_type == AdjustmentActionType.PARAMETER_TWEAK.value:
                    result["success"] = True
                    result["message"] = f"Parameter {target} adjusted by {delta}"

                elif action_type == AdjustmentActionType.LEARNING_RATE_ADJUST.value:
                    result["success"] = True
                    result["message"] = f"Learning rate adjusted by {delta}"

                elif action_type == AdjustmentActionType.EXPLORATION_RATE_ADJUST.value:
                    result["success"] = True
                    result["message"] = f"Exploration rate adjusted by {delta}"

            else:
                result["success"] = True
                result["message"] = "Adjustment logged (no engine provided)"

            self._adjustment_history.append({
                **result,
                "timestamp": time.time(),
            })
            results.append(result)

        return results

    def _map_to_adjustment_type(self, action_type: str):
        from .strategy_adjustment_engine import AdjustmentType
        if action_type == "periodic":
            return AdjustmentType.PERIODIC
        elif action_type == "event_triggered":
            return AdjustmentType.EVENT_TRIGGERED
        return AdjustmentType.REALTIME

    def get_feedback_stats(self) -> Dict[str, Any]:
        events = list(self._feedback_events)
        if events:
            avg_processing = np.mean([e.processing_time_ms for e in events])
            avg_adjustments = np.mean([len(e.adjustment_actions) for e in events])
        else:
            avg_processing = 0.0
            avg_adjustments = 0.0

        return {
            **self._loop_stats,
            "feedback_events_count": len(events),
            "avg_processing_time_ms": avg_processing,
            "avg_adjustments_per_event": avg_adjustments,
            "last_adjustment_time": self._last_adjustment_time,
            "next_adjustment_due": max(0, self._last_adjustment_time + self._adjustment_period - time.time()),
            "adjustment_period": self._adjustment_period,
        }

    def get_feedback_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in list(self._feedback_events)[-limit:]]

    def get_adjustment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._adjustment_history)[-limit:]

    def set_adjustment_period(self, period: int):
        self._adjustment_period = max(60, period)

    def set_quality_threshold(self, threshold: float):
        self._quality_threshold = max(0.0, min(1.0, threshold))

    def add_adjustment_mapping(self, mapping: QualityAdjustmentMapping):
        self._quality_adjustment_mappings.append(mapping)

    def remove_adjustment_mapping(self, metric_name: str) -> bool:
        original_length = len(self._quality_adjustment_mappings)
        self._quality_adjustment_mappings = [
            m for m in self._quality_adjustment_mappings if m.metric_name != metric_name
        ]
        return len(self._quality_adjustment_mappings) < original_length