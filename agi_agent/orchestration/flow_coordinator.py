"""
flow_coordinator.py - 主流程协调优化与多通路协调机制

实现：
- 主流程协调优化：增强事件触发机制和数据传递效率
- 多通路协调机制：明确反射/深思/元认知通路触发条件、数据交互规范和优先级处理策略
- 流程性能监控指标：实时监控流程性能并动态调整资源分配
"""
import time
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque

from .event_bus import EventBus, Event, EventPriority
from .automation_linkage import SystemState

logger = logging.getLogger(__name__)


class ProcessingPathway(Enum):
    REFLEX = "reflex"
    DELIBERATE = "deliberate"
    META_COGNITIVE = "meta_cognitive"


class PathwayPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class FlowPhase(Enum):
    PERCEPTION = "perception"
    COGNITION = "cognition"
    EXECUTION = "execution"
    FEEDBACK = "feedback"


class FlowControlMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    HYBRID = "hybrid"


class PathwayTriggerCondition:
    def __init__(
        self,
        pathway: ProcessingPathway,
        priority: PathwayPriority,
        confidence_threshold: Tuple[float, float] = (0.0, 1.0),
        free_energy_threshold: Tuple[float, float] = (0.0, 1.0),
        novelty_threshold: Tuple[float, float] = (0.0, 1.0),
        entropy_threshold: Tuple[float, float] = (0.0, 1.0),
        is_impasse: Optional[bool] = None,
        latency_threshold: float = 0.5,
        context_patterns: Optional[Set[str]] = None,
        required_features: Optional[Set[str]] = None,
    ):
        self.pathway = pathway
        self.priority = priority
        self.confidence_threshold = confidence_threshold
        self.free_energy_threshold = free_energy_threshold
        self.novelty_threshold = novelty_threshold
        self.entropy_threshold = entropy_threshold
        self.is_impasse = is_impasse
        self.latency_threshold = latency_threshold
        self.context_patterns = context_patterns or set()
        self.required_features = required_features or set()

    def evaluate(self, state: SystemState, context: Dict[str, Any] = None) -> bool:
        context = context or {}

        if not (self.confidence_threshold[0] <= state.confidence <= self.confidence_threshold[1]):
            return False

        if not (self.free_energy_threshold[0] <= state.free_energy <= self.free_energy_threshold[1]):
            return False

        if not (self.novelty_threshold[0] <= state.novelty <= self.novelty_threshold[1]):
            return False

        if not (self.entropy_threshold[0] <= state.entropy <= self.entropy_threshold[1]):
            return False

        if self.is_impasse is not None and state.is_impasse != self.is_impasse:
            return False

        if state.latency > self.latency_threshold:
            return False

        if self.context_patterns:
            context_str = str(context).lower()
            if not any(pattern.lower() in context_str for pattern in self.context_patterns):
                return False

        return True


class PathwayDataSpec:
    def __init__(
        self,
        pathway: ProcessingPathway,
        required_inputs: List[str],
        optional_inputs: List[str],
        output_format: str,
        data_transformation: Optional[Callable] = None,
        validation_schema: Optional[Dict[str, Any]] = None,
    ):
        self.pathway = pathway
        self.required_inputs = required_inputs
        self.optional_inputs = optional_inputs
        self.output_format = output_format
        self.data_transformation = data_transformation
        self.validation_schema = validation_schema

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        for required in self.required_inputs:
            if required not in data:
                errors.append(f"Missing required input: {required}")

        if self.validation_schema:
            for key, spec in self.validation_schema.items():
                if key in data:
                    if "type" in spec and not isinstance(data[key], spec["type"]):
                        errors.append(f"Invalid type for {key}: expected {spec['type'].__name__}")
                    if "min" in spec and data[key] < spec["min"]:
                        errors.append(f"{key} below minimum: {data[key]} < {spec['min']}")
                    if "max" in spec and data[key] > spec["max"]:
                        errors.append(f"{key} above maximum: {data[key]} > {spec['max']}")

        return len(errors) == 0, errors

    def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.data_transformation:
            return self.data_transformation(data)
        return data


class FlowPerformanceMetric:
    def __init__(self):
        self.phase_latencies: Dict[str, deque] = {}
        self.throughput: deque = deque(maxlen=100)
        self.pathway_utilization: Dict[str, deque] = {}
        self.event_processing_times: deque = deque(maxlen=100)
        self.queue_lengths: deque = deque(maxlen=100)

    def record_phase_latency(self, phase: str, latency: float):
        if phase not in self.phase_latencies:
            self.phase_latencies[phase] = deque(maxlen=50)
        self.phase_latencies[phase].append(latency)

    def record_throughput(self, events_per_second: float):
        self.throughput.append(events_per_second)

    def record_pathway_utilization(self, pathway: str, utilization: float):
        if pathway not in self.pathway_utilization:
            self.pathway_utilization[pathway] = deque(maxlen=50)
        self.pathway_utilization[pathway].append(utilization)

    def record_event_processing_time(self, processing_time: float):
        self.event_processing_times.append(processing_time)

    def record_queue_length(self, length: int):
        self.queue_lengths.append(length)

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "throughput": {
                "current": self.throughput[-1] if self.throughput else 0.0,
                "average": sum(self.throughput) / len(self.throughput) if self.throughput else 0.0,
                "peak": max(self.throughput) if self.throughput else 0.0,
            },
            "event_processing": {
                "avg_time": sum(self.event_processing_times) / len(self.event_processing_times) if self.event_processing_times else 0.0,
                "max_time": max(self.event_processing_times) if self.event_processing_times else 0.0,
                "min_time": min(self.event_processing_times) if self.event_processing_times else 0.0,
            },
            "queue_length": {
                "current": self.queue_lengths[-1] if self.queue_lengths else 0,
                "avg": int(sum(self.queue_lengths) / len(self.queue_lengths)) if self.queue_lengths else 0,
                "max": max(self.queue_lengths) if self.queue_lengths else 0,
            },
        }

        for phase, latencies in self.phase_latencies.items():
            if latencies:
                stats[f"{phase}_latency"] = {
                    "avg": sum(latencies) / len(latencies),
                    "max": max(latencies),
                    "min": min(latencies),
                }

        for pathway, utilizations in self.pathway_utilization.items():
            if utilizations:
                stats[f"{pathway}_utilization"] = {
                    "avg": sum(utilizations) / len(utilizations),
                    "max": max(utilizations),
                    "min": min(utilizations),
                }

        return stats


class FlowCoordinator:
    """主流程协调器

    优化主流程协调机制，完善多通路协调策略
    """

    def __init__(self, event_bus: EventBus = None):
        self._event_bus = event_bus or EventBus()
        self._performance_metrics = FlowPerformanceMetric()
        self._processing_mode = FlowControlMode.HYBRID

        self._pathway_triggers: Dict[ProcessingPathway, List[PathwayTriggerCondition]] = {}
        self._pathway_specs: Dict[ProcessingPathway, PathwayDataSpec] = {}
        self._pathway_handlers: Dict[ProcessingPathway, List[Callable]] = {}

        self._active_pathway: Optional[ProcessingPathway] = None
        self._pathway_history: deque = deque(maxlen=200)
        self._flow_history: deque = deque(maxlen=200)

        self._stats: Dict[str, Any] = {
            "total_flow_cycles": 0,
            "pathway_selections": {},
            "pathway_triggers": {},
            "flow_completions": 0,
            "flow_errors": 0,
            "dynamic_adjustments": 0,
        }

        self._lock = threading.RLock()
        self._running = False

        self._init_default_pathway_configs()

    def _init_default_pathway_configs(self):
        self._pathway_specs[ProcessingPathway.REFLEX] = PathwayDataSpec(
            pathway=ProcessingPathway.REFLEX,
            required_inputs=["perception_vector", "confidence", "free_energy"],
            optional_inputs=["context", "memory_snapshot"],
            output_format="action_vector",
            validation_schema={
                "confidence": {"type": float, "min": 0.0, "max": 1.0},
                "free_energy": {"type": float, "min": 0.0, "max": 2.0},
            },
        )

        self._pathway_specs[ProcessingPathway.DELIBERATE] = PathwayDataSpec(
            pathway=ProcessingPathway.DELIBERATE,
            required_inputs=["perception_vector", "goal_state", "confidence", "free_energy", "novelty"],
            optional_inputs=["context", "memory_snapshot", "knowledge_graph", "history"],
            output_format="plan_action_sequence",
            validation_schema={
                "confidence": {"type": float, "min": 0.0, "max": 1.0},
                "free_energy": {"type": float, "min": 0.0, "max": 2.0},
                "novelty": {"type": float, "min": 0.0, "max": 1.0},
            },
        )

        self._pathway_specs[ProcessingPathway.META_COGNITIVE] = PathwayDataSpec(
            pathway=ProcessingPathway.META_COGNITIVE,
            required_inputs=["system_state", "performance_metrics", "cognitive_event"],
            optional_inputs=["learning_history", "decision_history", "reflection_data"],
            output_format="optimization_request",
        )

        self._pathway_triggers[ProcessingPathway.REFLEX] = [
            PathwayTriggerCondition(
                pathway=ProcessingPathway.REFLEX,
                priority=PathwayPriority.CRITICAL,
                confidence_threshold=(0.7, 1.0),
                free_energy_threshold=(0.0, 0.3),
                novelty_threshold=(0.0, 0.2),
                latency_threshold=0.1,
            ),
            PathwayTriggerCondition(
                pathway=ProcessingPathway.REFLEX,
                priority=PathwayPriority.HIGH,
                confidence_threshold=(0.5, 0.7),
                free_energy_threshold=(0.0, 0.5),
                novelty_threshold=(0.0, 0.3),
                latency_threshold=0.2,
            ),
        ]

        self._pathway_triggers[ProcessingPathway.DELIBERATE] = [
            PathwayTriggerCondition(
                pathway=ProcessingPathway.DELIBERATE,
                priority=PathwayPriority.HIGH,
                confidence_threshold=(0.2, 0.5),
                free_energy_threshold=(0.3, 1.0),
                novelty_threshold=(0.3, 0.7),
            ),
            PathwayTriggerCondition(
                pathway=ProcessingPathway.DELIBERATE,
                priority=PathwayPriority.MEDIUM,
                confidence_threshold=(0.0, 0.3),
                free_energy_threshold=(0.5, 1.5),
                novelty_threshold=(0.5, 1.0),
                is_impasse=False,
            ),
        ]

        self._pathway_triggers[ProcessingPathway.META_COGNITIVE] = [
            PathwayTriggerCondition(
                pathway=ProcessingPathway.META_COGNITIVE,
                priority=PathwayPriority.MEDIUM,
                confidence_threshold=(0.0, 0.3),
                free_energy_threshold=(0.8, 2.0),
                is_impasse=True,
            ),
            PathwayTriggerCondition(
                pathway=ProcessingPathway.META_COGNITIVE,
                priority=PathwayPriority.LOW,
                free_energy_threshold=(1.0, 2.0),
                entropy_threshold=(0.7, 1.0),
            ),
        ]

        for pathway in ProcessingPathway:
            self._stats["pathway_selections"][pathway.value] = 0
            self._stats["pathway_triggers"][pathway.value] = 0

    def start(self):
        """启动流程协调器"""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._event_bus.subscribe("flow.*", self._handle_flow_event)
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("pathway.*", self._handle_pathway_event)

        logger.info("Flow coordinator started")

    def stop(self):
        """停止流程协调器"""
        with self._lock:
            self._running = False

        self._event_bus.unsubscribe("flow.*", self._handle_flow_event)
        self._event_bus.unsubscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.unsubscribe("pathway.*", self._handle_pathway_event)

        logger.info("Flow coordinator stopped")

    def _handle_flow_event(self, event: Event):
        """处理流程事件"""
        event_type = event.event_type

        if event_type == "flow.start":
            self._start_flow_cycle(event.data)
        elif event_type == "flow.phase.complete":
            self._handle_phase_complete(event.data)
        elif event_type == "flow.complete":
            self._handle_flow_complete(event.data)
        elif event_type == "flow.error":
            self._handle_flow_error(event.data)

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        try:
            system_state = self._extract_system_state(event.data)
            if system_state:
                self._select_pathway(system_state)
        except Exception as e:
            logger.error(f"Failed to handle cognitive event: {e}")

    def _handle_pathway_event(self, event: Event):
        """处理通路事件"""
        event_type = event.event_type

        if event_type == "pathway.register":
            self._register_pathway_handler(event.data)
        elif event_type == "pathway.select":
            self._handle_pathway_selection(event.data)

    def _extract_system_state(self, data: Dict[str, Any]) -> Optional[SystemState]:
        """从事件数据中提取系统状态"""
        try:
            return SystemState(
                step=data.get("step", 0),
                free_energy=data.get("free_energy", 0.0),
                confidence=data.get("confidence", 0.5),
                novelty=data.get("novelty", 0.0),
                entropy=data.get("entropy", 0.0),
                latency=data.get("latency", 0.0),
                is_impasse=data.get("is_impasse", False),
                system_used=data.get("system_used", ""),
            )
        except Exception:
            return None

    def _select_pathway(self, state: SystemState, context: Dict[str, Any] = None) -> Optional[ProcessingPathway]:
        """根据系统状态选择处理通路"""
        candidates = []

        for pathway, conditions in self._pathway_triggers.items():
            for condition in conditions:
                if condition.evaluate(state, context):
                    candidates.append((pathway, condition.priority))

        if not candidates:
            return self._select_default_pathway(state)

        candidates.sort(key=lambda x: x[1].value)
        selected_pathway = candidates[0][0]

        with self._lock:
            self._active_pathway = selected_pathway
            self._stats["pathway_selections"][selected_pathway.value] += 1
            self._stats["pathway_triggers"][selected_pathway.value] += len(
                [c for c in candidates if c[0] == selected_pathway]
            )

        self._pathway_history.append({
            "pathway": selected_pathway.value,
            "step": state.step,
            "confidence": state.confidence,
            "free_energy": state.free_energy,
            "novelty": state.novelty,
            "timestamp": time.time(),
        })

        self._event_bus.publish(Event(
            event_type="pathway.selected",
            data={
                "pathway": selected_pathway.value,
                "step": state.step,
                "confidence": state.confidence,
                "free_energy": state.free_energy,
                "novelty": state.novelty,
            },
            source="flow_coordinator",
            priority=EventPriority.HIGH,
        ))

        return selected_pathway

    def _select_default_pathway(self, state: SystemState) -> ProcessingPathway:
        """选择默认通路"""
        if state.confidence >= 0.7:
            return ProcessingPathway.REFLEX
        elif state.novelty >= 0.5:
            return ProcessingPathway.DELIBERATE
        else:
            return ProcessingPathway.DELIBERATE

    def _start_flow_cycle(self, data: Dict[str, Any]):
        """启动流程周期"""
        with self._lock:
            self._stats["total_flow_cycles"] += 1

        start_time = time.time()
        flow_id = data.get("flow_id", f"flow_{int(start_time * 1000)}")

        self._flow_history.append({
            "flow_id": flow_id,
            "start_time": start_time,
            "status": "running",
        })

        self._event_bus.publish(Event(
            event_type="flow.phase.start",
            data={
                "flow_id": flow_id,
                "phase": FlowPhase.PERCEPTION.value,
                "timestamp": start_time,
            },
            source="flow_coordinator",
        ))

    def _handle_phase_complete(self, data: Dict[str, Any]):
        """处理阶段完成"""
        phase = data.get("phase", "")
        latency = data.get("latency", 0.0)
        flow_id = data.get("flow_id", "")

        self._performance_metrics.record_phase_latency(phase, latency)

        phase_order = [FlowPhase.PERCEPTION, FlowPhase.COGNITION, FlowPhase.EXECUTION, FlowPhase.FEEDBACK]
        current_index = next((i for i, p in enumerate(phase_order) if p.value == phase), -1)

        if current_index >= 0 and current_index < len(phase_order) - 1:
            next_phase = phase_order[current_index + 1]
            self._event_bus.publish(Event(
                event_type="flow.phase.start",
                data={
                    "flow_id": flow_id,
                    "phase": next_phase.value,
                    "timestamp": time.time(),
                },
                source="flow_coordinator",
            ))

    def _handle_flow_complete(self, data: Dict[str, Any]):
        """处理流程完成"""
        flow_id = data.get("flow_id", "")
        duration = data.get("duration", 0.0)
        pathway = data.get("pathway", "")

        with self._lock:
            self._stats["flow_completions"] += 1

        self._performance_metrics.record_event_processing_time(duration)

        for entry in self._flow_history:
            if entry.get("flow_id") == flow_id:
                entry["status"] = "completed"
                entry["duration"] = duration
                entry["pathway"] = pathway
                break

        self._event_bus.publish(Event(
            event_type="flow.stats.update",
            data=self.get_stats(),
            source="flow_coordinator",
            priority=EventPriority.LOW,
        ))

    def _handle_flow_error(self, data: Dict[str, Any]):
        """处理流程错误"""
        flow_id = data.get("flow_id", "")
        error = data.get("error", "")

        with self._lock:
            self._stats["flow_errors"] += 1

        for entry in self._flow_history:
            if entry.get("flow_id") == flow_id:
                entry["status"] = "error"
                entry["error"] = error
                break

        logger.error(f"Flow error: {flow_id} - {error}")

    def _register_pathway_handler(self, data: Dict[str, Any]):
        """注册通路处理器"""
        pathway_name = data.get("pathway", "")
        handler = data.get("handler")

        try:
            pathway = ProcessingPathway(pathway_name)
            if pathway not in self._pathway_handlers:
                self._pathway_handlers[pathway] = []
            if handler not in self._pathway_handlers[pathway]:
                self._pathway_handlers[pathway].append(handler)
                logger.info(f"Registered handler for pathway: {pathway_name}")
        except ValueError:
            logger.error(f"Invalid pathway: {pathway_name}")

    def _handle_pathway_selection(self, data: Dict[str, Any]):
        """处理通路选择"""
        pathway_name = data.get("pathway", "")

        try:
            pathway = ProcessingPathway(pathway_name)
            spec = self._pathway_specs.get(pathway)

            if spec:
                input_data = data.get("input", {})
                valid, errors = spec.validate_input(input_data)

                if valid:
                    transformed_data = spec.transform_data(input_data)
                    self._event_bus.publish(Event(
                        event_type=f"pathway.{pathway_name}.data",
                        data={
                            "input": transformed_data,
                            "spec": {
                                "required_inputs": spec.required_inputs,
                                "output_format": spec.output_format,
                            },
                        },
                        source="flow_coordinator",
                    ))
                else:
                    logger.warning(f"Pathway input validation failed: {errors}")
        except ValueError:
            logger.error(f"Invalid pathway: {pathway_name}")

    def get_pathway_spec(self, pathway: ProcessingPathway) -> Optional[PathwayDataSpec]:
        """获取通路数据规范"""
        return self._pathway_specs.get(pathway)

    def add_pathway_trigger(self, pathway: ProcessingPathway, condition: PathwayTriggerCondition):
        """添加通路触发条件"""
        if pathway not in self._pathway_triggers:
            self._pathway_triggers[pathway] = []
        self._pathway_triggers[pathway].append(condition)

    def remove_pathway_trigger(self, pathway: ProcessingPathway, condition: PathwayTriggerCondition):
        """移除通路触发条件"""
        if pathway in self._pathway_triggers and condition in self._pathway_triggers[pathway]:
            self._pathway_triggers[pathway].remove(condition)

    def get_active_pathway(self) -> Optional[ProcessingPathway]:
        """获取当前激活的通路"""
        with self._lock:
            return self._active_pathway

    def set_processing_mode(self, mode: FlowControlMode):
        """设置处理模式"""
        with self._lock:
            self._processing_mode = mode
        logger.info(f"Flow processing mode set to: {mode.value}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats,
                "active_pathway": self._active_pathway.value if self._active_pathway else None,
                "processing_mode": self._processing_mode.value,
                "pathway_specs": {p.value: {
                    "required_inputs": s.required_inputs,
                    "optional_inputs": s.optional_inputs,
                    "output_format": s.output_format,
                } for p, s in self._pathway_specs.items()},
                "performance_metrics": self._performance_metrics.get_stats(),
            }

    def get_recent_pathway_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近通路选择历史"""
        with self._lock:
            return list(self._pathway_history)[-limit:]

    def get_recent_flow_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近流程历史"""
        with self._lock:
            return list(self._flow_history)[-limit:]

    def trigger_dynamic_adjustment(self, metrics: Dict[str, Any]):
        """触发动态调整"""
        throughput = metrics.get("throughput", {}).get("current", 0.0)
        avg_latency = metrics.get("event_processing", {}).get("avg_time", 0.0)
        queue_length = metrics.get("queue_length", {}).get("current", 0)

        adjustments = []

        if queue_length > 100:
            adjustments.append("reducing_non_critical_events")
        if avg_latency > 1.0:
            adjustments.append("switching_to_async_mode")
        if throughput < 10.0:
            adjustments.append("increasing_parallelism")

        if adjustments:
            with self._lock:
                self._stats["dynamic_adjustments"] += 1

            self._event_bus.publish(Event(
                event_type="flow.dynamic_adjustment",
                data={
                    "adjustments": adjustments,
                    "metrics": metrics,
                    "timestamp": time.time(),
                },
                source="flow_coordinator",
                priority=EventPriority.HIGH,
            ))

            logger.info(f"Dynamic adjustments triggered: {adjustments}")


_flow_coordinator_instance: Optional[FlowCoordinator] = None
_flow_coordinator_lock = threading.Lock()


def get_flow_coordinator(event_bus: EventBus = None) -> FlowCoordinator:
    """获取流程协调器单例"""
    global _flow_coordinator_instance
    with _flow_coordinator_lock:
        if _flow_coordinator_instance is None:
            if event_bus is None:
                event_bus = EventBus()
            _flow_coordinator_instance = FlowCoordinator(event_bus)
        return _flow_coordinator_instance