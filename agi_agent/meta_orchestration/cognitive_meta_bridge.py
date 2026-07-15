"""
cognitive_meta_bridge.py - 认知-元模块桥梁

实现认知流程与元模块之间的实时反馈通道，支持：
- 认知事件实时监听与转发
- 学习反馈收集与参数优化触发
- 决策效果评估与策略迭代
- 解析结果深度分析与语义理解增强
- 编程任务生成与代码质量监控
"""
import time
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from ..orchestration.event_bus import EventBus, Event, EventPriority
from .data_contract import (
    CognitiveEvent,
    LearningFeedback,
    DecisionFeedback,
    ParsingResult,
    ProgrammingTask,
    OptimizationRequest,
    OptimizationResult,
    ModuleStatusUpdate,
    EventCategory,
    EventAction,
    DataContractSerializer,
    DataContractFactory,
)

logger = logging.getLogger(__name__)


class BridgeChannel(Enum):
    """桥梁通道类型"""
    LEARNING = "learning"
    DECISION = "decision"
    PARSING = "parsing"
    PROGRAMMING = "programming"
    SYSTEM = "system"


class FeedbackTrigger(Enum):
    """反馈触发条件"""
    CONFIDENCE_LOW = "confidence_low"
    PERFORMANCE_DEGRADE = "performance_degrade"
    IMPASSE_DETECTED = "impasse_detected"
    ENTROPY_HIGH = "entropy_high"
    FREE_ENERGY_HIGH = "free_energy_high"
    PERIODIC = "periodic"
    MANUAL = "manual"


@dataclass
class BridgeConfig:
    """桥梁配置"""
    enabled_channels: Set[BridgeChannel] = field(default_factory=lambda: {
        BridgeChannel.LEARNING,
        BridgeChannel.DECISION,
        BridgeChannel.PARSING,
        BridgeChannel.PROGRAMMING,
        BridgeChannel.SYSTEM,
    })
    feedback_triggers: Set[FeedbackTrigger] = field(default_factory=lambda: {
        FeedbackTrigger.CONFIDENCE_LOW,
        FeedbackTrigger.PERFORMANCE_DEGRADE,
        FeedbackTrigger.IMPASSE_DETECTED,
        FeedbackTrigger.PERIODIC,
    })
    confidence_threshold: float = 0.3
    performance_degrade_threshold: float = 0.15
    entropy_threshold: float = 0.8
    free_energy_threshold: float = 0.5
    periodic_interval_sec: float = 60.0
    max_feedback_queue: int = 1000
    enable_real_time: bool = True


@dataclass
class FeedbackRecord:
    """反馈记录"""
    feedback_id: str
    channel: BridgeChannel
    trigger: FeedbackTrigger
    timestamp: float
    data: Dict[str, Any]
    processed: bool = False


class CognitiveMetaBridge:
    """认知-元模块桥梁

    实现认知流程与四大元模块（元学习、元决策、元编程、元解析）之间的实时反馈机制
    """

    def __init__(self, event_bus: EventBus, config: BridgeConfig = None):
        self._event_bus = event_bus
        self._config = config or BridgeConfig()
        self._channel_handlers: Dict[BridgeChannel, List[Callable]] = {}
        self._feedback_queue: List[FeedbackRecord] = []
        self._cognitive_context: Dict[str, Any] = {}
        self._last_periodic_trigger: float = 0.0
        self._stats: Dict[str, Any] = {
            "events_received": 0,
            "feedback_generated": 0,
            "feedback_processed": 0,
            "optimizations_requested": 0,
            "optimizations_completed": 0,
            "channel_activity": {},
        }
        self._lock = threading.RLock()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None

        for channel in BridgeChannel:
            self._channel_handlers[channel] = []
            self._stats["channel_activity"][channel.value] = {
                "events": 0,
                "feedback": 0,
                "optimizations": 0,
            }

    def start(self):
        """启动桥梁"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._register_event_listeners()

            if self._config.enable_real_time:
                self._processing_thread = threading.Thread(
                    target=self._feedback_processing_loop,
                    daemon=True,
                    name="cognitive-meta-bridge-processor"
                )
                self._processing_thread.start()

    def stop(self):
        """停止桥梁"""
        with self._lock:
            self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

    def _register_event_listeners(self):
        """注册事件监听器"""
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("learning.*", self._handle_learning_event)
        self._event_bus.subscribe("decision.*", self._handle_decision_event)
        self._event_bus.subscribe("parsing.*", self._handle_parsing_event)
        self._event_bus.subscribe("programming.*", self._handle_programming_event)
        self._event_bus.subscribe("system.*", self._handle_system_event)

    def subscribe_to_channel(self, channel: BridgeChannel, handler: Callable):
        """订阅桥梁通道"""
        if handler not in self._channel_handlers[channel]:
            self._channel_handlers[channel].append(handler)

    def unsubscribe_from_channel(self, channel: BridgeChannel, handler: Callable):
        """取消订阅桥梁通道"""
        if handler in self._channel_handlers[channel]:
            self._channel_handlers[channel].remove(handler)

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        with self._lock:
            self._stats["events_received"] += 1

        try:
            cognitive_event = self._parse_cognitive_event(event)
            if cognitive_event:
                self._update_cognitive_context(cognitive_event)
                self._evaluate_and_trigger_feedback(cognitive_event)
        except Exception as e:
            logger.error(f"Failed to handle cognitive event: {e}")

    def _handle_learning_event(self, event: Event):
        """处理学习事件"""
        with self._lock:
            self._stats["channel_activity"][BridgeChannel.LEARNING.value]["events"] += 1

        if BridgeChannel.LEARNING in self._config.enabled_channels:
            self._notify_channel(BridgeChannel.LEARNING, event.data)

    def _handle_decision_event(self, event: Event):
        """处理决策事件"""
        with self._lock:
            self._stats["channel_activity"][BridgeChannel.DECISION.value]["events"] += 1

        if BridgeChannel.DECISION in self._config.enabled_channels:
            self._notify_channel(BridgeChannel.DECISION, event.data)

    def _handle_parsing_event(self, event: Event):
        """处理解析事件"""
        with self._lock:
            self._stats["channel_activity"][BridgeChannel.PARSING.value]["events"] += 1

        if BridgeChannel.PARSING in self._config.enabled_channels:
            self._notify_channel(BridgeChannel.PARSING, event.data)

    def _handle_programming_event(self, event: Event):
        """处理编程事件"""
        with self._lock:
            self._stats["channel_activity"][BridgeChannel.PROGRAMMING.value]["events"] += 1

        if BridgeChannel.PROGRAMMING in self._config.enabled_channels:
            self._notify_channel(BridgeChannel.PROGRAMMING, event.data)

    def _handle_system_event(self, event: Event):
        """处理系统事件"""
        with self._lock:
            self._stats["channel_activity"][BridgeChannel.SYSTEM.value]["events"] += 1

        if BridgeChannel.SYSTEM in self._config.enabled_channels:
            self._notify_channel(BridgeChannel.SYSTEM, event.data)

    def _parse_cognitive_event(self, event: Event) -> Optional[CognitiveEvent]:
        """解析认知事件"""
        try:
            if "_contract_type" in event.data:
                return DataContractSerializer.deserialize(event.data)

            return CognitiveEvent(
                event_id=event.event_id,
                source_module=event.source,
                timestamp=event.timestamp,
                **event.data
            )
        except Exception as e:
            logger.warning(f"Failed to parse cognitive event: {e}")
            return None

    def _update_cognitive_context(self, event: CognitiveEvent):
        """更新认知上下文"""
        with self._lock:
            self._cognitive_context["last_event_time"] = event.timestamp
            self._cognitive_context["last_confidence"] = event.confidence
            self._cognitive_context["last_free_energy"] = event.free_energy
            self._cognitive_context["last_entropy"] = event.entropy
            self._cognitive_context["last_source"] = event.source_module
            self._cognitive_context["is_impasse"] = event.is_impasse

            if event.feature_vector:
                self._cognitive_context["last_features"] = event.feature_vector

    def _evaluate_and_trigger_feedback(self, event: CognitiveEvent):
        """评估事件并触发反馈"""
        triggers = []

        if FeedbackTrigger.CONFIDENCE_LOW in self._config.feedback_triggers:
            if event.confidence < self._config.confidence_threshold:
                triggers.append(FeedbackTrigger.CONFIDENCE_LOW)

        if FeedbackTrigger.IMPASSE_DETECTED in self._config.feedback_triggers:
            if event.is_impasse:
                triggers.append(FeedbackTrigger.IMPASSE_DETECTED)

        if FeedbackTrigger.ENTROPY_HIGH in self._config.feedback_triggers:
            if event.entropy > self._config.entropy_threshold:
                triggers.append(FeedbackTrigger.ENTROPY_HIGH)

        if FeedbackTrigger.FREE_ENERGY_HIGH in self._config.feedback_triggers:
            if event.free_energy > self._config.free_energy_threshold:
                triggers.append(FeedbackTrigger.FREE_ENERGY_HIGH)

        if FeedbackTrigger.PERIODIC in self._config.feedback_triggers:
            now = time.time()
            if now - self._last_periodic_trigger >= self._config.periodic_interval_sec:
                triggers.append(FeedbackTrigger.PERIODIC)
                self._last_periodic_trigger = now

        for trigger in triggers:
            self._generate_feedback(event, trigger)

    def _generate_feedback(self, event: CognitiveEvent, trigger: FeedbackTrigger):
        """生成反馈"""
        feedback_data = {
            "cognitive_event": DataContractSerializer.serialize(event),
            "trigger": trigger.value,
            "context": dict(self._cognitive_context),
        }

        record = FeedbackRecord(
            feedback_id=f"fb_{int(time.time() * 1000)}_{id(event)}",
            channel=self._determine_channel(event),
            trigger=trigger,
            timestamp=time.time(),
            data=feedback_data,
        )

        with self._lock:
            self._feedback_queue.append(record)
            self._stats["feedback_generated"] += 1
            self._stats["channel_activity"][record.channel.value]["feedback"] += 1

            if len(self._feedback_queue) > self._config.max_feedback_queue:
                self._feedback_queue = self._feedback_queue[-self._config.max_feedback_queue:]

        if self._config.enable_real_time:
            self._notify_channel(record.channel, feedback_data)

    def _determine_channel(self, event: CognitiveEvent) -> BridgeChannel:
        """确定反馈通道"""
        category = event.category
        if category == EventCategory.LEARNING:
            return BridgeChannel.LEARNING
        elif category == EventCategory.DECISION:
            return BridgeChannel.DECISION
        elif category == EventCategory.PARSING:
            return BridgeChannel.PARSING
        elif category == EventCategory.PROGRAMMING:
            return BridgeChannel.PROGRAMMING
        elif category == EventCategory.SYSTEM:
            return BridgeChannel.SYSTEM
        elif category == EventCategory.COGNITIVE:
            if event.is_impasse or event.confidence < 0.3:
                return BridgeChannel.LEARNING
            elif event.causal_effect > 0:
                return BridgeChannel.DECISION
        return BridgeChannel.SYSTEM

    def _notify_channel(self, channel: BridgeChannel, data: Dict[str, Any]):
        """通知通道订阅者"""
        for handler in self._channel_handlers[channel]:
            try:
                handler(channel, data)
            except Exception as e:
                logger.error(f"Channel handler error for {channel.value}: {e}")

    def _feedback_processing_loop(self):
        """反馈处理循环"""
        while self._running:
            try:
                record = None
                with self._lock:
                    if self._feedback_queue:
                        record = self._feedback_queue.pop(0)

                if record:
                    self._process_feedback_record(record)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Feedback processing loop error: {e}")
                time.sleep(1.0)

    def _process_feedback_record(self, record: FeedbackRecord):
        """处理反馈记录"""
        try:
            if record.channel == BridgeChannel.LEARNING:
                self._trigger_learning_optimization(record)
            elif record.channel == BridgeChannel.DECISION:
                self._trigger_decision_optimization(record)
            elif record.channel == BridgeChannel.PARSING:
                self._trigger_parsing_optimization(record)
            elif record.channel == BridgeChannel.PROGRAMMING:
                self._trigger_programming_optimization(record)

            record.processed = True
            with self._lock:
                self._stats["feedback_processed"] += 1
        except Exception as e:
            logger.error(f"Failed to process feedback record {record.feedback_id}: {e}")

    def _trigger_learning_optimization(self, record: FeedbackRecord):
        """触发学习优化"""
        cognitive_event = record.data.get("cognitive_event", {})

        feedback = DataContractFactory.create_learning_feedback(
            task_id=cognitive_event.get("task_id", ""),
            performance_metrics={
                "confidence": cognitive_event.get("confidence", 0.0),
                "free_energy": cognitive_event.get("free_energy", 0.0),
                "entropy": cognitive_event.get("entropy", 0.0),
            },
            learning_outcome="needs_adjustment" if cognitive_event.get("confidence", 1.0) < 0.5 else "stable",
            improvement=0.0,
            cognitive_context=record.data.get("context", {}),
        )

        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_learning",
            optimization_type="hyperparameter_adjustment",
            current_params=cognitive_event.get("hyperparameters", {}),
            performance_data={
                "confidence": cognitive_event.get("confidence", 0.0),
                "trigger": record.trigger.value,
            },
        )

        self._publish_optimization_request(optimization_request)

        with self._lock:
            self._stats["optimizations_requested"] += 1
            self._stats["channel_activity"][BridgeChannel.LEARNING.value]["optimizations"] += 1

    def _trigger_decision_optimization(self, record: FeedbackRecord):
        """触发决策优化"""
        cognitive_event = record.data.get("cognitive_event", {})

        feedback = DataContractFactory.create_decision_feedback(
            decision_id=cognitive_event.get("decision_id", ""),
            outcome="pending",
            confidence=cognitive_event.get("confidence", 0.5),
            quality_metrics={
                "confidence": cognitive_event.get("confidence", 0.5),
                "causal_effect": cognitive_event.get("causal_effect", 0.0),
                "entropy": cognitive_event.get("entropy", 0.0),
            },
        )

        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_decision",
            optimization_type="strategy_adjustment",
            current_params=cognitive_event.get("decision_params", {}),
            performance_data={
                "confidence": cognitive_event.get("confidence", 0.5),
                "causal_effect": cognitive_event.get("causal_effect", 0.0),
                "trigger": record.trigger.value,
            },
        )

        self._publish_optimization_request(optimization_request)

        with self._lock:
            self._stats["optimizations_requested"] += 1
            self._stats["channel_activity"][BridgeChannel.DECISION.value]["optimizations"] += 1

    def _trigger_parsing_optimization(self, record: FeedbackRecord):
        """触发解析优化"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_parsing",
            optimization_type="semantic_enhancement",
            current_params={},
            performance_data={
                "trigger": record.trigger.value,
                "context": record.data.get("context", {}),
            },
        )

        self._publish_optimization_request(optimization_request)

        with self._lock:
            self._stats["optimizations_requested"] += 1
            self._stats["channel_activity"][BridgeChannel.PARSING.value]["optimizations"] += 1

    def _trigger_programming_optimization(self, record: FeedbackRecord):
        """触发编程优化"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_programming",
            optimization_type="code_analysis",
            current_params={},
            performance_data={
                "trigger": record.trigger.value,
                "context": record.data.get("context", {}),
            },
        )

        self._publish_optimization_request(optimization_request)

        with self._lock:
            self._stats["optimizations_requested"] += 1
            self._stats["channel_activity"][BridgeChannel.PROGRAMMING.value]["optimizations"] += 1

    def _publish_optimization_request(self, request: OptimizationRequest):
        """发布优化请求"""
        serialized = DataContractSerializer.serialize(request)
        event = Event(
            event_type=f"optimization.request.{request.target_module}",
            data=serialized,
            source="cognitive_meta_bridge",
            priority=EventPriority.HIGH,
        )
        self._event_bus.publish(event)

    def publish_cognitive_event(self, event: CognitiveEvent):
        """发布认知事件到桥梁"""
        serialized = DataContractSerializer.serialize(event)
        event_obj = Event(
            event_type=f"cognitive.{event.category.value}",
            data=serialized,
            source=event.source_module,
            timestamp=event.timestamp,
        )
        self._event_bus.publish(event_obj)

    def publish_learning_feedback(self, feedback: LearningFeedback):
        """发布学习反馈"""
        serialized = DataContractSerializer.serialize(feedback)
        event = Event(
            event_type="learning.feedback",
            data=serialized,
            source="cognitive_meta_bridge",
            priority=EventPriority.NORMAL,
        )
        self._event_bus.publish(event)

    def publish_decision_feedback(self, feedback: DecisionFeedback):
        """发布决策反馈"""
        serialized = DataContractSerializer.serialize(feedback)
        event = Event(
            event_type="decision.feedback",
            data=serialized,
            source="cognitive_meta_bridge",
            priority=EventPriority.NORMAL,
        )
        self._event_bus.publish(event)

    def publish_parsing_result(self, result: ParsingResult):
        """发布解析结果"""
        serialized = DataContractSerializer.serialize(result)
        event = Event(
            event_type="parsing.result",
            data=serialized,
            source="cognitive_meta_bridge",
            priority=EventPriority.NORMAL,
        )
        self._event_bus.publish(event)

    def publish_programming_task(self, task: ProgrammingTask):
        """发布编程任务"""
        serialized = DataContractSerializer.serialize(task)
        event = Event(
            event_type="programming.task",
            data=serialized,
            source="cognitive_meta_bridge",
            priority=EventPriority.NORMAL,
        )
        self._event_bus.publish(event)

    def handle_optimization_result(self, result: OptimizationResult):
        """处理优化结果"""
        with self._lock:
            self._stats["optimizations_completed"] += 1

        if result.success:
            logger.info(f"Optimization successful for request {result.request_id}")
        else:
            logger.warning(f"Optimization failed for request {result.request_id}: {result.warnings}")

    def get_cognitive_context(self) -> Dict[str, Any]:
        """获取当前认知上下文"""
        with self._lock:
            return dict(self._cognitive_context)

    def get_stats(self) -> Dict[str, Any]:
        """获取桥梁统计信息"""
        with self._lock:
            return {
                "events_received": self._stats["events_received"],
                "feedback_generated": self._stats["feedback_generated"],
                "feedback_processed": self._stats["feedback_processed"],
                "optimizations_requested": self._stats["optimizations_requested"],
                "optimizations_completed": self._stats["optimizations_completed"],
                "channel_activity": dict(self._stats["channel_activity"]),
                "queue_size": len(self._feedback_queue),
            }

    def get_config(self) -> BridgeConfig:
        """获取桥梁配置"""
        return self._config

    def update_config(self, **kwargs):
        """更新桥梁配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated bridge config: {key} = {value}")


_bridge_instance: Optional[CognitiveMetaBridge] = None
_bridge_lock = threading.Lock()


def get_cognitive_meta_bridge(event_bus: EventBus = None) -> CognitiveMetaBridge:
    """获取认知-元模块桥梁单例"""
    global _bridge_instance
    with _bridge_lock:
        if _bridge_instance is None:
            if event_bus is None:
                from ..orchestration.event_bus import EventBus
                event_bus = EventBus()
            _bridge_instance = CognitiveMetaBridge(event_bus)
        return _bridge_instance