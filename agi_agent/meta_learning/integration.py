"""
integration.py - 元学习模块深度集成

实现元学习模块与核心认知流程的深度融合，包括：
- 认知事件实时监听与反馈处理
- 超参数动态调整与实时优化
- 学习效果评估与策略迭代
- 与认知-元模块桥梁的无缝对接
"""
import time
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

from ..meta_orchestration.cognitive_meta_bridge import (
    BridgeChannel,
    FeedbackTrigger,
    CognitiveMetaBridge,
    get_cognitive_meta_bridge,
)
from ..meta_orchestration.data_contract import (
    CognitiveEvent,
    LearningFeedback,
    OptimizationRequest,
    OptimizationResult,
    EventCategory,
    EventAction,
    DataContractSerializer,
    DataContractFactory,
)
from ..orchestration.event_bus import EventBus, Event, EventPriority

from .meta_learner import MetaLearner, MetaLearningTask, MetaLearningResult, MetaLearningMode
from .hyperparameter_controller import (
    HyperparameterController,
    ParameterType,
    AdjustmentStrategy,
)

logger = logging.getLogger(__name__)


class LearningIntegrationMode(Enum):
    PASSIVE = "passive"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"


class LearningAdaptationTrigger(Enum):
    CONFIDENCE_DROP = "confidence_drop"
    IMPASSE_DETECTED = "impasse_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PERIODIC_REVIEW = "periodic_review"
    TASK_SWITCH = "task_switch"
    KNOWLEDGE_GAP = "knowledge_gap"


class MetaLearningIntegration:
    """元学习模块深度集成控制器

    实现元学习模块与核心认知流程的深度融合，建立实时反馈机制
    """

    def __init__(self, event_bus: EventBus = None):
        self._event_bus = event_bus
        self._meta_learner = MetaLearner()
        self._hyperparameter_controller = HyperparameterController(
            adjustment_strategy=AdjustmentStrategy.REAL_TIME
        )
        self._bridge = get_cognitive_meta_bridge(event_bus)

        self._integration_mode = LearningIntegrationMode.REACTIVE
        self._enabled_triggers: Set[LearningAdaptationTrigger] = {
            LearningAdaptationTrigger.CONFIDENCE_DROP,
            LearningAdaptationTrigger.IMPASSE_DETECTED,
            LearningAdaptationTrigger.PERFORMANCE_DEGRADATION,
            LearningAdaptationTrigger.PERIODIC_REVIEW,
        }

        self._active_tasks: Dict[str, MetaLearningTask] = {}
        self._learning_results: Dict[str, MetaLearningResult] = {}
        self._feedback_history: deque = deque(maxlen=500)
        self._optimization_queue: deque = deque(maxlen=100)
        self._task_context_cache: Dict[str, Dict[str, Any]] = {}

        self._stats: Dict[str, Any] = {
            "events_processed": 0,
            "adaptations_triggered": 0,
            "optimizations_requested": 0,
            "optimizations_completed": 0,
            "hyperparameter_adjustments": 0,
            "knowledge_transfers": 0,
        }

        self._lock = threading.RLock()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._last_periodic_review = 0.0
        self._periodic_review_interval = 120.0

    def start(self):
        """启动元学习集成"""
        with self._lock:
            if self._running:
                return

            self._running = True

        self._bridge.subscribe_to_channel(BridgeChannel.LEARNING, self._handle_learning_channel)
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("optimization.request.meta_learning", self._handle_optimization_request)
        self._event_bus.subscribe("optimization.result.*", self._handle_optimization_result)

        self._processing_thread = threading.Thread(
            target=self._optimization_processing_loop,
            daemon=True,
            name="meta-learning-integration-processor"
        )
        self._processing_thread.start()

        logger.info("Meta-learning integration started")

    def stop(self):
        """停止元学习集成"""
        with self._lock:
            self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

        self._bridge.unsubscribe_from_channel(BridgeChannel.LEARNING, self._handle_learning_channel)
        logger.info("Meta-learning integration stopped")

    def _handle_learning_channel(self, channel: BridgeChannel, data: Dict[str, Any]):
        """处理学习通道消息"""
        try:
            trigger = data.get("trigger", "")
            cognitive_event = data.get("cognitive_event", {})
            context = data.get("context", {})

            if trigger == FeedbackTrigger.CONFIDENCE_LOW.value:
                self._handle_confidence_drop(cognitive_event, context)
            elif trigger == FeedbackTrigger.IMPASSE_DETECTED.value:
                self._handle_impasse_detected(cognitive_event, context)
            elif trigger == FeedbackTrigger.PERIODIC.value:
                self._handle_periodic_review(cognitive_event, context)
            elif trigger == FeedbackTrigger.ENTROPY_HIGH.value:
                self._handle_high_entropy(cognitive_event, context)

        except Exception as e:
            logger.error(f"Failed to handle learning channel message: {e}")

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        with self._lock:
            self._stats["events_processed"] += 1

        try:
            cognitive_event = self._parse_cognitive_event(event)
            if cognitive_event:
                self._process_cognitive_event(cognitive_event)
        except Exception as e:
            logger.error(f"Failed to process cognitive event: {e}")

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

    def _process_cognitive_event(self, event: CognitiveEvent):
        """处理认知事件并触发相应的学习自适应"""
        if event.is_impasse and LearningAdaptationTrigger.IMPASSE_DETECTED in self._enabled_triggers:
            self._trigger_adaptation(
                LearningAdaptationTrigger.IMPASSE_DETECTED,
                event,
                {"reason": "impasse_detected", "confidence": event.confidence}
            )

        if event.confidence < 0.3 and LearningAdaptationTrigger.CONFIDENCE_DROP in self._enabled_triggers:
            self._trigger_adaptation(
                LearningAdaptationTrigger.CONFIDENCE_DROP,
                event,
                {"reason": "confidence_drop", "confidence": event.confidence}
            )

        now = time.time()
        if now - self._last_periodic_review >= self._periodic_review_interval:
            if LearningAdaptationTrigger.PERIODIC_REVIEW in self._enabled_triggers:
                self._trigger_adaptation(
                    LearningAdaptationTrigger.PERIODIC_REVIEW,
                    event,
                    {"reason": "periodic_review"}
                )
            self._last_periodic_review = now

        self._update_hyperparameters_from_event(event)

    def _trigger_adaptation(self, trigger: LearningAdaptationTrigger,
                           event: CognitiveEvent, context: Dict[str, Any]):
        """触发学习自适应"""
        with self._lock:
            self._stats["adaptations_triggered"] += 1

        task_id = context.get("task_id", event.source_module)
        task = self._create_or_get_task(task_id, event)

        if trigger in (LearningAdaptationTrigger.IMPASSE_DETECTED, LearningAdaptationTrigger.CONFIDENCE_DROP):
            self._perform_rapid_adaptation(task, context)
        else:
            self._schedule_optimization(task, trigger, context)

    def _create_or_get_task(self, task_id: str, event: CognitiveEvent) -> MetaLearningTask:
        """创建或获取元学习任务"""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]

        task_type = event.metadata.get("task_type", "unknown")
        data_samples = event.metadata.get("data_samples", [])

        meta_context = {
            "confidence": event.confidence,
            "free_energy": event.free_energy,
            "entropy": event.entropy,
            "system_used": event.system_used,
        }

        task = MetaLearningTask(
            task_id=task_id,
            task_type=task_type,
            data_samples=data_samples,
            meta_context=meta_context
        )

        self._meta_learner.register_task(task)
        self._active_tasks[task_id] = task

        return task

    def _perform_rapid_adaptation(self, task: MetaLearningTask, context: Dict[str, Any]):
        """执行快速自适应"""
        learning_rate = self._hyperparameter_controller.get_parameter(ParameterType.LEARNING_RATE)
        inner_iterations = max(3, int(10 * (1 - context.get("confidence", 0.5))))

        result = self._meta_learner.adapt_to_task(
            task,
            num_inner_iterations=inner_iterations,
            learning_rate=learning_rate
        )

        self._learning_results[task.task_id] = result

        self._publish_learning_feedback(task, result, context)
        self._adjust_hyperparameters(result, context)

    def _schedule_optimization(self, task: MetaLearningTask,
                               trigger: LearningAdaptationTrigger, context: Dict[str, Any]):
        """调度优化任务"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_learning",
            optimization_type="hyperparameter_adjustment",
            current_params=self._hyperparameter_controller.get_parameter_summary(),
            performance_data={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "trigger": trigger.value,
                **context
            },
            priority=2 if trigger in (LearningAdaptationTrigger.IMPASSE_DETECTED, LearningAdaptationTrigger.CONFIDENCE_DROP) else 1,
        )

        with self._lock:
            self._optimization_queue.append(optimization_request)
            self._stats["optimizations_requested"] += 1

        self._event_bus.publish(Event(
            event_type="optimization.request.meta_learning",
            data=DataContractSerializer.serialize(optimization_request),
            source="meta_learning_integration",
            priority=EventPriority.HIGH
        ))

    def _handle_optimization_request(self, event: Event):
        """处理优化请求"""
        try:
            request = DataContractSerializer.deserialize(event.data)
            if not isinstance(request, OptimizationRequest):
                return

            result = self._execute_optimization(request)
            self._publish_optimization_result(request, result)

        except Exception as e:
            logger.error(f"Failed to handle optimization request: {e}")

    def _execute_optimization(self, request: OptimizationRequest) -> OptimizationResult:
        """执行优化"""
        optimization_type = request.optimization_type
        performance_data = request.performance_data
        task_id = performance_data.get("task_id", "")

        try:
            if optimization_type == "hyperparameter_adjustment":
                adjustments = self._hyperparameter_controller.adjust_all(
                    task_type=performance_data.get("task_type", "classification"),
                    force_adjust=True
                )

                with self._lock:
                    self._stats["hyperparameter_adjustments"] += 1

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=True,
                    optimized_params=adjustments,
                    performance_improvement=0.0,
                    validation_results={
                        "adjustments": adjustments,
                        "parameter_summary": self._hyperparameter_controller.get_parameter_summary()
                    }
                )

            elif optimization_type == "knowledge_transfer":
                source_task_id = performance_data.get("source_task", "")
                target_task_id = performance_data.get("target_task", task_id)

                transfer_result = self._meta_learner.transfer_knowledge(source_task_id, target_task_id)

                with self._lock:
                    self._stats["knowledge_transfers"] += 1

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=transfer_result.get("success", False),
                    optimized_params=transfer_result,
                    performance_improvement=transfer_result.get("transfer_effectiveness", 0.0)
                )

            elif optimization_type == "task_adaptation":
                if task_id in self._active_tasks:
                    task = self._active_tasks[task_id]
                    result = self._meta_learner.adapt_to_task(task)

                    return DataContractFactory.create_optimization_result(
                        request_id=request.request_id,
                        success=True,
                        optimized_params=result.to_dict(),
                        performance_improvement=result.get_best_metrics().get("accuracy", 0.0)
                    )

            return DataContractFactory.create_optimization_result(
                request_id=request.request_id,
                success=False,
                warnings=[f"Unknown optimization type: {optimization_type}"]
            )

        except Exception as e:
            return DataContractFactory.create_optimization_result(
                request_id=request.request_id,
                success=False,
                warnings=[str(e)]
            )

    def _publish_optimization_result(self, request: OptimizationRequest, result: OptimizationResult):
        """发布优化结果"""
        serialized = DataContractSerializer.serialize(result)
        self._event_bus.publish(Event(
            event_type=f"optimization.result.{request.target_module}",
            data=serialized,
            source="meta_learning_integration",
            priority=EventPriority.NORMAL
        ))

        self._bridge.handle_optimization_result(result)

    def _handle_optimization_result(self, event: Event):
        """处理优化结果"""
        try:
            result = DataContractSerializer.deserialize(event.data)
            if isinstance(result, OptimizationResult):
                with self._lock:
                    self._stats["optimizations_completed"] += 1

                if result.success:
                    logger.info(f"Meta-learning optimization successful: {result.result_id}")
                else:
                    logger.warning(f"Meta-learning optimization failed: {result.warnings}")

        except Exception as e:
            logger.error(f"Failed to handle optimization result: {e}")

    def _publish_learning_feedback(self, task: MetaLearningTask,
                                   result: MetaLearningResult, context: Dict[str, Any]):
        """发布学习反馈"""
        best_metrics = result.get_best_metrics()

        feedback = DataContractFactory.create_learning_feedback(
            task_id=task.task_id,
            performance_metrics={
                "accuracy": best_metrics.get("accuracy", 0.0),
                "loss": best_metrics.get("val_loss", 0.0),
                "confidence": context.get("confidence", 0.5),
            },
            learning_outcome="improved" if best_metrics.get("accuracy", 0.0) > 0.7 else "needs_more_training",
            improvement=best_metrics.get("accuracy", 0.0) - context.get("confidence", 0.0),
            confidence_change=best_metrics.get("accuracy", 0.0) - context.get("confidence", 0.0),
            hyperparameters=self._hyperparameter_controller.get_parameter_summary(),
            recommended_adjustments=self._hyperparameter_controller.adjust_all(),
            cognitive_context=context,
        )

        self._bridge.publish_learning_feedback(feedback)

        with self._lock:
            self._feedback_history.append(feedback.to_dict())

    def _update_hyperparameters_from_event(self, event: CognitiveEvent):
        """从认知事件更新超参数"""
        self._hyperparameter_controller.performance_metric.set_task_complexity(
            event.entropy
        )
        self._hyperparameter_controller.performance_metric.set_environment_uncertainty(
            event.free_energy
        )

        loss = 1.0 - event.confidence
        accuracy = event.confidence
        self._hyperparameter_controller.update_performance(loss, accuracy)

    def _adjust_hyperparameters(self, result: MetaLearningResult, context: Dict[str, Any]):
        """根据学习结果调整超参数"""
        best_metrics = result.get_best_metrics()

        if best_metrics.get("accuracy", 0.0) < 0.5:
            self._hyperparameter_controller.adjust_all(force_adjust=True)

    def _handle_confidence_drop(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理置信度下降"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_adaptation(
            LearningAdaptationTrigger.CONFIDENCE_DROP,
            event,
            {"reason": "confidence_drop", **context}
        )

    def _handle_impasse_detected(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理僵局检测"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_adaptation(
            LearningAdaptationTrigger.IMPASSE_DETECTED,
            event,
            {"reason": "impasse_detected", **context}
        )

    def _handle_periodic_review(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理定期回顾"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_adaptation(
            LearningAdaptationTrigger.PERIODIC_REVIEW,
            event,
            {"reason": "periodic_review", **context}
        )

    def _handle_high_entropy(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理高熵情况"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_adaptation(
            LearningAdaptationTrigger.KNOWLEDGE_GAP,
            event,
            {"reason": "high_entropy", **context}
        )

    def _optimization_processing_loop(self):
        """优化处理循环"""
        while self._running:
            try:
                request = None
                with self._lock:
                    if self._optimization_queue:
                        request = self._optimization_queue.popleft()

                if request:
                    self._execute_optimization_and_publish(request)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Optimization processing loop error: {e}")
                time.sleep(1.0)

    def _execute_optimization_and_publish(self, request: OptimizationRequest):
        """执行优化并发布结果"""
        result = self._execute_optimization(request)
        self._publish_optimization_result(request, result)

    def set_integration_mode(self, mode: LearningIntegrationMode):
        """设置集成模式"""
        self._integration_mode = mode

    def enable_trigger(self, trigger: LearningAdaptationTrigger):
        """启用触发器"""
        self._enabled_triggers.add(trigger)

    def disable_trigger(self, trigger: LearningAdaptationTrigger):
        """禁用触发器"""
        self._enabled_triggers.discard(trigger)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "events_processed": self._stats["events_processed"],
                "adaptations_triggered": self._stats["adaptations_triggered"],
                "optimizations_requested": self._stats["optimizations_requested"],
                "optimizations_completed": self._stats["optimizations_completed"],
                "hyperparameter_adjustments": self._stats["hyperparameter_adjustments"],
                "knowledge_transfers": self._stats["knowledge_transfers"],
                "active_tasks": len(self._active_tasks),
                "feedback_history_length": len(self._feedback_history),
                "optimization_queue_length": len(self._optimization_queue),
                "integration_mode": self._integration_mode.value,
                "enabled_triggers": [t.value for t in self._enabled_triggers],
            }

    def get_hyperparameter_summary(self) -> Dict[str, Any]:
        """获取超参数摘要"""
        return self._hyperparameter_controller.get_parameter_summary()

    def get_meta_knowledge(self) -> Dict[str, Any]:
        """获取元知识"""
        return self._meta_learner.get_meta_knowledge()

    def transfer_knowledge(self, source_task_id: str, target_task_id: str) -> Dict[str, Any]:
        """执行知识迁移"""
        return self._meta_learner.transfer_knowledge(source_task_id, target_task_id)

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """获取活跃任务"""
        return [task.to_dict() for task in self._active_tasks.values()]

    def get_feedback_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取反馈历史"""
        with self._lock:
            return list(self._feedback_history)[-limit:]


_integration_instance: Optional[MetaLearningIntegration] = None
_integration_lock = threading.Lock()


def get_meta_learning_integration(event_bus: EventBus = None) -> MetaLearningIntegration:
    """获取元学习集成单例"""
    global _integration_instance
    with _integration_lock:
        if _integration_instance is None:
            if event_bus is None:
                from ..orchestration.event_bus import EventBus
                event_bus = EventBus()
            _integration_instance = MetaLearningIntegration(event_bus)
        return _integration_instance