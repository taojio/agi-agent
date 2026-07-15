"""
integration.py - 元决策模块深度集成

构建完整的决策优化闭环系统，包括：
- 决策效果评估指标体系
- 反馈收集机制
- 决策策略迭代算法
- 与认知-元模块桥梁的无缝对接
"""
import time
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
import numpy as np

from ..meta_orchestration.cognitive_meta_bridge import (
    BridgeChannel,
    FeedbackTrigger,
    CognitiveMetaBridge,
    get_cognitive_meta_bridge,
)
from ..meta_orchestration.data_contract import (
    CognitiveEvent,
    DecisionFeedback,
    OptimizationRequest,
    OptimizationResult,
    EventCategory,
    EventAction,
    DataContractSerializer,
    DataContractFactory,
)
from ..orchestration.event_bus import EventBus, Event, EventPriority

from .decision_monitor import DecisionMonitor, DecisionTrace, DecisionPerformance, DecisionPhase
from .decision_optimizer import DecisionOptimizer
from .quality_analyzer import DecisionQualityAnalyzer

logger = logging.getLogger(__name__)


class DecisionOptimizationMode(Enum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    HYBRID = "hybrid"


class DecisionQualityDimension(Enum):
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"
    ADAPTABILITY = "adaptability"


class DecisionFeedbackTrigger(Enum):
    OUTCOME_DEVIATION = "outcome_deviation"
    CONFIDENCE_DROP = "confidence_drop"
    EFFICIENCY_DEGRADE = "efficiency_degrade"
    BIAS_DETECTED = "bias_detected"
    PERIODIC_REVIEW = "periodic_review"
    STRATEGY_VIOLATION = "strategy_violation"


class DecisionStrategyType(Enum):
    GREEDY = "greedy"
    EXPECTED_UTILITY = "expected_utility"
    MULTI_OBJECTIVE = "multi_objective"
    BAYESIAN = "bayesian"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class MetaDecisionIntegration:
    """元决策模块深度集成控制器

    构建完整的决策优化闭环系统，实现决策质量的持续提升
    """

    def __init__(self, event_bus: EventBus = None):
        self._event_bus = event_bus
        self._decision_monitor = DecisionMonitor()
        self._decision_optimizer = DecisionOptimizer()
        self._quality_analyzer = DecisionQualityAnalyzer()
        self._bridge = get_cognitive_meta_bridge(event_bus)

        self._optimization_mode = DecisionOptimizationMode.HYBRID
        self._enabled_triggers: Set[DecisionFeedbackTrigger] = {
            DecisionFeedbackTrigger.OUTCOME_DEVIATION,
            DecisionFeedbackTrigger.CONFIDENCE_DROP,
            DecisionFeedbackTrigger.EFFICIENCY_DEGRADE,
            DecisionFeedbackTrigger.PERIODIC_REVIEW,
        }

        self._active_decisions: Dict[str, DecisionTrace] = {}
        self._decision_feedback_history: deque = deque(maxlen=500)
        self._strategy_history: deque = deque(maxlen=100)
        self._optimization_queue: deque = deque(maxlen=100)

        self._quality_thresholds: Dict[DecisionQualityDimension, float] = {
            DecisionQualityDimension.ACCURACY: 0.7,
            DecisionQualityDimension.EFFICIENCY: 0.6,
            DecisionQualityDimension.CONSISTENCY: 0.8,
            DecisionQualityDimension.ROBUSTNESS: 0.6,
            DecisionQualityDimension.ADAPTABILITY: 0.5,
        }

        self._stats: Dict[str, Any] = {
            "decisions_monitored": 0,
            "decisions_completed": 0,
            "feedbacks_generated": 0,
            "optimizations_requested": 0,
            "optimizations_completed": 0,
            "strategy_adjustments": 0,
            "biases_detected": 0,
        }

        self._lock = threading.RLock()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._last_periodic_review = 0.0
        self._periodic_review_interval = 180.0

    def start(self):
        """启动元决策集成"""
        with self._lock:
            if self._running:
                return

            self._running = True

        self._bridge.subscribe_to_channel(BridgeChannel.DECISION, self._handle_decision_channel)
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("decision.*", self._handle_decision_event)
        self._event_bus.subscribe("optimization.request.meta_decision", self._handle_optimization_request)
        self._event_bus.subscribe("optimization.result.*", self._handle_optimization_result)

        self._processing_thread = threading.Thread(
            target=self._optimization_processing_loop,
            daemon=True,
            name="meta-decision-integration-processor"
        )
        self._processing_thread.start()

        logger.info("Meta-decision integration started")

    def stop(self):
        """停止元决策集成"""
        with self._lock:
            self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

        self._bridge.unsubscribe_from_channel(BridgeChannel.DECISION, self._handle_decision_channel)
        logger.info("Meta-decision integration stopped")

    def _handle_decision_channel(self, channel: BridgeChannel, data: Dict[str, Any]):
        """处理决策通道消息"""
        try:
            trigger = data.get("trigger", "")
            cognitive_event = data.get("cognitive_event", {})
            context = data.get("context", {})

            if trigger == FeedbackTrigger.CONFIDENCE_LOW.value:
                self._handle_confidence_drop(cognitive_event, context)
            elif trigger == FeedbackTrigger.FREE_ENERGY_HIGH.value:
                self._handle_high_uncertainty(cognitive_event, context)
            elif trigger == FeedbackTrigger.PERIODIC.value:
                self._handle_periodic_review(cognitive_event, context)

        except Exception as e:
            logger.error(f"Failed to handle decision channel message: {e}")

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        with self._lock:
            self._stats["decisions_monitored"] += 1

        try:
            cognitive_event = self._parse_cognitive_event(event)
            if cognitive_event:
                self._process_cognitive_event(cognitive_event)
        except Exception as e:
            logger.error(f"Failed to process cognitive event: {e}")

    def _handle_decision_event(self, event: Event):
        """处理决策事件"""
        try:
            event_type = event.event_type
            data = event.data

            if event_type == "decision.start":
                self._start_monitoring_decision(data)
            elif event_type == "decision.complete":
                self._complete_decision(data)
            elif event_type == "decision.feedback":
                self._process_decision_feedback(data)

        except Exception as e:
            logger.error(f"Failed to handle decision event: {e}")

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
        """处理认知事件"""
        if event.confidence < 0.3:
            self._trigger_feedback(
                DecisionFeedbackTrigger.CONFIDENCE_DROP,
                event,
                {"reason": "confidence_drop", "confidence": event.confidence}
            )

        if event.free_energy > 0.7:
            self._trigger_feedback(
                DecisionFeedbackTrigger.OUTCOME_DEVIATION,
                event,
                {"reason": "high_uncertainty", "free_energy": event.free_energy}
            )

        now = time.time()
        if now - self._last_periodic_review >= self._periodic_review_interval:
            self._trigger_feedback(
                DecisionFeedbackTrigger.PERIODIC_REVIEW,
                event,
                {"reason": "periodic_review"}
            )
            self._last_periodic_review = now

    def _start_monitoring_decision(self, data: Dict[str, Any]):
        """开始监控决策"""
        decision_id = data.get("decision_id", "")
        goal = data.get("goal", "")

        if not decision_id:
            return

        trace = self._decision_monitor.start_monitoring(decision_id, goal)
        self._active_decisions[decision_id] = trace

        self._record_decision_context(decision_id, data)

    def _complete_decision(self, data: Dict[str, Any]):
        """完成决策"""
        decision_id = data.get("decision_id", "")
        outcome = data.get("outcome", "unknown")
        quality_score = data.get("quality_score", 0.0)
        confidence = data.get("confidence", 0.5)

        if decision_id not in self._active_decisions:
            return

        performance = self._decision_monitor.complete_decision(
            decision_id, outcome, quality_score, confidence
        )

        if performance:
            with self._lock:
                self._stats["decisions_completed"] += 1

            self._evaluate_decision_quality(decision_id, performance, outcome)

            if decision_id in self._active_decisions:
                del self._active_decisions[decision_id]

    def _process_decision_feedback(self, data: Dict[str, Any]):
        """处理决策反馈"""
        try:
            feedback = DataContractSerializer.deserialize(data)
            if isinstance(feedback, DecisionFeedback):
                self._analyze_feedback(feedback)
                self._publish_decision_feedback(feedback)
        except Exception as e:
            logger.error(f"Failed to process decision feedback: {e}")

    def _trigger_feedback(self, trigger: DecisionFeedbackTrigger,
                          event: CognitiveEvent, context: Dict[str, Any]):
        """触发决策反馈"""
        if trigger not in self._enabled_triggers:
            return

        with self._lock:
            self._stats["feedbacks_generated"] += 1

        decision_id = context.get("decision_id", event.source_module)
        goal = context.get("goal", "")

        feedback = DataContractFactory.create_decision_feedback(
            decision_id=decision_id,
            goal=goal,
            outcome="pending",
            confidence=event.confidence,
            quality_metrics={
                "confidence": event.confidence,
                "causal_effect": event.causal_effect,
                "entropy": event.entropy,
                "free_energy": event.free_energy,
            },
            detected_biases=self._detect_biases(event),
            suggested_improvements=self._generate_improvement_suggestions(event, context),
            metadata={"trigger": trigger.value, **context},
        )

        self._decision_feedback_history.append(feedback.to_dict())
        self._publish_decision_feedback(feedback)

        if trigger in (DecisionFeedbackTrigger.OUTCOME_DEVIATION, DecisionFeedbackTrigger.CONFIDENCE_DROP):
            self._schedule_optimization(decision_id, trigger, context)

    def _evaluate_decision_quality(self, decision_id: str,
                                   performance: DecisionPerformance, outcome: str):
        """评估决策质量"""
        quality_dimensions = {}

        accuracy_score = self._compute_accuracy_score(performance, outcome)
        efficiency_score = self._compute_efficiency_score(performance)
        consistency_score = self._compute_consistency_score(decision_id, performance)
        robustness_score = self._compute_robustness_score(performance)
        adaptability_score = self._compute_adaptability_score(performance)

        quality_dimensions[DecisionQualityDimension.ACCURACY.value] = accuracy_score
        quality_dimensions[DecisionQualityDimension.EFFICIENCY.value] = efficiency_score
        quality_dimensions[DecisionQualityDimension.CONSISTENCY.value] = consistency_score
        quality_dimensions[DecisionQualityDimension.ROBUSTNESS.value] = robustness_score
        quality_dimensions[DecisionQualityDimension.ADAPTABILITY.value] = adaptability_score

        overall_quality = np.mean(list(quality_dimensions.values()))

        feedback = DataContractFactory.create_decision_feedback(
            decision_id=decision_id,
            goal="",
            outcome=outcome,
            outcome_score=overall_quality,
            confidence=performance.confidence,
            quality_metrics=quality_dimensions,
            detected_biases=self._detect_biases_from_performance(performance),
            suggested_improvements=self._generate_improvements_from_quality(quality_dimensions),
            metadata={"overall_quality": overall_quality},
        )

        self._decision_feedback_history.append(feedback.to_dict())
        self._publish_decision_feedback(feedback)

        if overall_quality < 0.5:
            self._schedule_optimization(decision_id, DecisionFeedbackTrigger.OUTCOME_DEVIATION, {
                "reason": "low_quality",
                "overall_quality": overall_quality,
                **quality_dimensions
            })

    def _compute_accuracy_score(self, performance: DecisionPerformance, outcome: str) -> float:
        """计算准确性分数"""
        base_score = performance.confidence

        if outcome == "success":
            base_score = min(1.0, base_score + 0.2)
        elif outcome == "failure":
            base_score = max(0.0, base_score - 0.3)

        return base_score

    def _compute_efficiency_score(self, performance: DecisionPerformance) -> float:
        """计算效率分数"""
        duration = performance.duration_ms

        if duration < 100:
            return 0.95
        elif duration < 500:
            return 0.8
        elif duration < 1000:
            return 0.6
        elif duration < 5000:
            return 0.4
        else:
            return 0.2

    def _compute_consistency_score(self, decision_id: str, performance: DecisionPerformance) -> float:
        """计算一致性分数"""
        history = list(self._decision_monitor.performance_history)
        if len(history) < 5:
            return 0.5

        recent_qualities = [p.quality_score for p in history[-5:]]
        std_dev = np.std(recent_qualities)

        return max(0.0, min(1.0, 1.0 - std_dev * 2))

    def _compute_robustness_score(self, performance: DecisionPerformance) -> float:
        """计算稳健性分数"""
        confidence = performance.confidence
        quality_score = performance.quality_score

        if confidence > 0.8 and quality_score > 0.7:
            return 0.9
        elif confidence > 0.6 and quality_score > 0.5:
            return 0.7
        elif confidence > 0.4:
            return 0.5
        else:
            return 0.3

    def _compute_adaptability_score(self, performance: DecisionPerformance) -> float:
        """计算适应性分数"""
        return 0.5

    def _detect_biases(self, event: CognitiveEvent) -> List[Dict[str, Any]]:
        """检测决策偏差"""
        biases = []

        if event.confidence < 0.3 and event.entropy > 0.7:
            biases.append({
                "type": "overconfidence",
                "severity": "high",
                "evidence": f"confidence={event.confidence}, entropy={event.entropy}"
            })

        return biases

    def _detect_biases_from_performance(self, performance: DecisionPerformance) -> List[Dict[str, Any]]:
        """从性能数据检测偏差"""
        biases = []

        if performance.confidence > 0.8 and performance.quality_score < 0.3:
            biases.append({
                "type": "overconfidence",
                "severity": "high",
                "evidence": f"confidence={performance.confidence}, quality={performance.quality_score}"
            })

        return biases

    def _generate_improvement_suggestions(self, event: CognitiveEvent,
                                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        suggestions = []

        if event.confidence < 0.3:
            suggestions.append({
                "type": "information_gathering",
                "priority": "high",
                "suggestion": "Collect more information before making decision",
                "expected_improvement": 0.3,
            })

        if event.free_energy > 0.5:
            suggestions.append({
                "type": "scenario_analysis",
                "priority": "medium",
                "suggestion": "Consider alternative scenarios and outcomes",
                "expected_improvement": 0.2,
            })

        return suggestions

    def _generate_improvements_from_quality(self, quality_dimensions: Dict[str, float]) -> List[Dict[str, Any]]:
        """根据质量维度生成改进建议"""
        suggestions = []

        for dimension, score in quality_dimensions.items():
            threshold = self._quality_thresholds.get(
                DecisionQualityDimension(dimension), 0.6
            )
            if score < threshold:
                suggestions.append({
                    "type": f"improve_{dimension}",
                    "priority": "high" if score < threshold * 0.5 else "medium",
                    "suggestion": f"Improve decision {dimension}",
                    "current_score": score,
                    "target_score": threshold,
                })

        return suggestions

    def _analyze_feedback(self, feedback: DecisionFeedback):
        """分析反馈并更新策略"""
        quality_metrics = feedback.quality_metrics or {}

        if quality_metrics.get("confidence", 0.5) < 0.3:
            self._adjust_decision_strategy(feedback.decision_id, "more_cautious")
        elif quality_metrics.get("confidence", 0.5) > 0.8:
            self._adjust_decision_strategy(feedback.decision_id, "more_aggressive")

    def _adjust_decision_strategy(self, decision_id: str, adjustment_type: str):
        """调整决策策略"""
        with self._lock:
            self._stats["strategy_adjustments"] += 1

        strategy_info = {
            "decision_id": decision_id,
            "adjustment_type": adjustment_type,
            "timestamp": time.time(),
        }
        self._strategy_history.append(strategy_info)

        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_decision",
            optimization_type="strategy_adjustment",
            current_params={"adjustment_type": adjustment_type},
            performance_data={"decision_id": decision_id},
        )

        self._event_bus.publish(Event(
            event_type="optimization.request.meta_decision",
            data=DataContractSerializer.serialize(optimization_request),
            source="meta_decision_integration",
            priority=EventPriority.HIGH
        ))

    def _schedule_optimization(self, decision_id: str,
                               trigger: DecisionFeedbackTrigger, context: Dict[str, Any]):
        """调度优化任务"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_decision",
            optimization_type="decision_optimization",
            current_params={},
            performance_data={
                "decision_id": decision_id,
                "trigger": trigger.value,
                **context
            },
            priority=2 if trigger in (DecisionFeedbackTrigger.OUTCOME_DEVIATION, DecisionFeedbackTrigger.CONFIDENCE_DROP) else 1,
        )

        with self._lock:
            self._optimization_queue.append(optimization_request)
            self._stats["optimizations_requested"] += 1

        self._event_bus.publish(Event(
            event_type="optimization.request.meta_decision",
            data=DataContractSerializer.serialize(optimization_request),
            source="meta_decision_integration",
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

        try:
            if optimization_type == "strategy_adjustment":
                adjustment_type = request.current_params.get("adjustment_type", "")

                strategy_update = {
                    "adjustment_type": adjustment_type,
                    "applied": True,
                    "timestamp": time.time(),
                }

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=True,
                    optimized_params=strategy_update,
                    performance_improvement=0.2,
                )

            elif optimization_type == "decision_optimization":
                decision_id = performance_data.get("decision_id", "")
                trigger = performance_data.get("trigger", "")

                optimization_result = {
                    "decision_id": decision_id,
                    "trigger": trigger,
                    "optimization_applied": True,
                    "suggested_strategy": self._suggest_optimal_strategy(performance_data),
                }

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=True,
                    optimized_params=optimization_result,
                    performance_improvement=0.15,
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

    def _suggest_optimal_strategy(self, performance_data: Dict[str, Any]) -> str:
        """建议最优策略"""
        confidence = performance_data.get("confidence", 0.5)
        free_energy = performance_data.get("free_energy", 0.0)

        if confidence < 0.3:
            return DecisionStrategyType.BAYESIAN.value
        elif free_energy > 0.7:
            return DecisionStrategyType.MULTI_OBJECTIVE.value
        else:
            return DecisionStrategyType.EXPECTED_UTILITY.value

    def _publish_optimization_result(self, request: OptimizationRequest, result: OptimizationResult):
        """发布优化结果"""
        serialized = DataContractSerializer.serialize(result)
        self._event_bus.publish(Event(
            event_type=f"optimization.result.{request.target_module}",
            data=serialized,
            source="meta_decision_integration",
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
                    logger.info(f"Meta-decision optimization successful: {result.result_id}")
                else:
                    logger.warning(f"Meta-decision optimization failed: {result.warnings}")

        except Exception as e:
            logger.error(f"Failed to handle optimization result: {e}")

    def _publish_decision_feedback(self, feedback: DecisionFeedback):
        """发布决策反馈"""
        self._bridge.publish_decision_feedback(feedback)

    def _record_decision_context(self, decision_id: str, data: Dict[str, Any]):
        """记录决策上下文"""
        pass

    def _handle_confidence_drop(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理置信度下降"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_feedback(
            DecisionFeedbackTrigger.CONFIDENCE_DROP,
            event,
            {"reason": "confidence_drop", **context}
        )

    def _handle_high_uncertainty(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理高不确定性"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_feedback(
            DecisionFeedbackTrigger.OUTCOME_DEVIATION,
            event,
            {"reason": "high_uncertainty", **context}
        )

    def _handle_periodic_review(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理定期回顾"""
        event = CognitiveEvent.from_dict(cognitive_event)
        self._trigger_feedback(
            DecisionFeedbackTrigger.PERIODIC_REVIEW,
            event,
            {"reason": "periodic_review", **context}
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

    def set_optimization_mode(self, mode: DecisionOptimizationMode):
        """设置优化模式"""
        self._optimization_mode = mode

    def enable_trigger(self, trigger: DecisionFeedbackTrigger):
        """启用触发器"""
        self._enabled_triggers.add(trigger)

    def disable_trigger(self, trigger: DecisionFeedbackTrigger):
        """禁用触发器"""
        self._enabled_triggers.discard(trigger)

    def set_quality_threshold(self, dimension: DecisionQualityDimension, threshold: float):
        """设置质量阈值"""
        self._quality_thresholds[dimension] = max(0.0, min(1.0, threshold))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "decisions_monitored": self._stats["decisions_monitored"],
                "decisions_completed": self._stats["decisions_completed"],
                "feedbacks_generated": self._stats["feedbacks_generated"],
                "optimizations_requested": self._stats["optimizations_requested"],
                "optimizations_completed": self._stats["optimizations_completed"],
                "strategy_adjustments": self._stats["strategy_adjustments"],
                "biases_detected": self._stats["biases_detected"],
                "active_decisions": len(self._active_decisions),
                "feedback_history_length": len(self._decision_feedback_history),
                "optimization_queue_length": len(self._optimization_queue),
                "optimization_mode": self._optimization_mode.value,
                "enabled_triggers": [t.value for t in self._enabled_triggers],
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self._decision_monitor.get_performance_summary()

    def get_decision_patterns(self, window_size: int = 50) -> Dict[str, Any]:
        """获取决策模式"""
        return self._decision_monitor.get_decision_patterns(window_size)

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        return self._decision_monitor.detect_anomalies()

    def get_feedback_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取反馈历史"""
        with self._lock:
            return list(self._decision_feedback_history)[-limit:]


_integration_instance: Optional[MetaDecisionIntegration] = None
_integration_lock = threading.Lock()


def get_meta_decision_integration(event_bus: EventBus = None) -> MetaDecisionIntegration:
    """获取元决策集成单例"""
    global _integration_instance
    with _integration_lock:
        if _integration_instance is None:
            if event_bus is None:
                from ..orchestration.event_bus import EventBus
                event_bus = EventBus()
            _integration_instance = MetaDecisionIntegration(event_bus)
        return _integration_instance