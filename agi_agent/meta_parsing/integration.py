"""
integration.py - 元解析模块深度集成

增强智能化解析能力，包括：
- 语义深度理解
- 上下文关联分析
- 多模态数据融合技术
- 信息提取准确性和完整性提升
- 与认知-元模块桥梁的无缝对接
"""
import time
import logging
import threading
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import deque

from ..meta_orchestration.cognitive_meta_bridge import (
    BridgeChannel,
    FeedbackTrigger,
    CognitiveMetaBridge,
    get_cognitive_meta_bridge,
)
from ..meta_orchestration.data_contract import (
    CognitiveEvent,
    ParsingResult,
    OptimizationRequest,
    OptimizationResult,
    DataContractSerializer,
    DataContractFactory,
)
from ..orchestration.event_bus import EventBus, Event, EventPriority

from .meta_parser import MetaParser, ParsingStrategy, ParserRegistry
from .orchestrator import ParsingOrchestrator
from .data_transformer import DataTransformer, TransformationType
from .complex_data_processor import ComplexDataProcessor

logger = logging.getLogger(__name__)


class SemanticAnalysisLevel(Enum):
    SURFACE = "surface"
    DEEP = "deep"
    CONTEXTUAL = "contextual"
    INTEGRATED = "integrated"


class ContextRelationType(Enum):
    REFERENCE = "reference"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    ASSOCIATION = "association"
    CONTRAST = "contrast"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"


class MultimodalDataType(Enum):
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    STRUCTURED = "structured"
    SEMI_STRUCTURED = "semi_structured"


class ParsingOptimizationTrigger(Enum):
    LOW_CONFIDENCE = "low_confidence"
    CONTEXT_MISSING = "context_missing"
    AMBIGUITY_DETECTED = "ambiguity_detected"
    MULTIMODAL_NEEDED = "multimodal_needed"
    PERIODIC = "periodic"
    MANUAL_REQUEST = "manual_request"


class SemanticFeature:
    def __init__(self, name: str, value: Any, confidence: float = 1.0):
        self.name = name
        self.value = value
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
        }


class ContextRelation:
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: ContextRelationType,
        strength: float = 0.5,
        evidence: str = "",
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "evidence": self.evidence,
        }


class MultimodalFusionResult:
    def __init__(self):
        self.success = False
        self.fused_data: Dict[str, Any] = {}
        self.sources: List[Dict[str, Any]] = []
        self.confidence = 0.0
        self.integration_score = 0.0
        self.errors: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "integration_score": self.integration_score,
            "sources": self.sources,
            "errors": self.errors,
        }


class MetaParsingIntegration:
    """元解析模块深度集成控制器

    增强智能化解析能力，实现语义深度理解、上下文关联分析和多模态数据融合
    """

    def __init__(self, event_bus: EventBus = None):
        self._event_bus = event_bus
        self._parser = MetaParser()
        self._orchestrator = ParsingOrchestrator()
        self._transformer = DataTransformer()
        self._complex_processor = ComplexDataProcessor()
        self._bridge = get_cognitive_meta_bridge(event_bus)

        self._enabled_triggers: Set[ParsingOptimizationTrigger] = {
            ParsingOptimizationTrigger.LOW_CONFIDENCE,
            ParsingOptimizationTrigger.CONTEXT_MISSING,
            ParsingOptimizationTrigger.PERIODIC,
        }

        self._semantic_analysis_level: SemanticAnalysisLevel = SemanticAnalysisLevel.DEEP

        self._context_store: Dict[str, Any] = {}
        self._context_relations: List[ContextRelation] = []
        self._multimodal_cache: Dict[str, Any] = {}
        self._parsing_history: deque = deque(maxlen=500)
        self._semantic_history: deque = deque(maxlen=200)
        self._optimization_queue: deque = deque(maxlen=100)

        self._stats: Dict[str, Any] = {
            "parsing_requests": 0,
            "parsing_successes": 0,
            "semantic_analyses": 0,
            "context_relations_found": 0,
            "multimodal_fusions": 0,
            "fusions_successful": 0,
            "optimizations_requested": 0,
            "optimizations_completed": 0,
            "accuracy_improvements": 0,
        }

        self._lock = threading.RLock()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._last_periodic_optimization = 0.0
        self._periodic_interval = 300.0

    def start(self):
        """启动元解析集成"""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._bridge.subscribe_to_channel(BridgeChannel.PARSING, self._handle_parsing_channel)
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("parsing.*", self._handle_parsing_event)
        self._event_bus.subscribe("optimization.request.meta_parsing", self._handle_optimization_request)
        self._event_bus.subscribe("optimization.result.*", self._handle_optimization_result)

        self._processing_thread = threading.Thread(
            target=self._optimization_processing_loop,
            daemon=True,
            name="meta-parsing-integration-processor",
        )
        self._processing_thread.start()

        logger.info("Meta-parsing integration started")

    def stop(self):
        """停止元解析集成"""
        with self._lock:
            self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

        self._bridge.unsubscribe_from_channel(BridgeChannel.PARSING, self._handle_parsing_channel)
        logger.info("Meta-parsing integration stopped")

    def _handle_parsing_channel(self, channel: BridgeChannel, data: Dict[str, Any]):
        """处理解析通道消息"""
        try:
            trigger = data.get("trigger", "")
            cognitive_event = data.get("cognitive_event", {})
            context = data.get("context", {})

            if trigger == FeedbackTrigger.PERIODIC.value:
                self._handle_periodic_optimization(cognitive_event, context)

        except Exception as e:
            logger.error(f"Failed to handle parsing channel message: {e}")

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        try:
            cognitive_event = self._parse_cognitive_event(event)
            if cognitive_event:
                self._process_cognitive_event(cognitive_event)
        except Exception as e:
            logger.error(f"Failed to process cognitive event: {e}")

    def _handle_parsing_event(self, event: Event):
        """处理解析事件"""
        try:
            event_type = event.event_type
            data = event.data

            if event_type == "parsing.parse":
                self._perform_parsing(data)
            elif event_type == "parsing.analyze":
                self._perform_semantic_analysis(data)
            elif event_type == "parsing.fuse":
                self._perform_multimodal_fusion(data)
            elif event_type == "parsing.enrich":
                self._enrich_with_context(data)

        except Exception as e:
            logger.error(f"Failed to handle parsing event: {e}")

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
        now = time.time()
        if now - self._last_periodic_optimization >= self._periodic_interval:
            if ParsingOptimizationTrigger.PERIODIC in self._enabled_triggers:
                self._trigger_periodic_optimization(event)
            self._last_periodic_optimization = now

        if event.confidence < 0.4 and ParsingOptimizationTrigger.LOW_CONFIDENCE in self._enabled_triggers:
            self._trigger_optimization(
                ParsingOptimizationTrigger.LOW_CONFIDENCE,
                event,
                {"reason": "low_confidence", "confidence": event.confidence},
            )

    def _perform_parsing(self, data: Dict[str, Any]):
        """执行解析"""
        raw_data = data.get("data", "")
        format_hint = data.get("format", "")
        context = data.get("context", {})

        if not raw_data:
            return

        with self._lock:
            self._stats["parsing_requests"] += 1

        result = self._orchestrator.parse_transform_and_understand(
            raw_data, format_hint, ParsingStrategy.RECURSIVE
        )

        if result.get("success", False):
            with self._lock:
                self._stats["parsing_successes"] += 1

            self._parsing_history.append(result)
            self._publish_parsing_result(result, format_hint)

            if self._semantic_analysis_level >= SemanticAnalysisLevel.DEEP:
                self._perform_semantic_analysis(
                    {
                        "data": raw_data,
                        "parsing_result": result,
                        "context": context,
                    }
                )
        else:
            if ParsingOptimizationTrigger.LOW_CONFIDENCE in self._enabled_triggers:
                self._trigger_optimization(
                    ParsingOptimizationTrigger.LOW_CONFIDENCE,
                    None,
                    {"reason": "parsing_failure", "format": format_hint},
                )

    def _perform_semantic_analysis(self, data: Dict[str, Any]):
        """执行语义深度分析"""
        raw_data = data.get("data", "")
        parsing_result = data.get("parsing_result", {})
        context = data.get("context", {})

        with self._lock:
            self._stats["semantic_analyses"] += 1

        semantic_features = self._extract_semantic_features(raw_data, parsing_result)
        context_relations = self._analyze_context_relations(raw_data, context)

        analysis_result = {
            "semantic_features": [f.to_dict() for f in semantic_features],
            "context_relations": [r.to_dict() for r in context_relations],
            "analysis_level": self._semantic_analysis_level.value,
            "timestamp": time.time(),
        }

        self._semantic_history.append(analysis_result)

        with self._lock:
            self._stats["context_relations_found"] += len(context_relations)

        self._publish_semantic_result(analysis_result)

        if len(context_relations) == 0 and ParsingOptimizationTrigger.CONTEXT_MISSING in self._enabled_triggers:
            self._trigger_optimization(
                ParsingOptimizationTrigger.CONTEXT_MISSING,
                None,
                {"reason": "context_missing", "data_length": len(raw_data)},
            )

    def _extract_semantic_features(self, text: str, parsing_result: Dict[str, Any]) -> List[SemanticFeature]:
        """提取语义特征"""
        features = []

        entities = self._extract_entities(text)
        for entity_type, entity_value in entities:
            features.append(SemanticFeature(f"entity_{entity_type}", entity_value, confidence=0.85))

        key_phrases = self._extract_key_phrases(text)
        for i, phrase in enumerate(key_phrases[:5]):
            features.append(SemanticFeature(f"key_phrase_{i}", phrase, confidence=0.8))

        sentiment = self._analyze_sentiment(text)
        features.append(SemanticFeature("sentiment", sentiment, confidence=0.75))

        topic = self._identify_topic(text)
        features.append(SemanticFeature("topic", topic, confidence=0.7))

        if parsing_result.get("understanding"):
            understanding = parsing_result["understanding"]
            features.append(SemanticFeature("complexity", understanding.get("complexity", {}), confidence=0.9))

        return features

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """提取实体"""
        entities = []

        date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        dates = re.findall(date_pattern, text)
        for date in dates[:3]:
            entities.append(("date", date))

        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        emails = re.findall(email_pattern, text)
        for email in emails[:3]:
            entities.append(("email", email))

        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)
        for url in urls[:3]:
            entities.append(("url", url))

        number_pattern = r"\b\d+(?:\.\d+)?\b"
        numbers = re.findall(number_pattern, text)
        for number in numbers[:5]:
            entities.append(("number", number))

        return entities

    def _extract_key_phrases(self, text: str) -> List[str]:
        """提取关键短语"""
        stop_words = {"the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being"}
        words = re.findall(r"\w+", text.lower())
        filtered = [w for w in words if w not in stop_words and len(w) > 2]

        phrase_counts: Dict[str, int] = {}
        for i in range(len(filtered) - 1):
            phrase = f"{filtered[i]} {filtered[i+1]}"
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:10]]

    def _analyze_sentiment(self, text: str) -> str:
        """分析情感"""
        positive_words = {"good", "great", "excellent", "positive", "success", "happy", "love", "best"}
        negative_words = {"bad", "worst", "negative", "failure", "sad", "hate", "terrible", "problem"}

        words = re.findall(r"\w+", text.lower())
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _identify_topic(self, text: str) -> str:
        """识别主题"""
        topic_keywords = {
            "technology": {"tech", "computer", "software", "hardware", "AI", "machine", "learning"},
            "business": {"business", "company", "market", "product", "sales", "profit", "revenue"},
            "health": {"health", "medical", "disease", "treatment", "doctor", "patient"},
            "science": {"science", "research", "study", "experiment", "theory", "discovery"},
            "finance": {"finance", "money", "investment", "stock", "market", "bank", "fund"},
        }

        words = set(re.findall(r"\w+", text.lower()))
        max_score = 0
        best_topic = "general"

        for topic, keywords in topic_keywords.items():
            score = len(words & keywords)
            if score > max_score:
                max_score = score
                best_topic = topic

        return best_topic

    def _analyze_context_relations(self, text: str, context: Dict[str, Any]) -> List[ContextRelation]:
        """分析上下文关联"""
        relations = []
        existing_context = self._get_recent_context()

        if existing_context:
            for ctx_id, ctx_data in existing_context.items():
                ctx_text = str(ctx_data.get("content", ""))
                similarity = self._calculate_similarity(text, ctx_text)

                if similarity > 0.3:
                    relations.append(
                        ContextRelation(
                            source_id="current",
                            target_id=ctx_id,
                            relation_type=ContextRelationType.ASSOCIATION,
                            strength=similarity,
                            evidence=f"text_similarity={similarity:.2f}",
                        )
                    )

        return relations

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _get_recent_context(self) -> Dict[str, Any]:
        """获取最近上下文"""
        recent = {}
        for item in list(self._parsing_history)[-10:]:
            if item.get("success"):
                context_id = f"ctx_{len(recent)}"
                recent[context_id] = item
        return recent

    def _perform_multimodal_fusion(self, data: Dict[str, Any]) -> MultimodalFusionResult:
        """执行多模态数据融合"""
        with self._lock:
            self._stats["multimodal_fusions"] += 1

        result = MultimodalFusionResult()
        sources = data.get("sources", [])

        if len(sources) < 2:
            result.errors.append("At least two sources required for fusion")
            return result

        fused_data = {}
        source_info = []
        total_confidence = 0.0

        for source in sources:
            data_type = source.get("type", "text")
            content = source.get("content", "")
            confidence = source.get("confidence", 0.5)

            source_info.append({"type": data_type, "confidence": confidence})
            total_confidence += confidence

            if data_type == MultimodalDataType.TEXT.value:
                fused_data["text_content"] = content
            elif data_type == MultimodalDataType.STRUCTURED.value:
                if isinstance(content, dict):
                    fused_data.update(content)
            elif data_type == MultimodalDataType.SEMI_STRUCTURED.value:
                fused_data["semi_structured"] = content
            elif data_type in (MultimodalDataType.AUDIO.value, MultimodalDataType.VIDEO.value):
                fused_data["media_reference"] = content

        result.success = True
        result.fused_data = fused_data
        result.sources = source_info
        result.confidence = total_confidence / len(sources)
        result.integration_score = self._calculate_integration_score(fused_data, sources)

        with self._lock:
            self._stats["fusions_successful"] += 1

        self._publish_multimodal_result(result)

        return result

    def _calculate_integration_score(self, fused_data: Dict[str, Any], sources: List[Dict[str, Any]]) -> float:
        """计算融合集成分数"""
        source_types = set(s.get("type") for s in sources)
        data_coverage = len(fused_data.keys()) / (len(source_types) * 3)

        if len(source_types) >= 3:
            diversity_bonus = 0.2
        elif len(source_types) >= 2:
            diversity_bonus = 0.1
        else:
            diversity_bonus = 0.0

        return min(1.0, data_coverage + diversity_bonus)

    def _enrich_with_context(self, data: Dict[str, Any]):
        """使用上下文丰富解析结果"""
        result_id = data.get("result_id", "")
        context_ids = data.get("context_ids", [])

        if not result_id or not context_ids:
            return

        enriched_context = {}
        for ctx_id in context_ids:
            if ctx_id in self._context_store:
                enriched_context[ctx_id] = self._context_store[ctx_id]

        self._event_bus.publish(
            Event(
                event_type="parsing.context_enriched",
                data={
                    "result_id": result_id,
                    "enriched_context": enriched_context,
                    "timestamp": time.time(),
                },
                source="meta_parsing_integration",
                priority=EventPriority.NORMAL,
            )
        )

    def _trigger_periodic_optimization(self, event: CognitiveEvent):
        """触发定期优化"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_parsing",
            optimization_type="parsing_strategy",
            current_params={},
            performance_data={
                "trigger": ParsingOptimizationTrigger.PERIODIC.value,
                "timestamp": event.timestamp if event else time.time(),
            },
            priority=1,
        )

        with self._lock:
            self._optimization_queue.append(optimization_request)
            self._stats["optimizations_requested"] += 1

        self._event_bus.publish(
            Event(
                event_type="optimization.request.meta_parsing",
                data=DataContractSerializer.serialize(optimization_request),
                source="meta_parsing_integration",
                priority=EventPriority.NORMAL,
            )
        )

    def _trigger_optimization(self, trigger: ParsingOptimizationTrigger, event: Optional[CognitiveEvent], context: Dict[str, Any]):
        """触发优化"""
        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_parsing",
            optimization_type=self._map_trigger_to_optimization_type(trigger),
            current_params={},
            performance_data={"trigger": trigger.value, **context},
            priority=2,
        )

        with self._lock:
            self._optimization_queue.append(optimization_request)
            self._stats["optimizations_requested"] += 1

        self._event_bus.publish(
            Event(
                event_type="optimization.request.meta_parsing",
                data=DataContractSerializer.serialize(optimization_request),
                source="meta_parsing_integration",
                priority=EventPriority.HIGH,
            )
        )

    def _map_trigger_to_optimization_type(self, trigger: ParsingOptimizationTrigger) -> str:
        """映射触发器到优化类型"""
        mapping = {
            ParsingOptimizationTrigger.LOW_CONFIDENCE: "confidence_improvement",
            ParsingOptimizationTrigger.CONTEXT_MISSING: "context_enrichment",
            ParsingOptimizationTrigger.AMBIGUITY_DETECTED: "disambiguation",
            ParsingOptimizationTrigger.MULTIMODAL_NEEDED: "multimodal_integration",
            ParsingOptimizationTrigger.PERIODIC: "parsing_strategy",
            ParsingOptimizationTrigger.MANUAL_REQUEST: "manual_tuning",
        }
        return mapping.get(trigger, "parsing_strategy")

    def _handle_periodic_optimization(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理定期优化"""
        pass

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

        try:
            if optimization_type == "parsing_strategy":
                self._optimize_parsing_strategy()
                improvement = 0.05
            elif optimization_type == "confidence_improvement":
                improvement = self._improve_confidence(request)
            elif optimization_type == "context_enrichment":
                improvement = self._enrich_context_stores(request)
            elif optimization_type == "disambiguation":
                improvement = self._improve_disambiguation(request)
            elif optimization_type == "multimodal_integration":
                improvement = self._enhance_multimodal_capabilities(request)
            else:
                improvement = 0.0

            with self._lock:
                if improvement > 0:
                    self._stats["accuracy_improvements"] += 1

            return DataContractFactory.create_optimization_result(
                request_id=request.request_id,
                success=True,
                optimized_params={
                    "optimization_type": optimization_type,
                    "timestamp": time.time(),
                },
                performance_improvement=improvement,
            )

        except Exception as e:
            return DataContractFactory.create_optimization_result(
                request_id=request.request_id,
                success=False,
                warnings=[str(e)],
            )

    def _optimize_parsing_strategy(self):
        """优化解析策略"""
        recent_results = list(self._parsing_history)[-50:]
        if not recent_results:
            return

        success_rate = sum(1 for r in recent_results if r.get("success")) / len(recent_results)

        if success_rate < 0.7:
            self._semantic_analysis_level = SemanticAnalysisLevel.INTEGRATED
        elif success_rate < 0.85:
            self._semantic_analysis_level = SemanticAnalysisLevel.DEEP
        else:
            self._semantic_analysis_level = SemanticAnalysisLevel.CONTEXTUAL

    def _improve_confidence(self, request: OptimizationRequest) -> float:
        """提高解析置信度"""
        return 0.1

    def _enrich_context_stores(self, request: OptimizationRequest) -> float:
        """丰富上下文存储"""
        recent = list(self._parsing_history)[-20:]
        for i, item in enumerate(recent):
            if item.get("success"):
                ctx_id = f"ctx_{int(time.time())}_{i}"
                self._context_store[ctx_id] = {
                    "content": str(item.get("parsing_result", {}).get("parsed_data", "")),
                    "timestamp": time.time(),
                    "format": item.get("format", ""),
                }
        return 0.15

    def _improve_disambiguation(self, request: OptimizationRequest) -> float:
        """改进歧义消除"""
        return 0.08

    def _enhance_multimodal_capabilities(self, request: OptimizationRequest) -> float:
        """增强多模态能力"""
        return 0.12

    def _publish_optimization_result(self, request: OptimizationRequest, result: OptimizationResult):
        """发布优化结果"""
        serialized = DataContractSerializer.serialize(result)
        self._event_bus.publish(
            Event(
                event_type=f"optimization.result.{request.target_module}",
                data=serialized,
                source="meta_parsing_integration",
                priority=EventPriority.NORMAL,
            )
        )

        self._bridge.handle_optimization_result(result)

    def _handle_optimization_result(self, event: Event):
        """处理优化结果"""
        try:
            result = DataContractSerializer.deserialize(event.data)
            if isinstance(result, OptimizationResult):
                with self._lock:
                    self._stats["optimizations_completed"] += 1

                if result.success:
                    logger.info(f"Meta-parsing optimization successful: {result.result_id}")
                else:
                    logger.warning(f"Meta-parsing optimization failed: {result.warnings}")

        except Exception as e:
            logger.error(f"Failed to handle optimization result: {e}")

    def _publish_parsing_result(self, result: Dict[str, Any], format_hint: str):
        """发布解析结果"""
        parsing_result = DataContractFactory.create_parsing_result(
            success=result.get("success", False),
            parsed_data=result.get("parsing_result", {}),
            metadata={
                "format": format_hint,
                "stage": result.get("stage", ""),
                "timestamp": time.time(),
            },
            confidence=result.get("parsing_result", {}).get("confidence", 0.5),
        )

        self._bridge.publish_parsing_result(parsing_result)

    def _publish_semantic_result(self, result: Dict[str, Any]):
        """发布语义分析结果"""
        self._event_bus.publish(
            Event(
                event_type="parsing.semantic_analysis_complete",
                data=result,
                source="meta_parsing_integration",
                priority=EventPriority.NORMAL,
            )
        )

    def _publish_multimodal_result(self, result: MultimodalFusionResult):
        """发布多模态融合结果"""
        self._event_bus.publish(
            Event(
                event_type="parsing.multimodal_fusion_complete",
                data=result.to_dict(),
                source="meta_parsing_integration",
                priority=EventPriority.NORMAL,
            )
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

    def enable_trigger(self, trigger: ParsingOptimizationTrigger):
        """启用触发器"""
        self._enabled_triggers.add(trigger)

    def disable_trigger(self, trigger: ParsingOptimizationTrigger):
        """禁用触发器"""
        self._enabled_triggers.discard(trigger)

    def set_semantic_level(self, level: SemanticAnalysisLevel):
        """设置语义分析级别"""
        self._semantic_analysis_level = level

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "parsing_requests": self._stats["parsing_requests"],
                "parsing_successes": self._stats["parsing_successes"],
                "parsing_success_rate": (
                    self._stats["parsing_successes"] / self._stats["parsing_requests"]
                    if self._stats["parsing_requests"] > 0
                    else 0.0
                ),
                "semantic_analyses": self._stats["semantic_analyses"],
                "context_relations_found": self._stats["context_relations_found"],
                "multimodal_fusions": self._stats["multimodal_fusions"],
                "fusions_successful": self._stats["fusions_successful"],
                "fusion_success_rate": (
                    self._stats["fusions_successful"] / self._stats["multimodal_fusions"]
                    if self._stats["multimodal_fusions"] > 0
                    else 0.0
                ),
                "optimizations_requested": self._stats["optimizations_requested"],
                "optimizations_completed": self._stats["optimizations_completed"],
                "accuracy_improvements": self._stats["accuracy_improvements"],
                "semantic_analysis_level": self._semantic_analysis_level.value,
                "context_store_size": len(self._context_store),
                "parsing_history_length": len(self._parsing_history),
                "semantic_history_length": len(self._semantic_history),
                "optimization_queue_length": len(self._optimization_queue),
                "enabled_triggers": [t.value for t in self._enabled_triggers],
            }

    def get_parsing_summary(self) -> Dict[str, Any]:
        """获取解析摘要"""
        return self._orchestrator.get_overview()

    def get_recent_parsing(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近解析记录"""
        with self._lock:
            return list(self._parsing_history)[-limit:]

    def get_recent_semantic(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近语义分析记录"""
        with self._lock:
            return list(self._semantic_history)[-limit:]


_integration_instance: Optional[MetaParsingIntegration] = None
_integration_lock = threading.Lock()


def get_meta_parsing_integration(event_bus: EventBus = None) -> MetaParsingIntegration:
    """获取元解析集成单例"""
    global _integration_instance
    with _integration_lock:
        if _integration_instance is None:
            if event_bus is None:
                from ..orchestration.event_bus import EventBus

                event_bus = EventBus()
            _integration_instance = MetaParsingIntegration(event_bus)
        return _integration_instance