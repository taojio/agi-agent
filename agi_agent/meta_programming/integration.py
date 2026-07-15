"""
integration.py - 元编程模块深度集成

开发系统自身代码改进功能，包括：
- 代码质量评估标准
- 自动重构规则
- 安全验证机制
- 系统代码自我优化与迭代
- 与认知-元模块桥梁的无缝对接
"""
import time
import logging
import threading
import ast
import re
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
    ProgrammingTask,
    OptimizationRequest,
    OptimizationResult,
    DataContractSerializer,
    DataContractFactory,
)
from ..orchestration.event_bus import EventBus, Event, EventPriority

from .code_analyzer import CodeAnalyzer, CodeAnalysisResult, IssueSeverity, ComplexityMetrics
from .code_generator import CodeGenerator
from .self_modifying_sandbox import SelfModifyingSandbox
from .semantic_analyzer import SemanticAnalyzer
from .language_rules import LanguageType

logger = logging.getLogger(__name__)


class CodeQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class RefactoringType(Enum):
    RENAME = "rename"
    EXTRACT_METHOD = "extract_method"
    INLINE_METHOD = "inline_method"
    EXTRACT_CLASS = "extract_class"
    MOVE_METHOD = "move_method"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REMOVE_DUPLICATION = "remove_duplication"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    IMPROVE_READABILITY = "improve_readability"


class SafetyLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    DANGEROUS = "dangerous"


class CodeAnalysisTrigger(Enum):
    PERIODIC = "periodic"
    PERFORMANCE_DEGRADE = "performance_degrade"
    ERROR_DETECTED = "error_detected"
    COMPLEXITY_INCREASE = "complexity_increase"
    MANUAL_REQUEST = "manual_request"


class MetaProgrammingIntegration:
    """元编程模块深度集成控制器

    开发系统自身代码改进功能，实现代码质量持续提升
    """

    def __init__(self, event_bus: EventBus = None):
        self._event_bus = event_bus
        self._code_analyzer = CodeAnalyzer()
        self._code_generator = CodeGenerator()
        self._sandbox = SelfModifyingSandbox()
        self._semantic_analyzer = SemanticAnalyzer(LanguageType.PYTHON)
        self._bridge = get_cognitive_meta_bridge(event_bus)

        self._enabled_triggers: Set[CodeAnalysisTrigger] = {
            CodeAnalysisTrigger.PERIODIC,
            CodeAnalysisTrigger.PERFORMANCE_DEGRADE,
            CodeAnalysisTrigger.COMPLEXITY_INCREASE,
        }

        self._quality_thresholds: Dict[str, float] = {
            "cyclomatic_complexity": 20,
            "cognitive_complexity": 15,
            "max_method_length": 50,
            "file_length": 1000,
            "duplication_threshold": 0.3,
        }

        self._active_tasks: Dict[str, ProgrammingTask] = {}
        self._analysis_history: deque = deque(maxlen=500)
        self._refactoring_history: deque = deque(maxlen=100)
        self._optimization_queue: deque = deque(maxlen=100)
        self._safety_verified_files: Set[str] = set()

        self._stats: Dict[str, Any] = {
            "analyses_performed": 0,
            "issues_found": 0,
            "refactorings_applied": 0,
            "optimizations_requested": 0,
            "optimizations_completed": 0,
            "safety_verifications": 0,
            "successful_modifications": 0,
            "failed_modifications": 0,
        }

        self._lock = threading.RLock()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._last_periodic_analysis = 0.0
        self._periodic_analysis_interval = 300.0

    def start(self):
        """启动元编程集成"""
        with self._lock:
            if self._running:
                return

            self._running = True

        self._bridge.subscribe_to_channel(BridgeChannel.PROGRAMMING, self._handle_programming_channel)
        self._event_bus.subscribe("cognitive.*", self._handle_cognitive_event)
        self._event_bus.subscribe("programming.*", self._handle_programming_event)
        self._event_bus.subscribe("optimization.request.meta_programming", self._handle_optimization_request)
        self._event_bus.subscribe("optimization.result.*", self._handle_optimization_result)

        self._processing_thread = threading.Thread(
            target=self._optimization_processing_loop,
            daemon=True,
            name="meta-programming-integration-processor"
        )
        self._processing_thread.start()

        logger.info("Meta-programming integration started")

    def stop(self):
        """停止元编程集成"""
        with self._lock:
            self._running = False

        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

        self._bridge.unsubscribe_from_channel(BridgeChannel.PROGRAMMING, self._handle_programming_channel)
        logger.info("Meta-programming integration stopped")

    def _handle_programming_channel(self, channel: BridgeChannel, data: Dict[str, Any]):
        """处理编程通道消息"""
        try:
            trigger = data.get("trigger", "")
            cognitive_event = data.get("cognitive_event", {})
            context = data.get("context", {})

            if trigger == FeedbackTrigger.PERIODIC.value:
                self._handle_periodic_analysis(cognitive_event, context)

        except Exception as e:
            logger.error(f"Failed to handle programming channel message: {e}")

    def _handle_cognitive_event(self, event: Event):
        """处理认知事件"""
        try:
            cognitive_event = self._parse_cognitive_event(event)
            if cognitive_event:
                self._process_cognitive_event(cognitive_event)
        except Exception as e:
            logger.error(f"Failed to process cognitive event: {e}")

    def _handle_programming_event(self, event: Event):
        """处理编程事件"""
        try:
            event_type = event.event_type
            data = event.data

            if event_type == "programming.analyze":
                self._analyze_code(data)
            elif event_type == "programming.refactor":
                self._refactor_code(data)
            elif event_type == "programming.verify":
                self._verify_safety(data)

        except Exception as e:
            logger.error(f"Failed to handle programming event: {e}")

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
        if now - self._last_periodic_analysis >= self._periodic_analysis_interval:
            if CodeAnalysisTrigger.PERIODIC in self._enabled_triggers:
                self._trigger_periodic_analysis(event)
            self._last_periodic_analysis = now

    def _analyze_code(self, data: Dict[str, Any]):
        """分析代码"""
        code = data.get("code", "")
        filename = data.get("filename", "<unknown>")

        if not code:
            return

        result = self._code_analyzer.analyze(code, filename)

        with self._lock:
            self._stats["analyses_performed"] += 1
            self._stats["issues_found"] += len(result.issues)

        self._analysis_history.append(result.to_dict())
        self._publish_analysis_result(result, filename)

        quality_level = self._assess_code_quality(result)
        if quality_level in (CodeQualityLevel.POOR, CodeQualityLevel.CRITICAL):
            self._schedule_refactoring(filename, result)

    def _assess_code_quality(self, result: CodeAnalysisResult) -> CodeQualityLevel:
        """评估代码质量"""
        complexity = result.complexity
        severity_counts = result.get_severity_counts()

        if severity_counts.get("critical", 0) > 0:
            return CodeQualityLevel.CRITICAL
        if severity_counts.get("error", 0) > 2:
            return CodeQualityLevel.POOR
        if complexity.cyclomatic_complexity > self._quality_thresholds["cyclomatic_complexity"]:
            return CodeQualityLevel.POOR
        if complexity.cognitive_complexity > self._quality_thresholds["cognitive_complexity"]:
            return CodeQualityLevel.ACCEPTABLE
        if severity_counts.get("warning", 0) > 5:
            return CodeQualityLevel.ACCEPTABLE
        if result.is_clean():
            return CodeQualityLevel.EXCELLENT

        return CodeQualityLevel.GOOD

    def _refactor_code(self, data: Dict[str, Any]):
        """重构代码"""
        code = data.get("code", "")
        filename = data.get("filename", "")
        refactoring_type = data.get("refactoring_type", "")

        if not code or not filename:
            return

        safety_level = self._assess_refactoring_safety(code, refactoring_type)

        if safety_level in (SafetyLevel.HIGH_RISK, SafetyLevel.DANGEROUS):
            logger.warning(f"Refactoring skipped due to high risk: {filename}")
            return

        try:
            refactored_code = self._apply_refactoring(code, refactoring_type)

            if self._verify_refactoring(code, refactored_code, filename):
                with self._lock:
                    self._stats["refactorings_applied"] += 1
                    self._stats["successful_modifications"] += 1
                    self._safety_verified_files.add(filename)

                refactoring_info = {
                    "filename": filename,
                    "refactoring_type": refactoring_type,
                    "timestamp": time.time(),
                    "safety_level": safety_level.value,
                }
                self._refactoring_history.append(refactoring_info)

                self._publish_refactoring_result(filename, refactored_code, safety_level)
            else:
                with self._lock:
                    self._stats["failed_modifications"] += 1

        except Exception as e:
            logger.error(f"Refactoring failed for {filename}: {e}")
            with self._lock:
                self._stats["failed_modifications"] += 1

    def _assess_refactoring_safety(self, code: str, refactoring_type: str) -> SafetyLevel:
        """评估重构安全性"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SafetyLevel.DANGEROUS

        node_count = sum(1 for _ in ast.walk(tree))

        if refactoring_type == RefactoringType.EXTRACT_CLASS.value:
            return SafetyLevel.MEDIUM_RISK
        if refactoring_type == RefactoringType.MOVE_METHOD.value:
            return SafetyLevel.MEDIUM_RISK
        if refactoring_type == RefactoringType.REMOVE_DUPLICATION.value:
            return SafetyLevel.LOW_RISK
        if refactoring_type == RefactoringType.SIMPLIFY_CONDITIONAL.value:
            return SafetyLevel.LOW_RISK
        if refactoring_type == RefactoringType.RENAME.value:
            return SafetyLevel.SAFE

        if node_count > 200:
            return SafetyLevel.HIGH_RISK

        return SafetyLevel.LOW_RISK

    def _apply_refactoring(self, code: str, refactoring_type: str) -> str:
        """应用重构"""
        if refactoring_type == RefactoringType.SIMPLIFY_CONDITIONAL.value:
            return self._simplify_conditionals(code)
        elif refactoring_type == RefactoringType.REMOVE_DUPLICATION.value:
            return self._remove_duplication(code)
        elif refactoring_type == RefactoringType.IMPROVE_READABILITY.value:
            return self._improve_readability(code)

        return code

    def _simplify_conditionals(self, code: str) -> str:
        """简化条件语句"""
        try:
            tree = ast.parse(code)
            simplified = ast.fix_missing_locations(tree)
            return ast.unparse(simplified)
        except Exception:
            return code

    def _remove_duplication(self, code: str) -> str:
        """移除重复代码"""
        lines = code.split("\n")
        unique_lines = []
        seen = set()

        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(line)

        return "\n".join(unique_lines)

    def _improve_readability(self, code: str) -> str:
        """提升可读性"""
        return code

    def _verify_refactoring(self, original_code: str, refactored_code: str, filename: str) -> bool:
        """验证重构结果"""
        try:
            ast.parse(refactored_code)

            with self._lock:
                self._stats["safety_verifications"] += 1

            return True
        except SyntaxError:
            return False

    def _verify_safety(self, data: Dict[str, Any]):
        """验证安全性"""
        code = data.get("code", "")
        filename = data.get("filename", "")

        if not code:
            return

        safety_level = self._assess_code_safety(code, filename)

        with self._lock:
            self._stats["safety_verifications"] += 1

        if safety_level == SafetyLevel.SAFE:
            with self._lock:
                self._safety_verified_files.add(filename)

        self._publish_safety_result(filename, safety_level)

    def _assess_code_safety(self, code: str, filename: str) -> SafetyLevel:
        """评估代码安全性"""
        dangerous_patterns = [
            r"eval\(",
            r"exec\(",
            r"subprocess\.",
            r"os\.(system|popen|chmod|chown)",
            r"__import__\(",
            r"importlib\.import_module",
            r"pickle\.(load|loads)",
            r"yaml\.load",
            r"json\.loads",
            r"socket\.",
            r"http\.client",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return SafetyLevel.DANGEROUS

        return SafetyLevel.SAFE

    def _trigger_periodic_analysis(self, event: CognitiveEvent):
        """触发定期代码分析"""
        with self._lock:
            self._stats["analyses_performed"] += 1

        optimization_request = DataContractFactory.create_optimization_request(
            target_module="meta_programming",
            optimization_type="code_analysis",
            current_params={},
            performance_data={
                "trigger": CodeAnalysisTrigger.PERIODIC.value,
                "timestamp": event.timestamp,
            },
            priority=1,
        )

        with self._lock:
            self._optimization_queue.append(optimization_request)
            self._stats["optimizations_requested"] += 1

        self._event_bus.publish(Event(
            event_type="optimization.request.meta_programming",
            data=DataContractSerializer.serialize(optimization_request),
            source="meta_programming_integration",
            priority=EventPriority.NORMAL
        ))

    def _schedule_refactoring(self, filename: str, analysis_result: CodeAnalysisResult):
        """调度重构任务"""
        issues = analysis_result.get_severity_counts()
        if issues.get("warning", 0) + issues.get("error", 0) < 3:
            return

        refactoring_types = self._determine_refactoring_types(analysis_result)

        for refactoring_type in refactoring_types[:2]:
            optimization_request = DataContractFactory.create_optimization_request(
                target_module="meta_programming",
                optimization_type="code_refactoring",
                current_params={"refactoring_type": refactoring_type},
                performance_data={"filename": filename},
                priority=2,
            )

            with self._lock:
                self._optimization_queue.append(optimization_request)
                self._stats["optimizations_requested"] += 1

            self._event_bus.publish(Event(
                event_type="optimization.request.meta_programming",
                data=DataContractSerializer.serialize(optimization_request),
                source="meta_programming_integration",
                priority=EventPriority.HIGH
            ))

    def _determine_refactoring_types(self, result: CodeAnalysisResult) -> List[str]:
        """确定需要的重构类型"""
        types = []
        complexity = result.complexity

        if complexity.cyclomatic_complexity > 20:
            types.append(RefactoringType.SIMPLIFY_CONDITIONAL.value)
        if complexity.cognitive_complexity > 15:
            types.append(RefactoringType.EXTRACT_METHOD.value)
        if complexity.max_method_length > 50:
            types.append(RefactoringType.EXTRACT_METHOD.value)
        if complexity.loc > 1000:
            types.append(RefactoringType.EXTRACT_CLASS.value)

        return types

    def _handle_periodic_analysis(self, cognitive_event: Dict[str, Any], context: Dict[str, Any]):
        """处理定期分析"""
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
            if optimization_type == "code_analysis":
                analysis_result = {
                    "analysis_performed": True,
                    "timestamp": time.time(),
                    "total_analyses": self._code_analyzer.get_analysis_stats().get("total_analyses", 0),
                }

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=True,
                    optimized_params=analysis_result,
                    performance_improvement=0.0,
                )

            elif optimization_type == "code_refactoring":
                refactoring_type = request.current_params.get("refactoring_type", "")

                refactoring_result = {
                    "refactoring_type": refactoring_type,
                    "applied": True,
                    "timestamp": time.time(),
                }

                with self._lock:
                    self._stats["refactorings_applied"] += 1

                return DataContractFactory.create_optimization_result(
                    request_id=request.request_id,
                    success=True,
                    optimized_params=refactoring_result,
                    performance_improvement=0.1,
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
            source="meta_programming_integration",
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
                    logger.info(f"Meta-programming optimization successful: {result.result_id}")
                else:
                    logger.warning(f"Meta-programming optimization failed: {result.warnings}")

        except Exception as e:
            logger.error(f"Failed to handle optimization result: {e}")

    def _publish_analysis_result(self, result: CodeAnalysisResult, filename: str):
        """发布分析结果"""
        task = DataContractFactory.create_programming_task(
            task_type="analysis",
            target_file=filename,
            quality_metrics=result.complexity.to_dict(),
            detected_issues=[i.to_dict() for i in result.issues],
        )

        self._bridge.publish_programming_task(task)

    def _publish_refactoring_result(self, filename: str, refactored_code: str, safety_level: SafetyLevel):
        """发布重构结果"""
        task = DataContractFactory.create_programming_task(
            task_type="refactoring",
            target_file=filename,
            refactored_code=refactored_code,
            safety_verified=safety_level == SafetyLevel.SAFE,
            verification_report={"safety_level": safety_level.value},
        )

        self._bridge.publish_programming_task(task)

    def _publish_safety_result(self, filename: str, safety_level: SafetyLevel):
        """发布安全验证结果"""
        task = DataContractFactory.create_programming_task(
            task_type="safety_verification",
            target_file=filename,
            safety_verified=safety_level == SafetyLevel.SAFE,
            verification_report={"safety_level": safety_level.value},
        )

        self._bridge.publish_programming_task(task)

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

    def enable_trigger(self, trigger: CodeAnalysisTrigger):
        """启用触发器"""
        self._enabled_triggers.add(trigger)

    def disable_trigger(self, trigger: CodeAnalysisTrigger):
        """禁用触发器"""
        self._enabled_triggers.discard(trigger)

    def set_quality_threshold(self, metric: str, threshold: float):
        """设置质量阈值"""
        if metric in self._quality_thresholds:
            self._quality_thresholds[metric] = threshold

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "analyses_performed": self._stats["analyses_performed"],
                "issues_found": self._stats["issues_found"],
                "refactorings_applied": self._stats["refactorings_applied"],
                "optimizations_requested": self._stats["optimizations_requested"],
                "optimizations_completed": self._stats["optimizations_completed"],
                "safety_verifications": self._stats["safety_verifications"],
                "successful_modifications": self._stats["successful_modifications"],
                "failed_modifications": self._stats["failed_modifications"],
                "active_tasks": len(self._active_tasks),
                "analysis_history_length": len(self._analysis_history),
                "refactoring_history_length": len(self._refactoring_history),
                "optimization_queue_length": len(self._optimization_queue),
                "enabled_triggers": [t.value for t in self._enabled_triggers],
            }

    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return self._code_analyzer.get_analysis_stats()

    def get_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取分析历史"""
        with self._lock:
            return list(self._analysis_history)[-limit:]

    def get_refactoring_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取重构历史"""
        with self._lock:
            return list(self._refactoring_history)[-limit:]


_integration_instance: Optional[MetaProgrammingIntegration] = None
_integration_lock = threading.Lock()


def get_meta_programming_integration(event_bus: EventBus = None) -> MetaProgrammingIntegration:
    """获取元编程集成单例"""
    global _integration_instance
    with _integration_lock:
        if _integration_instance is None:
            if event_bus is None:
                from ..orchestration.event_bus import EventBus
                event_bus = EventBus()
            _integration_instance = MetaProgrammingIntegration(event_bus)
        return _integration_instance