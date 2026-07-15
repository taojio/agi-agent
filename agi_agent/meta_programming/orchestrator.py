import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .code_generator import CodeGenerator, CodeGenerationResult, GenerationContext, GenerationMode
from .code_analyzer import CodeAnalyzer, CodeAnalysisResult, IssueSeverity
from .dynamic_executor import DynamicExecutor, ExecutionContext, ExecutionResult, SandboxConfig
from .cross_language_analyzer import CrossLanguageCodeAnalyzer
from .self_modifying_sandbox import (
    SelfModifyingSandbox, SandboxMode, ModificationRequest,
    ModificationType, ApprovalStatus,
)
from .performance_detector import (
    PerformanceDashboard, BottleneckType, SeverityLevel,
)
from .test_case_generator import (
    AutomatedTestSuite, TestStrategy, TestResultStatus,
)


class MetaProgrammingTaskType(Enum):
    GENERATE = "generate"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    GENERATE_AND_EXECUTE = "generate_and_execute"
    OPTIMIZE = "optimize"
    REFACTOR = "refactor"
    CROSS_LANG_ANALYZE = "cross_lang_analyze"
    SELF_MODIFY = "self_modify"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    TEST_GENERATION = "test_generation"
    SELF_IMPROVE = "self_improve"


class MetaProgrammingTask:
    def __init__(self, task_type: MetaProgrammingTaskType, code: str = "", 
                 target: str = "", context: Dict[str, Any] = None):
        self.task_id = f"mp_task_{int(time.time() * 1000)}"
        self.task_type = task_type
        self.code = code
        self.target = target
        self.context = context or {}
        self.status = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.created_at = time.time()
        self.completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "code": self.code,
            "target": self.target,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }


class MetaProgrammingOrchestrator:
    def __init__(self):
        self.code_generator = CodeGenerator()
        self.code_analyzer = CodeAnalyzer()
        self.dynamic_executor = DynamicExecutor()
        self.cross_lang_analyzer = CrossLanguageCodeAnalyzer()
        self.sandbox = SelfModifyingSandbox(mode=SandboxMode.ANALYSIS_ONLY)
        self.performance_dashboard = PerformanceDashboard()
        self.test_suite = AutomatedTestSuite()
        self.task_history: List[MetaProgrammingTask] = []
        self._active_tasks: Dict[str, MetaProgrammingTask] = {}
        self._lock = threading.Lock()

    def execute_task(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        with self._lock:
            self._active_tasks[task.task_id] = task
        task.status = "running"
        
        try:
            task_map = {
                MetaProgrammingTaskType.GENERATE: self._execute_generate,
                MetaProgrammingTaskType.ANALYZE: self._execute_analyze,
                MetaProgrammingTaskType.EXECUTE: self._execute_execute,
                MetaProgrammingTaskType.GENERATE_AND_EXECUTE: self._execute_generate_and_execute,
                MetaProgrammingTaskType.OPTIMIZE: self._execute_optimize,
                MetaProgrammingTaskType.REFACTOR: self._execute_refactor,
                MetaProgrammingTaskType.CROSS_LANG_ANALYZE: self._execute_cross_lang_analyze,
                MetaProgrammingTaskType.SELF_MODIFY: self._execute_self_modify,
                MetaProgrammingTaskType.PERFORMANCE_ANALYSIS: self._execute_performance_analysis,
                MetaProgrammingTaskType.TEST_GENERATION: self._execute_test_generation,
                MetaProgrammingTaskType.SELF_IMPROVE: self._execute_self_improve,
            }
            
            handler = task_map.get(task.task_type)
            if handler:
                result = handler(task)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            
        except Exception as e:
            task.result = {"error": str(e)}
            task.status = "failed"
        
        task.completed_at = time.time()
        with self._lock:
            self._active_tasks.pop(task.task_id, None)
        self.task_history.append(task)
        
        return task.result

    def _execute_generate(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        context = GenerationContext()
        
        if "imports" in task.context:
            for imp in task.context["imports"]:
                context.add_import(imp)
        
        if "constraints" in task.context:
            context.constraints.update(task.context["constraints"])
        
        if "existing_code" in task.context:
            context.existing_code = task.context["existing_code"]
        
        mode = task.context.get("mode", GenerationMode.SYNTHESIS)
        
        result = self.code_generator.generate_code(context, mode=mode, target=task.target)
        
        return {
            "generated_code": result.code,
            "confidence": result.confidence,
            "quality": result.quality.value,
            "issues": result.issues,
            "estimated_complexity": result.estimated_complexity,
            "is_valid": result.is_valid()
        }

    def _execute_analyze(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        analysis = self.code_analyzer.analyze(task.code, filename=task.target)
        
        return {
            "structure": analysis.structure.to_dict(),
            "complexity": analysis.complexity.to_dict(),
            "dependencies": analysis.dependencies.to_dict(),
            "issues": [i.to_dict() for i in analysis.issues],
            "is_clean": analysis.is_clean(),
            "severity_counts": analysis.get_severity_counts()
        }

    def _execute_execute(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        context = ExecutionContext()
        
        if "globals" in task.context:
            for name, value in task.context["globals"].items():
                context.set_global(name, value)
        
        if "sandbox" in task.context:
            config = task.context["sandbox"]
            if isinstance(config, SandboxConfig):
                context.sandbox_config = config
            elif isinstance(config, dict):
                sandbox = SandboxConfig()
                sandbox.allowed_modules = config.get("allowed_modules", [])
                sandbox.blocked_modules = config.get("blocked_modules", [])
                sandbox.max_execution_time = config.get("max_execution_time", 5.0)
                context.sandbox_config = sandbox
        
        result = self.dynamic_executor.execute(task.code, context)
        
        return {
            "success": result.is_success(),
            "return_value": result.return_value,
            "output": result.output,
            "error": result.error,
            "traceback": result.traceback,
            "execution_time_ms": result.execution_time_ms,
            "status": result.status.value
        }

    def _execute_generate_and_execute(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        gen_result = self._execute_generate(task)
        
        if not gen_result.get("is_valid"):
            return {
                "status": "generation_failed",
                "generation": gen_result
            }
        
        exec_task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.EXECUTE,
            code=gen_result["generated_code"],
            context=task.context.get("execution_context", {})
        )
        
        exec_result = self._execute_execute(exec_task)
        
        return {
            "status": "completed",
            "generation": gen_result,
            "execution": exec_result
        }

    def _execute_optimize(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        initial_analysis = self.code_analyzer.analyze(task.code)
        
        context = GenerationContext()
        context.existing_code = task.code
        context.constraints["optimize"] = True
        
        result = self.code_generator.generate_code(
            context, 
            mode=GenerationMode.OPTIMIZATION,
            target="optimization"
        )
        
        optimized_analysis = self.code_analyzer.analyze(result.code)
        
        improvement = {
            "cyclomatic_complexity": {
                "before": initial_analysis.complexity.cyclomatic_complexity,
                "after": optimized_analysis.complexity.cyclomatic_complexity,
                "improvement": initial_analysis.complexity.cyclomatic_complexity - optimized_analysis.complexity.cyclomatic_complexity
            },
            "cognitive_complexity": {
                "before": initial_analysis.complexity.cognitive_complexity,
                "after": optimized_analysis.complexity.cognitive_complexity,
                "improvement": initial_analysis.complexity.cognitive_complexity - optimized_analysis.complexity.cognitive_complexity
            }
        }
        
        return {
            "optimized_code": result.code,
            "initial_analysis": initial_analysis.to_dict(),
            "optimized_analysis": optimized_analysis.to_dict(),
            "improvement": improvement,
            "confidence": result.confidence
        }

    def _execute_refactor(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        context = GenerationContext()
        context.existing_code = task.code
        context.constraints["add_docstrings"] = True
        context.constraints["simplify"] = True
        
        result = self.code_generator.generate_code(
            context,
            mode=GenerationMode.TRANSFORMATION,
            target="refactoring"
        )
        
        return {
            "refactored_code": result.code,
            "confidence": result.confidence,
            "is_valid": result.is_valid()
        }

    def _execute_cross_lang_analyze(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        analysis = self.cross_lang_analyzer.analyze(task.code, filename=task.target)
        
        return {
            "language": analysis.language.value,
            "complexity": {
                "cyclomatic_complexity": analysis.complexity.cyclomatic_complexity,
                "cognitive_complexity": analysis.complexity.cognitive_complexity,
                "nesting_depth": analysis.complexity.nesting_depth,
            },
            "quality": {
                "score": analysis.quality_score,
                "grade": analysis.quality_level.value,
                "maintainability": 0.0,
                "reliability": 0.0,
            },
            "issues": [
                {"type": "code_issue", "severity": issue.severity, "description": issue.message}
                for issue in analysis.issues
            ],
            "optimization_suggestions": [
                {"priority": suggestion.priority.value, "category": suggestion.category, "description": suggestion.description}
                for suggestion in analysis.optimization_suggestions
            ],
        }

    def _execute_self_modify(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        target_path = task.context.get("target_path", task.target)
        content = task.context.get("content", task.code)
        modification_type = task.context.get("modification_type", "modify")
        reason = task.context.get("reason", "self-improvement")
        proposed_by = task.context.get("proposed_by", "system")
        
        type_map = {
            "add": ModificationType.ADD,
            "modify": ModificationType.MODIFY,
            "delete": ModificationType.DELETE,
            "rename": ModificationType.RENAME,
        }
        
        mod_type = type_map.get(modification_type, ModificationType.MODIFY)
        
        request = ModificationRequest(
            request_id=f"mod_{int(time.time() * 1000)}",
            type=mod_type,
            target_path=target_path,
            content=content,
            reason=reason,
            proposed_by=proposed_by,
        )
        
        request = self.sandbox.submit_modification(request)
        
        if self.sandbox.mode == SandboxMode.FULL_MODIFICATION:
            request.approve()
            success = self.sandbox.execute_modification(request)
        else:
            success = False
        
        return {
            "request_id": request.request_id,
            "type": modification_type,
            "target_path": target_path,
            "status": request.status.value,
            "approved": request.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED),
            "executed": success,
            "rollback_id": request.rollback_id,
        }

    def _execute_performance_analysis(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        action = task.context.get("action", "analyze")
        
        if action == "start":
            self.performance_dashboard.start_monitoring()
            return {"status": "monitoring_started"}
        
        if action == "stop":
            self.performance_dashboard.stop_monitoring()
            return {"status": "monitoring_stopped"}
        
        analysis = self.performance_dashboard.run_analysis()
        
        return {
            "status": analysis["status"],
            "current_metrics": analysis.get("current_metrics", {}),
            "detections": analysis.get("detections", []),
            "suggestions": analysis.get("suggestions", []),
            "total_detections": analysis.get("total_detections", 0),
        }

    def _execute_test_generation(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        target_func = task.context.get("target_function")
        strategies = task.context.get("strategies", ["boundary_value", "random_test"])
        count_per_strategy = task.context.get("count_per_strategy", 5)
        enable_coverage = task.context.get("enable_coverage", False)
        
        strategy_enums = []
        for s in strategies:
            try:
                strategy_enums.append(TestStrategy[s.upper()])
            except KeyError:
                pass
        
        if not target_func:
            return {"error": "target_function is required"}
        
        test_suite = self.test_suite.create_and_execute(
            target_func,
            strategy_enums,
            count_per_strategy=count_per_strategy,
            enable_coverage=enable_coverage
        )
        
        summary = self.test_suite.get_test_summary(test_suite)
        report = self.test_suite.generate_report(test_suite, format_type="text")
        
        return {
            "suite_id": test_suite.suite_id,
            "test_count": len(test_suite.test_cases),
            "summary": summary,
            "report": report,
        }

    def _execute_self_improve(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        target_code = task.code if task.code else task.context.get("target_code", "")
        target_file = task.target or task.context.get("target_file", "")
        
        results = {
            "phase": "self_improvement",
            "steps": [],
        }

        step = {
            "name": "cross_lang_analysis",
            "status": "running"
        }
        try:
            analysis_result = self._execute_cross_lang_analyze(task)
            step["status"] = "completed"
            step["result"] = analysis_result
        except Exception as e:
            step["status"] = "failed"
            step["error"] = str(e)
        results["steps"].append(step)

        step = {
            "name": "performance_analysis",
            "status": "running"
        }
        try:
            perf_task = MetaProgrammingTask(
                task_type=MetaProgrammingTaskType.PERFORMANCE_ANALYSIS,
                context={"action": "analyze"}
            )
            perf_result = self._execute_performance_analysis(perf_task)
            step["status"] = "completed"
            step["result"] = perf_result
        except Exception as e:
            step["status"] = "failed"
            step["error"] = str(e)
        results["steps"].append(step)

        has_issues = any(
            step.get("status") == "completed" and 
            step.get("result", {}).get("issues")
            for step in results["steps"]
        )
        
        if has_issues:
            step = {
                "name": "optimization",
                "status": "running"
            }
            try:
                opt_task = MetaProgrammingTask(
                    task_type=MetaProgrammingTaskType.OPTIMIZE,
                    code=target_code,
                    target=target_file
                )
                opt_result = self._execute_optimize(opt_task)
                step["status"] = "completed"
                step["result"] = opt_result
            except Exception as e:
                step["status"] = "failed"
                step["error"] = str(e)
            results["steps"].append(step)

        results["overall_status"] = "completed" if all(
            step.get("status") != "failed" for step in results["steps"]
        ) else "partial"
        
        return results

    def generate_and_evaluate(self, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.GENERATE_AND_EXECUTE,
            target=target,
            context=context or {}
        )
        return self.execute_task(task)

    def analyze_and_optimize(self, code: str, target: str = "") -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.OPTIMIZE,
            code=code,
            target=target
        )
        return self.execute_task(task)

    def cross_lang_analyze(self, code: str, filename: str = "") -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.CROSS_LANG_ANALYZE,
            code=code,
            target=filename
        )
        return self.execute_task(task)

    def self_modify(self, target_path: str, content: str, modification_type: str = "modify",
                    reason: str = "self-improvement") -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.SELF_MODIFY,
            target=target_path,
            code=content,
            context={
                "target_path": target_path,
                "content": content,
                "modification_type": modification_type,
                "reason": reason,
            }
        )
        return self.execute_task(task)

    def run_performance_analysis(self) -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.PERFORMANCE_ANALYSIS,
            context={"action": "analyze"}
        )
        return self.execute_task(task)

    def generate_tests(self, target_func: Callable, strategies: List[str] = None,
                       count_per_strategy: int = 5, enable_coverage: bool = False) -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.TEST_GENERATION,
            context={
                "target_function": target_func,
                "strategies": strategies or ["boundary_value", "random_test"],
                "count_per_strategy": count_per_strategy,
                "enable_coverage": enable_coverage,
            }
        )
        return self.execute_task(task)

    def run_self_improvement(self, code: str = "", target_file: str = "") -> Dict[str, Any]:
        task = MetaProgrammingTask(
            task_type=MetaProgrammingTaskType.SELF_IMPROVE,
            code=code,
            target=target_file,
            context={
                "target_code": code,
                "target_file": target_file,
            }
        )
        return self.execute_task(task)

    def set_sandbox_mode(self, mode: str):
        mode_map = {
            "read_only": SandboxMode.READ_ONLY,
            "analysis": SandboxMode.ANALYSIS_ONLY,
            "simulation": SandboxMode.SIMULATION,
            "limited": SandboxMode.LIMITED_MODIFICATION,
            "full": SandboxMode.FULL_MODIFICATION,
        }
        self.sandbox.mode = mode_map.get(mode, SandboxMode.ANALYSIS_ONLY)

    def start_performance_monitoring(self):
        self.performance_dashboard.start_monitoring()

    def stop_performance_monitoring(self):
        self.performance_dashboard.stop_monitoring()

    def get_task_history(self) -> List[Dict[str, Any]]:
        return [task.to_dict() for task in self.task_history]

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        for task in self.task_history:
            if task.task_id == task_id:
                return task.to_dict()
        for task in self._active_tasks.values():
            if task.task_id == task_id:
                return task.to_dict()
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "code_generator": self.code_generator.get_generation_stats(),
            "code_analyzer": self.code_analyzer.get_analysis_stats(),
            "dynamic_executor": self.dynamic_executor.get_execution_stats(),
            "total_tasks": len(self.task_history),
            "active_tasks": len(self._active_tasks),
            "sandbox_mode": self.sandbox.mode.value,
            "performance_monitoring": True,
        }