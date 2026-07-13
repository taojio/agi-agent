import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .code_generator import CodeGenerator, CodeGenerationResult, GenerationContext, GenerationMode
from .code_analyzer import CodeAnalyzer, CodeAnalysisResult, IssueSeverity
from .dynamic_executor import DynamicExecutor, ExecutionContext, ExecutionResult, SandboxConfig


class MetaProgrammingTaskType(Enum):
    GENERATE = "generate"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    GENERATE_AND_EXECUTE = "generate_and_execute"
    OPTIMIZE = "optimize"
    REFACTOR = "refactor"


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
        self.task_history: List[MetaProgrammingTask] = []
        self._active_tasks: Dict[str, MetaProgrammingTask] = {}

    def execute_task(self, task: MetaProgrammingTask) -> Dict[str, Any]:
        self._active_tasks[task.task_id] = task
        task.status = "running"
        
        try:
            if task.task_type == MetaProgrammingTaskType.GENERATE:
                result = self._execute_generate(task)
            elif task.task_type == MetaProgrammingTaskType.ANALYZE:
                result = self._execute_analyze(task)
            elif task.task_type == MetaProgrammingTaskType.EXECUTE:
                result = self._execute_execute(task)
            elif task.task_type == MetaProgrammingTaskType.GENERATE_AND_EXECUTE:
                result = self._execute_generate_and_execute(task)
            elif task.task_type == MetaProgrammingTaskType.OPTIMIZE:
                result = self._execute_optimize(task)
            elif task.task_type == MetaProgrammingTaskType.REFACTOR:
                result = self._execute_refactor(task)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            
        except Exception as e:
            task.result = {"error": str(e)}
            task.status = "failed"
        
        task.completed_at = time.time()
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
            "active_tasks": len(self._active_tasks)
        }