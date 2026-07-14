import ast
import io
import os
import sys
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


class SandboxConfig:
    def __init__(self):
        self.allowed_modules: List[str] = []
        self.blocked_modules: List[str] = ["os", "subprocess", "sys", "shutil", "ctypes"]
        self.max_execution_time: float = 5.0
        self.max_memory_mb: int = 100
        self.max_output_chars: int = 10000
        self.allow_network: bool = False
        self.allow_file_system: bool = False
        self.restrict_globals: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_modules": self.allowed_modules,
            "blocked_modules": self.blocked_modules,
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "max_output_chars": self.max_output_chars,
            "allow_network": self.allow_network,
            "allow_file_system": self.allow_file_system,
            "restrict_globals": self.restrict_globals
        }


class ExecutionContext:
    def __init__(self):
        self.globals: Dict[str, Any] = {}
        self.locals: Dict[str, Any] = {}
        self.sandbox_config: SandboxConfig = SandboxConfig()
        self.input_data: Dict[str, Any] = {}
        self.output_data: Dict[str, Any] = {}

    def set_global(self, name: str, value: Any):
        self.globals[name] = value

    def set_local(self, name: str, value: Any):
        self.locals[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "globals_keys": list(self.globals.keys()),
            "locals_keys": list(self.locals.keys()),
            "sandbox_config": self.sandbox_config.to_dict(),
            "input_data": self.input_data,
            "output_data": self.output_data
        }


class ExecutionResult:
    def __init__(self):
        self.status: ExecutionStatus = ExecutionStatus.PENDING
        self.return_value: Any = None
        self.output: str = ""
        self.error: str = ""
        self.traceback: str = ""
        self.execution_time_ms: float = 0.0
        self.memory_used_mb: float = 0.0
        self.context: Optional[ExecutionContext] = None

    def is_success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "output": self.output,
            "error": self.error,
            "traceback": self.traceback,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb
        }


class DynamicExecutor:
    def __init__(self):
        self.execution_history: List[ExecutionResult] = []
        self._active_executions: Dict[str, ExecutionResult] = {}
        self._execution_counter = 0

    def execute(self, code: str, context: ExecutionContext = None) -> ExecutionResult:
        if context is None:
            context = ExecutionContext()
        
        result = ExecutionResult()
        result.context = context
        result.status = ExecutionStatus.RUNNING
        
        execution_id = f"exec_{self._execution_counter}"
        self._execution_counter += 1
        self._active_executions[execution_id] = result
        
        start_time = time.time()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        output_buffer = io.StringIO()
        
        try:
            sys.stdout = output_buffer
            sys.stderr = output_buffer
            
            if context.sandbox_config.restrict_globals:
                safe_globals = self._create_safe_globals(context)
            else:
                safe_globals = context.globals.copy()
            
            safe_globals.update({
                "__name__": "__main__",
                "__doc__": "",
                "__package__": None,
                "__loader__": None,
                "__spec__": None,
            })
            
            tree = ast.parse(code)
            tree = self._sanitize_ast(tree, context.sandbox_config)
            
            compiled = compile(tree, "<dynamic_code>", "exec")
            
            exec(compiled, safe_globals, context.locals)
            
            result.status = ExecutionStatus.COMPLETED
            result.return_value = context.locals.get("_return_value")
            
        except SyntaxError as e:
            result.status = ExecutionStatus.FAILED
            result.error = f"Syntax error: {e.msg}"
            result.traceback = traceback.format_exc()
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.output = output_buffer.getvalue()[:context.sandbox_config.max_output_chars]
            
            self._active_executions.pop(execution_id, None)
            self.execution_history.append(result)
        
        return result

    def execute_function(self, code: str, args: Tuple[Any, ...] = (), 
                        kwargs: Dict[str, Any] = None) -> ExecutionResult:
        if kwargs is None:
            kwargs = {}
        
        context = ExecutionContext()
        
        call_code = f"""
def _dynamic_func{self._get_arg_string(args, kwargs)}:
{code}

_return_value = _dynamic_func{self._get_call_string(args, kwargs)}
"""
        
        return self.execute(call_code, context)

    def _create_safe_globals(self, context: ExecutionContext) -> Dict[str, Any]:
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'float': float,
            'int': int, 'len': len, 'max': max, 'min': min, 'pow': pow,
            'range': range, 'round': round, 'sum': sum, 'str': str, 'list': list,
            'dict': dict, 'tuple': tuple, 'set': set, 'type': type, 'isinstance': isinstance,
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'reversed': reversed, 'sorted': sorted, 'callable': callable, 'hash': hash,
        }
        
        return {
            "__builtins__": safe_builtins,
            **context.globals
        }

    def _sanitize_ast(self, tree: ast.AST, config: SandboxConfig) -> ast.AST:
        sanitized = ast.fix_missing_locations(ast.Module(body=[]))
        
        for node in tree.body:
            if self._is_safe_node(node, config):
                sanitized.body.append(node)
        
        return sanitized

    def _is_safe_node(self, node: ast.stmt, config: SandboxConfig) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.split('.')[0] in config.blocked_modules:
                        return False
            
            elif isinstance(child, ast.ImportFrom):
                if child.module and child.module.split('.')[0] in config.blocked_modules:
                    return False
            
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id in config.blocked_modules:
                    return False
            
            elif isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id in config.blocked_modules:
                    return False
        
        return True

    def _get_arg_string(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        arg_names = [f"arg_{i}" for i in range(len(args))]
        kwarg_names = list(kwargs.keys())
        
        all_args = arg_names + kwarg_names
        return f"({', '.join(all_args)})"

    def _get_call_string(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        arg_strings = []
        
        for i, arg in enumerate(args):
            arg_strings.append(f"arg_{i}={repr(arg)}")
        
        for name, value in kwargs.items():
            arg_strings.append(f"{name}={repr(value)}")
        
        return f"({', '.join(arg_strings)})"

    def execute_expression(self, expr: str, context: ExecutionContext = None) -> ExecutionResult:
        if context is None:
            context = ExecutionContext()
        
        code = f"_return_value = {expr}"
        return self.execute(code, context)

    def evaluate_code(self, code: str, context: ExecutionContext = None) -> Dict[str, Any]:
        result = self.execute(code, context)
        
        return {
            "success": result.is_success(),
            "return_value": result.return_value,
            "output": result.output,
            "error": result.error,
            "time_ms": result.execution_time_ms,
            "status": result.status.value
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history if r.is_success()])
        failed = len([r for r in self.execution_history if r.status == ExecutionStatus.FAILED])
        timed_out = len([r for r in self.execution_history if r.status == ExecutionStatus.TIMEOUT])
        
        avg_time = sum(r.execution_time_ms for r in self.execution_history) / total if total > 0 else 0.0
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "timed_out_executions": timed_out,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_execution_time_ms": avg_time,
            "active_executions": len(self._active_executions)
        }