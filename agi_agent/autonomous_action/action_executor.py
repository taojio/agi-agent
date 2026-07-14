import numpy as np
import time
from collections import deque
from enum import Enum
from .target_decomposer import TaskNode, DecompositionLevel


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ErrorLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CorrectionResult:
    def __init__(self, success, correction_level, error_message="", retry_count=0):
        self.success = success
        self.correction_level = correction_level
        self.error_message = error_message
        self.retry_count = retry_count

    def to_dict(self):
        return {
            "success": self.success,
            "correction_level": self.correction_level,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }


class ActionExecutor:
    def __init__(self, max_retries=3):
        self.execution_history = deque(maxlen=200)
        self.active_executions = {}
        self.max_retries = max_retries
        self.error_handler = self._default_error_handler

    def _default_error_handler(self, action_node, error):
        return {
            "correction_level": 1,
            "suggested_action": "retry",
            "params_modification": {}
        }

    def set_error_handler(self, handler):
        self.error_handler = handler

    def _validate_acceptance_criteria(self, node, result):
        criteria = node.acceptance_criteria
        if not criteria:
            return True, "No acceptance criteria defined"

        if "success" in criteria and not result.get("success", True):
            return False, f"Success check failed: expected {criteria['success']}, got {result.get('success')}"

        if "output_type" in criteria:
            expected_type = criteria["output_type"]
            actual_type = type(result.get("output")).__name__ if result.get("output") else "None"
            if expected_type != "any" and actual_type != expected_type:
                return False, f"Output type mismatch: expected {expected_type}, got {actual_type}"

        if "threshold" in criteria:
            if result.get("value", 0) < criteria["threshold"]:
                return False, f"Value below threshold: {result.get('value')} < {criteria['threshold']}"

        return True, "All criteria met"

    def _execute_action(self, action_node, context):
        action_name = action_node.name

        mock_actions = {
            "execute_scan": lambda: {"success": True, "output": ["file1.txt", "file2.log"], "value": 2},
            "collect_results": lambda: {"success": True, "output": {"count": 2}, "value": 2},
            "validate_input": lambda: {"success": True, "output": "validated", "value": 1},
            "execute_process": lambda: {"success": True, "output": "processed", "value": 1},
            "output_result": lambda: {"success": True, "output": "result", "value": 1},
            "check_criteria": lambda: {"success": True, "output": "criteria met", "value": 1},
            "generate_report": lambda: {"success": True, "output": {"status": "ok"}, "value": 1},
            "execute_action": lambda: {"success": True, "output": "executed", "value": 1},
            "check_result": lambda: {"success": True, "output": "checked", "value": 1}
        }

        if action_name in mock_actions:
            return mock_actions[action_name]()
        else:
            return {"success": True, "output": f"Action '{action_name}' executed", "value": 1}

    def _correct_action(self, action_node, error, retry_count):
        if retry_count >= self.max_retries:
            return CorrectionResult(
                success=False,
                correction_level=3,
                error_message=f"Max retries ({self.max_retries}) exceeded",
                retry_count=retry_count
            )

        handler_result = self.error_handler(action_node, error)
        correction_level = handler_result.get("correction_level", 1)

        if correction_level == 1:
            return CorrectionResult(
                success=True,
                correction_level=1,
                error_message="Retry with parameter adjustment",
                retry_count=retry_count
            )
        elif correction_level == 2:
            return CorrectionResult(
                success=True,
                correction_level=2,
                error_message="Switched to alternative path",
                retry_count=retry_count
            )
        elif correction_level == 3:
            return CorrectionResult(
                success=False,
                correction_level=3,
                error_message="Degrading to fallback",
                retry_count=retry_count
            )

        return CorrectionResult(
            success=False,
            correction_level=3,
            error_message="Unknown correction level",
            retry_count=retry_count
        )

    def execute_node(self, node, context=None):
        context = context or {}
        start_time = time.time()

        node.status = ExecutionStatus.RUNNING.value
        execution_id = f"exec_{np.random.randint(1000000)}"

        self.active_executions[execution_id] = {
            "execution_id": execution_id,
            "node_id": node.node_id,
            "status": ExecutionStatus.RUNNING.value,
            "start_time": start_time
        }

        try:
            if node.level == DecompositionLevel.ACTION:
                result = self._execute_action(node, context)

                is_valid, validation_message = self._validate_acceptance_criteria(node, result)

                if not is_valid:
                    raise ValueError(f"Acceptance criteria failed: {validation_message}")

                node.result = result
                node.status = ExecutionStatus.COMPLETED.value

            elif node.level == DecompositionLevel.STEP:
                for child in node.children:
                    self.execute_node(child, context)
                    if child.status == ExecutionStatus.FAILED.value:
                        node.status = ExecutionStatus.FAILED.value
                        node.error = f"Child {child.node_id} failed"
                        break

                if node.status == ExecutionStatus.RUNNING.value:
                    node.status = ExecutionStatus.COMPLETED.value
                    node.result = {"success": True, "child_results": [c.result for c in node.children]}

            elif node.level == DecompositionLevel.TASK:
                for child in node.children:
                    self.execute_node(child, context)
                    if child.status == ExecutionStatus.FAILED.value:
                        node.status = ExecutionStatus.FAILED.value
                        node.error = f"Child {child.node_id} failed"
                        break

                if node.status == ExecutionStatus.RUNNING.value:
                    node.status = ExecutionStatus.COMPLETED.value
                    node.result = {"success": True, "child_results": [c.result for c in node.children]}

            elif node.level == DecompositionLevel.GOAL:
                for child in node.children:
                    self.execute_node(child, context)

                node.status = ExecutionStatus.COMPLETED.value
                node.result = {"success": True, "child_results": [c.result for c in node.children]}

            execution_time = (time.time() - start_time) * 1000
            node.execution_time = execution_time

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            node.execution_time = execution_time
            node.status = ExecutionStatus.FAILED.value
            node.error = str(e)

            if node.level == DecompositionLevel.ACTION:
                retry_count = 0
                while retry_count < self.max_retries:
                    correction = self._correct_action(node, e, retry_count)
                    if correction.success:
                        try:
                            result = self._execute_action(node, context)
                            is_valid, _ = self._validate_acceptance_criteria(node, result)
                            if is_valid:
                                node.result = result
                                node.status = ExecutionStatus.COMPLETED.value
                                node.error = None
                                break
                        except Exception:
                            pass
                    retry_count += 1

        self.active_executions.pop(execution_id, None)

        self.execution_history.append({
            "execution_id": execution_id,
            "node_id": node.node_id,
            "node_name": node.name,
            "node_level": node.level.value,
            "status": node.status,
            "execution_time": node.execution_time,
            "error": node.error,
            "timestamp": np.random.randint(1000000)
        })

        return {
            "node_id": node.node_id,
            "status": node.status,
            "result": node.result,
            "error": node.error,
            "execution_time": node.execution_time
        }

    def pause_execution(self, execution_id):
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["status"] = ExecutionStatus.PAUSED.value
            return True
        return False

    def resume_execution(self, execution_id):
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["status"] = ExecutionStatus.RUNNING.value
            return True
        return False

    def cancel_execution(self, execution_id):
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["status"] = ExecutionStatus.CANCELLED.value
            return True
        return False

    def get_active_executions(self):
        return list(self.active_executions.values())

    def get_execution_history(self, limit=20):
        return list(self.execution_history)[-limit:]

    def get_execution_stats(self):
        if not self.execution_history:
            return {"total_executions": 0, "success_rate": 0.0, "avg_time": 0.0}

        total = len(self.execution_history)
        success_count = sum(1 for h in self.execution_history if h["status"] == ExecutionStatus.COMPLETED.value)
        avg_time = np.mean([h["execution_time"] for h in self.execution_history])

        return {
            "total_executions": total,
            "success_rate": success_count / total,
            "avg_time": avg_time,
            "active_count": len(self.active_executions)
        }