import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from enum import Enum


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMonitor:
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim

        self.status = ExecutionStatus.PENDING
        self.current_task_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.execution_history: deque = deque(maxlen=100)

        self.metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "success_rate": 0.0
        }

        self.checkpoints: List[Dict[str, Any]] = []
        self.progress_callbacks: List[Callable] = []
        self.failure_callbacks: List[Callable] = []
        self.success_callbacks: List[Callable] = []

        self._step_count = 0
        self._max_retries = 3
        self._retry_count = 0
        self._last_error: Optional[str] = None
        self._performance_samples: deque = deque(maxlen=50)

    def start_execution(self, task_id: str, task_description: str = "") -> bool:
        if self.status == ExecutionStatus.RUNNING:
            return False

        self.current_task_id = task_id
        self.status = ExecutionStatus.RUNNING
        self.start_time = time.time()
        self._step_count = 0
        self._retry_count = 0
        self._last_error = None
        self.checkpoints = []

        self.metrics["total_executions"] += 1

        self._emit_event("start", {"task_id": task_id, "description": task_description})

        return True

    def step_execution(self, step_data: Dict[str, Any] = None) -> Dict[str, Any]:
        if self.status != ExecutionStatus.RUNNING:
            return {"status": "not_running"}

        self._step_count += 1

        step_info = {
            "step": self._step_count,
            "timestamp": time.time(),
            "data": step_data or {}
        }

        self.checkpoints.append(step_info)
        self._emit_event("progress", step_info)

        return {
            "status": "running",
            "step": self._step_count,
            "elapsed": self.get_elapsed_time()
        }

    def complete_execution(self, result: Dict[str, Any] = None) -> Dict[str, Any]:
        if self.status != ExecutionStatus.RUNNING:
            return {"status": "not_running"}

        duration = self.get_elapsed_time()
        self.status = ExecutionStatus.COMPLETED

        self.metrics["successful_executions"] += 1
        self.metrics["total_duration"] += duration
        self._update_avg_metrics()

        self.execution_history.append({
            "task_id": self.current_task_id,
            "status": "completed",
            "duration": duration,
            "steps": self._step_count,
            "result": result or {}
        })

        self._performance_samples.append({"duration": duration, "success": True, "steps": self._step_count})

        self._emit_event("complete", {
            "task_id": self.current_task_id,
            "duration": duration,
            "result": result
        })

        return {
            "status": "completed",
            "task_id": self.current_task_id,
            "duration": duration,
            "steps": self._step_count
        }

    def fail_execution(self, error: str = "", recoverable: bool = False) -> Dict[str, Any]:
        if self.status != ExecutionStatus.RUNNING:
            return {"status": "not_running"}

        self._last_error = error
        self._retry_count += 1

        if recoverable and self._retry_count < self._max_retries:
            self._emit_event("retry", {
                "task_id": self.current_task_id,
                "error": error,
                "retry_count": self._retry_count
            })
            return {"status": "retrying", "retry_count": self._retry_count}

        duration = self.get_elapsed_time()
        self.status = ExecutionStatus.FAILED

        self.metrics["failed_executions"] += 1
        self.metrics["total_duration"] += duration
        self._update_avg_metrics()

        self.execution_history.append({
            "task_id": self.current_task_id,
            "status": "failed",
            "duration": duration,
            "steps": self._step_count,
            "error": error
        })

        self._performance_samples.append({"duration": duration, "success": False, "steps": self._step_count})

        self._emit_event("failure", {
            "task_id": self.current_task_id,
            "error": error,
            "duration": duration
        })

        return {
            "status": "failed",
            "task_id": self.current_task_id,
            "error": error,
            "retry_count": self._retry_count
        }

    def pause_execution(self) -> bool:
        if self.status != ExecutionStatus.RUNNING:
            return False
        self.status = ExecutionStatus.PAUSED
        self._emit_event("pause", {"task_id": self.current_task_id})
        return True

    def resume_execution(self) -> bool:
        if self.status != ExecutionStatus.PAUSED:
            return False
        self.status = ExecutionStatus.RUNNING
        self._emit_event("resume", {"task_id": self.current_task_id})
        return True

    def cancel_execution(self) -> bool:
        if self.status in (ExecutionStatus.RUNNING, ExecutionStatus.PAUSED):
            self.status = ExecutionStatus.CANCELLED
            self._emit_event("cancel", {"task_id": self.current_task_id})
            return True
        return False

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.status in (ExecutionStatus.RUNNING, ExecutionStatus.PAUSED):
            return time.time() - self.start_time
        return 0.0

    def get_progress(self) -> float:
        if not self.checkpoints:
            return 0.0
        return min(1.0, self._step_count / max(self._step_count + 10, 1))

    def _update_avg_metrics(self):
        total = self.metrics["successful_executions"] + self.metrics["failed_executions"]
        if total > 0:
            self.metrics["success_rate"] = self.metrics["successful_executions"] / total
            self.metrics["avg_duration"] = self.metrics["total_duration"] / total

    def register_callback(self, event_type: str, callback: Callable):
        if event_type == "progress":
            self.progress_callbacks.append(callback)
        elif event_type == "success":
            self.success_callbacks.append(callback)
        elif event_type == "failure":
            self.failure_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        callbacks = []
        if event_type == "progress":
            callbacks = self.progress_callbacks
        elif event_type == "complete":
            callbacks = self.success_callbacks
        elif event_type == "failure":
            callbacks = self.failure_callbacks

        for cb in callbacks:
            try:
                cb(data)
            except Exception:
                pass

    def get_execution_stats(self) -> Dict[str, Any]:
        recent_durations = [s["duration"] for s in self._performance_samples]
        recent_successes = [s["success"] for s in self._performance_samples]

        stats = dict(self.metrics)
        stats.update({
            "current_status": self.status.value,
            "current_task_id": self.current_task_id,
            "current_step": self._step_count,
            "elapsed_time": self.get_elapsed_time(),
            "last_error": self._last_error,
            "retry_count": self._retry_count,
            "recent_avg_duration": np.mean(recent_durations) if recent_durations else 0.0,
            "recent_success_rate": np.mean(recent_successes) if recent_successes else 0.0,
            "checkpoint_count": len(self.checkpoints)
        })

        return stats

    def resize(self, new_feature_dim: int):
        self.feature_dim = new_feature_dim
