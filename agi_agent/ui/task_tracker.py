"""
ui/task_tracker.py - 任务追踪器

跟踪任务进度、状态和执行结果
"""
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List
from collections import deque


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    COGNITIVE = "cognitive"
    PERCEPTION = "perception"
    EXECUTION = "execution"
    LEARNING = "learning"
    IMPROVEMENT = "improvement"
    SECURITY = "security"
    SYSTEM = "system"


class Task:
    def __init__(self, task_id: str, task_type: str, description: str, 
                 progress: float = 0.0, status: str = "pending",
                 metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.progress = progress
        self.status = status
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.subtasks: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []

    def start(self):
        self.status = TaskStatus.RUNNING.value
        self.started_at = time.time()

    def update_progress(self, progress: float):
        self.progress = min(1.0, max(0.0, progress))

    def complete(self, result: Optional[Dict[str, Any]] = None):
        self.status = TaskStatus.COMPLETED.value
        self.progress = 1.0
        self.completed_at = time.time()
        if result:
            self.metadata["result"] = result

    def fail(self, error: str):
        self.status = TaskStatus.FAILED.value
        self.completed_at = time.time()
        self.metadata["error"] = error

    def cancel(self):
        self.status = TaskStatus.CANCELLED.value
        self.completed_at = time.time()

    def add_log(self, message: str, level: str = "info"):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "level": level,
        })
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    def add_subtask(self, subtask_id: str, description: str):
        self.subtasks.append({
            "subtask_id": subtask_id,
            "description": description,
            "status": TaskStatus.PENDING.value,
            "progress": 0.0,
        })

    def update_subtask(self, subtask_id: str, status: str, progress: float = 0.0):
        for subtask in self.subtasks:
            if subtask["subtask_id"] == subtask_id:
                subtask["status"] = status
                subtask["progress"] = progress
                break

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "progress": self.progress,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "subtasks": self.subtasks,
            "logs": self.logs,
        }


class TaskTracker:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._task_history: deque = deque(maxlen=500)
        self._active_tasks: Dict[str, Task] = {}

    def create_task(self, task_type: str, description: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = Task(task_id, task_type, description, metadata=metadata)
        self._tasks[task_id] = task
        self._active_tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        task = self._tasks.get(task_id)
        if task:
            if "progress" in kwargs:
                task.update_progress(kwargs["progress"])
            if "status" in kwargs:
                task.status = kwargs["status"]
            if "metadata" in kwargs:
                task.metadata.update(kwargs["metadata"])
            if "started_at" in kwargs:
                task.started_at = kwargs["started_at"]
            if "completed_at" in kwargs:
                task.completed_at = kwargs["completed_at"]

    def start_task(self, task_id: str):
        task = self._tasks.get(task_id)
        if task:
            task.start()

    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None):
        task = self._tasks.get(task_id)
        if task:
            task.complete(result)
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            self._task_history.append(task)

    def fail_task(self, task_id: str, error: str):
        task = self._tasks.get(task_id)
        if task:
            task.fail(error)
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            self._task_history.append(task)

    def cancel_task(self, task_id: str):
        task = self._tasks.get(task_id)
        if task:
            task.cancel()
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]

    def add_task_log(self, task_id: str, message: str, level: str = "info"):
        task = self._tasks.get(task_id)
        if task:
            task.add_log(message, level)

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        return [task.to_dict() for task in self._active_tasks.values()]

    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [task.to_dict() for task in list(self._task_history)[-limit:]]

    def get_task_stats(self) -> Dict[str, Any]:
        stats = {
            "total_tasks": len(self._tasks),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": sum(1 for t in self._task_history if t.status == TaskStatus.COMPLETED.value),
            "failed_tasks": sum(1 for t in self._task_history if t.status == TaskStatus.FAILED.value),
            "by_type": {},
            "by_status": {},
        }

        for task in self._tasks.values():
            stats["by_type"][task.task_type] = stats["by_type"].get(task.task_type, 0) + 1
            stats["by_status"][task.status] = stats["by_status"].get(task.status, 0) + 1

        return stats

    def delete_task(self, task_id: str):
        if task_id in self._tasks:
            del self._tasks[task_id]
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]