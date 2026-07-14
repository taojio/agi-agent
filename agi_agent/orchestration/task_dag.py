"""
orchestration/task_dag.py - 任务 DAG 图

基于有向无环图的任务编排与依赖管理
"""
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from .event_bus import EventBus, Event


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    duration: float = 0.0


@dataclass
class TaskNode:
    """任务节点"""
    task_id: str
    name: str
    func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    priority: int = 0
    retries: int = 0
    max_retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """重置任务状态"""
        self.status = TaskStatus.PENDING
        self.result = None
        self.retries = 0


class TaskDAG:
    """任务 DAG 图

    功能：
    - 添加/移除任务节点
    - 依赖关系管理与环检测
    - 拓扑排序
    - 任务状态追踪
    - 就绪任务查询
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._tasks: Dict[str, TaskNode] = {}
        self._event_bus = event_bus

    def add_task(
        self,
        task_id: str,
        name: str = "",
        func: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskNode:
        """添加任务节点

        Args:
            task_id: 任务唯一 ID
            name: 任务名称
            func: 执行函数
            dependencies: 依赖的任务 ID 列表
            priority: 优先级（数值越大越优先）
            max_retries: 最大重试次数
            metadata: 附加元数据

        Returns:
            TaskNode
        """
        if task_id in self._tasks:
            raise ValueError(f"Task '{task_id}' already exists")

        node = TaskNode(
            task_id=task_id,
            name=name or task_id,
            func=func,
            dependencies=dependencies or [],
            priority=priority,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        self._tasks[task_id] = node
        self._emit("task.added", {"task_id": task_id})
        return node

    def remove_task(self, task_id: str) -> bool:
        """移除任务节点"""
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        for task in self._tasks.values():
            if task_id in task.dependencies:
                task.dependencies.remove(task_id)
        self._emit("task.removed", {"task_id": task_id})
        return True

    def add_dependency(self, task_id: str, dependency_id: str) -> bool:
        """添加依赖关系"""
        if task_id not in self._tasks or dependency_id not in self._tasks:
            return False
        if self._would_create_cycle(task_id, dependency_id):
            return False
        if dependency_id not in self._tasks[task_id].dependencies:
            self._tasks[task_id].dependencies.append(dependency_id)
        return True

    def _would_create_cycle(self, task_id: str, dep_id: str) -> bool:
        """检测是否会形成环"""
        visited = set()
        stack = [dep_id]
        while stack:
            current = stack.pop()
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            if current in self._tasks:
                stack.extend(self._tasks[current].dependencies)
        return False

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """获取任务节点"""
        return self._tasks.get(task_id)

    def has_task(self, task_id: str) -> bool:
        """检查任务是否存在"""
        return task_id in self._tasks

    def list_tasks(self) -> List[TaskNode]:
        """列出所有任务"""
        return list(self._tasks.values())

    def get_ready_tasks(self) -> List[TaskNode]:
        """获取所有就绪任务（依赖均已完成）

        按优先级排序
        """
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks.get(dep_id, TaskNode(task_id=dep_id, name=dep_id, status=TaskStatus.COMPLETED)).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def topological_sort(self) -> List[str]:
        """拓扑排序（Kahn 算法）

        Returns:
            任务 ID 的有序列表
        """
        in_degree = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for task in self._tasks.values():
                if current in task.dependencies:
                    in_degree[task.task_id] -= 1
                    if in_degree[task.task_id] == 0:
                        queue.append(task.task_id)

        if len(result) != len(self._tasks):
            raise ValueError("Cycle detected in task graph")
        return result

    def mark_running(self, task_id: str) -> None:
        """标记任务运行中"""
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._emit("task.started", {"task_id": task_id})

    def mark_completed(self, task_id: str, output: Any = None) -> None:
        """标记任务完成"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = TaskResult(
                task_id=task_id,
                success=True,
                output=output,
                completed_at=time.time(),
            )
            self._emit("task.completed", {"task_id": task_id, "output": output})

    def mark_failed(self, task_id: str, error: str = "") -> None:
        """标记任务失败"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self._emit("task.retrying", {"task_id": task_id, "attempt": task.retries})
            else:
                task.status = TaskStatus.FAILED
                task.result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=error,
                    completed_at=time.time(),
                )
                self._emit("task.failed", {"task_id": task_id, "error": error})

    def reset_all(self) -> None:
        """重置所有任务状态"""
        for task in self._tasks.values():
            task.reset()
        self._emit("dag.reset", {})

    def get_stats(self) -> Dict[str, Any]:
        """获取 DAG 统计"""
        stats = {status.value: 0 for status in TaskStatus}
        for task in self._tasks.values():
            stats[task.status.value] += 1
        stats["total"] = len(self._tasks)
        return stats

    def is_complete(self) -> bool:
        """是否所有任务都完成（成功或失败）"""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self._tasks.values()
        )

    def all_succeeded(self) -> bool:
        """是否所有任务都成功"""
        if not self._tasks:
            return True
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        if self._event_bus:
            self._event_bus.publish(Event(event_type=event_type, data=data, source="task_dag"))
