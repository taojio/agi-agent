"""
dag_engine.py - DAG 任务编排引擎

支持通过依赖关系定义任务的执行顺序，生成有向无环图，
系统按依赖顺序自动调度执行，前置任务失败自动阻断下游任务；
无依赖的子任务自动并行执行，最大化利用系统资源。
"""
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque
from dataclasses import dataclass, field


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskNode:
    """任务节点"""
    task_id: str
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    executor: Optional[Callable] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "metadata": self.metadata
        }


class DAGEngine:
    """DAG 任务编排引擎"""

    def __init__(self):
        self._nodes: Dict[str, TaskNode] = {}
        self._lock = threading.RLock()
        self._running_tasks = set()
        self._completed_tasks = set()
        self._failed_tasks = set()
        self._callback = None

        self._execute_lock = threading.RLock()
        self._event = threading.Event()
        self._shutdown = False

    def add_task(self, name: str, description: str = "", priority: int = 5,
                 executor: Callable = None, dependencies: List[str] = None,
                 task_id: str = None) -> TaskNode:
        """
        添加任务节点

        Args:
            name: 任务名称
            description: 任务描述
            priority: 优先级 (1-10)
            executor: 执行函数
            dependencies: 依赖任务ID列表
            task_id: 任务ID（可选，自动生成）

        Returns:
            任务节点
        """
        task_id = task_id or f"dag_{uuid.uuid4().hex[:8]}"

        node = TaskNode(
            task_id=task_id,
            name=name,
            description=description,
            priority=priority,
            executor=executor,
            dependencies=dependencies or []
        )

        with self._lock:
            self._nodes[task_id] = node

            for dep_id in node.dependencies:
                if dep_id in self._nodes:
                    self._nodes[dep_id].dependents.append(task_id)

        return node

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """获取任务节点"""
        return self._nodes.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        """移除任务节点"""
        with self._lock:
            if task_id not in self._nodes:
                return False

            node = self._nodes.pop(task_id)

            for dep_id in node.dependencies:
                if dep_id in self._nodes:
                    self._nodes[dep_id].dependents.remove(task_id)

            for dep_id in node.dependents:
                if dep_id in self._nodes:
                    self._nodes[dep_id].dependencies.remove(task_id)

            return True

    def has_cycle(self) -> bool:
        """检查是否存在循环依赖"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in self._nodes}

        def dfs(task_id):
            color[task_id] = GRAY
            for dep_id in self._nodes[task_id].dependents:
                if color[dep_id] == GRAY:
                    return True
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            color[task_id] = BLACK
            return False

        for task_id in self._nodes:
            if color[task_id] == WHITE and dfs(task_id):
                return True
        return False

    def get_topological_order(self) -> List[str]:
        """获取拓扑排序结果"""
        if self.has_cycle():
            return []

        in_degree = {task_id: len(node.dependencies) for task_id, node in self._nodes.items()}
        queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            task_id = queue.popleft()
            order.append(task_id)
            for dep_id in self._nodes[task_id].dependents:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        return order if len(order) == len(self._nodes) else []

    def execute(self, on_complete: Callable = None):
        """
        执行 DAG

        Args:
            on_complete: 完成回调函数
        """
        self._callback = on_complete
        order = self.get_topological_order()

        if not order:
            if self.has_cycle():
                if on_complete:
                    on_complete({"status": "error", "message": "DAG contains cycles"})
                return
            return

        def run():
            while order and not self._shutdown:
                ready_tasks = []
                with self._lock:
                    for task_id in order:
                        node = self._nodes[task_id]
                        if node.status == TaskStatus.PENDING:
                            all_deps_done = all(
                                self._nodes[dep_id].status == TaskStatus.COMPLETED
                                for dep_id in node.dependencies
                            )
                            if all_deps_done:
                                ready_tasks.append(task_id)

                threads = []
                for task_id in ready_tasks:
                    t = threading.Thread(target=self._execute_task, args=(task_id,))
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                with self._lock:
                    order[:] = [tid for tid in order if self._nodes[tid].status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]

                failed_count = sum(1 for node in self._nodes.values() if node.status == TaskStatus.FAILED)
                if failed_count > 0:
                    for task_id in order:
                        node = self._nodes[task_id]
                        node.status = TaskStatus.BLOCKED

            if on_complete:
                failed = [n for n in self._nodes.values() if n.status == TaskStatus.FAILED]
                blocked = [n for n in self._nodes.values() if n.status == TaskStatus.BLOCKED]
                completed = [n for n in self._nodes.values() if n.status == TaskStatus.COMPLETED]

                on_complete({
                    "status": "completed" if not failed else "failed",
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "blocked_count": len(blocked),
                    "total_count": len(self._nodes)
                })

        t = threading.Thread(target=run)
        t.start()

    def _execute_task(self, task_id: str):
        """执行单个任务"""
        with self._lock:
            node = self._nodes.get(task_id)
            if node is None or node.status != TaskStatus.PENDING:
                return
            node.status = TaskStatus.RUNNING
            node.started_at = time.time()

        try:
            if node.executor:
                result = node.executor(node)
                node.result = {"output": result} if result else {}
                node.progress = 1.0
                node.status = TaskStatus.COMPLETED
            else:
                node.result = {"message": "Task completed (no executor)"}
                node.progress = 1.0
                node.status = TaskStatus.COMPLETED
        except Exception as e:
            node.error = str(e)
            node.status = TaskStatus.FAILED

        node.completed_at = time.time()

    def get_dag_stats(self) -> Dict[str, Any]:
        """获取 DAG 统计信息"""
        with self._lock:
            status_counts = {}
            for node in self._nodes.values():
                s = node.status.value
                status_counts[s] = status_counts.get(s, 0) + 1

            return {
                "total_tasks": len(self._nodes),
                "status_counts": status_counts,
                "has_cycle": self.has_cycle(),
                "topological_order": self.get_topological_order()
            }

    def shutdown(self):
        """关闭引擎"""
        self._shutdown = True
        self._event.set()
