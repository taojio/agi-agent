"""
meta_orchestrator.py - 元层级统一编排器

实现元模块间的统一调度、资源分配和任务优先级管理，支持：
- 动态模块注册与即插即用
- 高并发任务调度
- 智能资源分配
- 实时状态监控
"""
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class ModuleStatus(Enum):
    """模块状态"""
    REGISTERED = "registered"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ModuleInfo:
    """模块信息"""
    module_id: str
    name: str
    type: str
    version: str
    status: ModuleStatus = ModuleStatus.REGISTERED
    capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    current_load: float = 0.0
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None

    def is_alive(self) -> bool:
        return (time.time() - self.last_heartbeat) < 30.0

    def update_heartbeat(self):
        self.last_heartbeat = time.time()


@dataclass
class ResourceAllocation:
    """资源分配"""
    cpu: float = 0.0
    memory: float = 0.0
    gpu: float = 0.0
    bandwidth: float = 0.0
    priority: TaskPriority = TaskPriority.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "bandwidth": self.bandwidth,
            "priority": self.priority.value,
        }


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    name: str
    type: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    target_module: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    resource_request: ResourceAllocation = field(default_factory=ResourceAllocation)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_running(self):
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: Dict[str, Any] = None):
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at) * 1000
        if result:
            self.result = result

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at) * 1000
        self.error = error

    def mark_cancelled(self):
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()

    def mark_timeout(self):
        self.status = TaskStatus.TIMEOUT
        self.completed_at = time.time()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at) * 1000


class PriorityQueueItem:
    """优先级队列项"""

    def __init__(self, priority: TaskPriority, task_id: str, task: TaskInfo):
        self.priority = priority
        self.task_id = task_id
        self.task = task

    def __lt__(self, other: "PriorityQueueItem") -> bool:
        return self.priority.value > other.priority.value


class MetaOrchestrator:
    """元层级统一编排器

    负责元模块的统一调度、资源分配和任务优先级管理
    """

    def __init__(self, max_workers: int = 8, max_modules: int = 20):
        self._modules: Dict[str, ModuleInfo] = {}
        self._tasks: Dict[str, TaskInfo] = {}
        self._active_tasks: Dict[str, TaskInfo] = {}
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._max_workers = max_workers
        self._max_modules = max_modules
        self._worker_threads: List[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()
        self._stats = {
            "modules_registered": 0,
            "modules_active": 0,
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time_ms": 0,
            "resource_usage": {"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
        }

    def start(self):
        """启动编排器"""
        with self._lock:
            if self._running:
                return

            self._running = True
            for _ in range(self._max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name=f"meta-orchestrator-worker-{_}"
                )
                worker.start()
                self._worker_threads.append(worker)

    def stop(self):
        """停止编排器"""
        with self._lock:
            self._running = False

        for _ in range(self._max_workers):
            self._task_queue.put(PriorityQueueItem(
                TaskPriority.CRITICAL,
                "__STOP__",
                TaskInfo(task_id="__STOP__", name="stop", type="internal", priority=TaskPriority.CRITICAL)
            ))

        for worker in self._worker_threads:
            worker.join(timeout=5)

        self._worker_threads.clear()

    def register_module(
        self,
        module_id: str,
        name: str,
        type: str,
        version: str = "1.0",
        capabilities: List[str] = None,
        resource_requirements: Dict[str, float] = None,
        handler: Callable = None,
        **metadata
    ) -> bool:
        """注册模块"""
        with self._lock:
            if len(self._modules) >= self._max_modules:
                return False

            if module_id in self._modules:
                return False

            self._modules[module_id] = ModuleInfo(
                module_id=module_id,
                name=name,
                type=type,
                version=version,
                capabilities=capabilities or [],
                resource_requirements=resource_requirements or {},
                status=ModuleStatus.ACTIVE,
                handler=handler,
                metadata=metadata,
            )
            self._stats["modules_registered"] += 1
            self._stats["modules_active"] += 1

        return True

    def unregister_module(self, module_id: str) -> bool:
        """注销模块"""
        with self._lock:
            if module_id not in self._modules:
                return False

            self._modules[module_id].status = ModuleStatus.SHUTDOWN
            del self._modules[module_id]
            self._stats["modules_active"] = max(0, self._stats["modules_active"] - 1)

        return True

    def get_module(self, module_id: str) -> Optional[ModuleInfo]:
        """获取模块信息"""
        return self._modules.get(module_id)

    def list_modules(self, status: ModuleStatus = None) -> List[ModuleInfo]:
        """列出模块"""
        with self._lock:
            if status:
                return [m for m in self._modules.values() if m.status == status]
            return list(self._modules.values())

    def submit_task(
        self,
        name: str,
        type: str,
        target_module: str = "",
        payload: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        resource_request: ResourceAllocation = None,
        **metadata
    ) -> str:
        """提交任务"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = TaskInfo(
            task_id=task_id,
            name=name,
            type=type,
            priority=priority,
            target_module=target_module,
            payload=payload or {},
            resource_request=resource_request or ResourceAllocation(),
            metadata=metadata,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._stats["tasks_submitted"] += 1

        self._task_queue.put(PriorityQueueItem(priority, task_id, task))
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus = None) -> List[TaskInfo]:
        """列出任务"""
        with self._lock:
            if status:
                return [t for t in self._tasks.values() if t.status == status]
            return list(self._tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status == TaskStatus.PENDING:
                task.mark_cancelled()
                self._stats["tasks_cancelled"] += 1
                return True

            if task.status == TaskStatus.RUNNING:
                task.mark_cancelled()
                self._stats["tasks_cancelled"] += 1
                return True

        return False

    def allocate_resources(self, task: TaskInfo) -> bool:
        """分配资源"""
        request = task.resource_request
        current_usage = self._stats["resource_usage"]

        if (current_usage["cpu"] + request.cpu > 100 or
            current_usage["memory"] + request.memory > 100 or
            current_usage["gpu"] + request.gpu > 100):
            return False

        with self._lock:
            self._stats["resource_usage"]["cpu"] += request.cpu
            self._stats["resource_usage"]["memory"] += request.memory
            self._stats["resource_usage"]["gpu"] += request.gpu

        return True

    def release_resources(self, task: TaskInfo):
        """释放资源"""
        request = task.resource_request
        with self._lock:
            self._stats["resource_usage"]["cpu"] = max(0, self._stats["resource_usage"]["cpu"] - request.cpu)
            self._stats["resource_usage"]["memory"] = max(0, self._stats["resource_usage"]["memory"] - request.memory)
            self._stats["resource_usage"]["gpu"] = max(0, self._stats["resource_usage"]["gpu"] - request.gpu)

    def _worker_loop(self):
        """工作线程循环"""
        while self._running:
            try:
                item = self._task_queue.get(timeout=1)
                if item.task_id == "__STOP__":
                    break

                self._execute_task(item.task)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _execute_task(self, task: TaskInfo):
        """执行任务"""
        with self._lock:
            if task.status not in (TaskStatus.PENDING,):
                return

            task.mark_running()
            self._active_tasks[task.task_id] = task

        if not self.allocate_resources(task):
            with self._lock:
                task.mark_failed("resource allocation failed")
                self._stats["tasks_failed"] += 1
                del self._active_tasks[task.task_id]
            return

        result = {}
        error = None

        try:
            if task.target_module:
                module = self._modules.get(task.target_module)
                if module and module.handler:
                    result = module.handler(task.payload)
                else:
                    error = f"target module {task.target_module} not found or no handler"
            else:
                result = self._execute_generic_task(task)
        except Exception as e:
            error = str(e)

        self.release_resources(task)

        with self._lock:
            if error:
                task.mark_failed(error)
                self._stats["tasks_failed"] += 1
            else:
                task.mark_completed(result)
                self._stats["tasks_completed"] += 1
                self._stats["total_execution_time_ms"] += task.duration_ms

            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]

    def _execute_generic_task(self, task: TaskInfo) -> Dict[str, Any]:
        """执行通用任务"""
        return {"status": "processed", "task_id": task.task_id, "type": task.type}

    def update_module_status(self, module_id: str, status: ModuleStatus):
        """更新模块状态"""
        with self._lock:
            module = self._modules.get(module_id)
            if module:
                module.status = status

    def send_heartbeat(self, module_id: str):
        """发送心跳"""
        with self._lock:
            module = self._modules.get(module_id)
            if module:
                module.update_heartbeat()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            avg_execution_time = (
                self._stats["total_execution_time_ms"] / max(self._stats["tasks_completed"], 1)
            )
            return {
                "modules": {
                    "registered": self._stats["modules_registered"],
                    "active": self._stats["modules_active"],
                    "capacity": self._max_modules,
                },
                "tasks": {
                    "submitted": self._stats["tasks_submitted"],
                    "completed": self._stats["tasks_completed"],
                    "failed": self._stats["tasks_failed"],
                    "cancelled": self._stats["tasks_cancelled"],
                    "active": len(self._active_tasks),
                    "pending": self._task_queue.qsize(),
                    "avg_execution_time_ms": avg_execution_time,
                },
                "resources": dict(self._stats["resource_usage"]),
                "workers": {
                    "count": len(self._worker_threads),
                    "max": self._max_workers,
                },
            }

    def find_module_by_capability(self, capability: str) -> List[ModuleInfo]:
        """根据能力查找模块"""
        with self._lock:
            return [m for m in self._modules.values()
                    if capability in m.capabilities and m.status == ModuleStatus.ACTIVE]

    def find_module_by_type(self, module_type: str) -> List[ModuleInfo]:
        """根据类型查找模块"""
        with self._lock:
            return [m for m in self._modules.values()
                    if m.type == module_type and m.status == ModuleStatus.ACTIVE]


_orchestrator_instance: Optional[MetaOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_meta_orchestrator() -> MetaOrchestrator:
    """获取元编排器单例"""
    global _orchestrator_instance
    with _orchestrator_lock:
        if _orchestrator_instance is None:
            _orchestrator_instance = MetaOrchestrator()
        return _orchestrator_instance