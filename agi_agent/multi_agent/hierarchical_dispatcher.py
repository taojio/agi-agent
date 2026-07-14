"""
hierarchical_dispatcher.py - 分层调度器

实现主子Agent分级调度：
- 主Agent承担「项目经理」角色，负责需求拆解、任务规划、资源分配与结果汇总
- 根据任务类型按需唤醒对应领域子Agent，子Agent专注执行专项任务
- 支持多级嵌套调度（子Agent可唤醒更细分的孙Agent）
- 支持平级并行协同（多个子Agent同时处理同一任务的不同模块）
- 权限委托机制：主Agent可向子Agent临时授予指定工作空间的访问权限
"""
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque
from dataclasses import dataclass, field

from .agent_swarm import AgentSwarm, AgentRole, AgentStatus, AgentInfo, CollaborationMode
from .workspace import WorkspaceManager, Workspace, WorkspacePermission, WorkspaceType


class DispatchStrategy(Enum):
    """调度策略"""
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"


@dataclass
class DispatchTask:
    """调度任务"""
    task_id: str
    parent_id: Optional[str] = None
    agent_id: Optional[str] = None
    workspace_id: Optional[str] = None
    description: str = ""
    type: str = "generic"
    status: str = "pending"
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    children: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "description": self.description,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "children": self.children,
            "dependencies": self.dependencies
        }


class HierarchicalDispatcher:
    """分层调度器"""

    def __init__(self, swarm: AgentSwarm, workspace_manager: WorkspaceManager = None):
        """
        初始化分层调度器

        Args:
            swarm: 智能体集群
            workspace_manager: 工作空间管理器
        """
        self.swarm = swarm
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self._lock = threading.RLock()

        self._tasks: Dict[str, DispatchTask] = {}
        self._task_queue = deque()
        self._dispatch_history: List[Dict[str, Any]] = []

        self._strategy = DispatchStrategy.CAPABILITY_MATCH
        self._round_robin_counter = 0

        self._dispatch_callbacks: List[Callable] = []
        self._completion_callbacks: List[Callable] = []

    def set_strategy(self, strategy: DispatchStrategy):
        """设置调度策略"""
        self._strategy = strategy

    def dispatch_task(self, description: str, task_type: str = "generic",
                      priority: int = 5, parent_id: str = None,
                      dependencies: List[str] = None) -> DispatchTask:
        """
        调度任务

        Args:
            description: 任务描述
            task_type: 任务类型
            priority: 优先级
            parent_id: 父任务ID（用于嵌套调度）
            dependencies: 依赖任务ID列表

        Returns:
            调度任务对象
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = DispatchTask(
            task_id=task_id,
            description=description,
            type=task_type,
            priority=priority,
            parent_id=parent_id,
            dependencies=dependencies or []
        )

        with self._lock:
            self._tasks[task_id] = task
            self._task_queue.append(task_id)

            if parent_id and parent_id in self._tasks:
                self._tasks[parent_id].children.append(task_id)

        self._try_dispatch()
        return task

    def _try_dispatch(self):
        """尝试调度任务"""
        while self._task_queue:
            task_id = self._task_queue.popleft()
            task = self._tasks.get(task_id)
            if task is None:
                continue

            if task.status != "pending":
                continue

            if task.dependencies:
                unresolved = [d for d in task.dependencies if self._tasks.get(d) and self._tasks[d].status != "completed"]
                if unresolved:
                    self._task_queue.append(task_id)
                    continue

            agent_id = self._select_agent(task)
            if agent_id:
                self._assign_to_agent(task, agent_id)
            else:
                self._task_queue.append(task_id)
                break

    def _select_agent(self, task: DispatchTask) -> Optional[str]:
        """选择合适的Agent"""
        with self._lock:
            available = self.swarm.get_available_agents()
            if not available:
                return None

            if self._strategy == DispatchStrategy.ROUND_ROBIN:
                index = self._round_robin_counter % len(available)
                self._round_robin_counter += 1
                return available[index].agent_id

            if self._strategy == DispatchStrategy.PERFORMANCE_BASED:
                best = max(available, key=lambda a: a.performance_score)
                return best.agent_id

            if self._strategy == DispatchStrategy.CAPABILITY_MATCH:
                capability_map = {
                    "code": ["code_review", "coding", "debugging"],
                    "data": ["data_analysis", "data_processing", "visualization"],
                    "writing": ["writing", "editing", "summarization"],
                    "research": ["research", "search", "analysis"],
                    "planning": ["planning", "strategy", "decision"]
                }
                required_caps = capability_map.get(task.type, [])
                for agent in available:
                    if any(cap in agent.capabilities for cap in required_caps):
                        return agent.agent_id
                return available[0].agent_id

            if self._strategy == DispatchStrategy.LOAD_BALANCED:
                return min(available, key=lambda a: a.workload).agent_id

            return available[0].agent_id

    def _assign_to_agent(self, task: DispatchTask, agent_id: str):
        """将任务分配给Agent"""
        with self._lock:
            task.agent_id = agent_id
            task.status = "running"
            task.started_at = time.time()

            agent = self.swarm.agents.get(agent_id)
            if agent:
                agent.current_task = task.task_id
                agent.status = AgentStatus.WORKING
                agent.workload = min(1.0, agent.workload + 0.2)

        self._emit_event("dispatch", {
            "task_id": task.task_id,
            "agent_id": agent_id,
            "description": task.description,
            "type": task.type
        })

    def delegate_to_sub_agent(self, parent_agent_id: str, task_description: str,
                              task_type: str = "generic", priority: int = 5,
                              workspace_id: str = None,
                              permission_duration: float = 24) -> DispatchTask:
        """
        主Agent委托任务给子Agent

        Args:
            parent_agent_id: 父Agent ID（授权者）
            task_description: 任务描述
            task_type: 任务类型
            priority: 优先级
            workspace_id: 工作空间ID（可选，用于权限委托）
            permission_duration: 权限有效期（小时）

        Returns:
            调度任务对象
        """
        task = self.dispatch_task(task_description, task_type, priority)

        if workspace_id and self.workspace_manager:
            workspace = self.workspace_manager.get_workspace(workspace_id)
            if workspace:
                workspace.delegate_permission(
                    from_agent_id=parent_agent_id,
                    to_agent_id=task.agent_id,
                    workspace_id=workspace_id,
                    permission=WorkspacePermission.READ_WRITE,
                    duration_hours=permission_duration
                )

        return task

    def complete_task(self, task_id: str, agent_id: str, result: Dict[str, Any] = None):
        """
        完成任务

        Args:
            task_id: 任务ID
            agent_id: Agent ID
            result: 任务结果
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            task.status = "completed"
            task.completed_at = time.time()
            task.result = result

            agent = self.swarm.agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.IDLE
                agent.current_task = None
                agent.workload = max(0.0, agent.workload - 0.2)
                agent.performance_score = min(1.0, agent.performance_score + 0.01)

            self.swarm.complete_task(task_id, agent_id, result)

            if task.workspace_id and self.workspace_manager:
                self.workspace_manager.revoke_delegated_permission(agent_id, task.workspace_id)

        self._emit_event("completion", {
            "task_id": task_id,
            "agent_id": agent_id,
            "result": result
        })

        self._try_dispatch()

    def fail_task(self, task_id: str, agent_id: str, error: str):
        """
        任务失败

        Args:
            task_id: 任务ID
            agent_id: Agent ID
            error: 错误信息
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            task.status = "failed"
            task.completed_at = time.time()
            task.result = {"error": error}

            agent = self.swarm.agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.IDLE
                agent.current_task = None
                agent.performance_score = max(0.0, agent.performance_score - 0.02)

        self._emit_event("completion", {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": "failed",
            "error": error
        })

        self._try_dispatch()

    def get_task(self, task_id: str) -> Optional[DispatchTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_tasks(self, status: str = None, agent_id: str = None) -> List[DispatchTask]:
        """获取任务列表"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if agent_id:
            tasks = [t for t in tasks if t.agent_id == agent_id]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def get_dispatch_stats(self) -> Dict[str, Any]:
        """获取调度统计信息"""
        with self._lock:
            status_counts = {}
            for task in self._tasks.values():
                s = task.status
                status_counts[s] = status_counts.get(s, 0) + 1

            total_tasks = len(self._tasks)
            completed_count = status_counts.get("completed", 0)
            success_rate = completed_count / total_tasks if total_tasks > 0 else 0.0

            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "success_rate": success_rate,
                "pending_tasks": len(self._task_queue),
                "strategy": self._strategy.value
            }

    def on(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type == "dispatch":
            self._dispatch_callbacks.append(callback)
        elif event_type == "completion":
            self._completion_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """触发事件"""
        cbs = self._dispatch_callbacks if event_type == "dispatch" else self._completion_callbacks
        for cb in cbs:
            try:
                cb(data)
            except Exception:
                pass
