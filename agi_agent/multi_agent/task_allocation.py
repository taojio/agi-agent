import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List["Task"] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None

    def get_priority_value(self) -> int:
        return self.priority.value


class TaskAllocator:
    def __init__(self, allocation_strategy: str = "balanced"):
        self.allocation_strategy = allocation_strategy
        self.task_history: deque = deque(maxlen=200)
        self.allocation_log: deque = deque(maxlen=200)

        self._strategies = {
            "balanced": self._balanced_allocation,
            "performance": self._performance_allocation,
            "capability": self._capability_allocation,
            "round_robin": self._round_robin_allocation,
        }

        self._round_robin_index = 0

    def allocate_task(self, task: Dict[str, Any], agents: List[Any]) -> Optional[str]:
        strategy_func = self._strategies.get(self.allocation_strategy, self._balanced_allocation)

        allocated_agent_id = strategy_func(task, agents)

        if allocated_agent_id:
            self.allocation_log.append({
                "task_id": task.get("task_id"),
                "agent_id": allocated_agent_id,
                "strategy": self.allocation_strategy,
                "timestamp": None
            })

        return allocated_agent_id

    def _balanced_allocation(self, task: Dict[str, Any], agents: List[Any]) -> Optional[str]:
        if not agents:
            return None

        required_caps = set(task.get("required_capabilities", []))

        eligible = [
            a for a in agents
            if a.status.value == "idle" and
            (not required_caps or required_caps.issubset(set(a.capabilities)))
        ]

        if not eligible:
            eligible = [
                a for a in agents
                if not required_caps or required_caps.issubset(set(a.capabilities))
            ]

        if not eligible:
            return None

        best = min(eligible, key=lambda a: a.workload)
        return best.agent_id

    def _performance_allocation(self, task: Dict[str, Any], agents: List[Any]) -> Optional[str]:
        if not agents:
            return None

        required_caps = set(task.get("required_capabilities", []))

        eligible = [
            a for a in agents
            if (not required_caps or required_caps.issubset(set(a.capabilities))) and
               a.workload < 0.9
        ]

        if not eligible:
            return None

        best = max(eligible, key=lambda a: a.performance_score)
        return best.agent_id

    def _capability_allocation(self, task: Dict[str, Any], agents: List[Any]) -> Optional[str]:
        if not agents:
            return None

        required_caps = set(task.get("required_capabilities", []))
        if not required_caps:
            return self._balanced_allocation(task, agents)

        best_match = None
        best_score = -1

        for agent in agents:
            if agent.status.value != "idle":
                continue

            agent_caps = set(agent.capabilities)
            match_count = len(required_caps & agent_caps)
            match_ratio = match_count / len(required_caps)
            workload_penalty = agent.workload * 0.3
            score = match_ratio - workload_penalty

            if score > best_score:
                best_score = score
                best_match = agent

        return best_match.agent_id if best_match else None

    def _round_robin_allocation(self, task: Dict[str, Any], agents: List[Any]) -> Optional[str]:
        if not agents:
            return None

        idle_agents = [a for a in agents if a.status.value == "idle"]
        if not idle_agents:
            return None

        sorted_agents = sorted(idle_agents, key=lambda a: a.agent_id)
        idx = self._round_robin_index % len(sorted_agents)
        self._round_robin_index += 1

        return sorted_agents[idx].agent_id

    def batch_allocate(self, tasks: List[Dict[str, Any]],
                        agents: List[Any]) -> Dict[str, str]:
        results = {}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.get("priority", TaskPriority.MEDIUM.value)
        )

        for task in sorted_tasks:
            agent_id = self.allocate_task(task, agents)
            if agent_id:
                results[task.get("task_id", "")] = agent_id

        return results

    def get_allocation_stats(self) -> Dict[str, Any]:
        return {
            "total_allocations": len(self.allocation_log),
            "strategy": self.allocation_strategy,
            "available_strategies": list(self._strategies.keys())
        }
