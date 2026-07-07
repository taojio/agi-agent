import uuid
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque
from dataclasses import dataclass, field


class AgentRole(Enum):
    LEADER = "leader"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"


class CollaborationMode(Enum):
    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class AgentInfo:
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    workload: float = 0.0
    performance_score: float = 0.5
    joined_at: float = 0.0
    last_heartbeat: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "current_task": self.current_task,
            "workload": self.workload,
            "performance_score": self.performance_score,
            "joined_at": self.joined_at
        }


class AgentSwarm:
    def __init__(self, swarm_id: str = None, mode: CollaborationMode = CollaborationMode.HYBRID):
        self.swarm_id = swarm_id or str(uuid.uuid4())[:8]
        self.mode = mode
        self.agents: Dict[str, AgentInfo] = {}
        self._lock = threading.RLock()

        self.task_queue: deque = deque(maxlen=200)
        self.completed_tasks: deque = deque(maxlen=100)
        self.message_bus: List[Dict[str, Any]] = []

        self._message_callbacks: List[Callable] = []
        self._agent_join_callbacks: List[Callable] = []
        self._agent_leave_callbacks: List[Callable] = []

        self.leader_id: Optional[str] = None
        self.shared_memory = None
        self.task_allocator = None
        self.conflict_resolver = None

        self._swarm_start_time = time.time()
        self._total_tasks_completed = 0

    def register_agent(self, name: str, role: AgentRole = AgentRole.WORKER,
                       capabilities: List[str] = None,
                       agent_id: str = None) -> AgentInfo:
        with self._lock:
            aid = agent_id or str(uuid.uuid4())[:8]
            now = time.time()

            agent = AgentInfo(
                agent_id=aid,
                name=name,
                role=role,
                status=AgentStatus.IDLE,
                capabilities=capabilities or [],
                joined_at=now,
                last_heartbeat=now
            )

            self.agents[aid] = agent

            if self.leader_id is None and role == AgentRole.LEADER:
                self.leader_id = aid

            self._emit_event("agent_join", {"agent": agent.to_dict()})

            return agent

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self.agents:
                return False

            agent = self.agents.pop(agent_id)

            if self.leader_id == agent_id:
                self._elect_new_leader()

            self._emit_event("agent_leave", {"agent_id": agent_id, "name": agent.name})

            return True

    def _elect_new_leader(self):
        candidates = [
            aid for aid, info in self.agents.items()
            if info.role in (AgentRole.LEADER, AgentRole.COORDINATOR)
        ]
        if not candidates:
            candidates = list(self.agents.keys())

        if candidates:
            self.leader_id = max(
                candidates,
                key=lambda aid: self.agents[aid].performance_score
            )
            self.agents[self.leader_id].role = AgentRole.LEADER

    def heartbeat(self, agent_id: str, status: AgentStatus = None,
                  workload: float = None, performance: float = None):
        with self._lock:
            if agent_id not in self.agents:
                return False

            agent = self.agents[agent_id]
            agent.last_heartbeat = time.time()

            if status is not None:
                agent.status = status
            if workload is not None:
                agent.workload = max(0.0, min(1.0, workload))
            if performance is not None:
                agent.performance_score = max(0.0, min(1.0, performance))

            return True

    def get_available_agents(self, capability: str = None) -> List[AgentInfo]:
        with self._lock:
            agents = [
                a for a in self.agents.values()
                if a.status == AgentStatus.IDLE
            ]

            if capability:
                agents = [a for a in agents if capability in a.capabilities]

            return sorted(agents, key=lambda a: a.workload)

    def broadcast_message(self, sender_id: str, message_type: str,
                           content: Any = None, target_id: str = None):
        msg = {
            "msg_id": str(uuid.uuid4())[:8],
            "sender_id": sender_id,
            "target_id": target_id,
            "message_type": message_type,
            "content": content,
            "timestamp": time.time()
        }

        self.message_bus.append(msg)
        if len(self.message_bus) > 500:
            self.message_bus.pop(0)

        self._emit_event("message", msg)

        return msg

    def get_messages(self, agent_id: str = None, since: float = 0.0) -> List[Dict[str, Any]]:
        msgs = [m for m in self.message_bus if m["timestamp"] > since]

        if agent_id:
            msgs = [
                m for m in msgs
                if m["target_id"] is None or m["target_id"] == agent_id or m["sender_id"] == agent_id
            ]

        return msgs

    def assign_task(self, task: Dict[str, Any], agent_id: str = None) -> Optional[str]:
        if agent_id and agent_id in self.agents:
            self.agents[agent_id].current_task = task.get("task_id")
            self.agents[agent_id].status = AgentStatus.WORKING
            task["assigned_to"] = agent_id
            self.task_queue.append(task)
            return agent_id

        if self.task_allocator:
            allocated = self.task_allocator.allocate_task(task, list(self.agents.values()))
            if allocated:
                self.agents[allocated].current_task = task.get("task_id")
                self.agents[allocated].status = AgentStatus.WORKING
                task["assigned_to"] = allocated
                self.task_queue.append(task)
                return allocated

        return None

    def complete_task(self, task_id: str, agent_id: str, result: Dict[str, Any] = None):
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.IDLE
                self.agents[agent_id].current_task = None
                self.agents[agent_id].workload = max(0.0, self.agents[agent_id].workload - 0.1)

            self._total_tasks_completed += 1
            self.completed_tasks.append({
                "task_id": task_id,
                "agent_id": agent_id,
                "result": result or {},
                "completed_at": time.time()
            })

    def get_swarm_stats(self) -> Dict[str, Any]:
        with self._lock:
            status_counts = {}
            for agent in self.agents.values():
                s = agent.status.value
                status_counts[s] = status_counts.get(s, 0) + 1

            avg_perf = np.mean([a.performance_score for a in self.agents.values()]) if self.agents else 0.0
            avg_workload = np.mean([a.workload for a in self.agents.values()]) if self.agents else 0.0

            return {
                "swarm_id": self.swarm_id,
                "mode": self.mode.value,
                "total_agents": len(self.agents),
                "leader_id": self.leader_id,
                "status_counts": status_counts,
                "avg_performance": float(avg_perf),
                "avg_workload": float(avg_workload),
                "total_tasks_completed": self._total_tasks_completed,
                "pending_tasks": len(self.task_queue),
                "uptime": time.time() - self._swarm_start_time
            }

    def on(self, event_type: str, callback: Callable):
        if event_type == "message":
            self._message_callbacks.append(callback)
        elif event_type == "agent_join":
            self._agent_join_callbacks.append(callback)
        elif event_type == "agent_leave":
            self._agent_leave_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        cbs = []
        if event_type == "message":
            cbs = self._message_callbacks
        elif event_type == "agent_join":
            cbs = self._agent_join_callbacks
        elif event_type == "agent_leave":
            cbs = self._agent_leave_callbacks

        for cb in cbs:
            try:
                cb(data)
            except Exception:
                pass

    def set_collaboration_mode(self, mode: CollaborationMode):
        self.mode = mode


import numpy as np
