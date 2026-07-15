"""
infrastructure/cluster_scheduler.py - 分布式节点调度子模块

包含任务：
- T003 分布式节点调度（DistributedNodeScheduler）

设计原则：
1. 维护节点心跳表 NodeHeartbeat（node_id, status, load, last_seen）
2. 将全局任务拆解分发至空闲节点
3. 单机模式下退化为单节点
4. 无重型依赖，纯 Python 实现
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.infrastructure")


# ====== 数据结构 ======

@dataclass
class NodeHeartbeat:
    """节点心跳（T003）"""
    node_id: str
    status: str = "online"          # online / offline / busy / draining
    load: float = 0.0               # 0-100
    capacity: int = 1               # 并发任务容量
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInfo:
    """注册节点信息"""
    node_id: str
    host: str = "127.0.0.1"
    port: int = 0
    capacity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalTask:
    """全局任务"""
    task_id: str
    total_units: int                # 任务可拆解的总单元数
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class SubTaskAssignment:
    """子任务分配"""
    subtask_id: str
    task_id: str
    node_id: str
    units: int
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class ClusterSchedulerConfig:
    """T003 配置"""
    heartbeat_timeout: float = 30.0     # 心跳超时（秒）
    enable_single_node_fallback: bool = True  # 无注册节点时退化为本机单节点
    self_node_id: str = ""              # 本机节点 ID，留空自动生成
    self_capacity: int = 1
    min_units_per_subtask: int = 1
    max_units_per_subtask: int = 1024


# ====== T003: 分布式节点调度 ======

class DistributedNodeScheduler(BaseModule):
    """T003 分布式节点调度

    动态调度：维护节点心跳表，将全局任务拆解分发至空闲节点。
    单机模式下退化为单节点。
    """

    name = "distributed_node_scheduler"
    version = "1.0.0"
    description = "T003 分布式节点调度：节点心跳表 + 全局任务拆解分发"

    def __init__(self, config: Optional[ClusterSchedulerConfig] = None) -> None:
        super().__init__()
        self.config: ClusterSchedulerConfig = config or ClusterSchedulerConfig()
        if not self.config.self_node_id:
            self.config.self_node_id = self._gen_self_node_id()
        self._nodes: Dict[str, NodeHeartbeat] = {}
        self._lock = threading.Lock()
        # 单机退化：注册本机节点
        if self.config.enable_single_node_fallback:
            self._register_self_node()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        if not self.config.self_node_id:
            self.config.self_node_id = self._gen_self_node_id()
        if self.config.enable_single_node_fallback:
            self._register_self_node()
        logger.info("DistributedNodeScheduler 初始化完成，self_node=%s", self.config.self_node_id)

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        with self._lock:
            self._nodes.clear()

    def _health_check(self) -> bool:
        with self._lock:
            return any(n.status == "online" for n in self._nodes.values())

    # ---- 公共方法 ----
    def register_node(self, node_info: NodeInfo) -> bool:
        """注册一个新节点"""
        with self._lock:
            if node_info.node_id in self._nodes:
                return False
            self._nodes[node_info.node_id] = NodeHeartbeat(
                node_id=node_info.node_id,
                status="online",
                load=0.0,
                capacity=node_info.capacity,
                last_seen=time.time(),
                metadata=dict(node_info.metadata),
            )
        logger.info("注册节点 id=%s host=%s capacity=%d",
                    node_info.node_id, node_info.host, node_info.capacity)
        return True

    def update_heartbeat(self, node_id: str, heartbeat: NodeHeartbeat) -> bool:
        """更新节点心跳"""
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is None:
                return False
            existing.status = heartbeat.status or existing.status
            existing.load = float(heartbeat.load)
            existing.capacity = heartbeat.capacity or existing.capacity
            existing.last_seen = time.time()
            if heartbeat.metadata:
                existing.metadata.update(heartbeat.metadata)
        return True

    def dispatch(self, global_task: GlobalTask) -> List[SubTaskAssignment]:
        """将全局任务拆解分发至空闲节点"""
        if not global_task.task_id:
            global_task.task_id = f"task_{uuid.uuid4().hex[:8]}"
        if not global_task.created_at:
            global_task.created_at = time.time()
        with self._lock:
            online_nodes = [
                n for n in self._nodes.values()
                if n.status == "online" and n.capacity > 0
            ]
        if not online_nodes:
            logger.warning("无在线节点，任务 %s 无法分发", global_task.task_id)
            return []
        # 按容量权重分配
        total_capacity = sum(n.capacity for n in online_nodes)
        units_left = max(1, global_task.total_units)
        assignments: List[SubTaskAssignment] = []
        for node in online_nodes:
            if units_left <= 0:
                break
            share = max(1, int(units_left * node.capacity / max(total_capacity, 1)))
            share = min(share, self.config.max_units_per_subtask)
            share = min(share, units_left)
            if share < self.config.min_units_per_subtask:
                continue
            sub_id = f"sub_{uuid.uuid4().hex[:8]}"
            assignments.append(SubTaskAssignment(
                subtask_id=sub_id,
                task_id=global_task.task_id,
                node_id=node.node_id,
                units=share,
                payload=dict(global_task.payload),
                timestamp=time.time(),
            ))
            units_left -= share
            # 提升节点负载估算
            node.load = min(100.0, node.load + share * 1.0)
            if node.load >= 90.0:
                node.status = "busy"
        logger.info(
            "任务 %s 拆解完成：总单元=%d，分配=%d，未分配=%d",
            global_task.task_id, global_task.total_units,
            sum(a.units for a in assignments), units_left,
        )
        return assignments

    def get_node_map(self) -> Dict[str, NodeHeartbeat]:
        """获取节点心跳表快照"""
        with self._lock:
            return {nid: NodeHeartbeat(**n.__dict__) for nid, n in self._nodes.items()}

    def remove_offline_nodes(self, timeout: Optional[float] = None) -> List[str]:
        """移除超时未心跳的节点，返回被移除的节点 ID 列表"""
        if timeout is None:
            timeout = self.config.heartbeat_timeout
        now = time.time()
        removed: List[str] = []
        with self._lock:
            for nid in list(self._nodes.keys()):
                hb = self._nodes[nid]
                # 本机节点不剔除
                if nid == self.config.self_node_id:
                    continue
                if now - hb.last_seen > timeout:
                    self._nodes.pop(nid, None)
                    removed.append(nid)
        if removed:
            logger.info("移除超时节点: %s", removed)
        return removed

    # ---- 内部 ----
    def _gen_self_node_id(self) -> str:
        try:
            hostname = socket.gethostname()
        except Exception:  # pragma: no cover
            hostname = "localhost"
        try:
            pid = os.getpid()
        except Exception:  # pragma: no cover
            pid = 0
        return f"self:{hostname}:{pid}"

    def _register_self_node(self) -> None:
        with self._lock:
            if self.config.self_node_id not in self._nodes:
                self._nodes[self.config.self_node_id] = NodeHeartbeat(
                    node_id=self.config.self_node_id,
                    status="online",
                    load=0.0,
                    capacity=self.config.self_capacity,
                    last_seen=time.time(),
                    metadata={"host": "self", "single_node_fallback": True},
                )
