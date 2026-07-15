"""
working_memory/task_state.py - 任务状态保存

实现任务清单：
- T046: 当前任务状态保存（动态调度）—— TaskStateSaver

设计要点：
- 内部按 task_id 维护快照栈（保留最近 N 个，默认 10），支持回退
- 内存实现 + 可选 pickle 持久化到 ./data/task_states/
- 配置统一使用 dataclass，参数有默认值
- 完整类型注解，中文 docstring
- 模块可独立 import，无副作用实例化
"""
from __future__ import annotations

import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core import BaseModule

logger = logging.getLogger("agi_agent.working_memory")


# ========================================================
# 数据结构
# ========================================================
@dataclass
class TaskState:
    """任务运行状态（T046）

    实时记录任务进度、临时变量、工具调用中间结果、未完成步骤。
    """

    task_id: str
    status: str = "pending"  # pending / running / paused / completed / failed
    progress: float = 0.0  # 0.0 - 1.0
    current_step: int = 0
    steps_total: int = 0
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    temp_vars: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    checkpoint_id: Optional[str] = None


@dataclass
class TaskSnapshot:
    """任务快照（T046）"""

    snapshot_id: str
    task_id: str
    state: TaskState
    timestamp: float


# ========================================================
# 配置
# ========================================================
@dataclass
class TaskStateConfig:
    """任务状态保存器配置（T046）"""

    max_snapshots: int = 10  # 每个 task 保留最近 N 个快照
    enable_persistence: bool = False  # 是否启用 pickle 持久化
    persist_dir: str = "./data/task_states"  # 持久化目录


# ========================================================
# T046: TaskStateSaver
# ========================================================
class TaskStateSaver(BaseModule):
    """任务状态保存器（T046 - 动态调度）

    任务 T046：实时记录任务进度、临时变量、工具调用中间结果、未完成步骤，
    生成状态快照缓存，支持断点续跑 / 异常恢复。

    触发机制：动态调度——在关键步骤调用 ``save_snapshot`` / ``update_progress`` 触发快照。

    存储实现：
        - 内存：``dict[task_id -> list[TaskSnapshot]]``（栈结构，末尾为最新）
        - 可选 pickle 持久化到 ``./data/task_states/``（``enable_persistence=True``）
    每个 task 保留最近 ``max_snapshots`` 个快照，支持回退与断点续跑。
    """

    name = "task_state_saver"
    version = "1.0.0"
    description = "T046 当前任务状态保存（动态调度）"

    def __init__(self, config: Optional[TaskStateConfig] = None) -> None:
        super().__init__()
        self.config: TaskStateConfig = config or TaskStateConfig()
        # task_id -> list[TaskSnapshot]（栈结构，末尾为最新）
        self._snapshots: Dict[str, List[TaskSnapshot]] = {}
        # task_id -> 当前最新 state（便于快速访问）
        self._latest: Dict[str, TaskState] = {}
        if self.config.enable_persistence:
            try:
                os.makedirs(self.config.persist_dir, exist_ok=True)
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning("persist_dir create failed: %s", e)

    # ---------- 生命周期 ----------
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    def _shutdown(self) -> None:
        try:
            self._snapshots.clear()
            self._latest.clear()
        except Exception:
            pass

    # ---------- 持久化 ----------
    def _persist_snapshot(self, snap: TaskSnapshot) -> None:
        if not self.config.enable_persistence:
            return
        try:
            path = os.path.join(
                self.config.persist_dir, f"{snap.task_id}_{snap.snapshot_id}.pkl"
            )
            with open(path, "wb") as f:
                pickle.dump(snap, f)
        except Exception as e:  # pragma: no cover - 环境相关
            logger.warning("persist snapshot failed: %s", e)

    def _load_persisted(self, task_id: str) -> List[TaskSnapshot]:
        if not self.config.enable_persistence:
            return []
        snaps: List[TaskSnapshot] = []
        try:
            for fn in os.listdir(self.config.persist_dir):
                if fn.startswith(f"{task_id}_") and fn.endswith(".pkl"):
                    path = os.path.join(self.config.persist_dir, fn)
                    try:
                        with open(path, "rb") as f:
                            snap = pickle.load(f)
                        snaps.append(snap)
                    except Exception:
                        continue
            snaps.sort(key=lambda s: s.timestamp)
        except Exception as e:  # pragma: no cover - 环境相关
            logger.warning("load persisted failed: %s", e)
        return snaps

    def _remove_persisted(self, task_id: str) -> None:
        if not self.config.enable_persistence:
            return
        try:
            for fn in list(os.listdir(self.config.persist_dir)):
                if fn.startswith(f"{task_id}_") and fn.endswith(".pkl"):
                    path = os.path.join(self.config.persist_dir, fn)
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover - 环境相关
            logger.warning("remove persisted failed: %s", e)

    # ---------- 核心方法 ----------
    def save_snapshot(self, task_id: str, state: TaskState) -> str:
        """保存任务状态快照

        Args:
            task_id: 任务 ID
            state: 任务状态

        Returns:
            生成的 snapshot_id
        """
        state.task_id = task_id
        state.updated_at = time.time()
        snapshot_id = f"snap_{uuid.uuid4().hex[:16]}"
        state.checkpoint_id = snapshot_id
        snap = TaskSnapshot(
            snapshot_id=snapshot_id,
            task_id=task_id,
            state=state,
            timestamp=time.time(),
        )
        stack = self._snapshots.setdefault(task_id, [])
        stack.append(snap)
        # 保留最近 N 个快照
        while len(stack) > self.config.max_snapshots:
            stack.pop(0)
        self._latest[task_id] = state
        self._persist_snapshot(snap)
        logger.debug(
            "save_snapshot task=%s snap=%s stack=%d",
            task_id,
            snapshot_id,
            len(stack),
        )
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Optional[TaskState]:
        """按 snapshot_id 加载快照

        Args:
            snapshot_id: 快照 ID

        Returns:
            对应的 TaskState；不存在则返回 None
        """
        for stack in self._snapshots.values():
            for snap in stack:
                if snap.snapshot_id == snapshot_id:
                    return snap.state
        return None

    def get_latest(self, task_id: str) -> Optional[TaskState]:
        """获取任务最新状态

        Args:
            task_id: 任务 ID

        Returns:
            最新 TaskState；不存在则返回 None
        """
        return self._latest.get(task_id)

    def list_snapshots(self, task_id: str) -> List[str]:
        """列出任务所有快照 ID（按时间正序）

        Args:
            task_id: 任务 ID

        Returns:
            snapshot_id 列表
        """
        return [s.snapshot_id for s in self._snapshots.get(task_id, [])]

    def update_progress(self, task_id: str, step: int, result: Any) -> None:
        """更新任务进度并自动保存快照

        Args:
            task_id: 任务 ID
            step: 当前步骤序号
            result: 当前步骤结果（写入 ``intermediate_results["step_{step}"]``）
        """
        state = self._latest.get(task_id)
        if state is None:
            state = TaskState(task_id=task_id, status="running")
        state.current_step = step
        state.intermediate_results[f"step_{step}"] = result
        if state.steps_total > 0:
            state.progress = min(1.0, step / state.steps_total)
        state.status = "running"
        self.save_snapshot(task_id, state)

    def mark_completed(self, task_id: str) -> None:
        """标记任务完成

        Args:
            task_id: 任务 ID
        """
        state = self._latest.get(task_id)
        if state is None:
            state = TaskState(task_id=task_id)
        state.status = "completed"
        state.progress = 1.0
        self.save_snapshot(task_id, state)

    def resume(self, task_id: str) -> Optional[TaskState]:
        """断点续跑：返回最新状态以供恢复

        若内存中无该任务状态，会尝试从持久化加载。

        Args:
            task_id: 任务 ID

        Returns:
            可恢复的 TaskState；不存在则返回 None
        """
        state = self._latest.get(task_id)
        if state is None:
            snaps = self._load_persisted(task_id)
            if snaps:
                self._snapshots[task_id] = snaps
                state = snaps[-1].state
                self._latest[task_id] = state
        if state is not None and state.status in ("paused", "running", "failed"):
            state.status = "running"
        return state

    def clear_task(self, task_id: str) -> int:
        """清理任务所有快照（供 ShortTermMemoryCleaner 协作调用）

        Args:
            task_id: 任务 ID

        Returns:
            清理的快照数
        """
        stack = self._snapshots.pop(task_id, [])
        self._latest.pop(task_id, None)
        self._remove_persisted(task_id)
        return len(stack)
