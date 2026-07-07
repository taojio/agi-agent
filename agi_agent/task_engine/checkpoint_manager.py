"""
checkpoint_manager.py - 检查点管理器

实现断点续跑机制：
- 任务执行过程中遭遇程序重启、设备断电、网络中断、运行报错等中断情况
- 恢复后自动从最近的检查点继续执行，无需从头开始
- 长任务每完成一个子步骤自动生成检查点
- 支持自定义检查点位置
"""
import time
import json
import os
import shutil
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class CheckpointStatus(Enum):
    """检查点状态"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Checkpoint:
    """检查点"""
    checkpoint_id: str
    task_id: str
    step: int
    step_name: str
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "step": self.step,
            "step_name": self.step_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "data": self.data,
            "metadata": self.metadata
        }


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, base_dir: str = None):
        """
        初始化检查点管理器

        Args:
            base_dir: 检查点存储基础目录
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser("~"), ".agi_checkpoints")
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self._checkpoints: Dict[str, Checkpoint] = {}
        self._task_checkpoints: Dict[str, List[str]] = {}

    def create_checkpoint(self, task_id: str, step: int, step_name: str,
                          data: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        创建检查点

        Args:
            task_id: 任务ID
            step: 步骤序号
            step_name: 步骤名称
            data: 步骤数据
            metadata: 元数据

        Returns:
            检查点对象
        """
        checkpoint_id = f"chk_{int(time.time() * 1000)}_{step}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            step=step,
            step_name=step_name,
            data=data or {},
            metadata=metadata or {}
        )

        self._checkpoints[checkpoint_id] = checkpoint
        if task_id not in self._task_checkpoints:
            self._task_checkpoints[task_id] = []
        self._task_checkpoints[task_id].append(checkpoint_id)

        self._save_checkpoint(checkpoint)
        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """保存检查点到文件"""
        task_dir = os.path.join(self.base_dir, checkpoint.task_id)
        os.makedirs(task_dir, exist_ok=True)

        filepath = os.path.join(task_dir, f"{checkpoint.checkpoint_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        latest_file = os.path.join(task_dir, "latest.json")
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """获取检查点"""
        return self._checkpoints.get(checkpoint_id)

    def get_latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """获取任务的最新检查点"""
        checkpoints = self._task_checkpoints.get(task_id, [])
        if not checkpoints:
            return self._load_latest_checkpoint(task_id)

        latest_id = checkpoints[-1]
        return self._checkpoints.get(latest_id)

    def _load_latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """从文件加载最新检查点"""
        latest_file = os.path.join(self.base_dir, task_id, "latest.json")
        if os.path.isfile(latest_file):
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = Checkpoint(
                checkpoint_id=data["checkpoint_id"],
                task_id=data["task_id"],
                step=data["step"],
                step_name=data["step_name"],
                status=CheckpointStatus(data["status"]),
                created_at=data["created_at"],
                data=data.get("data", {}),
                metadata=data.get("metadata", {})
            )
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            if task_id not in self._task_checkpoints:
                self._task_checkpoints[task_id] = []
            self._task_checkpoints[task_id].append(checkpoint.checkpoint_id)
            return checkpoint
        return None

    def get_checkpoints_for_task(self, task_id: str) -> List[Checkpoint]:
        """获取任务的所有检查点"""
        checkpoint_ids = self._task_checkpoints.get(task_id, [])
        return [self._checkpoints.get(cid) for cid in checkpoint_ids if self._checkpoints.get(cid)]

    def update_checkpoint(self, checkpoint_id: str, data: Dict[str, Any] = None,
                          status: CheckpointStatus = None):
        """更新检查点"""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return

        if data:
            checkpoint.data.update(data)
        if status:
            checkpoint.status = status

        self._save_checkpoint(checkpoint)

    def mark_completed(self, checkpoint_id: str):
        """标记检查点完成"""
        self.update_checkpoint(checkpoint_id, status=CheckpointStatus.COMPLETED)

    def mark_failed(self, checkpoint_id: str, error: str = ""):
        """标记检查点失败"""
        self.update_checkpoint(checkpoint_id, data={"error": error}, status=CheckpointStatus.FAILED)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        checkpoint = self._checkpoints.pop(checkpoint_id, None)
        if checkpoint:
            if checkpoint.task_id in self._task_checkpoints:
                self._task_checkpoints[checkpoint.task_id].remove(checkpoint_id)

            filepath = os.path.join(self.base_dir, checkpoint.task_id, f"{checkpoint_id}.json")
            if os.path.isfile(filepath):
                os.remove(filepath)
            return True
        return False

    def cleanup_task_checkpoints(self, task_id: str):
        """清理任务的所有检查点"""
        task_dir = os.path.join(self.base_dir, task_id)
        if os.path.isdir(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)

        if task_id in self._task_checkpoints:
            for cid in self._task_checkpoints[task_id]:
                self._checkpoints.pop(cid, None)
            del self._task_checkpoints[task_id]

    def restore_from_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        从检查点恢复任务状态

        Args:
            task_id: 任务ID

        Returns:
            检查点数据（包含 step, step_name, data 等）
        """
        checkpoint = self.get_latest_checkpoint(task_id)
        if checkpoint is None:
            return None

        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "step": checkpoint.step,
            "step_name": checkpoint.step_name,
            "data": checkpoint.data,
            "created_at": checkpoint.created_at
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._checkpoints)
        by_status = {}
        for checkpoint in self._checkpoints.values():
            s = checkpoint.status.value
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_checkpoints": total,
            "tasks_with_checkpoints": len(self._task_checkpoints),
            "by_status": by_status
        }
