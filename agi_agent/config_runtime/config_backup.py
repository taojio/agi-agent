"""
config_runtime/config_backup.py - 配置持久化备份 (T024)

事件触发的配置备份：生成快照、版本号、回滚、恢复。
持久化到 ./data/config_snapshots/。
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.config_runtime")


@dataclass
class ConfigBackupConfig:
    """配置备份配置"""
    snapshot_dir: str = "./data/config_snapshots"
    max_versions: int = 50  # 最多保留版本数


class ConfigBackup(BaseModule):
    """配置持久化备份器 (T024)

    提供 snapshot / list_versions / rollback / restore 方法。
    每个快照以 JSON 文件持久化，包含版本号、时间戳、配置内容。
    """

    name = "config_backup"
    version = "1.0.0"
    description = "配置持久化备份 (T024)"

    def __init__(self, config: Optional[ConfigBackupConfig] = None):
        super().__init__()
        self._cfg = config or ConfigBackupConfig()
        self._current_version: Optional[str] = None

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        os.makedirs(self._cfg.snapshot_dir, exist_ok=True)
        logger.info("ConfigBackup 初始化完成 (dir=%s)", self._cfg.snapshot_dir)

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return os.path.isdir(self._cfg.snapshot_dir)

    # ====== 公共方法 ======
    def snapshot(self, config: Any) -> str:
        """生成配置快照

        Args:
            config: AgentConfig 或 dict

        Returns:
            str: 版本号 version_id
        """
        os.makedirs(self._cfg.snapshot_dir, exist_ok=True)
        version_id = f"v_{int(time.time() * 1000)}_{os.getpid()}"
        data = {
            "version_id": version_id,
            "timestamp": time.time(),
            "config": self._as_dict(config),
        }
        path = os.path.join(self._cfg.snapshot_dir, f"{version_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.error("写入配置快照失败: %s", e)
            raise
        self._current_version = version_id
        self._enforce_max_versions()
        logger.info("配置快照已生成: %s", version_id)
        return version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出全部快照版本

        Returns:
            List[dict]: 版本信息列表（按时间降序）
        """
        versions: List[Dict[str, Any]] = []
        if not os.path.isdir(self._cfg.snapshot_dir):
            return versions
        for fname in os.listdir(self._cfg.snapshot_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._cfg.snapshot_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                versions.append(
                    {
                        "version_id": data.get("version_id", fname[:-5]),
                        "timestamp": data.get("timestamp", 0.0),
                        "path": path,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        versions.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
        return versions

    def rollback(self, version_id: str) -> Dict[str, Any]:
        """回滚到指定版本（等价于 restore）

        Args:
            version_id: 目标版本号

        Returns:
            dict: 恢复的配置字典
        """
        cfg = self.restore(version_id)
        self._current_version = version_id
        logger.info("配置已回滚到 %s", version_id)
        return cfg

    def restore(self, version_id: str) -> Dict[str, Any]:
        """从指定版本恢复配置

        Args:
            version_id: 目标版本号

        Returns:
            dict: 恢复的配置字典
        """
        path = os.path.join(self._cfg.snapshot_dir, f"{version_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置快照不存在: {version_id}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("config", {})

    @property
    def current_version(self) -> Optional[str]:
        return self._current_version

    # ====== 内部 ======
    def _enforce_max_versions(self) -> None:
        versions = self.list_versions()
        if len(versions) <= self._cfg.max_versions:
            return
        to_remove = versions[self._cfg.max_versions :]
        for item in to_remove:
            try:
                os.remove(item["path"])
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _as_dict(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(config, "__dict__"):
            return {k: v for k, v in vars(config).items() if not k.startswith("_")}
        return {"value": str(config)}
