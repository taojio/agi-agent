"""
storage/backup.py - 备份管理

提供数据备份、恢复、自动清理等功能
"""
import os
import json
import time
import shutil
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """备份信息"""
    name: str
    path: str
    size: int
    created_at: float
    checksum: str
    metadata: Dict[str, Any]


class BackupManager:
    """备份管理器

    功能：
    - 创建完整备份
    - 备份列表与查询
    - 备份恢复
    - 自动清理旧备份
    - 备份完整性校验
    """

    def __init__(self, backup_dir: str, max_backups: int = 10, max_age_days: int = 30):
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        self.max_age_days = max_age_days
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(
        self,
        source_dir: str,
        name: str = "backup",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[BackupInfo]:
        """创建备份

        Args:
            source_dir: 要备份的目录
            name: 备份名称前缀
            metadata: 附加元数据

        Returns:
            BackupInfo 或 None
        """
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory not found: {source_dir}")
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)

        try:
            if os.path.isdir(source_dir):
                shutil.copytree(source_dir, backup_path)
            else:
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(source_dir, backup_path)

            size = self._calc_dir_size(backup_path)
            checksum = self._calc_dir_checksum(backup_path)

            info = BackupInfo(
                name=backup_name,
                path=backup_path,
                size=size,
                created_at=time.time(),
                checksum=checksum,
                metadata=metadata or {},
            )

            info_path = os.path.join(self.backup_dir, f"{backup_name}.info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": info.name,
                    "path": info.path,
                    "size": info.size,
                    "created_at": info.created_at,
                    "checksum": info.checksum,
                    "metadata": info.metadata,
                }, f, ensure_ascii=False, indent=2)

            self._cleanup_old()
            logger.info(f"Backup created: {backup_name}")
            return info

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path, ignore_errors=True)
            return None

    def restore_backup(self, backup_name: str, target_dir: str) -> bool:
        """恢复备份

        Args:
            backup_name: 备份名称
            target_dir: 恢复目标目录

        Returns:
            是否成功
        """
        backup_path = os.path.join(self.backup_dir, backup_name)
        if not os.path.exists(backup_path):
            logger.warning(f"Backup not found: {backup_name}")
            return False

        try:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)
            logger.info(f"Backup restored: {backup_name} -> {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def list_backups(self) -> List[BackupInfo]:
        """列出所有备份"""
        backups = []
        if not os.path.exists(self.backup_dir):
            return backups

        for fname in os.listdir(self.backup_dir):
            if not fname.endswith(".info.json"):
                continue
            info_path = os.path.join(self.backup_dir, fname)
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                backups.append(BackupInfo(
                    name=data["name"],
                    path=data["path"],
                    size=data["size"],
                    created_at=data["created_at"],
                    checksum=data.get("checksum", ""),
                    metadata=data.get("metadata", {}),
                ))
            except (json.JSONDecodeError, IOError, KeyError):
                continue

        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups

    def get_latest(self) -> Optional[BackupInfo]:
        """获取最新备份"""
        backups = self.list_backups()
        return backups[0] if backups else None

    def verify_backup(self, backup_name: str) -> bool:
        """校验备份完整性"""
        info_path = os.path.join(self.backup_dir, f"{backup_name}.info.json")
        backup_path = os.path.join(self.backup_dir, backup_name)

        if not os.path.exists(info_path) or not os.path.exists(backup_path):
            return False

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored_checksum = data.get("checksum", "")
            actual_checksum = self._calc_dir_checksum(backup_path)
            return stored_checksum == actual_checksum
        except Exception:
            return False

    def delete_backup(self, backup_name: str) -> bool:
        """删除备份"""
        backup_path = os.path.join(self.backup_dir, backup_name)
        info_path = os.path.join(self.backup_dir, f"{backup_name}.info.json")
        deleted = False

        if os.path.exists(backup_path):
            try:
                shutil.rmtree(backup_path, ignore_errors=True)
                deleted = True
            except Exception:
                pass

        if os.path.exists(info_path):
            try:
                os.remove(info_path)
                deleted = True
            except OSError:
                pass

        return deleted

    def _cleanup_old(self) -> None:
        """清理旧备份"""
        backups = self.list_backups()

        if len(backups) > self.max_backups:
            for old in backups[self.max_backups:]:
                self.delete_backup(old.name)

        if self.max_age_days > 0:
            cutoff = time.time() - self.max_age_days * 86400
            for b in backups:
                if b.created_at < cutoff:
                    self.delete_backup(b.name)

    def _calc_dir_size(self, path: str) -> int:
        """计算目录大小"""
        total = 0
        if os.path.isfile(path):
            return os.path.getsize(path)
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def _calc_dir_checksum(self, path: str) -> str:
        """计算目录 checksum"""
        h = hashlib.sha256()
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
            return h.hexdigest()

        files = []
        for root, dirs, fnames in os.walk(path):
            for f in sorted(fnames):
                files.append(os.path.join(root, f))

        for fp in sorted(files):
            rel = os.path.relpath(fp, path)
            h.update(rel.encode("utf-8"))
            try:
                with open(fp, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
            except IOError:
                pass

        return h.hexdigest()
