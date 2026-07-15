"""
observability/log_archive.py - 日志归档清理 (T020)

定时轮询执行日志的归档：短期压缩、长期冷迁移、超期清理。
"""
import gzip
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.observability")


@dataclass
class LogArchiveConfig:
    """日志归档配置"""
    archive_dir: str = "./logs/archive"
    cold_storage_dir: str = "./logs/cold"
    compress_suffix: str = ".gz"
    default_retention_days: int = 30


class LogArchive(BaseModule):
    """日志归档清理器 (T020)

    提供 archive_older_than / compress / migrate_cold / cleanup_expired
    方法，供定时轮询调度调用。
    """

    name = "log_archive"
    version = "1.0.0"
    description = "日志归档清理 (T020)"

    def __init__(self, config: Optional[LogArchiveConfig] = None):
        super().__init__()
        self._cfg = config or LogArchiveConfig()
        self._last_run: float = 0.0
        self._stats: Dict[str, int] = {"archived": 0, "compressed": 0, "migrated": 0, "cleaned": 0}

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        os.makedirs(self._cfg.archive_dir, exist_ok=True)
        os.makedirs(self._cfg.cold_storage_dir, exist_ok=True)
        logger.info("LogArchive 初始化完成 (archive=%s, cold=%s)", self._cfg.archive_dir, self._cfg.cold_storage_dir)

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def archive_older_than(self, days: int, src_dir: str = "./logs") -> Dict[str, Any]:
        """将超过指定天数的日志文件归档

        Args:
            days: 天数阈值
            src_dir: 源日志目录

        Returns:
            dict: 归档统计
        """
        cutoff = time.time() - days * 86400.0
        archived_files: List[str] = []
        os.makedirs(self._cfg.archive_dir, exist_ok=True)
        for fname in self._list_log_files(src_dir):
            fpath = os.path.join(src_dir, fname)
            try:
                if os.path.getmtime(fpath) < cutoff:
                    dest = os.path.join(self._cfg.archive_dir, fname)
                    dest = self._unique_path(dest)
                    shutil.move(fpath, dest)
                    archived_files.append(dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("归档文件 %s 失败: %s", fpath, e)
        self._stats["archived"] += len(archived_files)
        self._last_run = time.time()
        return {"archived": len(archived_files), "files": archived_files}

    def compress(self, dir: str) -> Dict[str, Any]:
        """压缩指定目录下的日志文件

        Args:
            dir: 目标目录

        Returns:
            dict: 压缩统计
        """
        compressed: List[str] = []
        for fname in self._list_log_files(dir):
            fpath = os.path.join(dir, fname)
            if fpath.endswith(self._cfg.compress_suffix):
                continue
            dest = fpath + self._cfg.compress_suffix
            dest = self._unique_path(dest)
            try:
                with open(fpath, "rb") as src, gzip.open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(fpath)
                compressed.append(dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("压缩文件 %s 失败: %s", fpath, e)
        self._stats["compressed"] += len(compressed)
        self._last_run = time.time()
        return {"compressed": len(compressed), "files": compressed}

    def migrate_cold(self, dir: str) -> Dict[str, Any]:
        """将指定目录下的日志迁移到冷存储

        Args:
            dir: 源目录（默认归档目录）

        Returns:
            dict: 迁移统计
        """
        src_dir = dir or self._cfg.archive_dir
        os.makedirs(self._cfg.cold_storage_dir, exist_ok=True)
        migrated: List[str] = []
        for fname in self._list_log_files(src_dir):
            fpath = os.path.join(src_dir, fname)
            dest = os.path.join(self._cfg.cold_storage_dir, fname)
            dest = self._unique_path(dest)
            try:
                shutil.move(fpath, dest)
                migrated.append(dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("冷迁移文件 %s 失败: %s", fpath, e)
        self._stats["migrated"] += len(migrated)
        self._last_run = time.time()
        return {"migrated": len(migrated), "files": migrated}

    def cleanup_expired(self, retention_days: Optional[int] = None) -> Dict[str, Any]:
        """清理超过保留期的日志

        Args:
            retention_days: 保留天数，未指定时使用默认配置

        Returns:
            dict: 清理统计
        """
        days = self._cfg.default_retention_days if retention_days is None else int(retention_days)
        cutoff = time.time() - days * 86400.0
        cleaned: List[str] = []
        for target_dir in (self._cfg.archive_dir, self._cfg.cold_storage_dir, "./logs"):
            if not os.path.isdir(target_dir):
                continue
            for fname in self._list_log_files(target_dir):
                fpath = os.path.join(target_dir, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        cleaned.append(fpath)
                except Exception as e:  # noqa: BLE001
                    logger.warning("清理文件 %s 失败: %s", fpath, e)
        self._stats["cleaned"] += len(cleaned)
        self._last_run = time.time()
        return {"cleaned": len(cleaned), "files": cleaned}

    def stats(self) -> Dict[str, Any]:
        """返回归档统计"""
        return {
            "stats": dict(self._stats),
            "last_run": self._last_run,
            "archive_dir": self._cfg.archive_dir,
            "cold_storage_dir": self._cfg.cold_storage_dir,
        }

    # ====== 内部 ======
    @staticmethod
    def _list_log_files(dir: str) -> List[str]:
        if not os.path.isdir(dir):
            return []
        result: List[str] = []
        try:
            for entry in os.listdir(dir):
                fpath = os.path.join(dir, entry)
                if os.path.isfile(fpath) and entry.lower().endswith((".log", ".jsonl", ".txt", ".gz")):
                    result.append(entry)
        except Exception:  # noqa: BLE001
            return []
        return result

    @staticmethod
    def _unique_path(path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(f"{base}.{i}{ext}"):
            i += 1
        return f"{base}.{i}{ext}"
