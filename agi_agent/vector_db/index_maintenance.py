"""
vector_db/index_maintenance.py - 向量索引维护 (T016)

轮询执行向量索引的维护：清理过期向量、重建索引、分片优化、统计信息。
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.vector_db")


@dataclass
class IndexMaintenanceConfig:
    """索引维护配置"""
    default_ttl: float = 86400.0  # 默认 1 天
    optimize_threshold: int = 10000  # 超过该数量触发分片优化建议
    shard_size: int = 5000


class VectorIndexMaintenance(BaseModule):
    """向量索引维护器 (T016)

    绑定一个 VectorStoreWriter 实例，提供 cleanup_expired / rebuild_index /
    optimize_shards / stats 方法，供定时轮询调度调用。
    """

    name = "vector_index_maintenance"
    version = "1.0.0"
    description = "向量索引维护 (T016)"

    def __init__(self, store: Optional[Any] = None, config: Optional[IndexMaintenanceConfig] = None):
        super().__init__()
        self._store = store
        self._cfg = config or IndexMaintenanceConfig()
        self._last_cleanup: float = 0.0
        self._last_rebuild: float = 0.0

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._store is None:
            logger.info("VectorIndexMaintenance 未绑定 store，维护操作为 no-op")
        else:
            logger.info("VectorIndexMaintenance 绑定 store 后端: %s", getattr(self._store, "backend", "unknown"))

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def bind(self, store: Any) -> None:
        """绑定向量库写入器"""
        self._store = store

    def cleanup_expired(self, ttl: Optional[float] = None) -> int:
        """清理过期向量

        Args:
            ttl: 过期阈值（秒），未指定时使用默认配置

        Returns:
            int: 清理的记录数量
        """
        if self._store is None:
            return 0
        ttl_val = self._cfg.default_ttl if ttl is None else float(ttl)
        now = time.time()
        cutoff = now - ttl_val

        backend = getattr(self._store, "backend", "numpy")
        if backend == "chroma" and self._store._chroma_collection is not None:
            try:
                payload = self._store._chroma_collection.get(
                    where={"timestamp": {"$lt": cutoff}} if False else None,
                    include=["metadatas"],
                )
                ids = payload.get("ids", []) or []
                expired_ids: List[str] = []
                metas = payload.get("metadatas", []) or []
                for vid, meta in zip(ids, metas):
                    ts = float((meta or {}).get("timestamp", now))
                    if ts < cutoff:
                        expired_ids.append(str(vid))
                if expired_ids:
                    self._store.remove_ids(expired_ids)
                self._last_cleanup = now
                return len(expired_ids)
            except Exception as e:  # noqa: BLE001
                logger.warning("chromadb 过期清理失败，降级内存: %s", e)

        # numpy 后端：遍历记录
        expired: List[str] = []
        for rec in self._store.get_records():
            if rec.created_at < cutoff:
                expired.append(rec.vector_id)
        removed = self._store.remove_ids(expired) if expired else 0
        if removed:
            try:
                self._store.flush()
            except Exception:  # noqa: BLE001
                pass
        self._last_cleanup = now
        return removed

    def rebuild_index(self) -> bool:
        """重建索引

        Returns:
            bool: 是否成功重建
        """
        if self._store is None:
            return False
        backend = getattr(self._store, "backend", "numpy")
        try:
            if backend == "faiss" and self._store._faiss_index is not None:
                # 重建 faiss 索引：清空后重新灌入
                dim = self._store.dim
                records = self._store.get_records()
                self._store._faiss_index.reset()
                if records:
                    mat = __import__("numpy").vstack([r.vector for r in records]).astype("float32")
                    self._store._faiss_index.add(mat)
            elif backend == "numpy":
                self._store.rebuild_matrix()
            elif backend == "chroma":
                # chromadb 自行管理索引，无需显式重建
                pass
            self._last_rebuild = time.time()
            logger.info("索引重建完成 (backend=%s)", backend)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("索引重建失败: %s", e)
            return False

    def optimize_shards(self) -> Dict[str, Any]:
        """分片优化

        Returns:
            dict: 优化结果统计
        """
        if self._store is None:
            return {"optimized": False, "reason": "no_store"}
        count = self._store.count()
        shards = max(1, (count + self._cfg.shard_size - 1) // self._cfg.shard_size)
        optimized = count > self._cfg.optimize_threshold
        # numpy 后端：重建矩阵以紧凑内存
        if getattr(self._store, "backend", "numpy") == "numpy":
            self._store.rebuild_matrix()
        return {
            "optimized": optimized,
            "total_count": count,
            "estimated_shards": shards,
            "shard_size": self._cfg.shard_size,
        }

    def stats(self) -> Dict[str, Any]:
        """返回索引统计信息

        Returns:
            dict: 统计信息
        """
        if self._store is None:
            return {"count": 0, "backend": "none", "dim": 0}
        return {
            "count": self._store.count(),
            "backend": getattr(self._store, "backend", "unknown"),
            "dim": self._store.dim,
            "last_cleanup": self._last_cleanup,
            "last_rebuild": self._last_rebuild,
            "last_cleanup_ago": max(0.0, time.time() - self._last_cleanup) if self._last_cleanup else None,
        }
