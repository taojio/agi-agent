"""
cache/cache_reader.py - 缓存读取命中 (T026)

动态调度的缓存读取。未命中返回 None。可与 ShortTermCacheWriter 共享
同一个 InMemoryCacheStore 实例实现读写协同。
"""
import logging
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

from .cache_writer import InMemoryCacheStore

logger = logging.getLogger("agi_agent.cache")


class CacheReader(BaseModule):
    """缓存读取器 (T026)

    绑定一个 InMemoryCacheStore（默认自带）或共享 writer 的 store。
    提供 get / exists / get_context / mget 方法，未命中返回 None。
    """

    name = "cache_reader"
    version = "1.0.0"
    description = "缓存读取命中 (T026)"

    def __init__(self, store: Optional[InMemoryCacheStore] = None):
        super().__init__()
        self._store = store if store is not None else InMemoryCacheStore()
        self._hits: int = 0
        self._misses: int = 0

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        logger.info("CacheReader 初始化完成 (store_size=%d)", len(self._store))

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return self._store is not None

    # ====== 公共方法 ======
    def bind(self, store: InMemoryCacheStore) -> None:
        """绑定共享存储"""
        self._store = store

    @property
    def store(self) -> InMemoryCacheStore:
        return self._store

    def get(self, key: str) -> Optional[Any]:
        """读取缓存

        Args:
            key: 缓存键

        Returns:
            命中返回值，未命中返回 None
        """
        hit, value = self._store.get(key)
        if hit:
            self._hits += 1
            return value
        self._misses += 1
        return None

    def exists(self, key: str) -> bool:
        """判断缓存是否存在"""
        return self._store.exists(key)

    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """读取会话上下文

        Args:
            session_id: 会话 ID

        Returns:
            上下文字典，未命中返回 None
        """
        value = self.get(f"ctx:{session_id}")
        if isinstance(value, dict):
            return value
        return None

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """批量读取

        Args:
            keys: 缓存键列表

        Returns:
            与 keys 等长的值列表，未命中位置为 None
        """
        return [self.get(k) for k in keys]

    def stats(self) -> Dict[str, int]:
        """返回命中统计"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "store_size": len(self._store),
        }
