"""
cache/cache_eviction.py - 缓存过期淘汰 (T027)

轮询执行的缓存淘汰：基于 ttl 过期清理 + LRU 容量淘汰。
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agi_agent.core import BaseModule

from .cache_writer import InMemoryCacheStore

logger = logging.getLogger("agi_agent.cache")


@dataclass
class CacheEvictionConfig:
    """缓存淘汰配置"""
    default_max_size: int = 10000  # LRU 淘汰目标容量
    cycle_purge_expired: bool = True
    cycle_evict_lru: bool = True


class CacheEviction(BaseModule):
    """缓存过期淘汰器 (T027)

    绑定一个 InMemoryCacheStore（默认自带），提供 evict_expired /
    evict_lru / run_cycle 方法，供定时轮询调度调用。
    """

    name = "cache_eviction"
    version = "1.0.0"
    description = "缓存过期淘汰 (T027)"

    def __init__(self, store: Optional[InMemoryCacheStore] = None, config: Optional[CacheEvictionConfig] = None):
        super().__init__()
        self._store = store if store is not None else InMemoryCacheStore()
        self._cfg = config or CacheEvictionConfig()
        self._last_cycle: Dict[str, Any] = {}

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        logger.info("CacheEviction 初始化完成 (max_size=%d)", self._cfg.default_max_size)

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def bind(self, store: InMemoryCacheStore) -> None:
        """绑定共享存储"""
        self._store = store

    @property
    def store(self) -> InMemoryCacheStore:
        return self._store

    def evict_expired(self) -> int:
        """清理已过期条目

        Returns:
            int: 清理数量
        """
        removed = self._store.purge_expired()
        if removed:
            logger.info("过期清理淘汰 %d 条", removed)
        return removed

    def evict_lru(self, max_size: Optional[int] = None) -> int:
        """按 LRU 淘汰至 max_size

        Args:
            max_size: 目标容量，未指定时使用默认配置

        Returns:
            int: 淘汰数量
        """
        target = self._cfg.default_max_size if max_size is None else int(max_size)
        removed = self._store.evict_lru(target)
        if removed:
            logger.info("LRU 淘汰 %d 条至容量 %d", removed, target)
        return removed

    def run_cycle(self) -> Dict[str, Any]:
        """执行一次完整淘汰周期（过期清理 + LRU 淘汰）

        Returns:
            dict: 本次周期统计
        """
        expired = self.evict_expired() if self._cfg.cycle_purge_expired else 0
        lru = self.evict_lru() if self._cfg.cycle_evict_lru else 0
        self._last_cycle = {
            "expired": expired,
            "lru": lru,
            "remaining": len(self._store),
        }
        return self._last_cycle

    @property
    def last_cycle(self) -> Dict[str, Any]:
        return dict(self._last_cycle)
