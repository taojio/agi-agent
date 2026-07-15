"""
cache/cache_sync.py - 缓存一致性同步 (T028)

轮询执行的多实例缓存差异检测与同步。单实例降级为 no-op。
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agi_agent.core import BaseModule

from .cache_writer import InMemoryCacheStore

logger = logging.getLogger("agi_agent.cache")


@dataclass
class CacheSyncConfig:
    """缓存同步配置"""
    merge_strategy: str = "latest"  # latest | union


class CacheSync(BaseModule):
    """缓存一致性同步器 (T028)

    对多个 InMemoryCacheStore 实例进行差异检测与同步。单实例场景降级为
    no-op。提供 diff / sync / merge 方法。
    """

    name = "cache_sync"
    version = "1.0.0"
    description = "缓存一致性同步 (T028)"

    def __init__(self, config: Optional[CacheSyncConfig] = None):
        super().__init__()
        self._cfg = config or CacheSyncConfig()
        self._last_sync: Dict[str, Any] = {"synced": 0, "diff_keys": 0}

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        logger.info("CacheSync 初始化完成 (strategy=%s)", self._cfg.merge_strategy)

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def diff(
        self,
        node_a: InMemoryCacheStore,
        node_b: InMemoryCacheStore,
    ) -> Dict[str, Any]:
        """检测两个缓存实例的差异

        Args:
            node_a: 实例 A
            node_b: 实例 B

        Returns:
            dict: 差异信息（仅 A 有 / 仅 B 有 / 双方均有但不同）
        """
        a_items = {k: e for k, e in node_a.items()}
        b_items = {k: e for k, e in node_b.items()}
        a_keys = set(a_items.keys())
        b_keys = set(b_items.keys())
        only_a = sorted(a_keys - b_keys)
        only_b = sorted(b_keys - a_keys)
        common = a_keys & b_keys
        differing: List[str] = []
        for k in common:
            ea = a_items[k]
            eb = b_items[k]
            if ea.value != eb.value:
                differing.append(k)
        return {
            "only_in_a": only_a,
            "only_in_b": only_b,
            "differing": differing,
            "a_size": len(a_keys),
            "b_size": len(b_keys),
        }

    def sync(self, instances: Iterable[InMemoryCacheStore]) -> Dict[str, Any]:
        """同步多个缓存实例

        将所有实例合并为一致状态（latest 策略取最近写入，union 策略保留全部）。

        Args:
            instances: 缓存实例可迭代对象

        Returns:
            dict: 同步统计
        """
        stores = list(instances)
        if len(stores) <= 1:
            self._last_sync = {"synced": 0, "diff_keys": 0, "nodes": len(stores), "note": "single_node_noop"}
            return self._last_sync

        # 收集全部键及最新条目
        merged: Dict[str, Any] = {}
        for store in stores:
            for k, entry in store.items():
                if k not in merged:
                    merged[k] = entry
                else:
                    if self._cfg.merge_strategy == "latest":
                        if entry.last_access >= merged[k].last_access:
                            merged[k] = entry
                    else:  # union：保留已有（首次出现）
                        pass
        # 统计同步前差异键数
        all_keys = set(merged.keys())
        diff_keys = 0
        for store in stores:
            diff_keys += len(all_keys - set(store.keys()))
        # 写回每个实例
        synced = 0
        for store in stores:
            for k, entry in merged.items():
                existing = store.peek(k)
                if existing is None or existing.value != entry.value:
                    ttl = None
                    if entry.expiry is not None:
                        ttl = max(0.0, entry.expiry - entry.last_access)
                    store.set(k, entry.value, ttl)
                    synced += 1
        self._last_sync = {"synced": synced, "diff_keys": diff_keys, "nodes": len(stores)}
        logger.info("缓存同步完成: 同步 %d 项，差异键 %d", synced, diff_keys)
        return self._last_sync

    def merge(self, items: Iterable[Tuple[str, Any]], target: InMemoryCacheStore, ttl: Optional[float] = None) -> int:
        """将若干 (key, value) 合并写入目标实例

        Args:
            items: 待合并条目
            target: 目标缓存实例
            ttl: 过期时间（秒）

        Returns:
            int: 写入数量
        """
        count = 0
        for k, v in items:
            target.set(k, v, ttl)
            count += 1
        return count

    @property
    def last_sync(self) -> Dict[str, Any]:
        return dict(self._last_sync)
