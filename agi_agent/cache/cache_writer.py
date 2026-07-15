"""
cache/cache_writer.py - 短时上下文缓存写入 (T025)

动态调度的短时上下文缓存写入：对话轮次、临时计算结果、任务进度、
中间变量，生成唯一 cache_key。redis 可选导入，降级为内存 dict + 过期时间。

本模块定义共享的 InMemoryCacheStore，供 CacheReader / CacheEviction /
CacheSync 复用以实现多模块协同。
"""
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.cache")

# 可选依赖：redis
try:  # pragma: no cover - 环境相关
    import redis  # type: ignore

    _HAS_REDIS = True
except Exception:  # noqa: BLE001
    redis = None  # type: ignore
    _HAS_REDIS = False


@dataclass
class CacheEntry:
    """单条缓存条目"""
    value: Any
    expiry: Optional[float]  # 绝对过期时间戳，None 表示永不过期
    last_access: float
    created_at: float
    access_count: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expiry is None:
            return False
        return (now if now is not None else time.time()) >= self.expiry


class InMemoryCacheStore:
    """内存缓存存储（线程安全），作为 redis 不可用时的降级方案

    提供与 CacheReader / CacheEviction / CacheSync 协作的统一接口。
    """

    def __init__(self) -> None:
        self._data: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        now = time.time()
        expiry = (now + float(ttl)) if ttl is not None else None
        with self._lock:
            existing = self._data.get(key)
            self._data[key] = CacheEntry(
                value=value,
                expiry=expiry,
                last_access=now,
                created_at=existing.created_at if existing else now,
                access_count=existing.access_count if existing else 0,
            )

    def get(self, key: str, update_access: bool = True) -> Tuple[bool, Any]:
        """获取值，返回 (hit, value)"""
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False, None
            if entry.is_expired(now):
                del self._data[key]
                return False, None
            if update_access:
                entry.last_access = now
                entry.access_count += 1
            return True, entry.value

    def peek(self, key: str) -> Optional[CacheEntry]:
        """查看条目（不更新访问统计）"""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._data[key]
                return None
            return entry

    def exists(self, key: str) -> bool:
        return self.peek(key) is not None

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def keys(self) -> List[str]:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            return list(self._data.keys())

    def items(self) -> List[Tuple[str, CacheEntry]]:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            return [(k, e) for k, e in self._data.items()]

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def __len__(self) -> int:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            return len(self._data)

    def _purge_expired(self, now: float) -> int:
        expired = [k for k, e in self._data.items() if e.is_expired(now)]
        for k in expired:
            del self._data[k]
        return len(expired)

    def purge_expired(self) -> int:
        """清理过期条目，返回清理数量"""
        with self._lock:
            return self._purge_expired(time.time())

    def evict_lru(self, max_size: int) -> int:
        """按 LRU 淘汰至 max_size，返回淘汰数量"""
        with self._lock:
            if len(self._data) <= max_size:
                return 0
            # 按 last_access 升序淘汰
            ordered = sorted(self._data.items(), key=lambda kv: kv[1].last_access)
            remove_count = len(self._data) - max_size
            for k, _ in ordered[:remove_count]:
                del self._data[k]
            return remove_count


@dataclass
class CacheWriterConfig:
    """缓存写入配置"""
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: Optional[float] = 3600.0  # 默认 1 小时
    redis_prefix: str = "agi:cache:"


class ShortTermCacheWriter(BaseModule):
    """短时上下文缓存写入器 (T025)

    优先使用 redis；不可用或未配置时降级为 InMemoryCacheStore。
    提供 write / set_context / gen_key 方法，并暴露 .store 供协同模块使用。
    """

    name = "short_term_cache_writer"
    version = "1.0.0"
    description = "短时上下文缓存写入 (T025)"

    def __init__(self, config: Optional[CacheWriterConfig] = None, store: Optional[InMemoryCacheStore] = None):
        super().__init__()
        self._cfg = config or CacheWriterConfig()
        self._store = store if store is not None else InMemoryCacheStore()
        self._redis_client = None
        self._backend: str = "memory"

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._cfg.use_redis and _HAS_REDIS:
            try:
                self._redis_client = redis.from_url(self._cfg.redis_url)  # type: ignore[union-attr]
                self._redis_client.ping()
                self._backend = "redis"
                logger.info("ShortTermCacheWriter 使用 redis 后端: %s", self._cfg.redis_url)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("redis 连接失败，降级内存: %s", e)
                self._redis_client = None
        self._backend = "memory"
        logger.info("ShortTermCacheWriter 使用内存后端")

    def _shutdown(self) -> None:
        if self._redis_client is not None:
            try:
                self._redis_client.close()
            except Exception:  # noqa: BLE001
                pass
            self._redis_client = None

    def _health_check(self) -> bool:
        if self._backend == "redis" and self._redis_client is not None:
            try:
                self._redis_client.ping()
                return True
            except Exception:  # noqa: BLE001
                return False
        return True

    # ====== 公共方法 ======
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def store(self) -> InMemoryCacheStore:
        """返回内存存储（redis 后端时为镜像/降级存储）"""
        return self._store

    def gen_key(self, *parts: Any) -> str:
        """根据若干组成部分生成唯一 cache_key

        Args:
            *parts: 组成部分（任意可序列化对象）

        Returns:
            str: 32 位十六进制 cache_key
        """
        raw = "|".join(str(p) for p in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def write(self, key: str, value: Any, ttl: Optional[float] = None) -> str:
        """写入一条缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None 使用默认值，0 表示永不过期

        Returns:
            str: 写入的 cache_key
        """
        effective_ttl = self._cfg.default_ttl if ttl is None else ttl
        if effective_ttl is not None and effective_ttl <= 0:
            effective_ttl = None
        # 内存存储始终维护（便于 reader/eviction 协同与降级）
        self._store.set(key, value, effective_ttl)
        if self._backend == "redis" and self._redis_client is not None:
            try:
                rkey = self._cfg.redis_prefix + key
                import json as _json

                payload = _json.dumps(value, ensure_ascii=False, default=str)
                if effective_ttl is not None:
                    self._redis_client.setex(rkey, int(float(effective_ttl)), payload)
                else:
                    self._redis_client.set(rkey, payload)
            except Exception as e:  # noqa: BLE001
                logger.warning("redis 写入失败（内存已写入）: %s", e)
        return key

    def set_context(self, session_id: str, data: Dict[str, Any], ttl: Optional[float] = None) -> str:
        """设置某会话的上下文数据

        Args:
            session_id: 会话 ID
            data: 上下文数据字典
            ttl: 过期时间（秒）

        Returns:
            str: 上下文 cache_key
        """
        key = f"ctx:{session_id}"
        self.write(key, data, ttl)
        return key

    def count(self) -> int:
        """返回当前缓存条目数"""
        return len(self._store)

    def flush(self) -> None:
        """清空本地内存缓存（redis 数据不受影响）"""
        self._store.clear()
