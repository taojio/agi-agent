import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from enum import Enum


class MemoryPermission(Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ADMIN = "admin"


class SharedMemorySpace:
    def __init__(self, space_id: str = "default"):
        self.space_id = space_id
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._permissions: Dict[str, Dict[str, MemoryPermission]] = {}

        self._access_log: deque = deque(maxlen=500)
        self._change_callbacks: List[Callable] = []
        self._version_counter = 0

        self._ttl_store: Dict[str, float] = {}

    def register_agent(self, agent_id: str,
                       permission: MemoryPermission = MemoryPermission.READ_WRITE):
        with self._lock:
            self._permissions[agent_id] = permission

    def set_permission(self, agent_id: str, permission: MemoryPermission):
        with self._lock:
            self._permissions[agent_id] = permission

    def _check_permission(self, agent_id: str, required: str) -> bool:
        perm = self._permissions.get(agent_id)
        if perm == MemoryPermission.ADMIN:
            return True
        if required == "read" and perm in (MemoryPermission.READ, MemoryPermission.READ_WRITE):
            return True
        if required == "write" and perm in (MemoryPermission.WRITE, MemoryPermission.READ_WRITE):
            return True
        return False

    def put(self, key: str, value: Any, agent_id: str = "system",
            ttl_seconds: float = None) -> bool:
        with self._lock:
            if not self._check_permission(agent_id, "write"):
                return False

            self._data[key] = value
            self._version_counter += 1

            self._metadata[key] = {
                "created_by": agent_id,
                "updated_by": agent_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": self._version_counter,
                "access_count": self._metadata.get(key, {}).get("access_count", 0)
            }

            if ttl_seconds:
                self._ttl_store[key] = time.time() + ttl_seconds

            self._access_log.append({
                "operation": "put",
                "key": key,
                "agent_id": agent_id,
                "timestamp": time.time()
            })

            self._emit_change("put", key, value, agent_id)

            return True

    def get(self, key: str, default: Any = None, agent_id: str = "system") -> Any:
        with self._lock:
            if not self._check_permission(agent_id, "read"):
                return default

            if key in self._ttl_store and time.time() > self._ttl_store[key]:
                del self._data[key]
                del self._metadata[key]
                del self._ttl_store[key]
                return default

            if key in self._metadata:
                self._metadata[key]["access_count"] += 1

            self._access_log.append({
                "operation": "get",
                "key": key,
                "agent_id": agent_id,
                "timestamp": time.time()
            })

            return self._data.get(key, default)

    def delete(self, key: str, agent_id: str = "system") -> bool:
        with self._lock:
            if not self._check_permission(agent_id, "write"):
                return False

            if key in self._data:
                del self._data[key]
                self._metadata.pop(key, None)
                self._ttl_store.pop(key, None)

                self._access_log.append({
                    "operation": "delete",
                    "key": key,
                    "agent_id": agent_id,
                    "timestamp": time.time()
                })

                self._emit_change("delete", key, None, agent_id)

                return True

            return False

    def has_key(self, key: str) -> bool:
        with self._lock:
            if key in self._ttl_store and time.time() > self._ttl_store[key]:
                del self._data[key]
                del self._metadata[key]
                del self._ttl_store[key]
                return False
            return key in self._data

    def keys(self, agent_id: str = "system") -> List[str]:
        with self._lock:
            if not self._check_permission(agent_id, "read"):
                return []
            self._clean_expired()
            return list(self._data.keys())

    def get_all(self, agent_id: str = "system") -> Dict[str, Any]:
        with self._lock:
            if not self._check_permission(agent_id, "read"):
                return {}
            self._clean_expired()
            return dict(self._data)

    def _clean_expired(self):
        now = time.time()
        expired = [k for k, t in self._ttl_store.items() if now > t]
        for k in expired:
            self._data.pop(k, None)
            self._metadata.pop(k, None)
            self._ttl_store.pop(k, None)

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(key)

    def on_change(self, callback: Callable):
        self._change_callbacks.append(callback)

    def _emit_change(self, operation: str, key: str, value: Any, agent_id: str):
        for cb in self._change_callbacks:
            try:
                cb({
                    "operation": operation,
                    "key": key,
                    "value": value,
                    "agent_id": agent_id,
                    "space_id": self.space_id
                })
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            self._clean_expired()
            return {
                "space_id": self.space_id,
                "total_keys": len(self._data),
                "total_accesses": len(self._access_log),
                "version": self._version_counter,
                "registered_agents": len(self._permissions),
                "keys_with_ttl": len(self._ttl_store)
            }
