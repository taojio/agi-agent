"""
storage/backends.py - 存储后端抽象与实现

提供统一的存储抽象接口，支持多种后端实现
"""
import abc
import os
import json
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class StoredItem:
    """存储项"""
    key: str
    size: int
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


class StorageBackend(abc.ABC):
    """存储后端抽象基类"""

    @abc.abstractmethod
    def save(self, key: str, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """保存数据

        Args:
            key: 存储键
            data: 数据字节
            metadata: 附加元数据

        Returns:
            存储键（可能带版本号）
        """
        ...

    @abc.abstractmethod
    def load(self, key: str) -> Optional[bytes]:
        """加载数据"""
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> bool:
        """删除数据"""
        ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """检查是否存在"""
        ...

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """列出所有键"""
        ...

    @abc.abstractmethod
    def get_info(self, key: str) -> Optional[StoredItem]:
        """获取存储项信息"""
        ...

    @abc.abstractmethod
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """获取元数据"""
        ...

    def save_json(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """保存 JSON 数据"""
        return self.save(key, json.dumps(data, ensure_ascii=False).encode("utf-8"), metadata)

    def load_json(self, key: str) -> Optional[Any]:
        """加载 JSON 数据"""
        data = self.load(key)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class FileStorageBackend(StorageBackend):
    """本地文件存储后端"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._meta_dir = os.path.join(base_dir, ".meta")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self._meta_dir, exist_ok=True)

    def _data_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.base_dir, safe_key)

    def _meta_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self._meta_dir, f"{safe_key}.json")

    def save(self, key: str, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        path = self._data_path(key)
        meta_path = self._meta_path(key)
        now = time.time()

        with open(path, "wb") as f:
            f.write(data)

        checksum = hashlib.sha256(data).hexdigest()
        size = len(data)

        existing_meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        meta = {
            "key": key,
            "size": size,
            "created_at": existing_meta.get("created_at", now),
            "updated_at": now,
            "checksum": checksum,
            "metadata": metadata or {},
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return key

    def load(self, key: str) -> Optional[bytes]:
        path = self._data_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except IOError:
            return None

    def delete(self, key: str) -> bool:
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)
        deleted = False
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
                deleted = True
            except OSError:
                pass
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
                deleted = True
            except OSError:
                pass
        return deleted

    def exists(self, key: str) -> bool:
        return os.path.exists(self._data_path(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        if not os.path.exists(self.base_dir):
            return []
        keys = []
        for fname in os.listdir(self.base_dir):
            if fname.startswith("."):
                continue
            if prefix and not fname.startswith(prefix):
                continue
            if os.path.isfile(os.path.join(self.base_dir, fname)):
                keys.append(fname)
        return sorted(keys)

    def get_info(self, key: str) -> Optional[StoredItem]:
        meta_path = self._meta_path(key)
        if not os.path.exists(meta_path):
            if self.exists(key):
                path = self._data_path(key)
                stat = os.stat(path)
                return StoredItem(
                    key=key,
                    size=stat.st_size,
                    created_at=stat.st_ctime,
                    updated_at=stat.st_mtime,
                )
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return StoredItem(
                key=meta["key"],
                size=meta["size"],
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                metadata=meta.get("metadata", {}),
                checksum=meta.get("checksum", ""),
            )
        except (json.JSONDecodeError, IOError):
            return None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        info = self.get_info(key)
        return info.metadata if info else None


class VersionedStorage(StorageBackend):
    """版本化存储包装器

    为底层存储添加版本管理能力
    """

    def __init__(self, backend: StorageBackend, max_versions: int = 10):
        self.backend = backend
        self.max_versions = max_versions
        self._versions_key = "_versions"

    def _ver_key(self, key: str, version: int) -> str:
        return f"{key}@v{version:04d}"

    def _get_versions(self, key: str) -> List[int]:
        data = self.backend.load_json(f"{key}_{self._versions_key}")
        if data and isinstance(data, list):
            return sorted(data)
        return []

    def _save_versions(self, key: str, versions: List[int]) -> None:
        self.backend.save_json(f"{key}_{self._versions_key}", sorted(versions))

    def save(self, key: str, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        versions = self._get_versions(key)
        next_version = (max(versions) + 1) if versions else 1

        ver_key = self._ver_key(key, next_version)
        self.backend.save(ver_key, data, metadata)

        versions.append(next_version)
        if len(versions) > self.max_versions:
            old = versions[0]
            self.backend.delete(self._ver_key(key, old))
            versions = versions[1:]

        self._save_versions(key, versions)
        return ver_key

    def load(self, key: str, version: Optional[int] = None) -> Optional[bytes]:
        if "@v" in key:
            return self.backend.load(key)
        versions = self._get_versions(key)
        if not versions:
            return None
        if version is None:
            version = max(versions)
        if version not in versions:
            return None
        return self.backend.load(self._ver_key(key, version))

    def delete(self, key: str) -> bool:
        versions = self._get_versions(key)
        deleted = False
        for v in versions:
            if self.backend.delete(self._ver_key(key, v)):
                deleted = True
        self.backend.delete(f"{key}_{self._versions_key}")
        return deleted

    def exists(self, key: str) -> bool:
        if "@v" in key:
            return self.backend.exists(key)
        return len(self._get_versions(key)) > 0

    def list_keys(self, prefix: str = "") -> List[str]:
        all_keys = self.backend.list_keys(prefix)
        base_keys = set()
        for k in all_keys:
            if "@v" in k:
                base = k.split("@v")[0]
                base_keys.add(base)
            elif not k.endswith(f"_{self._versions_key}"):
                base_keys.add(k)
        return sorted(base_keys)

    def get_info(self, key: str) -> Optional[StoredItem]:
        if "@v" in key:
            return self.backend.get_info(key)
        versions = self._get_versions(key)
        if not versions:
            return None
        latest = max(versions)
        return self.backend.get_info(self._ver_key(key, latest))

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        info = self.get_info(key)
        return info.metadata if info else None

    def list_versions(self, key: str) -> List[int]:
        """列出所有版本号"""
        return self._get_versions(key)

    def rollback(self, key: str, version: int) -> bool:
        """回滚到指定版本

        通过将指定版本保存为新版本实现
        """
        data = self.load(key, version)
        if data is None:
            return False
        self.save(key, data)
        return True
