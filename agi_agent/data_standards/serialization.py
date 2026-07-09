"""
data_standards/serialization.py - 数据序列化

提供统一的序列化/反序列化能力，支持多种格式
"""
import abc
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Type
from dataclasses import is_dataclass, asdict

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class Serializer(abc.ABC):
    """序列化器基类"""

    @abc.abstractmethod
    def serialize(self, data: Any) -> bytes:
        """序列化数据为字节流"""
        ...

    @abc.abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """从字节流反序列化数据"""
        ...

    @abc.abstractmethod
    def serialize_to_str(self, data: Any) -> str:
        """序列化为字符串"""
        ...

    @abc.abstractmethod
    def deserialize_from_str(self, data: str) -> Any:
        """从字符串反序列化"""
        ...


class JSONSerializer(Serializer):
    """JSON 序列化器"""

    def __init__(self, indent: Optional[int] = None, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def serialize(self, data: Any) -> bytes:
        return self.serialize_to_str(data).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        return self.deserialize_from_str(data.decode('utf-8'))

    def serialize_to_str(self, data: Any) -> str:
        return json.dumps(
            self._prepare_for_json(data),
            indent=self.indent,
            ensure_ascii=self.ensure_ascii
        )

    def deserialize_from_str(self, data: str) -> Any:
        return json.loads(data)

    def _prepare_for_json(self, data: Any) -> Any:
        """准备数据用于 JSON 序列化"""
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, (list, tuple)):
            return [self._prepare_for_json(item) for item in data]
        if isinstance(data, dict):
            return {str(k): self._prepare_for_json(v) for k, v in data.items()}
        if hasattr(data, 'to_dict'):
            return self._prepare_for_json(data.to_dict())
        if is_dataclass(data):
            return self._prepare_for_json(asdict(data))
        if isinstance(data, set):
            return {'_set_': True, 'values': list(data)}
        if hasattr(data, 'isoformat'):
            return data.isoformat()
        if hasattr(data, 'item'):
            try:
                return data.item()
            except Exception:
                return str(data)
        return str(data)


class MessagePackSerializer(Serializer):
    """MessagePack 序列化器（高性能，二进制）"""

    def __init__(self, use_bin_type: bool = True, raw: bool = False):
        if not HAS_MSGPACK:
            raise ImportError(
                "msgpack is not installed. Install with: pip install msgpack"
            )
        self.use_bin_type = use_bin_type
        self.raw = raw

    def serialize(self, data: Any) -> bytes:
        prepared = self._prepare_for_pack(data)
        return msgpack.packb(prepared, use_bin_type=self.use_bin_type)

    def deserialize(self, data: bytes) -> Any:
        unpacked = msgpack.unpackb(data, raw=self.raw)
        return self._restore_from_pack(unpacked)

    def serialize_to_str(self, data: Any) -> str:
        return self.serialize(data).hex()

    def deserialize_from_str(self, data: str) -> Any:
        return self.deserialize(bytes.fromhex(data))

    def _prepare_for_pack(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool, bytes)):
            return data
        if isinstance(data, (list, tuple)):
            return [self._prepare_for_pack(item) for item in data]
        if isinstance(data, dict):
            return {str(k): self._prepare_for_pack(v) for k, v in data.items()}
        if hasattr(data, 'to_dict'):
            return self._prepare_for_pack(data.to_dict())
        if is_dataclass(data):
            return self._prepare_for_pack(asdict(data))
        return str(data)

    def _restore_from_pack(self, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get('_set_'):
                return set(data.get('values', []))
            return {k: self._restore_from_pack(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._restore_from_pack(item) for item in data]
        return data


_default_serializer = JSONSerializer()


def serialize(data: Any, fmt: str = "json", **kwargs) -> bytes:
    """便捷函数：序列化数据

    Args:
        data: 待序列化的数据
        fmt: 格式 ("json" 或 "msgpack")
        **kwargs: 传递给序列化器的参数

    Returns:
        序列化后的字节流
    """
    if fmt == "json":
        return JSONSerializer(**kwargs).serialize(data)
    elif fmt == "msgpack":
        return MessagePackSerializer(**kwargs).serialize(data)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def deserialize(data: bytes, fmt: str = "json", **kwargs) -> Any:
    """便捷函数：反序列化数据

    Args:
        data: 待反序列化的字节流
        fmt: 格式 ("json" 或 "msgpack")
        **kwargs: 传递给序列化器的参数

    Returns:
        反序列化后的数据
    """
    if fmt == "json":
        return JSONSerializer(**kwargs).deserialize(data)
    elif fmt == "msgpack":
        return MessagePackSerializer(**kwargs).deserialize(data)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def safe_serialize(data: Any, fmt: str = "json") -> Optional[bytes]:
    """安全序列化，失败返回 None"""
    try:
        return serialize(data, fmt)
    except Exception:
        return None


def safe_deserialize(data: bytes, fmt: str = "json") -> Optional[Any]:
    """安全反序列化，失败返回 None"""
    try:
        return deserialize(data, fmt)
    except Exception:
        return None
