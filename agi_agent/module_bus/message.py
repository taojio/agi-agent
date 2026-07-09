"""
module_bus/message.py - 消息模型

定义模块间通信的统一消息信封和响应格式
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    DATA = "data"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    def __lt__(self, other):
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value >= other.value


class ResponseStatus(Enum):
    """响应状态"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ModuleMessage:
    """模块消息信封

    所有模块间通信使用统一的消息信封格式，确保：
    - 可追踪性（message_id + correlation_id）
    - 可路由（source/target）
    - 优先级控制
    - 版本兼容
    - TTL 过期机制
    """

    message_id: str
    message_type: MessageType
    source_module: str
    target_module: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    version: str = "1.0"
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """消息是否已过期"""
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age(self) -> float:
        """消息年龄（秒）"""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "version": self.version,
            "priority": self.priority.value,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleMessage":
        """从字典创建消息"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "request")),
            source_module=data.get("source_module", "unknown"),
            target_module=data.get("target_module", "unknown"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id"),
            version=data.get("version", "1.0"),
            priority=MessagePriority(data.get("priority", 1)),
            ttl=data.get("ttl", 30.0),
            metadata=data.get("metadata", {}),
        )

    def clone(self, **overrides) -> "ModuleMessage":
        """克隆消息并覆盖部分字段"""
        data = self.to_dict()
        data.update(overrides)
        data["message_id"] = str(uuid.uuid4())
        if "correlation_id" not in overrides:
            data["correlation_id"] = self.message_id
        # target_module 若在 overrides 中，应取 overrides 的值
        if "target" in overrides and "target_module" not in overrides:
            data["target_module"] = overrides["target"]
        return ModuleMessage.from_dict(data)


@dataclass
class ModuleResponse:
    """模块响应

    对请求消息的标准化响应
    """

    response_id: str
    correlation_id: str
    status: ResponseStatus
    source_module: str
    target_module: str
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == ResponseStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """是否出错"""
        return self.status in (
            ResponseStatus.FAILED,
            ResponseStatus.TIMEOUT,
            ResponseStatus.NOT_FOUND,
            ResponseStatus.UNAUTHORIZED,
            ResponseStatus.VALIDATION_ERROR,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response_id": self.response_id,
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "data": self.data,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleResponse":
        """从字典创建响应"""
        return cls(
            response_id=data.get("response_id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id", ""),
            status=ResponseStatus(data.get("status", "success")),
            source_module=data.get("source_module", "unknown"),
            target_module=data.get("target_module", "unknown"),
            data=data.get("data", {}),
            errors=data.get("errors", []),
            timestamp=data.get("timestamp", time.time()),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


def create_message(
    source: str,
    target: str,
    payload: Dict[str, Any] = None,
    message_type: MessageType = MessageType.REQUEST,
    priority: MessagePriority = MessagePriority.NORMAL,
    ttl: float = 30.0,
    correlation_id: str = None,
    **metadata
) -> ModuleMessage:
    """便捷函数：创建消息

    Args:
        source: 来源模块名
        target: 目标模块名
        payload: 消息体
        message_type: 消息类型
        priority: 优先级
        ttl: 存活时间（秒）
        correlation_id: 关联ID
        **metadata: 额外元数据

    Returns:
        ModuleMessage 实例
    """
    return ModuleMessage(
        message_id=str(uuid.uuid4()),
        message_type=message_type,
        source_module=source,
        target_module=target,
        payload=payload or {},
        timestamp=time.time(),
        correlation_id=correlation_id,
        priority=priority,
        ttl=ttl,
        metadata=metadata,
    )


def create_response(
    request: ModuleMessage,
    status: ResponseStatus = ResponseStatus.SUCCESS,
    data: Dict[str, Any] = None,
    errors: List[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ModuleResponse:
    """便捷函数：创建响应

    Args:
        request: 请求消息
        status: 响应状态
        data: 响应数据
        errors: 错误列表
        duration_ms: 处理耗时（毫秒）

    Returns:
        ModuleResponse 实例
    """
    return ModuleResponse(
        response_id=str(uuid.uuid4()),
        correlation_id=request.message_id,
        status=status,
        source_module=request.target_module,
        target_module=request.source_module,
        data=data or {},
        errors=errors or [],
        timestamp=time.time(),
        duration_ms=duration_ms,
    )


def create_error_response(
    request: ModuleMessage,
    error_message: str,
    error_code: str = "UNKNOWN_ERROR",
    status: ResponseStatus = ResponseStatus.FAILED,
) -> ModuleResponse:
    """便捷函数：创建错误响应"""
    return create_response(
        request=request,
        status=status,
        errors=[{"code": error_code, "message": error_message}],
    )
