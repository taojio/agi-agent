"""
communication_protocol.py - 元模块通信协议

定义标准化的元模块间通信协议，支持：
- 同步/异步通信模式
- 消息可靠性保障
- 错误处理机制
- 版本兼容
"""
import time
import uuid
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class ProtocolVersion(Enum):
    """协议版本"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class MetaMessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"


class MetaResponseStatus(Enum):
    """响应状态"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MetaMessage:
    """元模块消息信封

    所有元模块间通信使用统一的消息格式，确保：
    - 可追踪性（message_id + correlation_id + trace_id）
    - 可路由（source_module/target_module）
    - 优先级控制
    - 版本兼容
    - TTL 过期机制
    - 消息完整性校验
    """

    message_id: str
    message_type: MetaMessageType
    source_module: str
    target_module: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    version: ProtocolVersion = ProtocolVersion.V1_0
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    def compute_checksum(self) -> str:
        data = json.dumps({
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "version": self.version.value,
            "priority": self.priority.value,
            "ttl": self.ttl,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_checksum(self) -> bool:
        if not self.checksum:
            return False
        return self.checksum == self.compute_checksum()

    def ensure_checksum(self):
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "version": self.version.value,
            "priority": self.priority.value,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaMessage":
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MetaMessageType(data.get("message_type", "request")),
            source_module=data.get("source_module", "unknown"),
            target_module=data.get("target_module", "unknown"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id"),
            trace_id=data.get("trace_id"),
            version=ProtocolVersion(data.get("version", "1.0")),
            priority=MessagePriority(data.get("priority", 1)),
            ttl=data.get("ttl", 30.0),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum"),
        )

    def clone(self, **overrides) -> "MetaMessage":
        data = self.to_dict()
        data.update(overrides)
        data["message_id"] = str(uuid.uuid4())
        if "correlation_id" not in overrides:
            data["correlation_id"] = self.message_id
        if "trace_id" not in overrides:
            data["trace_id"] = self.trace_id or self.message_id
        return MetaMessage.from_dict(data)


@dataclass
class MetaResponse:
    """元模块响应

    对请求消息的标准化响应
    """

    response_id: str
    correlation_id: str
    status: MetaResponseStatus
    source_module: str
    target_module: str
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == MetaResponseStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        return self.status in (
            MetaResponseStatus.FAILED,
            MetaResponseStatus.TIMEOUT,
            MetaResponseStatus.NOT_FOUND,
            MetaResponseStatus.UNAUTHORIZED,
            MetaResponseStatus.VALIDATION_ERROR,
            MetaResponseStatus.RATE_LIMITED,
        )

    def to_dict(self) -> Dict[str, Any]:
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
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaResponse":
        return cls(
            response_id=data.get("response_id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id", ""),
            status=MetaResponseStatus(data.get("status", "success")),
            source_module=data.get("source_module", "unknown"),
            target_module=data.get("target_module", "unknown"),
            data=data.get("data", {}),
            errors=data.get("errors", []),
            timestamp=data.get("timestamp", time.time()),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
            trace_id=data.get("trace_id"),
        )


def create_meta_message(
    source: str,
    target: str,
    payload: Dict[str, Any] = None,
    message_type: MetaMessageType = MetaMessageType.REQUEST,
    priority: MessagePriority = MessagePriority.NORMAL,
    ttl: float = 30.0,
    correlation_id: str = None,
    trace_id: str = None,
    version: ProtocolVersion = ProtocolVersion.V1_0,
    **metadata
) -> MetaMessage:
    msg = MetaMessage(
        message_id=str(uuid.uuid4()),
        message_type=message_type,
        source_module=source,
        target_module=target,
        payload=payload or {},
        timestamp=time.time(),
        correlation_id=correlation_id,
        trace_id=trace_id or str(uuid.uuid4()),
        version=version,
        priority=priority,
        ttl=ttl,
        metadata=metadata,
    )
    msg.ensure_checksum()
    return msg


def create_meta_response(
    request: MetaMessage,
    status: MetaResponseStatus = MetaResponseStatus.SUCCESS,
    data: Dict[str, Any] = None,
    errors: List[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> MetaResponse:
    return MetaResponse(
        response_id=str(uuid.uuid4()),
        correlation_id=request.message_id,
        status=status,
        source_module=request.target_module,
        target_module=request.source_module,
        data=data or {},
        errors=errors or [],
        timestamp=time.time(),
        duration_ms=duration_ms,
        trace_id=request.trace_id,
    )


def create_error_response(
    request: MetaMessage,
    error_message: str,
    error_code: str = "UNKNOWN_ERROR",
    status: MetaResponseStatus = MetaResponseStatus.FAILED,
) -> MetaResponse:
    return create_meta_response(
        request=request,
        status=status,
        errors=[{"code": error_code, "message": error_message}],
    )


def validate_meta_message(message: MetaMessage) -> Tuple[bool, List[str]]:
    errors = []
    
    if not message.message_id:
        errors.append("message_id is required")
    
    if not message.source_module:
        errors.append("source_module is required")
    
    if not message.target_module:
        errors.append("target_module is required")
    
    if message.ttl <= 0:
        errors.append("ttl must be positive")
    
    if message.checksum and not message.verify_checksum():
        errors.append("checksum verification failed")
    
    if message.is_expired:
        errors.append(f"message expired (age: {message.age:.2f}s, ttl: {message.ttl}s)")
    
    return (len(errors) == 0, errors)


class MessageRouter:
    """消息路由器

    负责消息的路由分发和可靠性保障
    """

    def __init__(self):
        self._routes: Dict[str, List[Tuple[str, Callable]]] = {}
        self._pending_messages: Dict[str, Tuple[MetaMessage, float]] = {}
        self._delivery_attempts: Dict[str, int] = {}
        self._max_retries = 3
        self._retry_delay = 1.0

    def register_route(self, target_module: str, handler: Callable) -> str:
        route_id = f"route_{uuid.uuid4().hex[:8]}"
        if target_module not in self._routes:
            self._routes[target_module] = []
        self._routes[target_module].append((route_id, handler))
        return route_id

    def unregister_route(self, route_id: str) -> bool:
        for target, routes in list(self._routes.items()):
            self._routes[target] = [(rid, h) for rid, h in routes if rid != route_id]
            if not self._routes[target]:
                del self._routes[target]
        return True

    def route_message(self, message: MetaMessage) -> bool:
        valid, errors = validate_meta_message(message)
        if not valid:
            return False

        handlers = []
        for target_pattern, route_list in self._routes.items():
            if self._matches_pattern(message.target_module, target_pattern):
                for _, handler in route_list:
                    handlers.append(handler)

        if not handlers:
            self._store_for_retry(message)
            return False

        success = False
        for handler in handlers:
            try:
                handler(message)
                success = True
            except Exception:
                pass

        if not success:
            self._store_for_retry(message)

        return success

    def _matches_pattern(self, target: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return target.startswith(pattern[:-2])
        return target == pattern

    def _store_for_retry(self, message: MetaMessage):
        attempts = self._delivery_attempts.get(message.message_id, 0)
        if attempts >= self._max_retries:
            return
        
        self._delivery_attempts[message.message_id] = attempts + 1
        self._pending_messages[message.message_id] = (message, time.time())

    def retry_pending(self):
        now = time.time()
        to_retry = []
        
        for msg_id, (message, stored_at) in self._pending_messages.items():
            if now - stored_at >= self._retry_delay:
                to_retry.append(msg_id)
        
        for msg_id in to_retry:
            message, _ = self._pending_messages.pop(msg_id)
            self.route_message(message)


class ProtocolStats:
    """协议统计

    追踪消息发送、接收、丢失、延迟等指标
    """

    def __init__(self):
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "messages_retried": 0,
            "responses_sent": 0,
            "responses_received": 0,
            "errors": 0,
            "total_latency_ms": 0,
            "latency_samples": 0,
        }

    def record_sent(self):
        self._stats["messages_sent"] += 1

    def record_received(self):
        self._stats["messages_received"] += 1

    def record_dropped(self):
        self._stats["messages_dropped"] += 1

    def record_retry(self):
        self._stats["messages_retried"] += 1

    def record_response_sent(self):
        self._stats["responses_sent"] += 1

    def record_response_received(self):
        self._stats["responses_received"] += 1

    def record_error(self):
        self._stats["errors"] += 1

    def record_latency(self, latency_ms: float):
        self._stats["total_latency_ms"] += latency_ms
        self._stats["latency_samples"] += 1

    def get_stats(self) -> Dict[str, Any]:
        samples = self._stats["latency_samples"]
        avg_latency = self._stats["total_latency_ms"] / samples if samples > 0 else 0
        success_rate = (self._stats["messages_received"] / max(self._stats["messages_sent"], 1)) * 100
        
        return {
            "messages": {
                "sent": self._stats["messages_sent"],
                "received": self._stats["messages_received"],
                "dropped": self._stats["messages_dropped"],
                "retried": self._stats["messages_retried"],
                "success_rate": success_rate,
            },
            "responses": {
                "sent": self._stats["responses_sent"],
                "received": self._stats["responses_received"],
            },
            "latency": {
                "avg_ms": avg_latency,
                "samples": samples,
            },
            "errors": self._stats["errors"],
        }