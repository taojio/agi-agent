"""
module_bus/__init__.py - 模块通信总线

提供模块间通信的统一基础设施：
- 事件驱动（发布/订阅）
- 请求响应（同步调用）
- 数据流（流式传输）
- 服务注册与发现
"""
from .message import (
    ModuleMessage,
    MessageType,
    MessagePriority,
    ModuleResponse,
    ResponseStatus,
    create_message,
    create_response,
)
from .event import (
    ModuleEvent,
    EventSeverity,
    EventCategory,
    create_event,
)
from .service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ModuleCapability,
    CapabilityType,
    get_service_registry,
)
from .data_stream import (
    DataStream,
    StreamConsumer,
    BackPressureStrategy,
    create_stream,
)
from .bus import (
    ModuleBus,
    get_module_bus,
)

__all__ = [
    "ModuleMessage",
    "MessageType",
    "MessagePriority",
    "ModuleResponse",
    "ResponseStatus",
    "create_message",
    "create_response",
    "ModuleEvent",
    "EventSeverity",
    "EventCategory",
    "create_event",
    "ServiceRegistry",
    "ServiceInfo",
    "ModuleCapability",
    "CapabilityType",
    "get_service_registry",
    "DataStream",
    "StreamConsumer",
    "BackPressureStrategy",
    "create_stream",
    "ModuleBus",
    "get_module_bus",
]
