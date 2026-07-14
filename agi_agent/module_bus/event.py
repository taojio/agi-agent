"""
module_bus/event.py - 事件模型

定义模块事件和严重级别，支持事件驱动架构
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventSeverity(Enum):
    """事件严重级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCategory(Enum):
    """事件分类"""
    SYSTEM = "system"          # 系统级事件（启动、停止、错误）
    DATA = "data"              # 数据变更事件
    STATE = "state"            # 状态变更事件
    METRIC = "metric"          # 指标事件
    ERROR = "error"            # 错误事件
    SECURITY = "security"      # 安全事件
    USER = "user"              # 用户行为事件


@dataclass
class ModuleEvent:
    """模块事件

    事件驱动架构的核心数据结构，用于模块间的解耦通知。
    特点：
    - 异步：发布者不等待处理结果
    - 一对多：多个订阅者可同时接收
    - 持久化：可记录事件历史用于审计和回放
    """

    event_id: str
    event_type: str
    source_module: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    severity: EventSeverity = EventSeverity.INFO
    category: EventCategory = EventCategory.SYSTEM
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """事件年龄（秒）"""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_module": self.source_module,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "category": self.category.value,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleEvent":
        """从字典创建事件"""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data.get("event_type", "unknown"),
            source_module=data.get("source_module", "unknown"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            severity=EventSeverity(data.get("severity", "info")),
            category=EventCategory(data.get("category", "system")),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )

    def matches(self, pattern: str) -> bool:
        """检查事件类型是否匹配模式

        支持通配符：
        - "*" 匹配任意
        - "system.*" 匹配 system 开头的所有类型

        Args:
            pattern: 匹配模式

        Returns:
            是否匹配
        """
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return self.event_type.startswith(prefix)
        return self.event_type == pattern


def create_event(
    event_type: str,
    source: str,
    payload: Dict[str, Any] = None,
    severity: EventSeverity = EventSeverity.INFO,
    category: EventCategory = EventCategory.SYSTEM,
    correlation_id: str = None,
    **metadata
) -> ModuleEvent:
    """便捷函数：创建事件

    Args:
        event_type: 事件类型（使用点号命名空间，如 "memory.entry_added"）
        source: 来源模块名
        payload: 事件数据
        severity: 严重级别
        category: 事件分类
        correlation_id: 关联ID
        **metadata: 额外元数据

    Returns:
        ModuleEvent 实例
    """
    return ModuleEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        source_module=source,
        payload=payload or {},
        timestamp=time.time(),
        severity=severity,
        category=category,
        correlation_id=correlation_id,
        metadata=metadata,
    )


def create_state_change_event(
    source: str,
    entity: str,
    old_state: Any,
    new_state: Any,
    **metadata
) -> ModuleEvent:
    """创建状态变更事件"""
    return create_event(
        event_type=f"state.{entity}.changed",
        source=source,
        payload={"entity": entity, "old": old_state, "new": new_state},
        severity=EventSeverity.INFO,
        category=EventCategory.STATE,
        **metadata,
    )


def create_error_event(
    source: str,
    error_type: str,
    error_message: str,
    details: Dict[str, Any] = None,
    severity: EventSeverity = EventSeverity.ERROR,
) -> ModuleEvent:
    """创建错误事件"""
    return create_event(
        event_type=f"error.{error_type}",
        source=source,
        payload={
            "error_type": error_type,
            "message": error_message,
            "details": details or {},
        },
        severity=severity,
        category=EventCategory.ERROR,
    )


def create_metric_event(
    source: str,
    metric_name: str,
    value: float,
    unit: str = "",
    tags: Dict[str, str] = None,
) -> ModuleEvent:
    """创建指标事件"""
    return create_event(
        event_type=f"metric.{metric_name}",
        source=source,
        payload={
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        },
        severity=EventSeverity.DEBUG,
        category=EventCategory.METRIC,
    )
