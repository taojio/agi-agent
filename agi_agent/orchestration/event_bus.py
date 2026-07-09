"""
orchestration/event_bus.py - 事件总线

发布-订阅模式的事件系统，用于模块间解耦通信
"""
import time
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """事件对象"""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{int(self.timestamp * 1000)}_{id(self)}"


class EventBus:
    """事件总线

    支持：
    - 发布/订阅模式
    - 同步/异步发布
    - 事件优先级
    - 通配符订阅
    - 事件历史记录
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._event_counts: Dict[str, int] = defaultdict(int)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """订阅事件

        Args:
            event_type: 事件类型，支持 "*" 通配符订阅所有事件
            handler: 处理函数，接收 Event 参数
        """
        if event_type == "*":
            if handler not in self._wildcard_subscribers:
                self._wildcard_subscribers.append(handler)
        else:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """取消订阅"""
        if event_type == "*":
            if handler in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(handler)
                return True
            return False
        else:
            if handler in self._subscribers.get(event_type, []):
                self._subscribers[event_type].remove(handler)
                return True
            return False

    def publish(self, event: Event) -> int:
        """同步发布事件

        Args:
            event: 事件对象

        Returns:
            触发的 handler 数量
        """
        self._record_event(event)
        count = 0

        for handler in self._wildcard_subscribers:
            try:
                handler(event)
                count += 1
            except Exception as e:
                logger.error(f"Wildcard handler error for event {event.event_type}: {e}")

        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
                count += 1
            except Exception as e:
                logger.error(f"Handler error for event {event.event_type}: {e}")

        return count

    async def publish_async(self, event: Event) -> int:
        """异步发布事件"""
        self._record_event(event)
        count = 0
        handlers = (
            self._wildcard_subscribers +
            self._subscribers.get(event.event_type, [])
        )
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                count += 1
            except Exception as e:
                logger.error(f"Async handler error for event {event.event_type}: {e}")
        return count

    def _record_event(self, event: Event) -> None:
        """记录事件到历史"""
        self._history.append(event)
        self._event_counts[event.event_type] += 1
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:] if limit > 0 else events

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        """获取事件计数"""
        if event_type:
            return self._event_counts.get(event_type, 0)
        return sum(self._event_counts.values())

    def get_event_types(self) -> Set[str]:
        """获取所有事件类型"""
        return set(self._subscribers.keys()) | set(self._event_counts.keys())

    def clear(self) -> None:
        """清空所有订阅者和历史"""
        self._subscribers.clear()
        self._wildcard_subscribers.clear()
        self._history.clear()
        self._event_counts.clear()

    @property
    def subscriber_count(self) -> int:
        """订阅者总数"""
        total = len(self._wildcard_subscribers)
        for handlers in self._subscribers.values():
            total += len(handlers)
        return total
