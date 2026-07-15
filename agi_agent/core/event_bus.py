"""
core/event_bus.py - 事件总线

提供统一的事件发布/订阅机制，支持：
- 同步事件分发
- 异步事件处理
- 事件过滤与优先级
- 事件溯源
- 分布式事件广播

使用方式：
    from agi_agent.core import EventBus, get_event_bus
    
    bus = get_event_bus()
    bus.subscribe("agent.*", handler_func)
    bus.publish("agent.initialized", {"agent_id": "test"})
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Pattern
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class EventSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCategory(Enum):
    SYSTEM = "system"
    AGENT = "agent"
    COGNITIVE = "cognitive"
    PERCEPTION = "perception"
    MEMORY = "memory"
    EXECUTION = "execution"
    SECURITY = "security"
    METRICS = "metrics"


@dataclass
class Event:
    """
    事件封装
    
    Attributes:
        event_type: 事件类型（如 "agent.initialized"）
        source: 事件源模块名
        payload: 事件数据
        severity: 严重级别
        category: 事件分类
        timestamp: 时间戳
        event_id: 唯一标识
    """
    event_type: str
    source: str
    payload: Dict[str, Any]
    severity: EventSeverity = EventSeverity.INFO
    category: EventCategory = EventCategory.SYSTEM
    timestamp: float = 0.0
    event_id: str = ""
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now().timestamp()
        if not self.event_id:
            self.event_id = f"{self.event_type}-{int(self.timestamp * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }


class Subscription:
    """
    订阅句柄
    
    用于管理事件订阅的生命周期
    """
    
    def __init__(self, topic: str, handler: Callable, bus: "EventBus"):
        self.topic = topic
        self.handler = handler
        self.bus = bus
        self.active = True
    
    def unsubscribe(self) -> None:
        """取消订阅"""
        if self.active:
            self.bus.unsubscribe(self.topic, self.handler)
            self.active = False


class EventBus:
    """
    事件总线
    
    提供统一的事件发布/订阅机制，支持同步和异步处理。
    
    Attributes:
        subscriptions: 订阅映射（主题 -> 处理器列表）
        event_history: 事件历史记录
        max_history: 最大历史记录数
        async_enabled: 是否启用异步处理
        event_counter: 事件计数器
    """
    
    def __init__(self, max_history: int = 1000, async_enabled: bool = True):
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = max_history
        self.async_enabled = async_enabled
        self.event_counter = 0
        self._lock = asyncio.Lock() if async_enabled else None
    
    def subscribe(self, topic: str, handler: Callable) -> Subscription:
        """
        订阅事件
        
        Args:
            topic: 事件主题（支持通配符 *）
            handler: 事件处理器
        
        Returns:
            Subscription: 订阅句柄
        """
        self.subscriptions[topic].append(handler)
        logger.debug(f"Subscribed to topic '{topic}'")
        return Subscription(topic, handler, self)
    
    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """
        取消订阅
        
        Args:
            topic: 事件主题
            handler: 事件处理器
        """
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                h for h in self.subscriptions[topic] if h != handler
            ]
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            logger.debug(f"Unsubscribed from topic '{topic}'")
    
    def publish(self, event_type: str, source: str, 
                payload: Dict[str, Any] = None,
                severity: EventSeverity = EventSeverity.INFO,
                category: EventCategory = EventCategory.SYSTEM) -> None:
        """
        发布事件（同步）
        
        Args:
            event_type: 事件类型
            source: 事件源
            payload: 事件数据
            severity: 严重级别
            category: 事件分类
        """
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            severity=severity,
            category=category,
        )
        
        self._store_event(event)
        self._dispatch(event)
    
    async def publish_async(self, event_type: str, source: str,
                            payload: Dict[str, Any] = None,
                            severity: EventSeverity = EventSeverity.INFO,
                            category: EventCategory = EventCategory.SYSTEM) -> None:
        """
        发布事件（异步）
        
        Args:
            event_type: 事件类型
            source: 事件源
            payload: 事件数据
            severity: 严重级别
            category: 事件分类
        """
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            severity=severity,
            category=category,
        )
        
        self._store_event(event)
        await self._dispatch_async(event)
    
    def _store_event(self, event: Event) -> None:
        """存储事件到历史记录"""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        self.event_counter += 1
    
    def _dispatch(self, event: Event) -> None:
        """同步分发事件"""
        matched_topics = self._match_topics(event.event_type)
        
        for topic in matched_topics:
            for handler in self.subscriptions[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        if self.async_enabled:
                            asyncio.create_task(handler(event))
                        else:
                            handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error handling event '{event.event_type}' in handler: {e}")
    
    async def _dispatch_async(self, event: Event) -> None:
        """异步分发事件"""
        matched_topics = self._match_topics(event.event_type)
        
        tasks = []
        for topic in matched_topics:
            for handler in self.subscriptions[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error handling event '{event.event_type}' in handler: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _match_topics(self, event_type: str) -> List[str]:
        """
        匹配订阅主题
        
        支持通配符匹配：
        - "agent.*" 匹配 "agent.initialized", "agent.shutdown"
        - "*" 匹配所有事件
        
        Args:
            event_type: 事件类型
        
        Returns:
            匹配的主题列表
        """
        matched = []
        
        for topic in self.subscriptions.keys():
            if topic == event_type:
                matched.append(topic)
            elif topic.endswith(".*"):
                prefix = topic[:-2]
                if event_type.startswith(prefix) and (
                    len(event_type) == len(prefix) or 
                    event_type[len(prefix)] == "."
                ):
                    matched.append(topic)
            elif topic == "*":
                matched.append(topic)
        
        return matched
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取事件历史
        
        Args:
            limit: 返回数量限制
        
        Returns:
            事件历史列表
        """
        return [e.to_dict() for e in self.event_history[-limit:]]
    
    def get_event_stats(self) -> Dict[str, Any]:
        """
        获取事件统计信息
        
        Returns:
            统计信息字典
        """
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for event in self.event_history:
            by_category[event.category.value] += 1
            by_severity[event.severity.value] += 1
            by_type[event.event_type] += 1
        
        return {
            "total_events": self.event_counter,
            "history_size": len(self.event_history),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "top_types": sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10],
        }
    
    def clear_history(self) -> None:
        """清空事件历史"""
        self.event_history.clear()
    
    def subscribe_multiple(self, topics: List[str], handler: Callable) -> List[Subscription]:
        """
        批量订阅多个主题
        
        Args:
            topics: 主题列表
            handler: 事件处理器
        
        Returns:
            订阅句柄列表
        """
        subscriptions = []
        for topic in topics:
            subscriptions.append(self.subscribe(topic, handler))
        return subscriptions
    
    def publish_with_trace(self, event_type: str, source: str,
                           payload: Dict[str, Any] = None, **kwargs) -> Event:
        """
        发布事件并返回事件对象（用于追踪）
        
        Args:
            event_type: 事件类型
            source: 事件源
            payload: 事件数据
        
        Returns:
            Event: 事件对象
        """
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            **kwargs,
        )
        
        self._store_event(event)
        self._dispatch(event)
        return event


_global_event_bus: Optional[EventBus] = None


def get_event_bus(max_history: int = 1000, async_enabled: bool = True) -> EventBus:
    """
    获取全局事件总线单例
    
    Args:
        max_history: 最大历史记录数
        async_enabled: 是否启用异步处理
    
    Returns:
        EventBus 实例
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus(max_history=max_history, async_enabled=async_enabled)
    return _global_event_bus