import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from functools import wraps


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


class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: EventSeverity = EventSeverity.INFO
    category: EventCategory = EventCategory.SYSTEM
    timestamp: float = 0.0
    event_id: str = ""
    priority: EventPriority = EventPriority.NORMAL
    cancellable: bool = False
    cancelled: bool = False
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now().timestamp()
        if not self.event_id:
            self.event_id = f"{self.event_type}-{int(self.timestamp * 1000)}"
    
    def cancel(self) -> None:
        if self.cancellable:
            self.cancelled = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "priority": self.priority.value,
            "cancellable": self.cancellable,
            "cancelled": self.cancelled,
        }


class Subscription:
    def __init__(self, topic: str, handler: Callable, bus: "EventBus", 
                 throttle_ms: int = 0, debounce_ms: int = 0):
        self.topic = topic
        self.handler = handler
        self.bus = bus
        self.active = True
        self.throttle_ms = throttle_ms
        self.debounce_ms = debounce_ms
        self._last_call = 0
        self._debounce_timer = None
    
    def unsubscribe(self) -> None:
        if self.active:
            self.bus.unsubscribe(self.topic, self.handler)
            self.active = False
    
    def _apply_throttle(self) -> bool:
        if self.throttle_ms <= 0:
            return True
        now = time.time() * 1000
        if now - self._last_call >= self.throttle_ms:
            self._last_call = now
            return True
        return False
    
    def _apply_debounce(self, event: Event) -> None:
        if self.debounce_ms <= 0:
            self._handle_event(event)
            return
        
        if self._debounce_timer:
            self._debounce_timer.cancel()
        
        async def debounce_handler():
            await asyncio.sleep(self.debounce_ms / 1000)
            self._handle_event(event)
        
        self._debounce_timer = asyncio.create_task(debounce_handler())
    
    def _handle_event(self, event: Event) -> None:
        if self.throttle_ms > 0 and not self._apply_throttle():
            return
        
        try:
            if asyncio.iscoroutinefunction(self.handler):
                asyncio.create_task(self.handler(event))
            else:
                self.handler(event)
        except Exception as e:
            logger.error(f"Error handling event '{event.event_type}' in handler: {e}")


def throttle(ms: int):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, '_last_call'):
                wrapper._last_call = 0
            
            now = time.time() * 1000
            if now - wrapper._last_call >= ms:
                wrapper._last_call = now
                return func(*args, **kwargs)
        return wrapper
    return decorator


def debounce(ms: int):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if hasattr(wrapper, '_debounce_timer') and wrapper._debounce_timer:
                wrapper._debounce_timer.cancel()
            
            async def debounce_handler():
                await asyncio.sleep(ms / 1000)
                func(*args, **kwargs)
            
            wrapper._debounce_timer = asyncio.create_task(debounce_handler())
        return wrapper
    return decorator


class EventBus:
    def __init__(self, max_history: int = 1000, async_enabled: bool = True):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = max_history
        self.async_enabled = async_enabled
        self.event_counter = 0
        self._lock = asyncio.Lock() if async_enabled else None
        self._priority_queue: Dict[EventPriority, List[Event]] = defaultdict(list)
        self._processing_queue = asyncio.Queue() if async_enabled else None
        self._processing_task = None
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_cancelled": 0,
            "handler_errors": 0,
            "async_handlers": 0,
            "sync_handlers": 0,
        }
        self._enable_monitoring = True
        
        if async_enabled:
            self._start_processing_loop()
    
    async def _start_processing_loop(self):
        self._processing_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        while True:
            try:
                event = await self._processing_queue.get()
                
                if event.cancelled:
                    self._stats["events_cancelled"] += 1
                    continue
                
                await self._dispatch_async(event)
                self._stats["events_processed"] += 1
                
                self._processing_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
    
    def subscribe(self, topic: str, handler: Callable, 
                  throttle_ms: int = 0, debounce_ms: int = 0) -> Subscription:
        sub = Subscription(topic, handler, self, throttle_ms, debounce_ms)
        self.subscriptions[topic].append(sub)
        logger.debug(f"Subscribed to topic '{topic}'")
        
        if asyncio.iscoroutinefunction(handler):
            self._stats["async_handlers"] += 1
        else:
            self._stats["sync_handlers"] += 1
        
        return sub
    
    def unsubscribe(self, topic: str, handler: Callable) -> None:
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                s for s in self.subscriptions[topic] if s.handler != handler
            ]
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            logger.debug(f"Unsubscribed from topic '{topic}'")
    
    def publish(self, event_type: str, source: str, 
                payload: Dict[str, Any] = None,
                severity: EventSeverity = EventSeverity.INFO,
                category: EventCategory = EventCategory.SYSTEM,
                priority: EventPriority = EventPriority.NORMAL,
                cancellable: bool = False) -> None:
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            severity=severity,
            category=category,
            priority=priority,
            cancellable=cancellable,
        )
        
        self._store_event(event)
        
        if priority >= EventPriority.HIGH:
            self._dispatch(event)
        else:
            self._dispatch(event)
    
    async def publish_async(self, event_type: str, source: str,
                            payload: Dict[str, Any] = None,
                            severity: EventSeverity = EventSeverity.INFO,
                            category: EventCategory = EventCategory.SYSTEM,
                            priority: EventPriority = EventPriority.NORMAL,
                            cancellable: bool = False) -> None:
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            severity=severity,
            category=category,
            priority=priority,
            cancellable=cancellable,
        )
        
        self._store_event(event)
        
        if self._processing_queue and priority >= EventPriority.HIGH:
            await self._processing_queue.put(event)
        elif self._processing_queue:
            await self._processing_queue.put(event)
        else:
            await self._dispatch_async(event)
    
    def publish_ordered(self, events: List[Event]) -> None:
        events_sorted = sorted(events, key=lambda e: e.priority.value, reverse=True)
        for event in events_sorted:
            self._store_event(event)
            if not event.cancelled:
                self._dispatch(event)
    
    async def publish_ordered_async(self, events: List[Event]) -> None:
        events_sorted = sorted(events, key=lambda e: e.priority.value, reverse=True)
        for event in events_sorted:
            self._store_event(event)
            if not event.cancelled:
                await self._dispatch_async(event)
    
    def cancel_event(self, event_id: str) -> bool:
        for event in self.event_history[-100:]:
            if event.event_id == event_id and event.cancellable:
                event.cancel()
                return True
        return False
    
    def _store_event(self, event: Event) -> None:
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        self.event_counter += 1
        self._stats["events_published"] += 1
    
    def _dispatch(self, event: Event) -> None:
        matched_topics = self._match_topics(event.event_type)
        
        for topic in matched_topics:
            for sub in self.subscriptions[topic]:
                if not sub.active:
                    continue
                try:
                    if event.cancelled:
                        break
                    sub._handle_event(event)
                except Exception as e:
                    self._stats["handler_errors"] += 1
                    logger.error(f"Error handling event '{event.event_type}' in handler: {e}")
    
    async def _dispatch_async(self, event: Event) -> None:
        matched_topics = self._match_topics(event.event_type)
        
        tasks = []
        for topic in matched_topics:
            for sub in self.subscriptions[topic]:
                if not sub.active:
                    continue
                try:
                    if event.cancelled:
                        break
                    if asyncio.iscoroutinefunction(sub.handler):
                        tasks.append(sub.handler(event))
                    else:
                        sub.handler(event)
                except Exception as e:
                    self._stats["handler_errors"] += 1
                    logger.error(f"Error handling event '{event.event_type}' in handler: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _match_topics(self, event_type: str) -> List[str]:
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
        return [e.to_dict() for e in self.event_history[-limit:]]
    
    def get_event_stats(self) -> Dict[str, Any]:
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        by_priority = defaultdict(int)
        
        for event in self.event_history:
            by_category[event.category.value] += 1
            by_severity[event.severity.value] += 1
            by_type[event.event_type] += 1
            by_priority[event.priority.value] += 1
        
        return {
            "total_events": self.event_counter,
            "history_size": len(self.event_history),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "by_priority": dict(by_priority),
            "top_types": sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10],
            **self._stats,
        }
    
    def clear_history(self) -> None:
        self.event_history.clear()
    
    def subscribe_multiple(self, topics: List[str], handler: Callable) -> List[Subscription]:
        subscriptions = []
        for topic in topics:
            subscriptions.append(self.subscribe(topic, handler))
        return subscriptions
    
    def publish_with_trace(self, event_type: str, source: str,
                           payload: Dict[str, Any] = None, **kwargs) -> Event:
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            **kwargs,
        )
        
        self._store_event(event)
        self._dispatch(event)
        return event
    
    def get_subscription_count(self) -> int:
        return sum(len(subs) for subs in self.subscriptions.values())
    
    def get_topics(self) -> Set[str]:
        return set(self.subscriptions.keys())
    
    def enable_monitoring(self, enabled: bool) -> None:
        self._enable_monitoring = enabled
    
    def shutdown(self) -> None:
        if self._processing_task:
            self._processing_task.cancel()
        self.clear_history()
        self.subscriptions.clear()


_global_event_bus: Optional[EventBus] = None


def get_event_bus(max_history: int = 1000, async_enabled: bool = True) -> EventBus:
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus(max_history=max_history, async_enabled=async_enabled)
    return _global_event_bus