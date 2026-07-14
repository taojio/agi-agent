"""
module_bus/bus.py - 模块通信总线

模块通信总线的主类，整合事件驱动、请求响应、数据流三种通信模式
"""
import time
import uuid
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .message import (
    ModuleMessage,
    MessageType,
    MessagePriority,
    ModuleResponse,
    ResponseStatus,
    create_message,
    create_response,
    create_error_response,
)
from .event import (
    ModuleEvent,
    EventSeverity,
    EventCategory,
    create_event,
)
from .service_registry import (
    ServiceRegistry,
    ModuleCapability,
    ServiceInfo,
    get_service_registry,
)
from .data_stream import (
    DataStream,
    StreamConsumer,
    BackPressureStrategy,
    create_stream,
    get_stream,
)


class Subscription:
    """事件订阅句柄"""

    def __init__(self, subscription_id: str, topic: str,
                 handler: Callable, bus: "ModuleBus"):
        self.subscription_id = subscription_id
        self.topic = topic
        self.handler = handler
        self._bus = bus
        self.active = True

    def unsubscribe(self) -> bool:
        """取消订阅"""
        if self.active:
            self.active = False
            return self._bus.unsubscribe(self.subscription_id)
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unsubscribe()


class PendingRequest:
    """挂起的请求（等待响应）"""

    def __init__(self, request: ModuleMessage, future: Future,
                 timeout: float = 30.0):
        self.request = request
        self.future = future
        self.timeout = timeout
        self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.timeout


class ModuleBus:
    """模块通信总线

    统一的模块间通信基础设施，支持三种通信模式：
    1. 事件驱动（发布/订阅）- 异步、一对多、解耦
    2. 请求响应（同步/异步）- 同步、一对一、需响应
    3. 数据流（流式传输）- 连续数据、多消费者、背压

    使用单例模式，全局共享一个总线实例。
    """

    _instance: Optional["ModuleBus"] = None

    def __init__(self, max_workers: int = 10):
        self._max_workers = max_workers

        # 事件订阅
        self._event_subscribers: Dict[str, List[Tuple[str, Callable]]] = {}
        self._subscriptions: Dict[str, Subscription] = {}

        # 请求响应
        self._service_handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[str, PendingRequest] = {}

        # 数据流
        self._streams: Dict[str, DataStream] = {}

        # 服务注册中心
        self._service_registry: ServiceRegistry = get_service_registry()

        # 线程池（异步处理）
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # 统计
        self._stats = {
            "events_published": 0,
            "requests_sent": 0,
            "responses_received": 0,
            "stream_messages": 0,
            "errors": 0,
            "dropped_events": 0,
        }

        # 锁
        self._lock = threading.Lock()

        # 启动过期清理线程
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired,
            daemon=True,
        )
        self._cleanup_thread.start()

    def __del__(self):
        try:
            self._stop_cleanup.set()
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ============================================================
    # 事件驱动模式
    # ============================================================

    def publish(self, event: ModuleEvent) -> None:
        """发布事件

        Args:
            event: 模块事件
        """
        self._stats["events_published"] += 1

        handlers = []
        with self._lock:
            for pattern, subs in self._event_subscribers.items():
                if event.matches(pattern):
                    for _, handler in subs:
                        handlers.append(handler)

        if not handlers:
            self._stats["dropped_events"] += 1
            return

        for handler in handlers:
            try:
                self._executor.submit(self._safe_handle, handler, event)
            except Exception:
                self._stats["errors"] += 1

    def subscribe(self, topic: str, handler: Callable) -> Subscription:
        """订阅事件

        Args:
            topic: 事件类型模式（支持 "*" 和 "prefix.*"）
            handler: 事件处理函数

        Returns:
            订阅句柄，可用于取消订阅
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if topic not in self._event_subscribers:
                self._event_subscribers[topic] = []
            self._event_subscribers[topic].append((subscription_id, handler))

            sub = Subscription(subscription_id, topic, handler, self)
            self._subscriptions[subscription_id] = sub

        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            是否成功取消
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                return False

            sub = self._subscriptions.pop(subscription_id)
            topic = sub.topic

            if topic in self._event_subscribers:
                self._event_subscribers[topic] = [
                    (sid, h) for sid, h in self._event_subscribers[topic]
                    if sid != subscription_id
                ]
                if not self._event_subscribers[topic]:
                    del self._event_subscribers[topic]

        return True

    # ============================================================
    # 请求响应模式
    # ============================================================

    def register_service(self, endpoint: str,
                         handler: Callable[[ModuleMessage], ModuleResponse]) -> bool:
        """注册服务端点

        Args:
            endpoint: 端点名称（如 "memory.query"）
            handler: 处理函数，接收 ModuleMessage 返回 ModuleResponse

        Returns:
            是否注册成功
        """
        with self._lock:
            if endpoint in self._service_handlers:
                return False
            self._service_handlers[endpoint] = handler
        return True

    def unregister_service(self, endpoint: str) -> bool:
        """注销服务端点"""
        with self._lock:
            if endpoint in self._service_handlers:
                del self._service_handlers[endpoint]
                return True
        return False

    def request(self, endpoint: str, message: ModuleMessage,
                timeout: float = 30.0) -> ModuleResponse:
        """同步请求

        Args:
            endpoint: 服务端点
            message: 请求消息
            timeout: 超时时间（秒）

        Returns:
            响应消息

        Raises:
            TimeoutError: 请求超时
            ValueError: 端点不存在
        """
        with self._lock:
            handler = self._service_handlers.get(endpoint)

        if handler is None:
            return create_error_response(
                message,
                f"Service endpoint '{endpoint}' not found",
                "ENDPOINT_NOT_FOUND",
                ResponseStatus.NOT_FOUND,
            )

        self._stats["requests_sent"] += 1
        start_time = time.time()

        try:
            response = handler(message)
            response.duration_ms = (time.time() - start_time) * 1000
            self._stats["responses_received"] += 1
            return response
        except Exception as e:
            self._stats["errors"] += 1
            return create_error_response(
                message,
                str(e),
                "HANDLER_EXCEPTION",
                ResponseStatus.FAILED,
            )

    def request_async(self, endpoint: str, message: ModuleMessage,
                      timeout: float = 30.0) -> Future:
        """异步请求

        Args:
            endpoint: 服务端点
            message: 请求消息
            timeout: 超时时间（秒）

        Returns:
            Future 对象，结果为 ModuleResponse
        """
        future = self._executor.submit(
            self.request, endpoint, message, timeout
        )
        return future

    # ============================================================
    # 数据流模式
    # ============================================================

    def create_stream(self, stream_id: str,
                      buffer_size: int = 1000,
                      back_pressure: BackPressureStrategy = BackPressureStrategy.DROP_OLDEST
                      ) -> DataStream:
        """创建数据流

        Args:
            stream_id: 流ID
            buffer_size: 缓冲区大小
            back_pressure: 背压策略

        Returns:
            数据流对象
        """
        if stream_id in self._streams:
            return self._streams[stream_id]

        stream = create_stream(stream_id, buffer_size, back_pressure)
        self._streams[stream_id] = stream
        return stream

    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        """获取数据流"""
        return self._streams.get(stream_id) or get_stream(stream_id)

    def send_to_stream(self, stream_id: str, data: Any) -> bool:
        """发送数据到流

        Args:
            stream_id: 流ID
            data: 数据

        Returns:
            是否成功
        """
        stream = self.get_stream(stream_id)
        if not stream:
            return False
        self._stats["stream_messages"] += 1
        return stream.publish(data)

    def subscribe_stream(self, stream_id: str,
                         callback: Callable,
                         consumer_id: str = None) -> Optional[str]:
        """订阅数据流

        Args:
            stream_id: 流ID
            callback: 回调函数
            consumer_id: 消费者ID

        Returns:
            消费者ID，None 表示流不存在
        """
        stream = self.get_stream(stream_id)
        if not stream:
            return None
        return stream.subscribe(callback, consumer_id)

    # ============================================================
    # 便捷方法
    # ============================================================

    def publish_event(self, event_type: str, source: str,
                      payload: Dict[str, Any] = None,
                      severity: EventSeverity = EventSeverity.INFO,
                      category: EventCategory = EventCategory.SYSTEM
                      ) -> None:
        """便捷方法：发布事件"""
        event = create_event(
            event_type=event_type,
            source=source,
            payload=payload,
            severity=severity,
            category=category,
        )
        self.publish(event)

    def send_request(self, source: str, target: str,
                     endpoint: str,
                     payload: Dict[str, Any] = None,
                     timeout: float = 30.0) -> ModuleResponse:
        """便捷方法：发送请求"""
        msg = create_message(
            source=source,
            target=target,
            payload=payload or {},
            message_type=MessageType.REQUEST,
        )
        return self.request(endpoint, msg, timeout)

    # ============================================================
    # 内部方法
    # ============================================================

    def _safe_handle(self, handler: Callable, event: ModuleEvent) -> None:
        """安全调用事件处理器"""
        try:
            handler(event)
        except Exception:
            self._stats["errors"] += 1

    def _cleanup_expired(self) -> None:
        """清理过期的挂起请求"""
        while not self._stop_cleanup.is_set():
            time.sleep(5)
            try:
                with self._lock:
                    expired = [
                        rid for rid, pr in self._pending_requests.items()
                        if pr.is_expired
                    ]
                    for rid in expired:
                        pr = self._pending_requests.pop(rid)
                        if not pr.future.done():
                            pr.future.set_exception(TimeoutError(
                                f"Request {rid} timed out"
                            ))
            except Exception:
                pass

    # ============================================================
    # 统计与状态
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取总线统计"""
        with self._lock:
            return {
                "events": {
                    "published": self._stats["events_published"],
                    "dropped": self._stats["dropped_events"],
                    "active_subscriptions": len(self._subscriptions),
                    "topics": len(self._event_subscribers),
                },
                "requests": {
                    "sent": self._stats["requests_sent"],
                    "responses": self._stats["responses_received"],
                    "pending": len(self._pending_requests),
                    "active_services": len(self._service_handlers),
                },
                "streams": {
                    "count": len(self._streams),
                    "total_messages": self._stats["stream_messages"],
                },
                "errors": self._stats["errors"],
                "service_registry": self._service_registry.get_stats(),
            }

    def shutdown(self) -> None:
        """关闭总线"""
        self._stop_cleanup.set()
        self._executor.shutdown(wait=True)
        for stream in self._streams.values():
            stream.close()


_global_bus: Optional[ModuleBus] = None
_init_lock = threading.Lock()


def get_module_bus(max_workers: int = 10) -> ModuleBus:
    """获取全局模块通信总线单例"""
    global _global_bus
    if _global_bus is None:
        with _init_lock:
            if _global_bus is None:
                _global_bus = ModuleBus(max_workers=max_workers)
    return _global_bus
