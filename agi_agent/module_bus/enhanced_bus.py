import time
import uuid
import threading
import asyncio
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from enum import Enum
from dataclasses import dataclass, field

from .message import (
    ModuleMessage,
    MessageType,
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
    CapabilityType,
    get_service_registry,
)
from .data_stream import (
    DataStream,
    StreamConsumer,
    BackPressureStrategy,
    create_stream,
    get_stream,
)

from agi_agent.injection import DependencyInjectionContainer, get_container


class ServiceStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class VersionMatch(Enum):
    EXACT = "exact"
    COMPATIBLE = "compatible"
    ANY = "any"


@dataclass
class ServiceInterceptor:
    name: str
    before: Optional[Callable] = None
    after: Optional[Callable] = None
    on_error: Optional[Callable] = None
    priority: int = 0


@dataclass
class ServiceVersion:
    major: int = 1
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, version_str: str) -> "ServiceVersion":
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: "ServiceVersion") -> bool:
        return self.major == other.major


class EnhancedModuleBus:
    def __init__(self, max_workers: int = 10, di_container: DependencyInjectionContainer = None):
        self._max_workers = max_workers
        self._di_container = di_container or get_container()

        self._event_subscribers: Dict[str, List[Tuple[str, Callable]]] = {}
        self._subscriptions: Dict[str, "Subscription"] = {}

        self._service_handlers: Dict[str, Dict[str, Callable]] = {}
        self._service_versions: Dict[str, Dict[str, ServiceVersion]] = {}
        self._service_interceptors: Dict[str, List[ServiceInterceptor]] = {}
        self._pending_requests: Dict[str, "PendingRequest"] = {}

        self._streams: Dict[str, DataStream] = {}
        self._service_registry: ServiceRegistry = get_service_registry()

        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._stats = {
            "events_published": 0,
            "requests_sent": 0,
            "responses_received": 0,
            "stream_messages": 0,
            "errors": 0,
            "dropped_events": 0,
            "interceptor_calls": 0,
        }

        self._lock = threading.Lock()
        self._stop_cleanup = threading.Event()
        self._stop_health_check = threading.Event()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired, daemon=True
        )
        self._cleanup_thread.start()

        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._health_check_thread.start()

        self._load_avg: Dict[str, deque] = {}

    def __del__(self):
        try:
            self._stop_cleanup.set()
            self._stop_health_check.set()
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def register_service(
        self,
        endpoint: str,
        handler: Callable[[ModuleMessage], ModuleResponse],
        version: str = "1.0.0",
        capability: Optional[ModuleCapability] = None,
        interceptors: Optional[List[ServiceInterceptor]] = None,
    ) -> bool:
        service_version = ServiceVersion.parse(version)

        with self._lock:
            if endpoint not in self._service_handlers:
                self._service_handlers[endpoint] = {}
                self._service_versions[endpoint] = {}

            version_str = str(service_version)
            if version_str in self._service_handlers[endpoint]:
                return False

            self._service_handlers[endpoint][version_str] = handler
            self._service_versions[endpoint][version_str] = service_version

            if interceptors:
                self._service_interceptors[endpoint] = sorted(
                    interceptors, key=lambda x: x.priority
                )

            if capability:
                capability.version = version
                self._service_registry.register(
                    endpoint.split(".")[0] if "." in endpoint else endpoint,
                    capability
                )

            self._load_avg[endpoint] = deque(maxlen=10)

        return True

    def register_service_with_di(
        self,
        endpoint: str,
        service_type: Type,
        version: str = "1.0.0",
        capability: Optional[ModuleCapability] = None,
    ) -> bool:
        self._di_container.register(service_type, lifecycle="singleton")

        def di_handler(message: ModuleMessage) -> ModuleResponse:
            service = self._di_container.resolve(service_type)
            method_name = capability.endpoint.split(".")[-1] if capability else "handle"
            if hasattr(service, method_name):
                return getattr(service, method_name)(message)
            return create_error_response(
                message, "Method not found", "METHOD_NOT_FOUND", ResponseStatus.FAILED
            )

        return self.register_service(endpoint, di_handler, version, capability)

    def unregister_service(self, endpoint: str, version: str = None) -> bool:
        with self._lock:
            if endpoint not in self._service_handlers:
                return False

            if version:
                version_str = str(ServiceVersion.parse(version))
                if version_str in self._service_handlers[endpoint]:
                    del self._service_handlers[endpoint][version_str]
                    del self._service_versions[endpoint][version_str]
                    if not self._service_handlers[endpoint]:
                        del self._service_handlers[endpoint]
                        del self._service_versions[endpoint]
                    return True
                return False

            del self._service_handlers[endpoint]
            del self._service_versions[endpoint]
            if endpoint in self._service_interceptors:
                del self._service_interceptors[endpoint]
            return True

    def register_interceptor(self, endpoint: str, interceptor: ServiceInterceptor) -> None:
        with self._lock:
            if endpoint not in self._service_interceptors:
                self._service_interceptors[endpoint] = []
            self._service_interceptors[endpoint].append(interceptor)
            self._service_interceptors[endpoint].sort(key=lambda x: x.priority)

    def _select_version(self, endpoint: str, target_version: str = None,
                        match_strategy: VersionMatch = VersionMatch.COMPATIBLE) -> Optional[str]:
        with self._lock:
            if endpoint not in self._service_versions:
                return None

            versions = list(self._service_versions[endpoint].keys())
            if not versions:
                return None

            if not target_version:
                return max(versions, key=lambda v: self._service_versions[endpoint][v])

            target = ServiceVersion.parse(target_version)
            if match_strategy == VersionMatch.EXACT:
                if target_version in versions:
                    return target_version
                return None

            compatible = [
                v for v in versions
                if self._service_versions[endpoint][v].is_compatible(target)
            ]
            if compatible:
                return max(compatible, key=lambda v: self._service_versions[endpoint][v])

            if match_strategy == VersionMatch.ANY:
                return max(versions, key=lambda v: self._service_versions[endpoint][v])

            return None

    def request(
        self,
        endpoint: str,
        message: ModuleMessage,
        timeout: float = 30.0,
        version: str = None,
        match_strategy: VersionMatch = VersionMatch.COMPATIBLE,
    ) -> ModuleResponse:
        version_str = self._select_version(endpoint, version, match_strategy)
        if not version_str:
            return create_error_response(
                message,
                f"No compatible version found for '{endpoint}'",
                "VERSION_NOT_FOUND",
                ResponseStatus.NOT_FOUND,
            )

        with self._lock:
            handlers = self._service_handlers.get(endpoint, {})
            handler = handlers.get(version_str)

        if handler is None:
            return create_error_response(
                message,
                f"Service endpoint '{endpoint}' not found",
                "ENDPOINT_NOT_FOUND",
                ResponseStatus.NOT_FOUND,
            )

        self._stats["requests_sent"] += 1
        start_time = time.time()

        interceptors = self._service_interceptors.get(endpoint, [])

        for interceptor in interceptors:
            if interceptor.before:
                try:
                    interceptor.before(endpoint, message)
                    self._stats["interceptor_calls"] += 1
                except Exception:
                    self._stats["errors"] += 1

        try:
            response = handler(message)
            response.duration_ms = (time.time() - start_time) * 1000

            for interceptor in interceptors:
                if interceptor.after:
                    try:
                        interceptor.after(endpoint, message, response)
                        self._stats["interceptor_calls"] += 1
                    except Exception:
                        self._stats["errors"] += 1

            self._stats["responses_received"] += 1

            with self._lock:
                if endpoint in self._load_avg:
                    self._load_avg[endpoint].append(response.duration_ms)

            return response
        except Exception as e:
            self._stats["errors"] += 1

            for interceptor in interceptors:
                if interceptor.on_error:
                    try:
                        interceptor.on_error(endpoint, message, e)
                        self._stats["interceptor_calls"] += 1
                    except Exception:
                        self._stats["errors"] += 1

            return create_error_response(
                message, str(e), "HANDLER_EXCEPTION", ResponseStatus.FAILED
            )

    def request_async(
        self,
        endpoint: str,
        message: ModuleMessage,
        timeout: float = 30.0,
        version: str = None,
        match_strategy: VersionMatch = VersionMatch.COMPATIBLE,
    ) -> Future:
        future = self._executor.submit(
            self.request, endpoint, message, timeout, version, match_strategy
        )
        return future

    async def request_asyncio(
        self,
        endpoint: str,
        message: ModuleMessage,
        timeout: float = 30.0,
        version: str = None,
        match_strategy: VersionMatch = VersionMatch.COMPATIBLE,
    ) -> ModuleResponse:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            self.request,
            endpoint,
            message,
            timeout,
            version,
            match_strategy,
        )
        return response

    def publish(self, event: ModuleEvent) -> None:
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

    def subscribe(self, topic: str, handler: Callable) -> "Subscription":
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if topic not in self._event_subscribers:
                self._event_subscribers[topic] = []
            self._event_subscribers[topic].append((subscription_id, handler))

            sub = Subscription(subscription_id, topic, handler, self)
            self._subscriptions[subscription_id] = sub

        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
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

    def create_stream(self, stream_id: str, buffer_size: int = 1000,
                      back_pressure: BackPressureStrategy = BackPressureStrategy.DROP_OLDEST) -> DataStream:
        if stream_id in self._streams:
            return self._streams[stream_id]

        stream = create_stream(stream_id, buffer_size, back_pressure)
        self._streams[stream_id] = stream
        return stream

    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        return self._streams.get(stream_id) or get_stream(stream_id)

    def send_to_stream(self, stream_id: str, data: Any) -> bool:
        stream = self.get_stream(stream_id)
        if not stream:
            return False
        self._stats["stream_messages"] += 1
        return stream.publish(data)

    def subscribe_stream(self, stream_id: str, callback: Callable,
                         consumer_id: str = None) -> Optional[str]:
        stream = self.get_stream(stream_id)
        if not stream:
            return None
        return stream.subscribe(callback, consumer_id)

    def get_service_versions(self, endpoint: str) -> List[str]:
        with self._lock:
            if endpoint not in self._service_versions:
                return []
            return list(self._service_versions[endpoint].keys())

    def get_service_load(self, endpoint: str) -> float:
        with self._lock:
            if endpoint not in self._load_avg:
                return 0.0
            times = list(self._load_avg[endpoint])
            if not times:
                return 0.0
            return sum(times) / len(times)

    def health_check(self, endpoint: str) -> Dict[str, Any]:
        versions = self.get_service_versions(endpoint)
        load = self.get_service_load(endpoint)

        status = ServiceStatus.INACTIVE
        if versions:
            status = ServiceStatus.ACTIVE if load < 500 else ServiceStatus.DEGRADED

        return {
            "endpoint": endpoint,
            "versions": versions,
            "load_ms": load,
            "status": status.value,
        }

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        with self._lock:
            for endpoint in self._service_handlers:
                result[endpoint] = self.health_check(endpoint)
        return result

    def _health_check_loop(self) -> None:
        while not self._stop_health_check.is_set():
            time.sleep(10)
            try:
                health = self.health_check_all()
                for endpoint, info in health.items():
                    if info["status"] == ServiceStatus.DEGRADED.value:
                        self.publish_event(
                            "service.degraded",
                            "module_bus",
                            {"endpoint": endpoint, "load_ms": info["load_ms"]},
                            severity=EventSeverity.WARNING,
                        )
                    elif info["status"] == ServiceStatus.INACTIVE.value:
                        self.publish_event(
                            "service.inactive",
                            "module_bus",
                            {"endpoint": endpoint},
                            severity=EventSeverity.ERROR,
                        )
            except Exception:
                pass

    def publish_event(self, event_type: str, source: str,
                      payload: Dict[str, Any] = None,
                      severity: EventSeverity = EventSeverity.INFO,
                      category: EventCategory = EventCategory.SYSTEM) -> None:
        event = create_event(
            event_type=event_type,
            source=source,
            payload=payload,
            severity=severity,
            category=category,
        )
        self.publish(event)

    def send_request(self, source: str, target: str, endpoint: str,
                     payload: Dict[str, Any] = None, timeout: float = 30.0,
                     version: str = None) -> ModuleResponse:
        msg = create_message(
            source=source,
            target=target,
            payload=payload or {},
            message_type=MessageType.REQUEST,
        )
        return self.request(endpoint, msg, timeout, version)

    def _safe_handle(self, handler: Callable, event: ModuleEvent) -> None:
        try:
            handler(event)
        except Exception:
            self._stats["errors"] += 1

    def _cleanup_expired(self) -> None:
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
                            pr.future.set_exception(TimeoutError(f"Request {rid} timed out"))
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
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
                    "active_services": sum(len(v) for v in self._service_handlers.values()),
                },
                "streams": {
                    "count": len(self._streams),
                    "total_messages": self._stats["stream_messages"],
                },
                "errors": self._stats["errors"],
                "interceptors": {
                    "calls": self._stats["interceptor_calls"],
                    "registered": sum(len(v) for v in self._service_interceptors.values()),
                },
                "service_registry": self._service_registry.get_stats(),
            }

    def shutdown(self) -> None:
        self._stop_cleanup.set()
        self._stop_health_check.set()
        self._executor.shutdown(wait=True)
        for stream in self._streams.values():
            stream.close()


class Subscription:
    def __init__(self, subscription_id: str, topic: str,
                 handler: Callable, bus: EnhancedModuleBus):
        self.subscription_id = subscription_id
        self.topic = topic
        self.handler = handler
        self._bus = bus
        self.active = True

    def unsubscribe(self) -> bool:
        if self.active:
            self.active = False
            return self._bus.unsubscribe(self.subscription_id)
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unsubscribe()


class PendingRequest:
    def __init__(self, request: ModuleMessage, future: Future, timeout: float = 30.0):
        self.request = request
        self.future = future
        self.timeout = timeout
        self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.timeout


_global_enhanced_bus: Optional[EnhancedModuleBus] = None
_init_lock = threading.Lock()


def get_enhanced_bus(max_workers: int = 10) -> EnhancedModuleBus:
    global _global_enhanced_bus
    if _global_enhanced_bus is None:
        with _init_lock:
            if _global_enhanced_bus is None:
                _global_enhanced_bus = EnhancedModuleBus(max_workers=max_workers)
    return _global_enhanced_bus