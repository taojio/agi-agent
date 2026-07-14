"""
core/base_module.py - 模块基类

所有功能模块的统一基类，提供标准化的生命周期管理
v2.0 升级：增加模块总线集成、能力声明、消息处理
"""
import abc
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .exceptions import InitializationException, ModuleException


class ModuleStatus(Enum):
    """模块状态"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class BaseModule(abc.ABC):
    """模块基类

    所有功能模块都应继承此类，提供统一的生命周期管理接口

    子类需要实现的方法：
    - _initialize: 初始化逻辑
    - _shutdown: 关闭逻辑
    - _health_check: 健康检查

    v2.0 新增：
    - 模块总线集成（_bus 属性）
    - 能力声明（get_capabilities）
    - 消息处理（handle_message）
    - 事件发布与订阅
    """

    name: str = "base_module"
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = []

    def __init__(self):
        self._status: ModuleStatus = ModuleStatus.UNINITIALIZED
        self._started_at: Optional[float] = None
        self._error: Optional[str] = None
        self._metrics: Dict[str, Any] = {}
        self._bus = None
        self._subscriptions: List[Any] = []
        self._service_endpoints: Dict[str, Callable] = {}

    @property
    def status(self) -> ModuleStatus:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status == ModuleStatus.READY

    @property
    def is_running(self) -> bool:
        return self._status == ModuleStatus.RUNNING

    @property
    def uptime(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.time() - self._started_at

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化模块

        Args:
            config: 配置字典
        """
        if self._status not in (ModuleStatus.UNINITIALIZED, ModuleStatus.ERROR):
            raise InitializationException(
                f"Module '{self.name}' cannot be initialized from state {self._status.value}",
                module_name=self.name,
            )

        self._status = ModuleStatus.INITIALIZING
        try:
            self._initialize(config or {})
            self._status = ModuleStatus.READY
            self._started_at = time.time()
            self._error = None
        except Exception as e:
            self._status = ModuleStatus.ERROR
            self._error = str(e)
            raise InitializationException(
                f"Failed to initialize module '{self.name}': {e}",
                module_name=self.name,
            ) from e

    def start(self) -> None:
        """启动模块"""
        if self._status != ModuleStatus.READY:
            raise ModuleException(
                f"Module '{self.name}' is not ready (status: {self._status.value})",
                module_name=self.name,
            )
        self._status = ModuleStatus.RUNNING
        self._start()

    def stop(self) -> None:
        """停止模块"""
        if self._status == ModuleStatus.RUNNING:
            self._stop()
            self._status = ModuleStatus.READY

    def shutdown(self) -> None:
        """关闭模块，释放资源"""
        try:
            if self._status != ModuleStatus.SHUTDOWN:
                self._shutdown()
        finally:
            self._status = ModuleStatus.SHUTDOWN

    def health_check(self) -> bool:
        """健康检查

        Returns:
            bool: 模块是否健康
        """
        if self._status in (ModuleStatus.UNINITIALIZED, ModuleStatus.ERROR, ModuleStatus.SHUTDOWN):
            return False
        try:
            return self._health_check()
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取模块状态信息

        Returns:
            dict: 状态信息字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "status": self._status.value,
            "uptime": self.uptime,
            "error": self._error,
            "metrics": self._metrics,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取模块指标"""
        return dict(self._metrics)

    def update_metric(self, key: str, value: Any) -> None:
        """更新指标"""
        self._metrics[key] = value

    # ====== 子类需实现的方法 ======

    def _initialize(self, config: Dict[str, Any]) -> None:
        """初始化逻辑（子类实现）"""
        pass

    def _start(self) -> None:
        """启动逻辑（子类实现）"""
        pass

    def _stop(self) -> None:
        """停止逻辑（子类实现）"""
        pass

    def _shutdown(self) -> None:
        """关闭逻辑（子类实现）"""
        pass

    def _health_check(self) -> bool:
        """健康检查逻辑（子类实现）"""
        return True

    # ====== v2.0: 模块总线集成 ======

    def attach_bus(self, bus) -> None:
        """挂载模块总线

        Args:
            bus: ModuleBus 实例
        """
        self._bus = bus
        self._register_with_bus()

    def _register_with_bus(self) -> None:
        """向总线注册服务和能力（子类可重写）"""
        if self._bus is None:
            return

        for endpoint, handler in self._service_endpoints.items():
            self._bus.register_service(endpoint, handler)

        capabilities = self.get_capabilities()
        if hasattr(self._bus, '_service_registry'):
            for cap in capabilities:
                self._bus._service_registry.register(self.name, cap)

    def detach_bus(self) -> None:
        """从总线注销"""
        if self._bus is None:
            return

        for sub in self._subscriptions:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()

        for endpoint in list(self._service_endpoints.keys()):
            try:
                self._bus.unregister_service(endpoint)
            except Exception:
                pass

        if hasattr(self._bus, '_service_registry'):
            self._bus._service_registry.unregister(self.name)

        self._bus = None

    def get_capabilities(self) -> List[Any]:
        """声明模块提供的能力

        子类应重写此方法声明自己的能力列表。

        Returns:
            List[ModuleCapability]: 能力描述列表
        """
        return []

    def handle_message(self, message: Any) -> Optional[Any]:
        """处理来自总线的消息

        子类可重写此方法处理自定义消息。

        Args:
            message: ModuleMessage 实例

        Returns:
            ModuleResponse 或 None
        """
        return None

    def publish_event(self, event_type: str, payload: Dict[str, Any] = None,
                      severity: Any = None, category: Any = None) -> None:
        """通过总线发布事件

        Args:
            event_type: 事件类型
            payload: 事件数据
            severity: 严重级别
            category: 事件分类
        """
        if self._bus is None:
            return

        try:
            from ..module_bus.event import EventSeverity, EventCategory
            sev = severity or EventSeverity.INFO
            cat = category or EventCategory.SYSTEM
            self._bus.publish_event(
                event_type=event_type,
                source=self.name,
                payload=payload,
                severity=sev,
                category=cat,
            )
        except Exception:
            pass

    def subscribe_event(self, topic: str, handler: Callable) -> Optional[Any]:
        """订阅总线事件

        Args:
            topic: 事件主题模式
            handler: 处理函数

        Returns:
            订阅句柄
        """
        if self._bus is None:
            return None

        sub = self._bus.subscribe(topic, handler)
        self._subscriptions.append(sub)
        return sub

    def send_request(self, endpoint: str, target: str,
                     payload: Dict[str, Any] = None,
                     timeout: float = 30.0) -> Optional[Any]:
        """发送请求到其他模块

        Args:
            endpoint: 服务端点
            target: 目标模块名
            payload: 请求数据
            timeout: 超时时间

        Returns:
            ModuleResponse 或 None
        """
        if self._bus is None:
            return None

        try:
            return self._bus.send_request(
                source=self.name,
                target=target,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
            )
        except Exception:
            return None

    def get_health_status(self) -> HealthStatus:
        """获取详细健康状态

        Returns:
            HealthStatus: 健康状态枚举
        """
        if self._status in (ModuleStatus.UNINITIALIZED, ModuleStatus.SHUTDOWN):
            return HealthStatus.UNKNOWN
        if self._status == ModuleStatus.ERROR:
            return HealthStatus.UNHEALTHY

        try:
            healthy = self._health_check()
            if healthy:
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY

    # ====== 便捷方法：注册服务端点 ======

    def _register_endpoint(self, endpoint: str, handler: Callable) -> None:
        """注册服务端点（在 _register_with_bus 中会自动注册到总线）

        Args:
            endpoint: 端点名称
            handler: 处理函数
        """
        self._service_endpoints[endpoint] = handler

    def _unregister_endpoint(self, endpoint: str) -> None:
        """注销服务端点"""
        self._service_endpoints.pop(endpoint, None)
