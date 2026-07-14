"""
module_bus/service_registry.py - 服务注册与发现

提供模块能力的注册和发现机制，支持动态服务查找
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CapabilityType(Enum):
    """能力类型"""
    DATA_PROCESSING = "data_processing"      # 数据处理
    DECISION = "decision"                    # 决策
    PREDICTION = "prediction"                # 预测
    ANALYSIS = "analysis"                    # 分析
    STORAGE = "storage"                      # 存储
    SECURITY = "security"                    # 安全
    MONITORING = "monitoring"                # 监控
    LEARNING = "learning"                    # 学习
    EVOLUTION = "evolution"                  # 进化
    COMMUNICATION = "communication"          # 通信
    UI = "ui"                                # 界面
    OTHER = "other"                          # 其他


@dataclass
class ModuleCapability:
    """模块能力描述

    描述一个模块对外提供的能力，用于服务发现
    """

    name: str
    capability_type: CapabilityType
    description: str = ""
    version: str = "1.0.0"
    endpoint: str = ""                        # 服务端点（用于请求响应模式）
    events_published: List[str] = field(default_factory=list)  # 发布的事件类型
    events_subscribed: List[str] = field(default_factory=list)  # 订阅的事件类型
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    quality_of_service: Dict[str, Any] = field(default_factory=dict)  # 服务质量参数
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "version": self.version,
            "endpoint": self.endpoint,
            "events_published": self.events_published,
            "events_subscribed": self.events_subscribed,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "quality_of_service": self.quality_of_service,
            "tags": self.tags,
        }


@dataclass
class ServiceInfo:
    """服务信息"""

    module_name: str
    capability: ModuleCapability
    status: str = "active"  # active / degraded / inactive
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    load: float = 0.0       # 负载 0.0 - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "capability": self.capability.to_dict(),
            "status": self.status,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "load": self.load,
        }


class ServiceRegistry:
    """服务注册中心

    管理所有模块提供的服务，支持：
    - 服务注册与注销
    - 按类型/名称/标签发现服务
    - 负载均衡选择
    - 健康状态监控
    """

    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._module_services: Dict[str, List[str]] = {}

    def register(self, module_name: str, capability: ModuleCapability) -> bool:
        """注册服务

        Args:
            module_name: 模块名称
            capability: 能力描述

        Returns:
            是否注册成功
        """
        service_id = f"{module_name}.{capability.name}"

        if service_id in self._services:
            return False

        self._services[service_id] = ServiceInfo(
            module_name=module_name,
            capability=capability,
            status="active",
        )

        if module_name not in self._module_services:
            self._module_services[module_name] = []
        self._module_services[module_name].append(service_id)

        return True

    def unregister(self, module_name: str, capability_name: str = None) -> bool:
        """注销服务

        Args:
            module_name: 模块名称
            capability_name: 能力名称，None 则注销该模块所有服务

        Returns:
            是否注销成功
        """
        if capability_name:
            service_id = f"{module_name}.{capability_name}"
            if service_id in self._services:
                del self._services[service_id]
                if module_name in self._module_services:
                    if service_id in self._module_services[module_name]:
                        self._module_services[module_name].remove(service_id)
                return True
            return False

        if module_name in self._module_services:
            for service_id in self._module_services[module_name]:
                if service_id in self._services:
                    del self._services[service_id]
            del self._module_services[module_name]
            return True

        return False

    def discover(self, capability_type: CapabilityType = None,
                 tags: List[str] = None) -> List[ServiceInfo]:
        """发现服务

        Args:
            capability_type: 能力类型过滤
            tags: 标签过滤（需全部匹配）

        Returns:
            匹配的服务列表
        """
        results = list(self._services.values())

        if capability_type:
            results = [
                s for s in results
                if s.capability.capability_type == capability_type
            ]

        if tags:
            results = [
                s for s in results
                if all(tag in s.capability.tags for tag in tags)
            ]

        return results

    def discover_by_name(self, capability_name: str) -> List[ServiceInfo]:
        """按能力名称发现服务"""
        return [
            s for s in self._services.values()
            if s.capability.name == capability_name
        ]

    def select_service(self, capability_name: str,
                       strategy: str = "round_robin") -> Optional[ServiceInfo]:
        """选择一个服务实例（负载均衡）

        Args:
            capability_name: 能力名称
            strategy: 选择策略 (round_robin / least_load / random)

        Returns:
            选中的服务，None 表示没有可用服务
        """
        services = [
            s for s in self.discover_by_name(capability_name)
            if s.status == "active"
        ]

        if not services:
            return None

        if strategy == "least_load":
            services.sort(key=lambda s: s.load)
            return services[0]

        if strategy == "random":
            import random
            return random.choice(services)

        # round_robin (默认)
        service_id = getattr(self, f"_rr_{capability_name}", 0)
        service = services[service_id % len(services)]
        setattr(self, f"_rr_{capability_name}", service_id + 1)
        return service

    def update_service_status(self, module_name: str,
                              capability_name: str,
                              status: str) -> bool:
        """更新服务状态"""
        service_id = f"{module_name}.{capability_name}"
        if service_id in self._services:
            self._services[service_id].status = status
            return True
        return False

    def update_load(self, module_name: str,
                    capability_name: str,
                    load: float) -> bool:
        """更新服务负载"""
        service_id = f"{module_name}.{capability_name}"
        if service_id in self._services:
            self._services[service_id].load = max(0.0, min(1.0, load))
            self._services[service_id].last_heartbeat = time.time()
            return True
        return False

    def heartbeat(self, module_name: str) -> int:
        """模块心跳，更新所有该模块服务的心跳时间"""
        count = 0
        if module_name in self._module_services:
            for service_id in self._module_services[module_name]:
                if service_id in self._services:
                    self._services[service_id].last_heartbeat = time.time()
                    count += 1
        return count

    def get_all_services(self) -> Dict[str, ServiceInfo]:
        """获取所有服务"""
        return dict(self._services)

    def get_module_services(self, module_name: str) -> List[ServiceInfo]:
        """获取指定模块的所有服务"""
        if module_name not in self._module_services:
            return []
        return [
            self._services[sid]
            for sid in self._module_services[module_name]
            if sid in self._services
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取服务注册中心统计"""
        active = sum(1 for s in self._services.values() if s.status == "active")
        by_type = {}
        for s in self._services.values():
            t = s.capability.capability_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_services": len(self._services),
            "active_services": active,
            "modules_count": len(self._module_services),
            "by_type": by_type,
        }


_global_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """获取全局服务注册中心单例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry
