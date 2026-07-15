"""
core/registry.py - 模块注册中心

管理所有模块的注册、发现、依赖解析与生命周期管理
v2.0 升级：增加模块分组、动态加载、启动顺序、监控回调、事件总线集成
"""
import time
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from collections import defaultdict

from .base_module import BaseModule, ModuleStatus, HealthStatus
from .exceptions import ModuleException, InitializationException


_global_registry: Optional["ModuleRegistry"] = None


class ModuleRegistry:
    """模块注册中心

    负责模块的注册、查找、初始化和生命周期管理

    v2.0 新增功能：
    - 模块分组管理（按功能域分组）
    - 动态模块加载/卸载
    - 启动顺序控制
    - 生命周期监控回调
    - 配置管理集成
    - 事件总线集成
    """

    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._module_classes: Dict[str, Type[BaseModule]] = {}
        self._groups: Dict[str, List[str]] = defaultdict(list)
        self._initialized: bool = False
        self._start_order: List[str] = []
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._event_bus = None

    def register_class(self, name: str, module_class: Type[BaseModule], 
                       group: str = "default") -> None:
        """注册模块类

        Args:
            name: 模块名称
            module_class: 模块类
            group: 分组名称
        """
        if name in self._module_classes:
            raise ModuleException(
                f"Module class '{name}' already registered",
                module_name=name,
            )
        self._module_classes[name] = module_class
        if group not in self._groups or name not in self._groups[group]:
            self._groups[group].append(name)

    def register_instance(self, name: str, instance: BaseModule, 
                          group: str = "default") -> None:
        """直接注册模块实例

        Args:
            name: 模块名称
            instance: 模块实例
            group: 分组名称
        """
        if name in self._modules:
            raise ModuleException(
                f"Module '{name}' already registered",
                module_name=name,
            )
        self._modules[name] = instance
        if group not in self._groups or name not in self._groups[group]:
            self._groups[group].append(name)

    def get(self, name: str) -> BaseModule:
        """获取模块实例

        Args:
            name: 模块名称

        Returns:
            BaseModule: 模块实例

        Raises:
            ModuleException: 模块未找到
        """
        if name not in self._modules:
            raise ModuleException(
                f"Module '{name}' not found",
                module_name=name,
            )
        return self._modules[name]

    def has(self, name: str) -> bool:
        """检查模块是否存在

        Args:
            name: 模块名称

        Returns:
            bool: 是否存在
        """
        return name in self._modules

    def list_modules(self) -> List[str]:
        """列出所有已注册模块"""
        return list(self._modules.keys())

    def list_groups(self) -> List[str]:
        """列出所有分组"""
        return list(self._groups.keys())

    def get_modules_in_group(self, group: str) -> List[str]:
        """获取指定分组的模块列表

        Args:
            group: 分组名称

        Returns:
            List[str]: 模块名称列表
        """
        return self._groups.get(group, [])

    def set_start_order(self, order: List[str]) -> None:
        """设置模块启动顺序

        Args:
            order: 模块名称列表（按启动顺序排列）
        """
        self._start_order = order

    def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """初始化所有已注册模块

        按依赖顺序初始化

        Args:
            config: 配置字典

        Returns:
            Dict[str, bool]: 初始化结果（模块名 -> 是否成功）
        """
        config = config or {}
        results: Dict[str, bool] = {}
        initialized: set = set()
        init_order = self._resolve_dependencies()

        for name in init_order:
            try:
                module = self._modules[name]
                if module.status == ModuleStatus.UNINITIALIZED:
                    module_config = config.get(name, {})
                    module.initialize(module_config)
                    self._notify_callback("module_initialized", name, module)
                    results[name] = True
                else:
                    results[name] = True
                initialized.add(name)
            except Exception as e:
                results[name] = False
                self._notify_callback("module_init_failed", name, str(e))

        self._initialized = True
        self._notify_callback("registry_initialized", initialized=initialized)
        return results

    def _resolve_dependencies(self) -> List[str]:
        """解析模块依赖顺序（拓扑排序）

        Returns:
            List[str]: 模块名称列表（按依赖顺序）
        """
        visited: set = set()
        order: List[str] = []

        def visit(name: str):
            if name in visited:
                return
            if name not in self._modules:
                return
            visited.add(name)
            module = self._modules[name]
            for dep in module.dependencies:
                visit(dep)
            order.append(name)

        if self._start_order:
            for name in self._start_order:
                if name in self._modules:
                    visit(name)

        for name in self._modules:
            if name not in visited:
                visit(name)

        return order

    def start_all(self) -> Dict[str, bool]:
        """启动所有已初始化模块

        Returns:
            Dict[str, bool]: 启动结果（模块名 -> 是否成功）
        """
        results: Dict[str, bool] = {}
        
        start_order = self._start_order or list(self._modules.keys())
        
        for name in start_order:
            if name not in self._modules:
                continue
            try:
                module = self._modules[name]
                if module.status == ModuleStatus.READY:
                    module.start()
                    self._notify_callback("module_started", name, module)
                    results[name] = True
                else:
                    results[name] = False
            except Exception as e:
                results[name] = False
                self._notify_callback("module_start_failed", name, str(e))

        self._notify_callback("registry_started")
        return results

    def stop_all(self) -> None:
        """停止所有运行中的模块"""
        for name, module in self._modules.items():
            if module.status == ModuleStatus.RUNNING:
                try:
                    module.stop()
                    self._notify_callback("module_stopped", name, module)
                except Exception as e:
                    self._notify_callback("module_stop_failed", name, str(e))

        self._notify_callback("registry_stopped")

    def shutdown_all(self) -> None:
        """关闭所有模块（按相反顺序）"""
        for name in reversed(self._start_order or list(self._modules.keys())):
            if name not in self._modules:
                continue
            module = self._modules[name]
            try:
                module.shutdown()
                self._notify_callback("module_shutdown", name, module)
            except Exception:
                pass

        self._notify_callback("registry_shutdown")

    def health_check_all(self) -> Dict[str, HealthStatus]:
        """所有模块健康检查

        Returns:
            Dict[str, HealthStatus]: 健康状态（模块名 -> 状态）
        """
        results = {}
        for name, module in self._modules.items():
            results[name] = module.get_health_status()
        return results

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块状态

        Returns:
            Dict[str, Dict[str, Any]]: 状态信息
        """
        return {name: module.get_status() for name, module in self._modules.items()}

    def unregister(self, name: str) -> None:
        """注销模块

        Args:
            name: 模块名称
        """
        if name not in self._modules:
            raise ModuleException(
                f"Module '{name}' not found",
                module_name=name,
            )
        
        module = self._modules[name]
        
        if module.status == ModuleStatus.RUNNING:
            module.stop()
        if module.status != ModuleStatus.SHUTDOWN:
            module.shutdown()
        
        del self._modules[name]
        
        for group in self._groups.values():
            if name in group:
                group.remove(name)
        
        if name in self._module_classes:
            del self._module_classes[name]
        
        self._notify_callback("module_unregistered", name)

    def reload_module(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """重新加载模块

        Args:
            name: 模块名称
            config: 新配置

        Returns:
            bool: 是否成功
        """
        if name not in self._modules:
            return False

        try:
            module = self._modules[name]
            if module.status == ModuleStatus.RUNNING:
                module.stop()
            if module.status != ModuleStatus.SHUTDOWN:
                module.shutdown()
            
            module.initialize(config or {})
            module.start()
            
            self._notify_callback("module_reloaded", name, module)
            return True
        except Exception as e:
            self._notify_callback("module_reload_failed", name, str(e))
            return False

    def on(self, event: str, callback: Callable) -> None:
        """注册生命周期回调

        支持的事件：
        - module_initialized
        - module_init_failed
        - module_started
        - module_start_failed
        - module_stopped
        - module_stop_failed
        - module_shutdown
        - module_unregistered
        - module_reloaded
        - module_reload_failed
        - registry_initialized
        - registry_started
        - registry_stopped
        - registry_shutdown

        Args:
            event: 事件名称
            callback: 回调函数
        """
        self._callbacks[event].append(callback)

    def _notify_callback(self, event: str, *args, **kwargs) -> None:
        """通知回调

        Args:
            event: 事件名称
            *args: 位置参数
            **kwargs: 关键字参数
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                pass

    def attach_event_bus(self, event_bus) -> None:
        """挂载事件总线

        Args:
            event_bus: EventBus 实例
        """
        self._event_bus = event_bus
        
        for name, module in self._modules.items():
            if hasattr(module, 'attach_bus'):
                module.attach_bus(event_bus)

    def get_stats(self) -> Dict[str, Any]:
        """获取注册中心统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        status_counts = defaultdict(int)
        for module in self._modules.values():
            status_counts[module.status.value] += 1

        return {
            "total_modules": len(self._modules),
            "total_classes": len(self._module_classes),
            "total_groups": len(self._groups),
            "initialized": self._initialized,
            "status_counts": dict(status_counts),
            "groups": {g: len(m) for g, m in self._groups.items()},
        }

    @property
    def all_ready(self) -> bool:
        """是否所有模块都已就绪"""
        if not self._modules:
            return False
        return all(
            m.status in (ModuleStatus.READY, ModuleStatus.RUNNING)
            for m in self._modules.values()
        )

    @property
    def all_running(self) -> bool:
        """是否所有模块都在运行"""
        if not self._modules:
            return False
        return all(m.status == ModuleStatus.RUNNING for m in self._modules.values())


def get_registry() -> ModuleRegistry:
    """获取全局模块注册中心单例

    Returns:
        ModuleRegistry: 注册中心实例
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModuleRegistry()
    return _global_registry