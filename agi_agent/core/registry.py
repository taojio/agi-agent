"""
core/registry.py - 模块注册中心

管理所有模块的注册、发现与依赖解析
"""
import time
from typing import Any, Dict, List, Optional, Type

from .base_module import BaseModule, ModuleStatus
from .exceptions import ModuleException, InitializationException


_global_registry: Optional["ModuleRegistry"] = None


class ModuleRegistry:
    """模块注册中心

    负责模块的注册、查找、初始化和生命周期管理
    """

    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._module_classes: Dict[str, Type[BaseModule]] = {}
        self._initialized: bool = False

    def register_class(self, name: str, module_class: Type[BaseModule]) -> None:
        """注册模块类"""
        if name in self._module_classes:
            raise ModuleException(
                f"Module class '{name}' already registered",
                module_name=name,
            )
        self._module_classes[name] = module_class

    def register_instance(self, name: str, instance: BaseModule) -> None:
        """直接注册模块实例"""
        if name in self._modules:
            raise ModuleException(
                f"Module '{name}' already registered",
                module_name=name,
            )
        self._modules[name] = instance

    def get(self, name: str) -> BaseModule:
        """获取模块实例"""
        if name not in self._modules:
            raise ModuleException(
                f"Module '{name}' not found",
                module_name=name,
            )
        return self._modules[name]

    def has(self, name: str) -> bool:
        """检查模块是否存在"""
        return name in self._modules

    def list_modules(self) -> List[str]:
        """列出所有已注册模块"""
        return list(self._modules.keys())

    def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化所有已注册模块

        按依赖顺序初始化
        """
        config = config or {}
        initialized: set = set()
        init_order = self._resolve_dependencies()

        for name in init_order:
            module = self._modules[name]
            if module.status == ModuleStatus.UNINITIALIZED:
                module_config = config.get(name, {})
                module.initialize(module_config)
                initialized.add(name)

        self._initialized = True

    def _resolve_dependencies(self) -> List[str]:
        """解析模块依赖顺序（拓扑排序）"""
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

        for name in self._modules:
            visit(name)

        return order

    def start_all(self) -> None:
        """启动所有已初始化模块"""
        for name, module in self._modules.items():
            if module.status == ModuleStatus.READY:
                module.start()

    def stop_all(self) -> None:
        """停止所有运行中的模块"""
        for module in self._modules.values():
            if module.status == ModuleStatus.RUNNING:
                module.stop()

    def shutdown_all(self) -> None:
        """关闭所有模块"""
        for module in reversed(list(self._modules.values())):
            try:
                module.shutdown()
            except Exception:
                pass

    def health_check_all(self) -> Dict[str, bool]:
        """所有模块健康检查"""
        results = {}
        for name, module in self._modules.items():
            results[name] = module.health_check()
        return results

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块状态"""
        return {name: module.get_status() for name, module in self._modules.items()}

    @property
    def all_ready(self) -> bool:
        """是否所有模块都已就绪"""
        if not self._modules:
            return False
        return all(
            m.status in (ModuleStatus.READY, ModuleStatus.RUNNING)
            for m in self._modules.values()
        )


def get_registry() -> ModuleRegistry:
    """获取全局模块注册中心单例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModuleRegistry()
    return _global_registry
