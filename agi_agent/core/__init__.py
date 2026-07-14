"""
core/__init__.py - 核心抽象层

提供架构基础组件：模块基类、统一异常、配置管理、模块注册等
v2.0 升级：增加 HealthStatus、模块总线集成
"""
from .base_module import BaseModule, ModuleStatus, HealthStatus
from .exceptions import (
    CoreException,
    ModuleException,
    ConfigurationException,
    InitializationException,
    ValidationException as CoreValidationException,
)
from .config import ConfigManager, get_config
from .registry import ModuleRegistry, get_registry

__all__ = [
    "BaseModule",
    "ModuleStatus",
    "HealthStatus",
    "CoreException",
    "ModuleException",
    "ConfigurationException",
    "InitializationException",
    "CoreValidationException",
    "ConfigManager",
    "get_config",
    "ModuleRegistry",
    "get_registry",
]
