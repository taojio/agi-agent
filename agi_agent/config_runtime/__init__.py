"""
config_runtime/__init__.py - 配置管理子模块 (T021-T024)

提供配置加载解析、配置热更新、配置校验、配置持久化备份能力。
"""
from .config_backup import ConfigBackup, ConfigBackupConfig
from .config_hot_reloader import ConfigHotReloader, HotReloaderConfig
from .config_loader import (
    AgentConfig,
    ConfigLoader,
    ConfigLoaderConfig,
    ModuleSwitch,
    ToolPermission,
)
from .config_validator import (
    ConfigValidator,
    ConfigValidatorConfig,
    ValidationResult,
)

__all__ = [
    "AgentConfig",
    "ConfigLoader",
    "ConfigLoaderConfig",
    "ToolPermission",
    "ModuleSwitch",
    "ConfigHotReloader",
    "HotReloaderConfig",
    "ConfigValidator",
    "ConfigValidatorConfig",
    "ValidationResult",
    "ConfigBackup",
    "ConfigBackupConfig",
]
