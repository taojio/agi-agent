"""
core/config.py - 统一配置管理

集中管理所有配置，支持环境变量、配置文件、默认值三级覆盖
"""
import os
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .exceptions import ConfigurationException


_global_config: Optional["ConfigManager"] = None


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"
    path: str = "data/security.db"
    host: str = "localhost"
    port: int = 5432
    name: str = "agi_agent"
    user: str = ""
    password: str = ""


@dataclass
class RedisConfig:
    """Redis 配置"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""


@dataclass
class JWTConfig:
    """JWT 配置"""
    algorithm: str = "HS256"
    access_token_expiry: int = 900  # 15 分钟
    refresh_token_expiry: int = 604800  # 7 天
    issuer: str = "agi-agent"
    audience: str = "agi-agent-api"


@dataclass
class SecurityConfig:
    """安全配置"""
    enable_rate_limiting: bool = True
    enable_security_headers: bool = True
    enable_audit_logging: bool = True
    rate_limit_login_per_minute: int = 5
    rate_limit_register_per_hour: int = 3
    jwt: JWTConfig = field(default_factory=JWTConfig)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: str = "logs/agi_agent.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """应用总配置"""
    app_name: str = "AGI Agent"
    version: str = "1.0.0"
    environment: str = "development"
    data_dir: str = "data"
    log_dir: str = "logs"
    host: str = "0.0.0.0"
    port: int = 8090

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigManager:
    """配置管理器

    支持从环境变量、配置文件加载配置，并提供类型安全的访问
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = AppConfig()
        self._config_path = config_path
        self._load_from_env()
        if config_path:
            if not os.path.exists(config_path):
                raise ConfigurationException(
                    f"Config file not found: {config_path}",
                    config_key=config_path,
                )
            self._load_from_file(config_path)

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        env_map = {
            "APP_ENV": ("environment", str),
            "DATA_DIR": ("data_dir", str),
            "LOG_DIR": ("log_dir", str),
            "APP_HOST": ("host", str),
            "APP_PORT": ("port", int),
        }
        for env_key, (cfg_path, type_fn) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    setattr(self._config, cfg_path, type_fn(val))
                except (ValueError, TypeError) as e:
                    raise ConfigurationException(
                        f"Invalid environment variable {env_key}: {e}",
                        config_key=env_key,
                    )

        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            self._config.data_dir = data_dir
            self._config.database.path = os.path.join(data_dir, "security.db")

    def _load_from_file(self, path: str) -> None:
        """从 JSON 配置文件加载"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigurationException(
                f"Failed to load config from {path}: {e}",
                config_key=path,
            )

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """应用字典配置"""
        self._apply_recursive(self._config, data)

    def _apply_recursive(self, target: Any, data: Dict[str, Any]) -> None:
        """递归应用配置到目标对象"""
        import dataclasses
        for key, value in data.items():
            if not hasattr(target, key):
                continue
            current = getattr(target, key)
            if isinstance(value, dict) and dataclasses.is_dataclass(current):
                self._apply_recursive(current, value)
            else:
                setattr(target, key, value)

    @property
    def config(self) -> AppConfig:
        """获取配置对象"""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """按键获取配置值

        支持点号分隔的嵌套键，如 "security.jwt.algorithm"
        """
        parts = key.split(".")
        obj: Any = self._config
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                return default
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return _dataclass_to_dict(self._config)


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """递归转换 dataclass 为字典"""
    import dataclasses
    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if dataclasses.is_dataclass(val):
                result[f.name] = _dataclass_to_dict(val)
            else:
                result[f.name] = val
        return result
    return obj


def get_config() -> ConfigManager:
    """获取全局配置管理器单例"""
    global _global_config
    if _global_config is None:
        config_path = os.environ.get("CONFIG_PATH")
        _global_config = ConfigManager(config_path)
    return _global_config
