import os
import json
import logging
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("agi_agent.config_runtime")


class ConfigSource(Enum):
    DEFAULT = "default"
    ENVIRONMENT = "environment"
    FILE = "file"
    LOCAL = "local"
    DYNAMIC = "dynamic"

    @property
    def priority(self) -> int:
        priorities = {
            ConfigSource.DEFAULT: 1,
            ConfigSource.FILE: 2,
            ConfigSource.ENVIRONMENT: 3,
            ConfigSource.LOCAL: 4,
            ConfigSource.DYNAMIC: 5,
        }
        return priorities[self]


class ConfigScope(Enum):
    GLOBAL = "global"
    MODULE = "module"
    INSTANCE = "instance"


@dataclass
class ConfigEntry:
    key: str
    value: Any
    source: ConfigSource
    scope: ConfigScope
    module: str = ""
    description: str = ""
    sensitive: bool = False
    version: int = 1
    type_hint: str = ""


@dataclass
class ConfigChange:
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: float = 0.0
    user: str = ""


class ConfigEncryption:
    def __init__(self, key: Optional[str] = None):
        self._key = key or os.environ.get("AGI_CONFIG_KEY", "default_encryption_key_123")

    def encrypt(self, value: str) -> str:
        if not value:
            return value
        try:
            from cryptography.fernet import Fernet
            import base64
            key_bytes = base64.urlsafe_b64encode(self._key.ljust(32)[:32].encode())
            f = Fernet(key_bytes)
            return f.encrypt(value.encode()).decode()
        except ImportError:
            logger.warning("cryptography not available, using simple encoding")
            return value[::-1]

    def decrypt(self, value: str) -> str:
        if not value:
            return value
        try:
            from cryptography.fernet import Fernet
            import base64
            key_bytes = base64.urlsafe_b64encode(self._key.ljust(32)[:32].encode())
            f = Fernet(key_bytes)
            return f.decrypt(value.encode()).decode()
        except ImportError:
            return value[::-1]


class ConfigSchema:
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, key: str, schema: Dict[str, Any]) -> None:
        self._schemas[key] = schema

    def validate_type(self, key: str, value: Any) -> Tuple[bool, Optional[str]]:
        if key not in self._schemas:
            return True, None

        schema = self._schemas[key]
        expected_type = schema.get("type", "any")

        if expected_type == "int":
            if not isinstance(value, int):
                return False, f"Expected int, got {type(value).__name__}"
        elif expected_type == "float":
            if not isinstance(value, (int, float)):
                return False, f"Expected float, got {type(value).__name__}"
        elif expected_type == "bool":
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
        elif expected_type == "str":
            if not isinstance(value, str):
                return False, f"Expected str, got {type(value).__name__}"
        elif expected_type == "list":
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        elif expected_type == "dict":
            if not isinstance(value, dict):
                return False, f"Expected dict, got {type(value).__name__}"

        if "min" in schema and value < schema["min"]:
            return False, f"Value {value} below minimum {schema['min']}"
        if "max" in schema and value > schema["max"]:
            return False, f"Value {value} above maximum {schema['max']}"
        if "enum" in schema and value not in schema["enum"]:
            return False, f"Value {value} not in enum {schema['enum']}"

        return True, None


class ConfigManager:
    def __init__(self, config_dir: str = "config", env_prefix: str = "AGI_"):
        self._config_dir = os.path.abspath(config_dir)
        self._env_prefix = env_prefix
        self._configs: Dict[str, ConfigEntry] = {}
        self._module_configs: Dict[str, Dict[str, ConfigEntry]] = defaultdict(dict)
        self._listeners: Dict[str, List[Callable[[ConfigChange], None]]] = defaultdict(list)
        self._change_history: List[ConfigChange] = []
        self._max_history = 1000
        self._version = 1
        self._lock = threading.RLock()
        self._encryption = ConfigEncryption()
        self._schema = ConfigSchema()
        self._reload_count = 0

        os.makedirs(config_dir, exist_ok=True)

    def load_defaults(self, defaults: Dict[str, Any], module: str = "") -> None:
        with self._lock:
            for key, value in defaults.items():
                full_key = self._get_full_key(key, module)
                if full_key not in self._configs:
                    self._configs[full_key] = ConfigEntry(
                        key=full_key,
                        value=value,
                        source=ConfigSource.DEFAULT,
                        scope=ConfigScope.GLOBAL if not module else ConfigScope.MODULE,
                        module=module,
                    )

    def load_from_env(self) -> None:
        with self._lock:
            for env_key, env_value in os.environ.items():
                if env_key.startswith(self._env_prefix):
                    config_key = env_key[len(self._env_prefix):].lower().replace("_", ".")
                    old_entry = self._configs.get(config_key)

                    if old_entry and old_entry.source.priority >= ConfigSource.ENVIRONMENT.priority:
                        continue

                    try:
                        value = json.loads(env_value)
                    except json.JSONDecodeError:
                        value = env_value

                    sensitive = env_key.lower().endswith(("_password", "_secret", "_token", "_key"))

                    self._configs[config_key] = ConfigEntry(
                        key=config_key,
                        value=self._encryption.encrypt(str(value)) if sensitive else value,
                        source=ConfigSource.ENVIRONMENT,
                        scope=ConfigScope.GLOBAL,
                        sensitive=sensitive,
                        version=self._version,
                    )
                    logger.debug(f"Loaded from env: {config_key}")

    def load_from_file(self, filename: str, source_type: ConfigSource = ConfigSource.FILE) -> None:
        filepath = os.path.join(self._config_dir, filename)
        if not os.path.exists(filepath):
            logger.debug(f"Config file not found: {filepath}")
            return

        ext = os.path.splitext(filename)[1].lower()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if ext in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(content) or {}
                except ImportError:
                    logger.warning("PyYAML not available, skipping YAML file")
                    return
            else:
                data = json.loads(content) if content.strip() else {}

            if isinstance(data, dict):
                self._apply_config_dict(data, source_type)
                logger.info(f"Loaded config from: {filename}")
        except Exception as e:
            logger.error(f"Failed to load config file {filename}: {e}")

    def load_local_config(self) -> None:
        self.load_from_file("local.json", ConfigSource.LOCAL)

    def load_config_chain(self) -> None:
        self.load_from_file("defaults.json", ConfigSource.DEFAULT)
        self.load_from_file("config.json", ConfigSource.FILE)
        self.load_from_env()
        self.load_local_config()

    def _apply_config_dict(self, data: Dict[str, Any], source: ConfigSource) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                self._apply_nested(key, value, source)
            else:
                self._set_value(key, value, source)

    def _apply_nested(self, prefix: str, data: Dict[str, Any], source: ConfigSource) -> None:
        for key, value in data.items():
            full_key = f"{prefix}.{key}"
            if isinstance(value, dict):
                self._apply_nested(full_key, value, source)
            else:
                self._set_value(full_key, value, source)

    def _set_value(self, key: str, value: Any, source: ConfigSource,
                   module: str = "", sensitive: bool = False, description: str = "") -> bool:
        with self._lock:
            old_entry = self._configs.get(key)

            if old_entry and old_entry.source.priority > source.priority:
                return False

            is_sensitive = sensitive or key.lower().endswith(("_password", "_secret", "_token", "_key"))

            if is_sensitive and not isinstance(value, str):
                value = str(value)

            validated, err_msg = self._schema.validate_type(key, value)
            if not validated:
                logger.warning(f"Config validation failed for {key}: {err_msg}")

            stored_value = self._encryption.encrypt(str(value)) if is_sensitive else value

            self._configs[key] = ConfigEntry(
                key=key,
                value=stored_value,
                source=source,
                scope=ConfigScope.GLOBAL if not module else ConfigScope.MODULE,
                module=module,
                sensitive=is_sensitive,
                version=self._version,
                description=description,
            )

            if module:
                self._module_configs[module][key] = self._configs[key]

            if old_entry is None or old_entry.value != stored_value:
                change = ConfigChange(
                    key=key,
                    old_value=self._get_decrypted_value(old_entry) if old_entry else None,
                    new_value=self._get_decrypted_value(self._configs[key]),
                    source=source,
                    timestamp=time.time(),
                )
                self._record_change(change)
                self._notify_listeners(change)

            return True

    def _get_decrypted_value(self, entry: ConfigEntry) -> Any:
        if entry.sensitive:
            return self._encryption.decrypt(str(entry.value))
        return entry.value

    def _record_change(self, change: ConfigChange) -> None:
        self._change_history.append(change)
        if len(self._change_history) > self._max_history:
            self._change_history = self._change_history[-self._max_history:]

    def _notify_listeners(self, change: ConfigChange) -> None:
        key = change.key
        parts = key.split(".")

        for listener_key, callbacks in list(self._listeners.items()):
            match = False
            if listener_key == key:
                match = True
            elif listener_key.endswith(".*"):
                prefix = listener_key[:-2]
                if key.startswith(prefix):
                    match = True
            elif listener_key == "*":
                match = True

            if match:
                for cb in callbacks:
                    try:
                        cb(change)
                    except Exception as e:
                        logger.error(f"Listener error for {key}: {e}")

    def get(self, key: str, default: Any = None, module: str = "") -> Any:
        full_key = self._get_full_key(key, module)

        if full_key in self._configs:
            entry = self._configs[full_key]
            return self._get_decrypted_value(entry)

        if module:
            module_key = f"{module}.{key}"
            if module_key in self._configs:
                entry = self._configs[module_key]
                return self._get_decrypted_value(entry)

        return default

    def _get_full_key(self, key: str, module: str = "") -> str:
        if module and not key.startswith(module):
            return f"{module}.{key}"
        return key

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.DYNAMIC,
            module: str = "", sensitive: bool = False, description: str = "") -> bool:
        return self._set_value(key, value, source, module, sensitive, description)

    def get_module_config(self, module: str) -> Dict[str, Any]:
        configs = self._module_configs.get(module, {})
        return {key: self._get_decrypted_value(entry) for key, entry in configs.items()}

    def register_listener(self, key_pattern: str, callback: Callable[[ConfigChange], None]) -> None:
        with self._lock:
            self._listeners[key_pattern].append(callback)
        logger.debug(f"Registered listener for {key_pattern}")

    def unregister_listener(self, key_pattern: str, callback: Callable) -> None:
        with self._lock:
            if key_pattern in self._listeners:
                self._listeners[key_pattern] = [
                    cb for cb in self._listeners[key_pattern] if cb != callback
                ]

    def register_schema(self, key: str, schema: Dict[str, Any]) -> None:
        self._schema.register_schema(key, schema)

    def validate(self, key: str, validator: Callable[[Any], bool]) -> bool:
        value = self.get(key)
        return validator(value)

    def validate_all(self, validators: Dict[str, Callable[[Any], bool]]) -> Dict[str, bool]:
        results = {}
        for key, validator in validators.items():
            results[key] = self.validate(key, validator)
        return results

    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        return self._change_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        by_source = defaultdict(int)
        by_scope = defaultdict(int)
        sensitive_count = 0

        for entry in self._configs.values():
            by_source[entry.source.value] += 1
            by_scope[entry.scope.value] += 1
            if entry.sensitive:
                sensitive_count += 1

        return {
            "total_configs": len(self._configs),
            "total_modules": len(self._module_configs),
            "by_source": dict(by_source),
            "by_scope": dict(by_scope),
            "sensitive_count": sensitive_count,
            "change_history_size": len(self._change_history),
            "version": self._version,
            "listeners_count": sum(len(v) for v in self._listeners.values()),
            "reload_count": self._reload_count,
        }

    def save_to_file(self, filename: str = "config.json") -> None:
        filepath = os.path.join(self._config_dir, filename)
        config_dict = {}
        for key, entry in self._configs.items():
            if not entry.sensitive:
                config_dict[key] = entry.value

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved config to: {filename}")

    def increment_version(self) -> int:
        with self._lock:
            self._version += 1
        return self._version

    def reset(self) -> None:
        with self._lock:
            self._configs.clear()
            self._module_configs.clear()
            self._change_history.clear()
            self._version = 1

    def apply_update(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        changed_keys = []
        for key, value in new_config.items():
            old_value = self.get(key)
            if self.set(key, value):
                if old_value != value:
                    changed_keys.append(key)

        self._reload_count += 1
        logger.info(f"Config update applied (changed_keys={len(changed_keys)})")
        return {"changed_keys": changed_keys, "reload_count": self._reload_count}

    def snapshot(self) -> str:
        snapshot_dir = os.path.join(self._config_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        version_id = f"v_{int(time.time() * 1000)}_{os.getpid()}"
        data = {
            "version_id": version_id,
            "timestamp": time.time(),
            "version": self._version,
            "config": {key: entry.value for key, entry in self._configs.items()},
        }

        filepath = os.path.join(snapshot_dir, f"{version_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Config snapshot created: {version_id}")
        return version_id

    def list_snapshots(self) -> List[Dict[str, Any]]:
        snapshot_dir = os.path.join(self._config_dir, "snapshots")
        snapshots = []

        if not os.path.isdir(snapshot_dir):
            return snapshots

        for fname in os.listdir(snapshot_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(snapshot_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                snapshots.append({
                    "version_id": data.get("version_id", fname[:-5]),
                    "timestamp": data.get("timestamp", 0.0),
                    "path": path,
                })
            except Exception:
                continue

        snapshots.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
        return snapshots

    def restore_snapshot(self, version_id: str) -> bool:
        snapshot_dir = os.path.join(self._config_dir, "snapshots")
        filepath = os.path.join(snapshot_dir, f"{version_id}.json")

        if not os.path.exists(filepath):
            logger.error(f"Snapshot not found: {version_id}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            config_data = data.get("config", {})
            self.reset()

            for key, value in config_data.items():
                self._configs[key] = ConfigEntry(
                    key=key,
                    value=value,
                    source=ConfigSource.FILE,
                    scope=ConfigScope.GLOBAL,
                )

            self._version = data.get("version", 1)
            logger.info(f"Config restored from snapshot: {version_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore snapshot {version_id}: {e}")
            return False


_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "config", env_prefix: str = "AGI_") -> ConfigManager:
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_dir, env_prefix)
    return _global_config_manager