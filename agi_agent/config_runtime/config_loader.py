"""
config_runtime/config_loader.py - 配置加载解析 (T021)

加载本地/云端 YAML/JSON 配置，解析角色参数、工具权限、推理阈值、
Prompt 模板、模块开关为结构化 dataclass。PyYAML 可选，降级仅支持 JSON。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.config_runtime")

# 可选依赖：PyYAML
try:  # pragma: no cover - 环境相关
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore
    _HAS_YAML = False


@dataclass
class ToolPermission:
    """工具权限"""
    name: str
    enabled: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleSwitch:
    """模块开关"""
    name: str
    enabled: bool = True
    priority: str = "normal"  # low | normal | high


@dataclass
class AgentConfig:
    """Agent 结构化配置"""
    role: str = "default"
    role_params: Dict[str, Any] = field(default_factory=dict)
    tool_permissions: List[ToolPermission] = field(default_factory=list)
    inference_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "confidence": 0.5,
            "novelty": 0.5,
            "free_energy": 0.3,
            "max_steps": 5,
        }
    )
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    module_switches: List[ModuleSwitch] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "role_params": dict(self.role_params),
            "tool_permissions": [
                {"name": t.name, "enabled": t.enabled, "constraints": dict(t.constraints)}
                for t in self.tool_permissions
            ],
            "inference_thresholds": dict(self.inference_thresholds),
            "prompt_templates": dict(self.prompt_templates),
            "module_switches": [
                {"name": m.name, "enabled": m.enabled, "priority": m.priority}
                for m in self.module_switches
            ],
            "extra": dict(self.extra),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentConfig":
        tool_perms = [
            ToolPermission(
                name=t.get("name", ""),
                enabled=bool(t.get("enabled", True)),
                constraints=dict(t.get("constraints", {}) or {}),
            )
            for t in (d.get("tool_permissions") or [])
        ]
        module_switches = [
            ModuleSwitch(
                name=m.get("name", ""),
                enabled=bool(m.get("enabled", True)),
                priority=str(m.get("priority", "normal")),
            )
            for m in (d.get("module_switches") or [])
        ]
        return cls(
            role=str(d.get("role", "default")),
            role_params=dict(d.get("role_params", {}) or {}),
            tool_permissions=tool_perms,
            inference_thresholds=dict(d.get("inference_thresholds", {}) or {}),
            prompt_templates=dict(d.get("prompt_templates", {}) or {}),
            module_switches=module_switches,
            extra=dict(d.get("extra", {}) or {}),
            source=str(d.get("source", "")),
        )


@dataclass
class ConfigLoaderConfig:
    """配置加载器配置"""
    allow_yaml: bool = True
    default_role: str = "default"


class ConfigLoader(BaseModule):
    """配置加载解析器 (T021)

    单次触发（启动时）加载 YAML/JSON 配置文件，解析为 AgentConfig。
    PyYAML 不可用时仅支持 JSON。
    """

    name = "config_loader"
    version = "1.0.0"
    description = "配置加载解析 (T021)"

    def __init__(self, config: Optional[ConfigLoaderConfig] = None):
        super().__init__()
        self._cfg = config or ConfigLoaderConfig()
        self._last_loaded: Optional[AgentConfig] = None

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._cfg.allow_yaml and not _HAS_YAML:
            logger.info("PyYAML 不可用，ConfigLoader 降级为仅支持 JSON")
        else:
            logger.info("ConfigLoader 初始化完成 (yaml=%s)", _HAS_YAML)

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    @property
    def yaml_available(self) -> bool:
        return _HAS_YAML

    def load(self, path: str) -> AgentConfig:
        """从文件路径加载配置

        Args:
            path: 配置文件路径（.yaml/.yml/.json）

        Returns:
            AgentConfig: 结构化配置
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if ext in (".yaml", ".yml"):
            if not (self._cfg.allow_yaml and _HAS_YAML):
                raise ValueError("PyYAML 不可用，无法解析 YAML 配置")
            data = yaml.safe_load(raw) or {}
        elif ext == ".json":
            data = json.loads(raw) if raw.strip() else {}
        else:
            # 尝试 JSON 优先，再尝试 YAML
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:  # noqa: BLE001
                if self._cfg.allow_yaml and _HAS_YAML:
                    data = yaml.safe_load(raw) or {}
                else:
                    raise
        if not isinstance(data, dict):
            raise ValueError(f"配置文件根节点必须是字典: {path}")
        cfg = AgentConfig.from_dict(data)
        cfg.source = path
        self._last_loaded = cfg
        logger.info("配置加载完成: %s (role=%s)", path, cfg.role)
        return cfg

    def load_from_dict(self, d: Dict[str, Any]) -> AgentConfig:
        """从字典加载配置

        Args:
            d: 配置字典

        Returns:
            AgentConfig: 结构化配置
        """
        cfg = AgentConfig.from_dict(d or {})
        cfg.source = "<dict>"
        self._last_loaded = cfg
        return cfg

    @property
    def last_loaded(self) -> Optional[AgentConfig]:
        """最近一次加载的配置"""
        return self._last_loaded
