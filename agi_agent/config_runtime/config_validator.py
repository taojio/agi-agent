"""
config_runtime/config_validator.py - 配置校验 (T023)

动态调度的配置校验：校验参数越界、格式错误、配置冲突、权限异常。
支持注册自定义校验规则，返回结构化 ValidationResult。
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.config_runtime")

# 校验规则函数签名：(config_dict, context) -> List[str]
# 返回错误消息列表（空表示无问题）
RuleFn = Callable[[Dict[str, Any], Dict[str, Any]], List[str]]


@dataclass
class ValidationResult:
    """校验结果"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.errors:
            self.valid = False


@dataclass
class ConfigValidatorConfig:
    """配置校验器配置"""
    enable_builtin_rules: bool = True
    # 推理阈值合理区间
    confidence_range: tuple = (0.0, 1.0)
    novelty_range: tuple = (0.0, 1.0)
    free_energy_range: tuple = (0.0, 10.0)
    max_steps_range: tuple = (1, 1000)


class ConfigValidator(BaseModule):
    """配置校验器 (T023)

    提供 validate / register_rule 方法，内置常见规则（参数越界、格式错误、
    配置冲突、权限异常），并支持注册自定义规则函数。
    """

    name = "config_validator"
    version = "1.0.0"
    description = "配置校验 (T023)"

    def __init__(self, config: Optional[ConfigValidatorConfig] = None):
        super().__init__()
        self._cfg = config or ConfigValidatorConfig()
        self._rules: List[RuleFn] = []
        self._rule_names: List[str] = []
        if self._cfg.enable_builtin_rules:
            self._register_builtin_rules()

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        logger.info("ConfigValidator 初始化完成 (rules=%d)", len(self._rules))

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def register_rule(self, rule_fn: RuleFn, name: Optional[str] = None) -> None:
        """注册自定义校验规则

        Args:
            rule_fn: 规则函数，签名为 (config_dict, context) -> List[str]
            name: 规则名称
        """
        self._rules.append(rule_fn)
        self._rule_names.append(name or getattr(rule_fn, "__name__", f"rule_{len(self._rules)}"))

    def validate(self, config: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """校验配置

        Args:
            config: AgentConfig 或 dict
            context: 校验上下文

        Returns:
            ValidationResult: 校验结果
        """
        result = ValidationResult(valid=True)
        cfg_dict = self._as_dict(config)
        ctx = context or {}
        for name, rule in zip(self._rule_names, self._rules):
            try:
                errors = rule(cfg_dict, ctx) or []
                for msg in errors:
                    if msg.startswith("WARN:"):
                        result.add_warning(msg[5:].strip())
                    else:
                        result.add_error(f"[{name}] {msg}")
            except Exception as e:  # noqa: BLE001
                result.add_error(f"[{name}] 规则执行异常: {e}")
        logger.info("配置校验完成 (valid=%s, errors=%d, warnings=%d)", result.valid, len(result.errors), len(result.warnings))
        return result

    @property
    def rules(self) -> List[str]:
        return list(self._rule_names)

    # ====== 内置规则 ======
    def _register_builtin_rules(self) -> None:
        self._rules.append(self._rule_threshold_bounds)
        self._rule_names.append("threshold_bounds")
        self._rules.append(self._rule_role_format)
        self._rule_names.append("role_format")
        self._rules.append(self._rule_permission_conflict)
        self._rule_names.append("permission_conflict")
        self._rules.append(self._rule_module_switch_format)
        self._rule_names.append("module_switch_format")

    def _rule_threshold_bounds(self, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        thresholds = cfg.get("inference_thresholds", {}) or {}
        checks = [
            ("confidence", self._cfg.confidence_range),
            ("novelty", self._cfg.novelty_range),
            ("free_energy", self._cfg.free_energy_range),
            ("max_steps", self._cfg.max_steps_range),
        ]
        for key, (lo, hi) in checks:
            if key in thresholds:
                try:
                    val = float(thresholds[key])
                    if val < lo or val > hi:
                        errors.append(f"inference_thresholds.{key}={val} 越界，应在 [{lo}, {hi}]")
                except (TypeError, ValueError):
                    errors.append(f"inference_thresholds.{key} 格式错误，应为数值")
        return errors

    def _rule_role_format(self, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        role = cfg.get("role")
        if role is None:
            errors.append("WARN: 未设置 role，将使用默认值")
        elif not isinstance(role, str) or not role.strip():
            errors.append("role 必须为非空字符串")
        return errors

    def _rule_permission_conflict(self, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        perms = cfg.get("tool_permissions", []) or []
        names: Dict[str, int] = {}
        for p in perms:
            if not isinstance(p, dict):
                continue
            nm = p.get("name", "")
            names[nm] = names.get(nm, 0) + 1
        for nm, cnt in names.items():
            if cnt > 1:
                errors.append(f"tool_permissions 存在重复项: {nm} (x{cnt})")
        # 开关同时启用与禁用同名模块视为冲突
        switches = cfg.get("module_switches", []) or []
        sw_state: Dict[str, List[bool]] = {}
        for s in switches:
            if not isinstance(s, dict):
                continue
            nm = s.get("name", "")
            sw_state.setdefault(nm, []).append(bool(s.get("enabled", True)))
        for nm, states in sw_state.items():
            if len(set(states)) > 1:
                errors.append(f"module_switches 冲突: {nm} 同时存在启用与禁用状态")
        return errors

    def _rule_module_switch_format(self, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for s in cfg.get("module_switches", []) or []:
            if not isinstance(s, dict):
                errors.append("module_switches 项必须为字典")
                continue
            if not s.get("name"):
                errors.append("module_switches 项缺少 name 字段")
            priority = s.get("priority", "normal")
            if priority not in ("low", "normal", "high"):
                errors.append(f"module_switches priority 非法: {priority}")
        return errors

    # ====== 工具 ======
    @staticmethod
    def _as_dict(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(config, "__dict__"):
            return {k: v for k, v in vars(config).items() if not k.startswith("_")}
        return {}
