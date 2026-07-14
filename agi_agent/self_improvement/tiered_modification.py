from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import time


class SelfModificationTier(Enum):
    PARAM_RULE = "param_rule"
    MODULE = "module"
    ARCHITECTURE = "architecture"


class ParamType(Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    LIST = "list"


@dataclass
class ModifiableParamSpec:
    param_path: str
    param_type: ParamType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_options: Optional[List[Any]] = None
    tier: SelfModificationTier = SelfModificationTier.PARAM_RULE
    description: str = ""
    current_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.param_path,
            "type": self.param_type.value,
            "default": self.default_value,
            "min": self.min_value,
            "max": self.max_value,
            "options": self.valid_options,
            "tier": self.tier.value,
            "description": self.description,
            "current": self.current_value
        }


@dataclass
class RuleSpec:
    rule_id: str
    condition_schema: Dict[str, Any]
    action_schema: Dict[str, Any]
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    max_rules: int = 100
    tier: SelfModificationTier = SelfModificationTier.PARAM_RULE
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "condition_schema": self.condition_schema,
            "action_schema": self.action_schema,
            "confidence_range": list(self.confidence_range),
            "max_rules": self.max_rules,
            "tier": self.tier.value,
            "description": self.description
        }


@dataclass
class ModuleInterfaceSpec:
    module_id: str
    module_name: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    performance_metrics: Dict[str, str]
    replaceable: bool = True
    tier: SelfModificationTier = SelfModificationTier.MODULE
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "name": self.module_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.performance_metrics,
            "replaceable": self.replaceable,
            "tier": self.tier.value,
            "description": self.description
        }


@dataclass
class TieredModificationRequest:
    request_id: str
    tier: SelfModificationTier
    target: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    proposed_by: str = "self_improver"
    created_at: float = 0.0
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier": self.tier.value,
            "target": self.target,
            "action": self.action,
            "details": self.details,
            "proposed_by": self.proposed_by,
            "created_at": self.created_at,
            "status": self.status
        }


class TieredSelfModifier:
    def __init__(self):
        self.current_tier: SelfModificationTier = SelfModificationTier.PARAM_RULE
        self.unlocked_tiers: Set[SelfModificationTier] = {SelfModificationTier.PARAM_RULE}
        
        self.modifiable_params: Dict[str, ModifiableParamSpec] = {}
        self.rule_systems: Dict[str, RuleSpec] = {}
        self.module_interfaces: Dict[str, ModuleInterfaceSpec] = {}
        
        self._param_callbacks: Dict[str, Callable[[Any], bool]] = {}
        self._rule_add_callbacks: Dict[str, Callable[[Dict, Dict, float], bool]] = {}
        self._rule_remove_callbacks: Dict[str, Callable[[str], bool]] = {}
        self._module_replace_callbacks: Dict[str, Callable[[str], bool]] = {}
        
        self._modification_history: deque = deque(maxlen=200)
        self._tier_unlock_metrics: Dict[SelfModificationTier, Dict[str, float]] = {
            SelfModificationTier.PARAM_RULE: {"success_rate": 0.0, "total_attempts": 0, "required_for_next": 10},
            SelfModificationTier.MODULE: {"success_rate": 0.0, "total_attempts": 0, "required_for_next": 20},
            SelfModificationTier.ARCHITECTURE: {"success_rate": 0.0, "total_attempts": 0, "required_for_next": 0},
        }
        
        self._request_counter = 0

    def register_param(self, spec: ModifiableParamSpec,
                       set_callback: Callable[[Any], bool] = None):
        self.modifiable_params[spec.param_path] = spec
        if set_callback:
            self._param_callbacks[spec.param_path] = set_callback

    def register_rule_system(self, spec: RuleSpec,
                             add_callback: Callable[[Dict, Dict, float], bool] = None,
                             remove_callback: Callable[[str], bool] = None):
        self.rule_systems[spec.rule_id] = spec
        if add_callback:
            self._rule_add_callbacks[spec.rule_id] = add_callback
        if remove_callback:
            self._rule_remove_callbacks[spec.rule_id] = remove_callback

    def register_module(self, spec: ModuleInterfaceSpec,
                        replace_callback: Callable[[str], bool] = None):
        self.module_interfaces[spec.module_id] = spec
        if replace_callback:
            self._module_replace_callbacks[spec.module_id] = replace_callback

    def can_modify(self, tier: SelfModificationTier) -> bool:
        return tier in self.unlocked_tiers

    def request_param_change(self, param_path: str, new_value: Any) -> Optional[TieredModificationRequest]:
        if param_path not in self.modifiable_params:
            return None
        spec = self.modifiable_params[param_path]
        if not self.can_modify(spec.tier):
            return None
        if not self._validate_param_value(spec, new_value):
            return None
        
        self._request_counter += 1
        req = TieredModificationRequest(
            request_id=f"mod_req_{self._request_counter}",
            tier=spec.tier,
            target=param_path,
            action="set_param",
            details={"new_value": new_value, "old_value": spec.current_value},
            created_at=time.time()
        )
        return req

    def request_rule_add(self, system_id: str, condition: Dict,
                         action: Dict, confidence: float) -> Optional[TieredModificationRequest]:
        if system_id not in self.rule_systems:
            return None
        spec = self.rule_systems[system_id]
        if not self.can_modify(spec.tier):
            return None
        
        min_c, max_c = spec.confidence_range
        if not (min_c <= confidence <= max_c):
            return None
        
        self._request_counter += 1
        req = TieredModificationRequest(
            request_id=f"mod_req_{self._request_counter}",
            tier=spec.tier,
            target=system_id,
            action="add_rule",
            details={"condition": condition, "action": action, "confidence": confidence},
            created_at=time.time()
        )
        return req

    def request_rule_remove(self, system_id: str, rule_index: int) -> Optional[TieredModificationRequest]:
        if system_id not in self.rule_systems:
            return None
        spec = self.rule_systems[system_id]
        if not self.can_modify(spec.tier):
            return None
        
        self._request_counter += 1
        req = TieredModificationRequest(
            request_id=f"mod_req_{self._request_counter}",
            tier=spec.tier,
            target=system_id,
            action="remove_rule",
            details={"rule_index": rule_index},
            created_at=time.time()
        )
        return req

    def execute_request(self, request: TieredModificationRequest) -> Tuple[bool, str]:
        if not self.can_modify(request.tier):
            return False, "tier_not_unlocked"
        
        success = False
        message = ""
        
        if request.action == "set_param":
            callback = self._param_callbacks.get(request.target)
            if callback:
                try:
                    success = callback(request.details["new_value"])
                    if success:
                        self.modifiable_params[request.target].current_value = request.details["new_value"]
                    message = "success" if success else "callback_failed"
                except Exception as e:
                    success = False
                    message = f"error: {str(e)}"
            else:
                success = False
                message = "no_callback"
        
        elif request.action == "add_rule":
            callback = self._rule_add_callbacks.get(request.target)
            if callback:
                try:
                    success = callback(
                        request.details["condition"],
                        request.details["action"],
                        request.details["confidence"]
                    )
                    message = "success" if success else "callback_failed"
                except Exception as e:
                    success = False
                    message = f"error: {str(e)}"
            else:
                success = False
                message = "no_callback"
        
        elif request.action == "remove_rule":
            callback = self._rule_remove_callbacks.get(request.target)
            if callback:
                try:
                    success = callback(str(request.details["rule_index"]))
                    message = "success" if success else "callback_failed"
                except Exception as e:
                    success = False
                    message = f"error: {str(e)}"
            else:
                success = False
                message = "no_callback"
        
        request.status = "applied" if success else "failed"
        self._modification_history.append({
            "request": request.to_dict(),
            "success": success,
            "message": message,
            "timestamp": time.time()
        })
        
        self._update_tier_metrics(request.tier, success)
        return success, message

    def _validate_param_value(self, spec: ModifiableParamSpec, value: Any) -> bool:
        if spec.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if spec.min_value is not None and value < spec.min_value:
                return False
            if spec.max_value is not None and value > spec.max_value:
                return False
            return True
        elif spec.param_type == ParamType.INT:
            if not isinstance(value, int):
                return False
            if spec.min_value is not None and value < spec.min_value:
                return False
            if spec.max_value is not None and value > spec.max_value:
                return False
            return True
        elif spec.param_type == ParamType.BOOL:
            return isinstance(value, bool)
        elif spec.param_type == ParamType.STRING:
            if spec.valid_options and value not in spec.valid_options:
                return False
            return isinstance(value, str)
        elif spec.param_type == ParamType.LIST:
            return isinstance(value, list)
        return False

    def _update_tier_metrics(self, tier: SelfModificationTier, success: bool):
        metrics = self._tier_unlock_metrics.get(tier)
        if not metrics:
            return
        metrics["total_attempts"] += 1
        total = metrics["total_attempts"]
        old_rate = metrics["success_rate"]
        metrics["success_rate"] = (old_rate * (total - 1) + (1.0 if success else 0.0)) / total

    def maybe_unlock_next_tier(self) -> Optional[SelfModificationTier]:
        tier_order = [SelfModificationTier.PARAM_RULE, SelfModificationTier.MODULE, SelfModificationTier.ARCHITECTURE]
        for i, tier in enumerate(tier_order[:-1]):
            next_tier = tier_order[i + 1]
            if next_tier in self.unlocked_tiers:
                continue
            metrics = self._tier_unlock_metrics.get(tier)
            if not metrics:
                continue
            if metrics["total_attempts"] >= metrics["required_for_next"] and metrics["success_rate"] > 0.6:
                self.unlocked_tiers.add(next_tier)
                self.current_tier = next_tier
                return next_tier
        return None

    def get_available_modifications(self) -> Dict[str, Any]:
        available = {"params": [], "rules": [], "modules": []}
        for path, spec in self.modifiable_params.items():
            if self.can_modify(spec.tier):
                available["params"].append(spec.to_dict())
        for sid, spec in self.rule_systems.items():
            if self.can_modify(spec.tier):
                available["rules"].append(spec.to_dict())
        for mid, spec in self.module_interfaces.items():
            if self.can_modify(spec.tier):
                available["modules"].append(spec.to_dict())
        return available

    def get_tier_status(self) -> Dict[str, Any]:
        return {
            "current_tier": self.current_tier.value,
            "unlocked_tiers": [t.value for t in self.unlocked_tiers],
            "tier_metrics": {
                t.value: m for t, m in self._tier_unlock_metrics.items()
            },
            "history_count": len(self._modification_history)
        }

    def sync_param_value(self, param_path: str, value: Any):
        if param_path in self.modifiable_params:
            self.modifiable_params[param_path].current_value = value
