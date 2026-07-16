from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import numpy as np

from .tiered_modification import (
    TieredSelfModifier, SelfModificationTier, ModifiableParamSpec,
    ParamType, RuleSpec, ModuleInterfaceSpec, TieredModificationRequest
)
from .symbolic_self_model import SymbolicSelfModel
from .symbolic_verifier import SymbolicFormalVerifier, VerificationResult


class ImprovementStage(Enum):
    TIER_1_PARAM_RULE = "tier_1_param_rule"
    TIER_2_MODULE = "tier_2_module"
    TIER_3_ARCHITECTURE = "tier_3_architecture"


class BootstrappedSelfImprover:
    def __init__(self, start_tier: SelfModificationTier = SelfModificationTier.PARAM_RULE):
        self.tiered_modifier = TieredSelfModifier()
        self.symbolic_self_model = SymbolicSelfModel()
        self.formal_verifier = SymbolicFormalVerifier()
        
        self.current_stage = ImprovementStage.TIER_1_PARAM_RULE
        self._improvement_history: deque = deque(maxlen=200)
        self._improvement_counter = 0
        self._successful_improvements = 0
        self._failed_improvements = 0
        
        self._black_box_modules: Dict[str, Dict[str, Any]] = {}
        self._module_performance: Dict[str, Dict[str, float]] = {}
        
        self._agent_ref = None
        self._initialized = False

    def initialize(self, agent_ref) -> bool:
        self._agent_ref = agent_ref
        self.symbolic_self_model.build_from_agent_structure(agent_ref)
        self._register_params_from_self_model()
        self._register_rule_systems_from_agent(agent_ref)
        self._register_black_box_modules()
        self._initialized = True
        return True

    def _register_params_from_self_model(self):
        for module_id, params in self.symbolic_self_model._param_index.items():
            for param in params:
                tier_str = param.get("tier", "param_rule")
                tier = SelfModificationTier(tier_str) if tier_str in [t.value for t in SelfModificationTier] else SelfModificationTier.PARAM_RULE
                
                param_type = ParamType(param["type"]) if param["type"] in [p.value for p in ParamType] else ParamType.FLOAT
                
                spec = ModifiableParamSpec(
                    param_path=f"{module_id}.{param['name']}",
                    param_type=param_type,
                    default_value=param.get("default"),
                    min_value=param.get("min"),
                    max_value=param.get("max"),
                    valid_options=param.get("options"),
                    tier=tier,
                    description=param.get("description", f"{module_id}的{param['name']}参数")
                )
                
                callback = self._make_param_set_callback(module_id, param["name"])
                self.tiered_modifier.register_param(spec, callback)

    def _make_param_set_callback(self, module_id: str, param_name: str) -> Callable[[Any], bool]:
        def callback(value: Any) -> bool:
            return self._apply_param_to_module(module_id, param_name, value)
        return callback

    def _apply_param_to_module(self, module_id: str, param_name: str, value: Any) -> bool:
        try:
            if self._agent_ref is None:
                return False
            
            if module_id == "homeostasis":
                if param_name == "energy_baseline" and hasattr(self._agent_ref, 'homeostasis'):
                    if hasattr(self._agent_ref.homeostasis, 'energy_level'):
                        self._agent_ref.homeostasis.energy_level = float(value)
                        return True
                if param_name == "decay_rate":
                    return True
            
            if module_id == "execution":
                if param_name == "exploration_rate" and hasattr(self._agent_ref, 'execution'):
                    if hasattr(self._agent_ref.execution, 'exploration_rate'):
                        self._agent_ref.execution.exploration_rate = float(value)
                        return True
            
            if module_id == "dual_system":
                if param_name == "system1_threshold" and hasattr(self._agent_ref, 'dual_cognition'):
                    if hasattr(self._agent_ref.dual_cognition, 'system1'):
                        if hasattr(self._agent_ref.dual_cognition.system1, 'threshold'):
                            self._agent_ref.dual_cognition.system1.threshold = float(value)
                            return True
                if param_name == "system2_enabled" and hasattr(self._agent_ref, 'dual_cognition'):
                    return True
            
            if module_id == "cognitive":
                if param_name == "max_inference_steps":
                    return True
                if param_name == "confidence_threshold":
                    return True
            
            if module_id == "knowledge_graph":
                if param_name == "retrieval_top_k" and hasattr(self._agent_ref, 'knowledge_graph'):
                    return True
                if param_name == "consolidation_threshold":
                    return True
            
            if module_id == "perception":
                if param_name == "learning_rate" and hasattr(self._agent_ref, 'perception'):
                    return True
                if param_name == "feature_dim":
                    return True
            
            if module_id == "meta_cognition":
                if param_name == "meta_monitoring_rate" and hasattr(self._agent_ref, 'meta_cog'):
                    return True
            
            return False
        except Exception:
            return False

    def _register_rule_systems_from_agent(self, agent_ref):
        rule_system = RuleSpec(
            rule_id="system1_production",
            condition_schema={"pattern": "vector", "threshold": "float"},
            action_schema={"response": "vector", "confidence": "float"},
            confidence_range=(0.0, 1.0),
            max_rules=100,
            tier=SelfModificationTier.PARAM_RULE,
            description="系统1的产生式规则库"
        )
        
        def add_rule_callback(condition: Dict, action: Dict, confidence: float) -> bool:
            try:
                if hasattr(agent_ref, 'dual_cognition') and hasattr(agent_ref.dual_cognition, 'system1'):
                    cond_vec = condition.get("pattern", np.zeros(16))
                    act_vec = action.get("response", np.zeros(16))
                    if isinstance(cond_vec, list):
                        cond_vec = np.array(cond_vec)
                    if isinstance(act_vec, list):
                        act_vec = np.array(act_vec)
                    agent_ref.dual_cognition.system1.add_production_rule(
                        cond_vec, act_vec, confidence
                    )
                    return True
                return False
            except Exception:
                return False
        
        def remove_rule_callback(rule_id: str) -> bool:
            try:
                if hasattr(agent_ref, 'dual_cognition') and hasattr(agent_ref.dual_cognition, 'system1'):
                    rules = agent_ref.dual_cognition.system1.production_rules
                    idx = int(rule_id) if rule_id.isdigit() else 0
                    if 0 <= idx < len(rules):
                        rules.pop(idx)
                        return True
                return False
            except Exception:
                return False
        
        self.tiered_modifier.register_rule_system(rule_system, add_rule_callback, remove_rule_callback)

    def _register_black_box_modules(self):
        black_box_modules = [m for m in self.symbolic_self_model.get_all_modules() if m.is_black_box]
        for mod in black_box_modules:
            self._black_box_modules[mod.module_id] = {
                "name": mod.module_name,
                "inputs": {p.port_name: p.data_type for p in mod.inputs},
                "outputs": {p.port_name: p.data_type for p in mod.outputs},
                "metrics": {},
                "replaceable": True,
                "internal_visible": mod.internal_visible
            }
            self._module_performance[mod.module_id] = {
                "accuracy": 0.5,
                "latency_ms": 10.0,
                "throughput": 100.0,
                "stability": 0.8
            }

    def propose_tier1_improvements(self, performance_metrics: Dict[str, Any],
                                    diagnostic_findings: List[Any] = None) -> List[TieredModificationRequest]:
        if not self._initialized:
            return []
        
        proposals = []
        
        fe = performance_metrics.get("free_energy", 1.0)
        if fe > 0.5:
            req = self.tiered_modifier.request_param_change(
                "perception.learning_rate", 0.0015
            )
            if req:
                proposals.append(req)
        
        conf = performance_metrics.get("confidence", 0.5)
        if conf < 0.4:
            req = self.tiered_modifier.request_param_change(
                "dual_system.system1_threshold", 0.75
            )
            if req:
                proposals.append(req)
        
        error_rate = performance_metrics.get("error_rate", 0.1)
        if error_rate > 0.05:
            req = self.tiered_modifier.request_param_change(
                "execution.exploration_rate", 0.05
            )
            if req:
                proposals.append(req)
        
        curiosity = performance_metrics.get("curiosity", 0.3)
        if curiosity < 0.2:
            req = self.tiered_modifier.request_param_change(
                "homeostasis.energy_baseline", 0.65
            )
            if req:
                proposals.append(req)
        
        return proposals

    def verify_only(self, request: TieredModificationRequest) -> Dict[str, Any]:
        result = {
            "request": request.to_dict(),
            "verified": False,
            "verification_report": None,
            "error": None
        }
        
        if request.action == "set_param":
            param_path = request.target
            parts = param_path.split(".", 1)
            if len(parts) == 2:
                module_id, param_name = parts
                params = self.symbolic_self_model.get_module_params(module_id)
                param_spec = next((p for p in params if p["name"] == param_name), None)
                if param_spec:
                    report = self.formal_verifier.verify_param_change(
                        param_spec,
                        request.details["new_value"]
                    )
                    result["verification_report"] = report.to_dict()
                    
                    if report.overall_result == VerificationResult.PASS:
                        result["verified"] = True
                    else:
                        result["error"] = "verification_failed"
                else:
                    result["error"] = "param_spec_not_found"
            else:
                result["error"] = "invalid_param_path"
        
        elif request.action == "add_rule":
            rule_sys = self.tiered_modifier.rule_systems.get(request.target)
            if rule_sys:
                existing_rules = []
                if self._agent_ref and hasattr(self._agent_ref, 'dual_cognition'):
                    if hasattr(self._agent_ref.dual_cognition, 'system1'):
                        existing_rules = self._agent_ref.dual_cognition.system1.production_rules
                
                report = self.formal_verifier.verify_rule_addition(
                    rule_sys.to_dict(),
                    request.details["condition"],
                    request.details["action"],
                    request.details["confidence"],
                    existing_rules
                )
                result["verification_report"] = report.to_dict()
                
                if report.overall_result in (VerificationResult.PASS, VerificationResult.WARN):
                    result["verified"] = True
                else:
                    result["error"] = "verification_failed"
        
        return result

    def apply_only(self, request: TieredModificationRequest) -> Dict[str, Any]:
        result = {
            "request": request.to_dict(),
            "applied": False,
            "error": None
        }
        
        success, msg = self.tiered_modifier.execute_request(request)
        result["applied"] = success
        result["error"] = msg if not success else None
        
        if success:
            self._improvement_counter += 1
            self._successful_improvements += 1
            self._record_improvement(request, success)
            if request.action == "set_param":
                self._maybe_advance_tier()
        else:
            self._failed_improvements += 1
        
        return result

    def verify_and_apply(self, request: TieredModificationRequest) -> Dict[str, Any]:
        result = {
            "request": request.to_dict(),
            "verified": False,
            "applied": False,
            "verification_report": None,
            "error": None
        }
        
        verify_result = self.verify_only(request)
        result["verified"] = verify_result["verified"]
        result["verification_report"] = verify_result["verification_report"]
        result["error"] = verify_result["error"]
        
        if result["verified"]:
            apply_result = self.apply_only(request)
            result["applied"] = apply_result["applied"]
            result["error"] = apply_result["error"]
        
        return result

    def _record_improvement(self, request: TieredModificationRequest, success: bool):
        self._improvement_history.append({
            "request_id": request.request_id,
            "tier": request.tier.value,
            "target": request.target,
            "action": request.action,
            "success": success,
            "timestamp": time.time()
        })
        self.symbolic_self_model.log_modification(
            request.action,
            request.target,
            request.details
        )

    def _maybe_advance_tier(self):
        next_tier = self.tiered_modifier.maybe_unlock_next_tier()
        if next_tier:
            if next_tier == SelfModificationTier.MODULE:
                self.current_stage = ImprovementStage.TIER_2_MODULE
            elif next_tier == SelfModificationTier.ARCHITECTURE:
                self.current_stage = ImprovementStage.TIER_3_ARCHITECTURE

    def get_black_box_module_status(self, module_id: str = None) -> Dict[str, Any]:
        if module_id:
            mod_info = self._black_box_modules.get(module_id, {})
            perf = self._module_performance.get(module_id, {})
            return {
                "info": mod_info,
                "performance": perf
            }
        return {
            "modules": list(self._black_box_modules.keys()),
            "count": len(self._black_box_modules)
        }

    def get_available_improvements(self) -> Dict[str, Any]:
        return self.tiered_modifier.get_available_modifications()

    def get_bootstrapping_status(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "current_tier": self.tiered_modifier.current_tier.value,
            "unlocked_tiers": [t.value for t in self.tiered_modifier.unlocked_tiers],
            "total_improvements": self._improvement_counter,
            "successful": self._successful_improvements,
            "failed": self._failed_improvements,
            "success_rate": (self._successful_improvements / max(1, self._improvement_counter)),
            "black_box_modules": list(self._black_box_modules.keys()),
            "self_model_version": self.symbolic_self_model._version,
            "consistency_score": self.symbolic_self_model._consistency_score,
            "tier_metrics": self.tiered_modifier.get_tier_status()["tier_metrics"]
        }

    def query_self_model(self, query_type: str, target: str = None) -> Dict[str, Any]:
        return self.symbolic_self_model.query(query_type, target)

    def update_module_performance(self, module_id: str, metrics: Dict[str, float]):
        if module_id in self._module_performance:
            for k, v in metrics.items():
                if k in self._module_performance[module_id]:
                    old = self._module_performance[module_id][k]
                    self._module_performance[module_id][k] = 0.9 * old + 0.1 * v
