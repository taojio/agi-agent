from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class ComponentType(Enum):
    PERCEPTION = "perception"
    COGNITION = "cognition"
    EXECUTION = "execution"
    MEMORY = "memory"
    HOMEOSTASIS = "homeostasis"
    LEARNING = "learning"
    META = "meta"


class InterfaceType(Enum):
    SYMBOLIC = "symbolic"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class SymbolPortSpec:
    port_name: str
    direction: str
    data_type: str
    description: str = ""
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.port_name,
            "direction": self.direction,
            "type": self.data_type,
            "description": self.description,
            "required": self.required
        }


@dataclass
class SymbolRuleSpec:
    rule_id: str
    condition_pattern: str
    action_pattern: str
    confidence: str
    priority: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.rule_id,
            "condition": self.condition_pattern,
            "action": self.action_pattern,
            "confidence": self.confidence,
            "priority": self.priority,
            "description": self.description
        }


@dataclass
class SymbolModuleSpec:
    module_id: str
    module_name: str
    component_type: ComponentType
    interface_type: InterfaceType
    inputs: List[SymbolPortSpec]
    outputs: List[SymbolPortSpec]
    parameters: List[Dict[str, Any]]
    rules: List[SymbolRuleSpec]
    performance_children: List[str]
    description: str = ""
    is_black_box: bool = False
    internal_visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.module_id,
            "name": self.module_name,
            "type": self.component_type.value,
            "interface": self.interface_type.value,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "parameters": self.parameters,
            "rules": [r.to_dict() for r in self.rules],
            "performance_children": self.performance_children,
            "description": self.description,
            "is_black_box": self.is_black_box,
            "internal_visible": self.internal_visible
        }


@dataclass
class SymbolConnectionSpec:
    connection_id: str
    source_module: str
    source_port: str
    target_module: str
    target_port: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.connection_id,
            "from": f"{self.source_module}.{self.source_port}",
            "to": f"{self.target_module}.{self.target_port}",
            "description": self.description
        }


@dataclass
class SymbolArchitectureSpec:
    architecture_id: str
    name: str
    modules: List[SymbolModuleSpec]
    connections: List[SymbolConnectionSpec]
    data_flow: List[str]
    control_flow: List[str]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.architecture_id,
            "name": self.name,
            "modules": [m.to_dict() for m in self.modules],
            "connections": [c.to_dict() for c in self.connections],
            "data_flow": self.data_flow,
            "control_flow": self.control_flow,
            "description": self.description
        }


class SymbolicSelfModel:
    def __init__(self):
        self.architecture: Optional[SymbolArchitectureSpec] = None
        self._module_index: Dict[str, SymbolModuleSpec] = {}
        self._connection_index: Dict[str, SymbolConnectionSpec] = {}
        self._param_index: Dict[str, Dict[str, Any]] = {}
        self._rule_index: Dict[str, List[SymbolRuleSpec]] = {}
        self._sync_callbacks: Dict[str, List] = {}
        self._modification_log: List[Dict[str, Any]] = []
        self._version = 1
        self._consistency_score = 1.0

    def build_from_agent_structure(self, agent_ref):
        modules = []
        connections = []

        perception_mod = SymbolModuleSpec(
            module_id="perception",
            module_name="感知层",
            component_type=ComponentType.PERCEPTION,
            interface_type=InterfaceType.HYBRID,
            inputs=[
                SymbolPortSpec("sensory_input", "in", "vector", "原始传感器输入"),
            ],
            outputs=[
                SymbolPortSpec("feature_vector", "out", "vector", "编码后的特征向量"),
                SymbolPortSpec("free_energy", "out", "float", "自由能估计"),
                SymbolPortSpec("confidence", "out", "float", "感知置信度"),
            ],
            parameters=[
                {"name": "learning_rate", "type": "float", "min": 1e-5, "max": 0.1, "default": 1e-3, "tier": "param_rule"},
                {"name": "feature_dim", "type": "int", "min": 8, "max": 512, "default": 512, "tier": "module"},
            ],
            rules=[],
            performance_children=["multimodal_fusion"],
            description="负责将原始感官输入编码为抽象特征向量",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(perception_mod)

        multimodal_mod = SymbolModuleSpec(
            module_id="multimodal_fusion",
            module_name="多模态融合",
            component_type=ComponentType.PERCEPTION,
            interface_type=InterfaceType.VECTOR,
            inputs=[
                SymbolPortSpec("primary", "in", "vector", "主模态特征"),
            ],
            outputs=[
                SymbolPortSpec("fused", "out", "vector", "融合特征"),
            ],
            parameters=[
                {"name": "fusion_method", "type": "string", "options": ["concat", "attention", "gated"], "default": "concat", "tier": "module"},
            ],
            rules=[],
            performance_children=[],
            description="多模态特征融合模块",
            is_black_box=True,
            internal_visible=False
        )
        modules.append(multimodal_mod)

        cognitive_mod = SymbolModuleSpec(
            module_id="cognitive",
            module_name="认知推理层",
            component_type=ComponentType.COGNITION,
            interface_type=InterfaceType.HYBRID,
            inputs=[
                SymbolPortSpec("feature_input", "in", "vector", "输入特征"),
                SymbolPortSpec("goal_input", "in", "symbol", "目标符号"),
            ],
            outputs=[
                SymbolPortSpec("inference_result", "out", "vector", "推理结果"),
                SymbolPortSpec("confidence", "out", "float", "推理置信度"),
            ],
            parameters=[
                {"name": "max_inference_steps", "type": "int", "min": 1, "max": 20, "default": 5, "tier": "param_rule"},
                {"name": "confidence_threshold", "type": "float", "min": 0.1, "max": 0.99, "default": 0.7, "tier": "param_rule"},
            ],
            rules=[
                SymbolRuleSpec("rule_high_confidence_accept", "confidence > threshold", "accept_result", "0.9", 10,
                               "高置信度直接接受结论"),
                SymbolRuleSpec("rule_low_confidence_deliberate", "confidence < threshold", "deeper_reasoning", "0.8", 5,
                               "低置信度启动深度思考"),
            ],
            performance_children=["dual_system", "predictive_coding", "causal_reasoning"],
            description="核心认知推理引擎",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(cognitive_mod)

        dual_mod = SymbolModuleSpec(
            module_id="dual_system",
            module_name="双系统认知",
            component_type=ComponentType.COGNITION,
            interface_type=InterfaceType.HYBRID,
            inputs=[
                SymbolPortSpec("input_feature", "in", "vector", "输入特征"),
            ],
            outputs=[
                SymbolPortSpec("system1_output", "out", "vector", "系统1直觉输出"),
                SymbolPortSpec("system2_output", "out", "vector", "系统2深思输出"),
                SymbolPortSpec("final_output", "out", "vector", "最终输出"),
            ],
            parameters=[
                {"name": "system1_threshold", "type": "float", "min": 0.5, "max": 0.95, "default": 0.85, "tier": "param_rule"},
                {"name": "system2_enabled", "type": "bool", "default": True, "tier": "param_rule"},
            ],
            rules=[
                SymbolRuleSpec("rule_system1_fast", "familiar_pattern", "quick_response", "0.9", 10,
                               "熟悉模式用系统1快速响应"),
                SymbolRuleSpec("rule_system2_slow", "novel_pattern", "deliberate_thinking", "0.7", 5,
                               "新颖模式用系统2深思熟虑"),
            ],
            performance_children=["system1_intuitive", "system2_deliberative"],
            description="卡尼曼双系统认知架构",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(dual_mod)

        memory_mod = SymbolModuleSpec(
            module_id="knowledge_graph",
            module_name="知识图谱记忆",
            component_type=ComponentType.MEMORY,
            interface_type=InterfaceType.SYMBOLIC,
            inputs=[
                SymbolPortSpec("query", "in", "symbol", "查询符号"),
                SymbolPortSpec("new_knowledge", "in", "symbol", "新知识"),
            ],
            outputs=[
                SymbolPortSpec("retrieved", "out", "symbol", "检索结果"),
                SymbolPortSpec("associations", "out", "list", "关联知识列表"),
            ],
            parameters=[
                {"name": "retrieval_top_k", "type": "int", "min": 1, "max": 50, "default": 10, "tier": "param_rule"},
                {"name": "consolidation_threshold", "type": "float", "min": 0.1, "max": 0.9, "default": 0.5, "tier": "param_rule"},
            ],
            rules=[
                SymbolRuleSpec("rule_high_confidence_store", "confidence > 0.8", "permanent_store", "0.95", 8,
                               "高置信度知识永久存储"),
                SymbolRuleSpec("rule_low_confidence_cache", "confidence < 0.5", "temporary_cache", "0.6", 3,
                               "低置信度知识临时缓存"),
            ],
            performance_children=[],
            description="符号化知识图谱与长期记忆",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(memory_mod)

        homeo_mod = SymbolModuleSpec(
            module_id="homeostasis",
            module_name="内稳态引擎",
            component_type=ComponentType.HOMEOSTASIS,
            interface_type=InterfaceType.SYMBOLIC,
            inputs=[
                SymbolPortSpec("internal_state", "in", "dict", "内部状态信号"),
                SymbolPortSpec("external_event", "in", "symbol", "外部事件"),
            ],
            outputs=[
                SymbolPortSpec("need_signals", "out", "dict", "需求信号"),
                SymbolPortSpec("drive_levels", "out", "dict", "驱力水平"),
            ],
            parameters=[
                {"name": "energy_baseline", "type": "float", "min": 0.1, "max": 0.9, "default": 0.7, "tier": "param_rule"},
                {"name": "decay_rate", "type": "float", "min": 0.001, "max": 0.1, "default": 0.01, "tier": "param_rule"},
            ],
            rules=[
                SymbolRuleSpec("rule_energy_low_seek", "energy < 0.3", "seek_energy_source", "0.9", 10,
                               "能量低时主动寻找能量源"),
                SymbolRuleSpec("rule_curiosity_high_explore", "curiosity > 0.7", "explore_environment", "0.8", 7,
                               "好奇心强时探索环境"),
            ],
            performance_children=["need_regulator", "drive_generator"],
            description="维持内部稳态调节与驱力产生",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(homeo_mod)

        exec_mod = SymbolModuleSpec(
            module_id="execution",
            module_name="执行层",
            component_type=ComponentType.EXECUTION,
            interface_type=InterfaceType.HYBRID,
            inputs=[
                SymbolPortSpec("action_plan", "in", "list", "行动计划"),
                SymbolPortSpec("goal", "in", "symbol", "目标"),
            ],
            outputs=[
                SymbolPortSpec("action_output", "out", "vector", "动作输出"),
                SymbolPortSpec("execution_result", "out", "symbol", "执行结果"),
            ],
            parameters=[
                {"name": "exploration_rate", "type": "float", "min": 0.0, "max": 1.0, "default": 0.1, "tier": "param_rule"},
                {"name": "max_action_space", "type": "int", "min": 2, "max": 100, "default": 8, "tier": "module"},
            ],
            rules=[
                SymbolRuleSpec("rule_high_confidence_exploit", "confidence > 0.8", "exploit_best", "0.9", 10,
                               "高置信度时利用最优动作"),
                SymbolRuleSpec("rule_uncertain_explore", "uncertainty > 0.5", "explore_options", "0.7", 5,
                               "不确定时探索新选项"),
            ],
            performance_children=["action_selector", "motor_control"],
            description="动作选择与执行控制",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(exec_mod)

        meta_mod = SymbolModuleSpec(
            module_id="meta_cognition",
            module_name="元认知",
            component_type=ComponentType.META,
            interface_type=InterfaceType.SYMBOLIC,
            inputs=[
                SymbolPortSpec("cognitive_state", "in", "dict", "认知状态"),
                SymbolPortSpec("performance_feedback", "in", "float", "性能反馈"),
            ],
            outputs=[
                SymbolPortSpec("strategy_adjustment", "out", "symbol", "策略调整指令"),
                SymbolPortSpec("attention_focus", "out", "symbol", "注意力焦点"),
            ],
            parameters=[
                {"name": "meta_monitoring_rate", "type": "float", "min": 0.01, "max": 1.0, "default": 0.3, "tier": "param_rule"},
            ],
            rules=[
                SymbolRuleSpec("rule_performance_drop_adjust", "performance_decline", "strategy_shift", "0.85", 9,
                               "性能下降时调整策略"),
                SymbolRuleSpec("rule_stuck_try_new_strategy", "repeated_failure", "try_new_approach", "0.8", 7,
                               "重复失败时尝试新方法"),
            ],
            performance_children=["self_monitor", "strategy_selector"],
            description="监控和调节认知过程的元认知",
            is_black_box=False,
            internal_visible=True
        )
        modules.append(meta_mod)

        connections = [
            SymbolConnectionSpec("conn_001", "perception", "feature_vector", "cognitive", "feature_input",
                                 "感知特征到认知层"),
            SymbolConnectionSpec("conn_002", "perception", "feature_vector", "dual_system", "input_feature",
                                 "感知特征到双系统"),
            SymbolConnectionSpec("conn_003", "cognitive", "inference_result", "execution", "action_plan",
                                 "认知结果到执行层"),
            SymbolConnectionSpec("conn_004", "homeostasis", "need_signals", "meta_cognition", "cognitive_state",
                                 "内稳态信号到元认知"),
            SymbolConnectionSpec("conn_005", "knowledge_graph", "retrieved", "cognitive", "feature_input",
                                 "记忆检索到认知层"),
            SymbolConnectionSpec("conn_006", "meta_cognition", "strategy_adjustment", "cognitive", "goal_input",
                                 "元认知调节认知目标"),
            SymbolConnectionSpec("conn_007", "execution", "execution_result", "meta_cognition", "performance_feedback",
                                 "执行结果反馈到元认知"),
        ]

        data_flow = [
            "sensory_input → perception.feature_vector → cognitive.inference_result → execution.action_output",
            "internal_state → homeostasis.need_signals → meta_cognition.strategy_adjustment → cognitive.goal_input",
        ]

        control_flow = [
            "元认知监控所有模块性能",
            "内稳态驱动目标优先级",
            "双系统在直觉与深思间切换",
        ]

        arch = SymbolArchitectureSpec(
            architecture_id="agi_main_v1",
            name="自进化AGI主架构",
            modules=modules,
            connections=connections,
            data_flow=data_flow,
            control_flow=control_flow,
            description="基于主动推理 + 预测编码 + 双系统认知 + 内稳态 + 元认知的综合架构"
        )

        self.architecture = arch
        for m in modules:
            self._module_index[m.module_id] = m
        for c in connections:
            self._connection_index[c.connection_id] = c
        
        for m in modules:
            self._param_index[m.module_id] = m.parameters
            self._rule_index[m.module_id] = m.rules

        return True

    def _rebuild_indices(self):
        if not self.architecture:
            return
        self._module_index = {m.module_id: m for m in self.architecture.modules}
        self._connection_index = {c.connection_id: c for c in self.architecture.connections}
        for m in self.architecture.modules:
            self._param_index[m.module_id] = m.parameters
            self._rule_index[m.module_id] = m.rules

    def get_module_spec(self, module_id: str) -> Optional[SymbolModuleSpec]:
        return self._module_index.get(module_id)

    def get_all_modules(self, component_type: ComponentType = None) -> List[SymbolModuleSpec]:
        if not self.architecture:
            return []
        if component_type is None:
            return list(self._module_index.values())
        return [m for m in self._module_index.values() if m.component_type == component_type]

    def get_module_params(self, module_id: str) -> List[Dict[str, Any]]:
        return self._param_index.get(module_id, [])

    def get_module_rules(self, module_id: str) -> List[SymbolRuleSpec]:
        return self._rule_index.get(module_id, [])

    def get_connections_for(self, module_id: str) -> Dict[str, List[SymbolConnectionSpec]]:
        incoming = []
        outgoing = []
        for c in self._connection_index.values():
            if c.target_module == module_id:
                incoming.append(c)
            if c.source_module == module_id:
                outgoing.append(c)
        return {"incoming": incoming, "outgoing": outgoing}

    def validate_param_change(self, module_id: str, param_name: str, new_value: Any) -> Tuple[bool, str]:
        params = self._param_index.get(module_id)
        if not params:
            return False, "module_not_found"
        for p in params:
            if p["name"] == param_name:
                if p["type"] == "float":
                    if not isinstance(new_value, (int, float)):
                        return False, "type_mismatch"
                    if "min" in p and new_value < p["min"]:
                        return False, "below_min"
                    if "max" in p and new_value > p["max"]:
                        return False, "above_max"
                    return True, "valid"
                elif p["type"] == "int":
                    if not isinstance(new_value, int):
                        return False, "type_mismatch"
                    if "min" in p and new_value < p["min"]:
                        return False, "below_min"
                    if "max" in p and new_value > p["max"]:
                        return False, "above_max"
                    return True, "valid"
                elif p["type"] == "bool":
                    return isinstance(new_value, bool), "valid" if isinstance(new_value, bool) else "type_mismatch"
                elif p["type"] == "string":
                    if "options" in p and new_value not in p["options"]:
                        return False, "invalid_option"
                    return True, "valid"
        return False, "param_not_found"

    def check_connector_consistency(self) -> Dict[str, Any]:
        issues = []
        if not self.architecture:
            return {"consistent": False, "issues": ["no_architecture"]}
        
        module_ids = set(self._module_index.keys())
        
        for conn in self._connection_index.values():
            src_mod = self._module_index.get(conn.source_module)
            tgt_mod = self._module_index.get(conn.target_module)
            if not src_mod:
                issues.append(f"missing_source_module:{conn.source_module}")
                continue
            if not tgt_mod:
                issues.append(f"missing_target_module:{conn.target_module}")
                continue
            src_ports = {p.port_name for p in src_mod.outputs}
            tgt_ports = {p.port_name for p in tgt_mod.inputs}
            if conn.source_port not in src_ports:
                issues.append(f"missing_source_port:{conn.source_module}.{conn.source_port}")
            if conn.target_port not in tgt_ports:
                issues.append(f"missing_target_port:{conn.target_module}.{conn.target_port}")
        
        self._consistency_score = 1.0 - (len(issues) / max(1, len(self._connection_index)))
        
        return {
            "consistent": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues,
            "consistency_score": self._consistency_score
        }

    def get_self_description(self, depth: str = "summary") -> Dict[str, Any]:
        if not self.architecture:
            return {}
        
        if depth == "summary":
            return {
                "architecture_id": self.architecture.architecture_id,
                "name": self.architecture.name,
                "module_count": len(self._module_index),
                "connection_count": len(self._connection_index),
                "component_types": list(set(m.component_type.value for m in self._module_index.values())),
                "version": self._version,
                "description": self.architecture.description
            }
        elif depth == "full":
            return self.architecture.to_dict()
        else:
            return self.architecture.to_dict()

    def query(self, query_type: str, target: str = None) -> Dict[str, Any]:
        if query_type == "module_list":
            return {"modules": [m.module_id for m in self._module_index.values()]}
        elif query_type == "module_detail" and target:
            mod = self._module_index.get(target)
            return mod.to_dict() if mod else {}
        elif query_type == "module_params" and target:
            return {"params": self._param_index.get(target, [])}
        elif query_type == "module_rules" and target:
            return {"rules": [r.to_dict() for r in self._rule_index.get(target, [])]}
        elif query_type == "connections" and target:
            conn = self.get_connections_for(target)
            return {
                "incoming": [c.to_dict() for c in conn["incoming"]],
                "outgoing": [c.to_dict() for c in conn["outgoing"]]
            }
        elif query_type == "data_flow":
            return {"data_flow": self.architecture.data_flow if self.architecture else []}
        elif query_type == "control_flow":
            return {"control_flow": self.architecture.control_flow if self.architecture else []}
        return {}

    def log_modification(self, mod_type: str, target: str, details: Dict[str, Any]):
        self._modification_log.append({
            "type": mod_type,
            "target": target,
            "details": details,
            "timestamp": time.time(),
            "version": self._version
        })
        self._version += 1

    def get_modification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._modification_log)[-limit:]
