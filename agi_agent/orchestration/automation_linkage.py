"""
自动化联动引擎 (Automation Linkage Engine)

基于事件驱动的跨模块自动化联动机制：
- 条件触发：监控系统状态，当满足预设条件时自动触发
- 事件链：支持多步骤联动（A检测→B分析→C执行）
- 优先级响应：高优先级规则可抢占低优先级
- 冷却机制：防止规则频繁重复触发
- 动态注册：运行时可添加/移除/禁用规则
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Any, Optional
from collections import deque

logger = logging.getLogger("agi_agent.automation_linkage")


class TriggerPriority(Enum):
    """联动规则优先级"""
    CRITICAL = 0   # 安全/紧急 - 立即执行
    HIGH = 1       # 重要 - 优先执行
    MEDIUM = 2     # 常规 - 队列执行
    LOW = 3        # 后台 - 空闲执行


class LinkageRuleType(Enum):
    """联动规则类型"""
    THRESHOLD = "threshold"         # 阈值触发
    PATTERN = "pattern"             # 模式匹配
    CUMULATIVE = "cumulative"       # 累积条件
    EVENT_CHAIN = "event_chain"     # 事件链


@dataclass
class SystemState:
    """系统状态快照 - 每步采集"""
    step: int = 0
    free_energy: float = 0.0
    confidence: float = 0.5
    novelty: float = 0.0
    entropy: float = 0.0
    latency: float = 0.0
    is_impasse: bool = False
    system_used: str = ""
    action_success: bool = True
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph_stats: Dict[str, Any] = field(default_factory=dict)
    evolution_stats: Dict[str, Any] = field(default_factory=dict)
    reflex_stats: Dict[str, Any] = field(default_factory=dict)
    safety_status: Dict[str, Any] = field(default_factory=dict)
    meta_cognitive_stats: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkageRule:
    """联动规则定义"""
    rule_id: str
    name: str
    description: str
    condition: Callable[[SystemState], bool]
    actions: List[Callable[[Any, SystemState], Dict[str, Any]]]
    priority: TriggerPriority = TriggerPriority.MEDIUM
    cooldown_steps: int = 50        # 最小触发间隔
    last_triggered: int = -999999   # 上次触发的step
    enabled: bool = True
    max_consecutive: int = 3        # 最大连续触发次数
    consecutive_count: int = 0      # 当前连续触发次数
    rule_type: LinkageRuleType = LinkageRuleType.THRESHOLD

    def can_trigger(self, current_step: int) -> bool:
        if not self.enabled:
            return False
        if self.consecutive_count >= self.max_consecutive:
            return False
        if current_step - self.last_triggered < self.cooldown_steps:
            return False
        return True

    def mark_triggered(self, current_step: int):
        self.last_triggered = current_step
        self.consecutive_count += 1

    def reset_consecutive(self):
        self.consecutive_count = 0


@dataclass
class LinkageResult:
    """单次联动执行结果"""
    rule_id: str
    rule_name: str
    triggered: bool
    actions_executed: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None


class AutomationLinkageEngine:
    """自动化联动引擎

    核心职责：
    1. 监控系统状态，检测触发条件
    2. 按优先级执行联动动作
    3. 管理规则生命周期（注册/禁用/冷却）
    4. 记录联动历史供分析
    """

    def __init__(self, max_history: int = 500):
        self._rules: Dict[str, LinkageRule] = {}
        self._execution_history: deque = deque(maxlen=max_history)
        self._stats: Dict[str, Any] = {
            "total_checks": 0,
            "total_triggered": 0,
            "total_actions_executed": 0,
            "rule_stats": {},
        }
        self._state_history: deque = deque(maxlen=100)
        self._consecutive_reset_steps = 200  # 连续触发计数重置间隔

    def register_rule(self, rule: LinkageRule) -> bool:
        if rule.rule_id in self._rules:
            logger.warning(f"Rule {rule.rule_id} already exists, overwriting")
        self._rules[rule.rule_id] = rule
        self._stats["rule_stats"][rule.rule_id] = {
            "triggered": 0,
            "last_triggered_step": -1,
            "total_actions": 0,
        }
        logger.info(f"Registered linkage rule: {rule.name} ({rule.rule_id})")
        return True

    def remove_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            del self._rules[rule_id]
            if rule_id in self._stats["rule_stats"]:
                del self._stats["rule_stats"][rule_id]
            return True
        return False

    def enable_rule(self, rule_id: str):
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False

    def collect_state(self, agent, step_metrics: Dict[str, Any]) -> SystemState:
        """从agent和步骤指标中采集系统状态"""
        state = SystemState(
            step=step_metrics.get("step", 0),
            free_energy=step_metrics.get("free_energy", 0.0),
            confidence=step_metrics.get("confidence", 0.5),
            novelty=step_metrics.get("novelty", 0.0),
            entropy=step_metrics.get("entropy", 0.0),
            latency=step_metrics.get("latency", 0.0),
            is_impasse=step_metrics.get("is_impasse", False),
            system_used=step_metrics.get("system_used", ""),
            action_success=step_metrics.get("confidence", 0.5) > 0.5,
            memory_stats=step_metrics.get("memory_tiers", {}),
            knowledge_graph_stats=step_metrics.get("knowledge_graph", {}),
            evolution_stats=step_metrics.get("evolution_stats", {}),
            reflex_stats=step_metrics.get("reflex_stats", {}),
            safety_status=step_metrics.get("safety", {}),
            meta_cognitive_stats=step_metrics.get("meta_cognitive_stats", {}),
            extra=step_metrics.get("extra", {}),
        )
        self._state_history.append(state)
        return state

    def check_and_execute(self, agent, state: SystemState) -> List[LinkageResult]:
        """检查所有规则条件并执行触发的联动动作"""
        self._stats["total_checks"] += 1

        # 定期重置连续触发计数
        if state.step > 0 and state.step % self._consecutive_reset_steps == 0:
            for rule in self._rules.values():
                rule.reset_consecutive()

        # 按优先级排序
        sorted_rules = sorted(
            self._rules.values(),
            key=lambda r: (r.priority.value, -r.last_triggered)
        )

        results: List[LinkageResult] = []

        for rule in sorted_rules:
            if not rule.can_trigger(state.step):
                continue

            try:
                condition_met = rule.condition(state)
            except Exception as e:
                logger.error(f"Rule {rule.rule_id} condition error: {e}")
                continue

            if not condition_met:
                continue

            # 条件满足，执行联动动作
            result = LinkageResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                triggered=True,
                actions_executed=0,
            )

            start_time = time.time()

            for action_fn in rule.actions:
                try:
                    action_result = action_fn(agent, state)
                    if action_result is None:
                        action_result = {"status": "ok"}
                    result.results.append(action_result)
                    result.actions_executed += 1
                    self._stats["total_actions_executed"] += 1
                except Exception as e:
                    logger.error(f"Rule {rule.rule_id} action error: {e}")
                    result.error = str(e)
                    break

            result.execution_time = time.time() - start_time
            rule.mark_triggered(state.step)

            # 更新统计
            self._stats["total_triggered"] += 1
            rule_stat = self._stats["rule_stats"].get(rule.rule_id, {})
            rule_stat["triggered"] = rule_stat.get("triggered", 0) + 1
            rule_stat["last_triggered_step"] = state.step
            rule_stat["total_actions"] = rule_stat.get("total_actions", 0) + result.actions_executed
            self._stats["rule_stats"][rule.rule_id] = rule_stat

            self._execution_history.append({
                "step": state.step,
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "priority": rule.priority.name,
                "actions_executed": result.actions_executed,
                "execution_time": result.execution_time,
                "error": result.error,
                "timestamp": time.time(),
            })

            results.append(result)

            logger.info(
                f"[Linkage] Rule '{rule.name}' triggered at step {state.step}, "
                f"executed {result.actions_executed} actions in {result.execution_time:.3f}s"
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "active_rules": sum(1 for r in self._rules.values() if r.enabled),
            "total_rules": len(self._rules),
            "recent_triggers": list(self._execution_history)[-10:],
        }

    def get_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        recent = list(self._execution_history)[-limit:]
        return recent[::-1]


def create_default_linkage_rules() -> List[LinkageRule]:
    """创建默认联动规则集

    覆盖系统核心自动化联动场景：
    1. 高自由能 → 触发进化优化 + 自我改进
    2. 低置信度 → 触发元学习策略 + 深度推理
    3. 高新颖度 → 触发知识图谱更新 + 元解析
    4. 性能停滞 → 触发探索 + 架构变异
    5. 安全威胁 → 触发断路器 + 紧急诊断
    6. 进化成功 → 传播参数 + 通知元学习
    7. 记忆过载 → 触发记忆巩固
    8. 反射中断 → 紧急诊断 + 安全通知
    """
    rules = []

    # ========== 规则1: 高自由能 → 进化优化 + 自我改进 ==========
    def _high_fe_condition(state: SystemState) -> bool:
        return state.free_energy > 1.5 and state.confidence < 0.5

    def _trigger_evolution(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'dual_loop_evolution'):
            result = agent.dual_loop_evolution.evolve_outer(
                population=agent.perception.parameters(),
                fitness_score=state.confidence
            )
            return {"action": "evolution_triggered", "result": "completed"}
        return {"action": "evolution_triggered", "result": "skipped"}

    def _trigger_self_improvement(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'self_diagnostic'):
            findings = agent.self_diagnostic.run_diagnostics(
                system_state={"step": state.step, "free_energy": state.free_energy},
                metrics={"free_energy": state.free_energy, "confidence": state.confidence}
            )
            if findings and hasattr(agent, 'self_improver'):
                agent.self_improver.generate_proposals(findings, {
                    "free_energy": state.free_energy,
                    "confidence": state.confidence,
                })
            return {"action": "self_improvement_triggered", "findings_count": len(findings) if findings else 0}
        return {"action": "self_improvement_triggered", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="high_fe_evolution",
        name="高自由能联动：进化+自我改进",
        description="当自由能>1.5且置信度<0.5时，立即触发进化优化和自我改进诊断",
        condition=_high_fe_condition,
        actions=[_trigger_evolution, _trigger_self_improvement],
        priority=TriggerPriority.HIGH,
        cooldown_steps=30,
        max_consecutive=2,
    ))

    # ========== 规则2: 低置信度 → 元学习策略 + 深度推理 ==========
    def _low_confidence_condition(state: SystemState) -> bool:
        return state.confidence < 0.35 and not state.is_impasse

    def _trigger_meta_learning(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'meta_learning_orchestrator'):
            task_type = "exploration" if state.novelty > 0.5 else "exploitation"
            recommendation = agent.meta_learning_orchestrator.get_strategy_recommendation(
                task_type, min(1.0, state.novelty * 1.5)
            )
            strategy = recommendation.get("recommended_strategy", "balanced")
            if hasattr(agent, 'perception') and agent.perception.optimizer is not None:
                current_lr = agent.perception.optimizer.param_groups[0]['lr']
                if strategy == "exploration":
                    new_lr = min(0.05, current_lr * 1.2)
                elif strategy == "exploitation":
                    new_lr = max(1e-5, current_lr * 0.85)
                else:
                    new_lr = current_lr
                if abs(new_lr - current_lr) / (current_lr + 1e-8) > 0.01:
                    agent.perception.optimizer.param_groups[0]['lr'] = new_lr
            return {"action": "meta_learning_adjusted", "strategy": strategy}
        return {"action": "meta_learning_adjusted", "result": "skipped"}

    def _trigger_deep_reasoning(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'thinking_orchestrator'):
            context = {
                "risk_level": state.novelty,
                "threat_detected": state.free_energy > 0.8,
                "goal_detected": False,
                "novelty": state.novelty,
                "free_energy": state.free_energy,
                "confidence": state.confidence,
            }
            result = agent.thinking_orchestrator.process(
                input_vector=[state.free_energy, state.confidence, state.novelty, state.entropy],
                context=context
            )
            return {"action": "deep_reasoning_triggered", "mode": result.get("mode", "unknown")}
        return {"action": "deep_reasoning_triggered", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="low_confidence_reasoning",
        name="低置信度联动：元学习+深度推理",
        description="当置信度<0.35时，立即调整学习策略并启动深度推理",
        condition=_low_confidence_condition,
        actions=[_trigger_meta_learning, _trigger_deep_reasoning],
        priority=TriggerPriority.HIGH,
        cooldown_steps=20,
        max_consecutive=3,
    ))

    # ========== 规则3: 高新颖度 → 知识图谱更新 + 元解析 ==========
    def _high_novelty_condition(state: SystemState) -> bool:
        return state.novelty > 0.7 and state.step > 10

    def _trigger_kg_update(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'knowledge_graph') and hasattr(agent, 'memory_harness'):
            agent.memory_harness.add_context_memory(
                content=f"High novelty event at step {state.step}: novelty={state.novelty:.3f}, fe={state.free_energy:.3f}",
                category="KNOWLEDGE" if hasattr(agent.memory_harness, '__module__') else "knowledge",
                source_agent="automation_linkage"
            )
            return {"action": "knowledge_update", "novelty": state.novelty}
        return {"action": "knowledge_update", "result": "skipped"}

    def _trigger_meta_parsing(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'meta_parsing'):
            data_str = f"novelty={state.novelty:.3f},fe={state.free_energy:.3f},conf={state.confidence:.3f}"
            result = agent.meta_parsing.parse_and_understand(data_str, format_hint="text")
            return {"action": "meta_parsing", "success": result.get("success", False)}
        return {"action": "meta_parsing", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="high_novelty_knowledge",
        name="高新颖度联动：知识更新+元解析",
        description="当新颖度>0.7时，自动更新知识图谱并触发元解析理解",
        condition=_high_novelty_condition,
        actions=[_trigger_kg_update, _trigger_meta_parsing],
        priority=TriggerPriority.MEDIUM,
        cooldown_steps=25,
        max_consecutive=3,
    ))

    # ========== 规则4: 性能停滞 → 探索 + 架构变异 ==========
    def _stagnation_condition(state: SystemState) -> bool:
        """检测性能停滞：置信度持续低且新颖度也低"""
        return (state.confidence < 0.45 and
                state.novelty < 0.15 and
                state.step > 100)

    def _trigger_exploration(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'learning_planner'):
            try:
                from agi_agent.learning.learning_planner import LearningGoalType, LearningPriority
                goal_id = agent.learning_planner.create_learning_goal(
                    goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
                    description=f"Stagnation recovery at step {state.step}",
                    priority=LearningPriority.HIGH,
                    target_confidence=0.8
                )
                plan_id = agent.learning_planner.create_learning_plan([goal_id])
                agent.learning_planner.execute_plan(plan_id)
                return {"action": "exploration_triggered", "goal_id": goal_id}
            except Exception as e:
                return {"action": "exploration_triggered", "error": str(e)}
        return {"action": "exploration_triggered", "result": "skipped"}

    def _trigger_architecture_mutation(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'architecture_mutator'):
            trigger = agent.architecture_mutator.should_mutate({
                'confidence': state.confidence,
                'impasse_count': 0,
                'stagnation_score': max(0.0, 1.0 - state.novelty),
            })
            if trigger:
                agent.architecture_mutator.mutate(trigger, {
                    'confidence': state.confidence,
                    'step': state.step,
                })
                return {"action": "architecture_mutation", "type": trigger}
            return {"action": "architecture_mutation", "result": "not_needed"}
        return {"action": "architecture_mutation", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="stagnation_exploration",
        name="性能停滞联动：探索+架构变异",
        description="当置信度持续低且新颖度也低时，触发主动探索和架构变异",
        condition=_stagnation_condition,
        actions=[_trigger_exploration, _trigger_architecture_mutation],
        priority=TriggerPriority.MEDIUM,
        cooldown_steps=80,
        max_consecutive=2,
    ))

    # ========== 规则5: 安全威胁 → 断路器 + 紧急诊断 ==========
    def _safety_threat_condition(state: SystemState) -> bool:
        safety = state.safety_status
        cb_state = safety.get("circuit_breaker", {}) if isinstance(safety, dict) else {}
        if isinstance(cb_state, dict):
            return cb_state.get("state") == "open" or cb_state.get("state") == "tripped"
        return state.free_energy > 2.5

    def _trigger_circuit_breaker(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'circuit_breaker'):
            agent.circuit_breaker.record_failure()
            return {"action": "circuit_breaker_activated"}
        return {"action": "circuit_breaker_activated", "result": "skipped"}

    def _trigger_emergency_diagnostic(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'audit_trail'):
            agent.audit_trail.log_entry("security", "emergency_linkage_trigger", {
                "step": state.step,
                "free_energy": state.free_energy,
                "confidence": state.confidence,
                "safety_status": state.safety_status,
            })
        if hasattr(agent, 'self_diagnostic'):
            findings = agent.self_diagnostic.run_diagnostics(
                system_state={"step": state.step, "emergency": True},
                metrics={"free_energy": state.free_energy, "confidence": state.confidence}
            )
            return {"action": "emergency_diagnostic", "findings": len(findings) if findings else 0}
        return {"action": "emergency_diagnostic", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="safety_emergency",
        name="安全威胁联动：断路器+紧急诊断",
        description="当安全状态异常时，立即触发断路器和紧急诊断",
        condition=_safety_threat_condition,
        actions=[_trigger_circuit_breaker, _trigger_emergency_diagnostic],
        priority=TriggerPriority.CRITICAL,
        cooldown_steps=5,
        max_consecutive=5,
    ))

    # ========== 规则6: 进化成功 → 传播参数 + 通知元学习 ==========
    def _evolution_success_condition(state: SystemState) -> bool:
        evo = state.evolution_stats
        if isinstance(evo, dict):
            best_fitness = evo.get("best_fitness", 0)
            gen = evo.get("generation", 0)
            return best_fitness > 0.7 and gen > 0 and state.step > 50
        return False

    def _propagate_evolution_params(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'meta_learn') and hasattr(agent.meta_learn, 'best_lr'):
            if hasattr(agent, 'perception') and agent.perception.optimizer is not None:
                current_lr = agent.perception.optimizer.param_groups[0]['lr']
                best_lr = agent.meta_learn.best_lr
                if best_lr > 0 and abs(best_lr - current_lr) / (current_lr + 1e-8) > 0.05:
                    agent.perception.optimizer.param_groups[0]['lr'] = best_lr
                    return {"action": "params_propagated", "old_lr": current_lr, "new_lr": best_lr}
        return {"action": "params_propagated", "result": "no_change"}

    def _notify_meta_learning(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'meta_learning_orchestrator'):
            evo = state.evolution_stats
            if isinstance(evo, dict) and hasattr(agent.meta_learning_orchestrator, 'register_task'):
                try:
                    agent.meta_learning_orchestrator.register_task(
                        task_id=f"evo_task_{state.step}",
                        task_type="evolution",
                        data_samples=[],
                        meta_context={"best_fitness": evo.get("best_fitness", 0)}
                    )
                    return {"action": "meta_learning_notified"}
                except Exception:
                    pass
        return {"action": "meta_learning_notified", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="evolution_success_propagate",
        name="进化成功联动：参数传播+元学习通知",
        description="当进化达到高适应度时，自动传播参数并通知元学习系统",
        condition=_evolution_success_condition,
        actions=[_propagate_evolution_params, _notify_meta_learning],
        priority=TriggerPriority.LOW,
        cooldown_steps=100,
        max_consecutive=1,
    ))

    # ========== 规则7: 记忆过载 → 记忆巩固 ==========
    def _memory_overload_condition(state: SystemState) -> bool:
        mem = state.memory_stats
        if isinstance(mem, dict):
            l1_count = 0
            for key, val in mem.items():
                if isinstance(val, dict):
                    l1_count += val.get("count", 0)
                elif isinstance(val, (int, float)):
                    l1_count += val
            return l1_count > 500
        return False

    def _trigger_memory_consolidation(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'memory_harness'):
            try:
                agent.memory_harness.consolidate()
                return {"action": "memory_consolidated"}
            except Exception as e:
                return {"action": "memory_consolidated", "error": str(e)}
        return {"action": "memory_consolidated", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="memory_overload_consolidation",
        name="记忆过载联动：自动巩固",
        description="当记忆条目过多时，自动触发记忆巩固",
        condition=_memory_overload_condition,
        actions=[_trigger_memory_consolidation],
        priority=TriggerPriority.MEDIUM,
        cooldown_steps=60,
        max_consecutive=2,
    ))

    # ========== 规则8: 反射中断 → 紧急诊断 + 安全通知 ==========
    def _reflex_shutdown_condition(state: SystemState) -> bool:
        reflex = state.reflex_stats
        if isinstance(reflex, dict):
            shutdown = reflex.get("shutdown_triggered", False)
            return shutdown
        return False

    def _trigger_reflex_emergency(agent, state: SystemState) -> Dict[str, Any]:
        if hasattr(agent, 'audit_trail'):
            agent.audit_trail.log_entry("reflex", "reflex_shutdown_linkage", {
                "step": state.step,
                "free_energy": state.free_energy,
                "confidence": state.confidence,
            })
        if hasattr(agent, 'self_diagnostic'):
            findings = agent.self_diagnostic.run_diagnostics(
                system_state={"step": state.step, "reflex_shutdown": True},
                metrics={"free_energy": state.free_energy, "confidence": state.confidence}
            )
            return {"action": "reflex_emergency_diagnostic", "findings": len(findings) if findings else 0}
        return {"action": "reflex_emergency_diagnostic", "result": "skipped"}

    rules.append(LinkageRule(
        rule_id="reflex_shutdown_emergency",
        name="反射中断联动：紧急诊断",
        description="当反射系统触发关闭时，立即进行紧急诊断",
        condition=_reflex_shutdown_condition,
        actions=[_trigger_reflex_emergency],
        priority=TriggerPriority.CRITICAL,
        cooldown_steps=3,
        max_consecutive=5,
    ))

    return rules
