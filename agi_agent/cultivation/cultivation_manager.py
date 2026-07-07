import time
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from agi_agent.evolution.quad_level_evolution import EvolutionTier


class CultivationPhase(Enum):
    FOUNDATION = "foundation"
    GROWTH = "growth"
    AUTONOMOUS = "autonomous"


@dataclass
class PhaseConfig:
    phase: CultivationPhase
    enabled_features: List[str]
    disabled_features: List[str]
    description: str
    objectives: List[str]
    acceptance_criteria: List[str]
    duration_days: int = 0


@dataclass
class PhaseTransition:
    from_phase: CultivationPhase
    to_phase: CultivationPhase
    timestamp: float
    reason: str
    operator: str = "system"


@dataclass
class CultivationStats:
    phase_start_time: float
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    rule_matches: int = 0
    fallback_triggered: int = 0
    avg_response_time_ms: float = 0.0
    compliance_rate: float = 0.0
    performance_score: float = 0.0


class CultivationManager:
    def __init__(self, agent=None, config_path: str = "./data/cultivation"):
        self.config_path = config_path
        self.agent = agent
        self.current_phase = CultivationPhase.FOUNDATION
        self.phase_history: List[PhaseTransition] = []
        self.phase_stats: Dict[CultivationPhase, CultivationStats] = {}
        
        self._init_phase_configs()
        self._load_state()
        self._apply_phase_config()
        
    def _init_phase_configs(self):
        self.phase_configs: Dict[CultivationPhase, PhaseConfig] = {
            CultivationPhase.FOUNDATION: PhaseConfig(
                phase=CultivationPhase.FOUNDATION,
                enabled_features=[
                    "rule_matching",
                    "basic_execution",
                    "memory_l1_l3",
                    "safety_monitor",
                    "manual_save",
                    "version_lock"
                ],
                disabled_features=[
                    "auto_evolution",
                    "unsupervised_learning",
                    "autonomous_decision",
                    "self_improvement",
                    "meta_learning",
                    "snn_stdp",
                    "skill_auto_generate",
                    "rule_auto_generate",
                    "multi_agent",
                    "memory_l4_l5",
                    "auto_save"
                ],
                description="冷启动筑基期：跑通输入输出全链路，杜绝静默无响应，基础任务100%可复现",
                objectives=[
                    "跑通输入接收→规则匹配→执行动作→结果输出全链路",
                    "标准指令响应率100%，无静默失败",
                    "异常输入100%触发兜底回复",
                    "核心任务执行成功率100%，结果可复现"
                ],
                acceptance_criteria=[
                    "标准指令响应率 ≥ 100%",
                    "异常输入兜底触发率 ≥ 100%",
                    "核心任务执行成功率 ≥ 100%",
                    "无任何静默无响应情况",
                    "全链路日志完整可追溯"
                ],
                duration_days=14
            ),
            CultivationPhase.GROWTH: PhaseConfig(
                phase=CultivationPhase.GROWTH,
                enabled_features=[
                    "rule_matching",
                    "basic_execution",
                    "memory_l1_l4",
                    "safety_monitor",
                    "manual_save",
                    "snn_supervised",
                    "rule_proposal",
                    "human_review",
                    "meta_cognition_monitor",
                    "skill_manual_add"
                ],
                disabled_features=[
                    "auto_evolution",
                    "unsupervised_learning",
                    "autonomous_decision",
                    "self_improvement",
                    "snn_stdp",
                    "skill_auto_generate",
                    "rule_auto_apply",
                    "multi_agent",
                    "memory_l5_auto",
                    "auto_save"
                ],
                description="场景成长期：定向积累场景经验，提升规则覆盖率和执行准确率，实现越用越稳",
                objectives=[
                    "定向投喂高质量标注数据",
                    "半自动规则补全，人工审核上线",
                    "SNN渐进式调优，提升匹配精度",
                    "建立元认知安全边界"
                ],
                acceptance_criteria=[
                    "高频场景规则覆盖率 ≥ 90%",
                    "执行成功率 ≥ 90%",
                    "异常场景均有明确反馈",
                    "新增规则无冲突导致的逻辑混乱",
                    "元认知拦截率可控"
                ],
                duration_days=28
            ),
            CultivationPhase.AUTONOMOUS: PhaseConfig(
                phase=CultivationPhase.AUTONOMOUS,
                enabled_features=[
                    "rule_matching",
                    "basic_execution",
                    "memory_full",
                    "safety_monitor",
                    "auto_save",
                    "snn_stdp",
                    "rule_auto_apply",
                    "meta_learning",
                    "meta_cognition",
                    "auto_evolution",
                    "self_improvement",
                    "skill_auto_generate",
                    "multi_agent",
                    "dual_loop_evolution"
                ],
                disabled_features=[
                    "version_lock",
                    "high_risk_autonomous"
                ],
                description="自主进阶期：逐步放开自主权限，启动闭环进化，实现越用越聪明",
                objectives=[
                    "分层开启进化权限",
                    "阶梯放开自主决策与行动",
                    "启动双循环进化",
                    "实现复利成长"
                ],
                acceptance_criteria=[
                    "低风险任务自主完成率 ≥ 80%",
                    "规则与技能自动生成通过率 ≥ 70%",
                    "系统执行效率随使用持续提升",
                    "安全边界无突破",
                    "进化可回滚、可审计"
                ],
                duration_days=0
            )
        }
    
    def _apply_phase_config(self):
        config = self.phase_configs[self.current_phase]
        
        if self.agent:
            self.agent.autonomous_mode = "autonomous_decision" in config.enabled_features
            
            if hasattr(self.agent, 'reflex_controller'):
                if hasattr(self.agent.reflex_controller, 'spiking_core'):
                    snn_learning = "snn_stdp" in config.enabled_features or "snn_supervised" in config.enabled_features
                    self.agent.reflex_controller.spiking_core.learning_enabled = snn_learning
                
                if hasattr(self.agent.reflex_controller, 'rule_engine'):
                    pass
            
            if hasattr(self.agent, 'meta_learn'):
                self.agent.meta_learn.enabled = "meta_learning" in config.enabled_features
            
            if hasattr(self.agent, 'meta_cog'):
                self.agent.meta_cog.enabled = "meta_cognition" in config.enabled_features or "meta_cognition_monitor" in config.enabled_features
            
            if hasattr(self.agent, 'evolve_engine'):
                self.agent.evolve_engine.enabled = "auto_evolution" in config.enabled_features
            
            if hasattr(self.agent, 'quad_level_evolution'):
                qle = self.agent.quad_level_evolution
                auto_evolution_enabled = "auto_evolution" in config.enabled_features
                if auto_evolution_enabled:
                    qle.enable_tier(EvolutionTier.MICRO, True)
                    qle.enable_tier(EvolutionTier.MESO, True)
                else:
                    qle.enable_tier(EvolutionTier.MICRO, False)
                    qle.enable_tier(EvolutionTier.MESO, False)
                    qle.enable_tier(EvolutionTier.MACRO, False)
                    qle.enable_tier(EvolutionTier.META, False)
            
            if hasattr(self.agent, 'dual_loop_evolution'):
                self.agent.dual_loop_evolution.enabled = "dual_loop_evolution" in config.enabled_features
            
            if hasattr(self.agent, 'self_improver'):
                self.agent.self_improver.enabled = "self_improvement" in config.enabled_features
            
            if hasattr(self.agent, 'memory_harness'):
                memory_levels = config.enabled_features
                self.agent.memory_harness.set_active_tiers(
                    l1="memory_l1_l3" in memory_levels or "memory_l1_l4" in memory_levels or "memory_full" in memory_levels,
                    l2="memory_l1_l3" in memory_levels or "memory_l1_l4" in memory_levels or "memory_full" in memory_levels,
                    l3="memory_l1_l3" in memory_levels or "memory_l1_l4" in memory_levels or "memory_full" in memory_levels,
                    l4="memory_l1_l4" in memory_levels or "memory_full" in memory_levels,
                    l5="memory_full" in memory_levels
                )
    
    def transition_to_phase(self, new_phase: CultivationPhase, reason: str = "", operator: str = "user") -> bool:
        if new_phase == self.current_phase:
            return False
        
        transition = PhaseTransition(
            from_phase=self.current_phase,
            to_phase=new_phase,
            timestamp=time.time(),
            reason=reason,
            operator=operator
        )
        self.phase_history.append(transition)
        
        self.current_phase = new_phase
        self._apply_phase_config()
        self._save_state()
        
        return True
    
    def get_phase_status(self) -> Dict[str, Any]:
        config = self.phase_configs[self.current_phase]
        stats = self.phase_stats.get(self.current_phase, CultivationStats(phase_start_time=time.time()))
        
        phase_duration = time.time() - stats.phase_start_time
        success_rate = stats.successful_interactions / stats.total_interactions if stats.total_interactions > 0 else 0.0
        
        return {
            "current_phase": self.current_phase.value,
            "current_phase_name": self._get_phase_name(self.current_phase),
            "description": config.description,
            "objectives": config.objectives,
            "acceptance_criteria": config.acceptance_criteria,
            "duration_days": config.duration_days,
            "elapsed_hours": round(phase_duration / 3600, 2),
            "enabled_features": config.enabled_features,
            "disabled_features": config.disabled_features,
            "stats": {
                "total_interactions": stats.total_interactions,
                "successful_interactions": stats.successful_interactions,
                "failed_interactions": stats.failed_interactions,
                "success_rate": round(success_rate, 4),
                "rule_matches": stats.rule_matches,
                "fallback_triggered": stats.fallback_triggered,
                "avg_response_time_ms": round(stats.avg_response_time_ms, 2),
                "compliance_rate": round(stats.compliance_rate, 4),
                "performance_score": round(stats.performance_score, 4)
            },
            "phase_history": [self._transition_to_dict(t) for t in self.phase_history]
        }
    
    def record_interaction(self, success: bool, rule_matched: bool = False, 
                          fallback: bool = False, response_time_ms: float = 0.0):
        if self.current_phase not in self.phase_stats:
            self.phase_stats[self.current_phase] = CultivationStats(phase_start_time=time.time())
        
        stats = self.phase_stats[self.current_phase]
        stats.total_interactions += 1
        
        if success:
            stats.successful_interactions += 1
        else:
            stats.failed_interactions += 1
        
        if rule_matched:
            stats.rule_matches += 1
        
        if fallback:
            stats.fallback_triggered += 1
        
        if stats.avg_response_time_ms == 0:
            stats.avg_response_time_ms = response_time_ms
        else:
            stats.avg_response_time_ms = (stats.avg_response_time_ms * (stats.total_interactions - 1) + response_time_ms) / stats.total_interactions
    
    def update_performance(self, compliance_rate: float, performance_score: float):
        if self.current_phase not in self.phase_stats:
            self.phase_stats[self.current_phase] = CultivationStats(phase_start_time=time.time())
        
        stats = self.phase_stats[self.current_phase]
        stats.compliance_rate = compliance_rate
        stats.performance_score = performance_score
    
    def _get_phase_name(self, phase: CultivationPhase) -> str:
        names = {
            CultivationPhase.FOUNDATION: "冷启动筑基期",
            CultivationPhase.GROWTH: "场景成长期",
            CultivationPhase.AUTONOMOUS: "自主进阶期"
        }
        return names.get(phase, "")
    
    def _transition_to_dict(self, transition: PhaseTransition) -> Dict[str, Any]:
        return {
            "from_phase": self._get_phase_name(transition.from_phase),
            "to_phase": self._get_phase_name(transition.to_phase),
            "timestamp": transition.timestamp,
            "time_str": datetime.fromtimestamp(transition.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "reason": transition.reason,
            "operator": transition.operator
        }
    
    def _save_state(self):
        os.makedirs(self.config_path, exist_ok=True)
        state = {
            "current_phase": self.current_phase.value,
            "phase_history": [self._transition_to_dict(t) for t in self.phase_history],
            "phase_stats": {
                phase.value: {
                    "phase_start_time": stats.phase_start_time,
                    "total_interactions": stats.total_interactions,
                    "successful_interactions": stats.successful_interactions,
                    "failed_interactions": stats.failed_interactions,
                    "rule_matches": stats.rule_matches,
                    "fallback_triggered": stats.fallback_triggered,
                    "avg_response_time_ms": stats.avg_response_time_ms,
                    "compliance_rate": stats.compliance_rate,
                    "performance_score": stats.performance_score
                }
                for phase, stats in self.phase_stats.items()
            }
        }
        
        with open(os.path.join(self.config_path, "cultivation_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def _load_state(self):
        state_path = os.path.join(self.config_path, "cultivation_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                
                self.current_phase = CultivationPhase(state.get("current_phase", "foundation"))
                
                for phase_str, stats_data in state.get("phase_stats", {}).items():
                    try:
                        phase = CultivationPhase(phase_str)
                        stats = CultivationStats(
                            phase_start_time=stats_data.get("phase_start_time", time.time()),
                            total_interactions=stats_data.get("total_interactions", 0),
                            successful_interactions=stats_data.get("successful_interactions", 0),
                            failed_interactions=stats_data.get("failed_interactions", 0),
                            rule_matches=stats_data.get("rule_matches", 0),
                            fallback_triggered=stats_data.get("fallback_triggered", 0),
                            avg_response_time_ms=stats_data.get("avg_response_time_ms", 0.0),
                            compliance_rate=stats_data.get("compliance_rate", 0.0),
                            performance_score=stats_data.get("performance_score", 0.0)
                        )
                        self.phase_stats[phase] = stats
                    except ValueError:
                        pass
            except Exception:
                pass
    
    def get_all_phases(self) -> List[Dict[str, Any]]:
        result = []
        for phase in CultivationPhase:
            config = self.phase_configs[phase]
            is_current = phase == self.current_phase
            stats = self.phase_stats.get(phase)
            
            result.append({
                "phase": phase.value,
                "name": self._get_phase_name(phase),
                "description": config.description,
                "is_current": is_current,
                "duration_days": config.duration_days,
                "objectives": config.objectives,
                "acceptance_criteria": config.acceptance_criteria
            })
        
        return result