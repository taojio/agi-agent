"""
quad_level_evolution.py - 四级迭代进化体系

实现从微观突触到宏观机制的四级闭环进化：
1. 微进化：突触级（实时生效，零风险）- SNN突触权重
2. 中进化：规则技能级（周期生效，低风险）- 规则库、技能库、知识图谱
3. 宏进化：架构级（手动授权，中风险）- 思考框架、行动流程、SNN网络结构
4. 元进化：机制级（严格授权，高风险）- 元学习算法、进化规则本身
"""
import numpy as np
import time
import uuid
from collections import deque
from enum import Enum
from typing import Dict, List, Any, Optional


class EvolutionTier(Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"
    META = "meta"


class EvolutionTrigger(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    STAGNATION = "stagnation"
    MANUAL = "manual"


class EvolutionRecord:
    def __init__(self, record_id, tier, trigger_type, content, source="system"):
        self.record_id = record_id
        self.tier = tier
        self.trigger_type = trigger_type
        self.content = content
        self.source = source
        self.created_at = time.time()
        self.applied = False
        self.effect = None

    def to_dict(self):
        return {
            "record_id": self.record_id,
            "tier": self.tier.value,
            "trigger_type": self.trigger_type.value,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "applied": self.applied,
            "effect": self.effect
        }


class SynapticUpdate:
    def __init__(self, connection_id, weight_delta, reason):
        self.connection_id = connection_id
        self.weight_delta = weight_delta
        self.reason = reason
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "connection_id": self.connection_id,
            "weight_delta": self.weight_delta,
            "reason": self.reason,
            "timestamp": self.timestamp
        }


class RuleUpdate:
    def __init__(self, rule_id, update_type, content, priority_change=0):
        self.rule_id = rule_id
        self.update_type = update_type
        self.content = content
        self.priority_change = priority_change
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "update_type": self.update_type,
            "content": self.content,
            "priority_change": self.priority_change,
            "timestamp": self.timestamp
        }


class ArchitectureMutation:
    def __init__(self, mutation_id, target_component, mutation_type, parameters):
        self.mutation_id = mutation_id
        self.target_component = target_component
        self.mutation_type = mutation_type
        self.parameters = parameters
        self.status = "proposed"
        self.approved = False
        self.applied_at = None
        self.timestamp = time.time()

    def approve(self):
        self.approved = True
        self.status = "approved"

    def apply(self):
        self.applied_at = time.time()
        self.status = "applied"

    def to_dict(self):
        return {
            "mutation_id": self.mutation_id,
            "target_component": self.target_component,
            "mutation_type": self.mutation_type,
            "parameters": self.parameters,
            "status": self.status,
            "approved": self.approved,
            "applied_at": self.applied_at,
            "timestamp": self.timestamp
        }


class MetaEvolutionConfig:
    def __init__(self):
        self.trigger_thresholds = {
            EvolutionTier.MICRO: 0.05,
            EvolutionTier.MESO: 0.15,
            EvolutionTier.MACRO: 0.30,
            EvolutionTier.META: 0.50
        }
        self.review_periods = {
            EvolutionTier.MICRO: 1,
            EvolutionTier.MESO: 50,
            EvolutionTier.MACRO: 500,
            EvolutionTier.META: 2000
        }
        self.validation_standards = {
            EvolutionTier.MICRO: 0.6,
            EvolutionTier.MESO: 0.75,
            EvolutionTier.MACRO: 0.85,
            EvolutionTier.META: 0.95
        }
        self.enabled_tiers = {
            EvolutionTier.MICRO: True,
            EvolutionTier.MESO: True,
            EvolutionTier.MACRO: False,
            EvolutionTier.META: False
        }

    def to_dict(self):
        return {
            "trigger_thresholds": {k.value: v for k, v in self.trigger_thresholds.items()},
            "review_periods": {k.value: v for k, v in self.review_periods.items()},
            "validation_standards": {k.value: v for k, v in self.validation_standards.items()},
            "enabled_tiers": {k.value: v for k, v in self.enabled_tiers.items()}
        }


class QuadLevelEvolution:
    def __init__(self):
        self.config = MetaEvolutionConfig()

        self.micro_records = deque(maxlen=500)
        self.meso_records = deque(maxlen=200)
        self.macro_records = deque(maxlen=100)
        self.meta_records = deque(maxlen=50)

        self.synaptic_updates = deque(maxlen=1000)
        self.rule_updates = deque(maxlen=200)
        self.architecture_mutations = {}
        self.meta_config_history = deque(maxlen=50)

        self.cycle_counter = 0
        self.stagnation_counter = 0
        self.last_performance_score = 0.5

        self.snn_interface = None
        self.rule_engine_interface = None
        self.knowledge_graph_interface = None

    def set_interfaces(self, snn=None, rule_engine=None, knowledge_graph=None):
        self.snn_interface = snn
        self.rule_engine_interface = rule_engine
        self.knowledge_graph_interface = knowledge_graph

    def enable_tier(self, tier, enabled=True):
        self.config.enabled_tiers[tier] = enabled

    def is_tier_enabled(self, tier):
        return self.config.enabled_tiers.get(tier, False)

    def process_micro_evolution(self, action_result, confidence):
        if not self.is_tier_enabled(EvolutionTier.MICRO):
            return {"tier": "micro", "status": "disabled"}

        success = action_result.get("success", False)
        weight_delta = 0.01 if success else -0.015

        updates = []
        for i in range(10):
            connection_id = f"synapse_{i}"
            update = SynapticUpdate(
                connection_id=connection_id,
                weight_delta=weight_delta * (0.8 + np.random.uniform(0, 0.4)),
                reason=f"{'reinforcement' if success else 'attenuation'} based on action result"
            )
            updates.append(update)
            self.synaptic_updates.append(update)

        if self.snn_interface and hasattr(self.snn_interface, 'update_weights'):
            updates_dict = {update.connection_id: update.weight_delta for update in updates}
            self.snn_interface.update_weights(updates_dict)

        record = EvolutionRecord(
            record_id=f"micro_{uuid.uuid4().hex[:8]}",
            tier=EvolutionTier.MICRO,
            trigger_type=EvolutionTrigger.SUCCESS if success else EvolutionTrigger.FAILURE,
            content=f"Synaptic update: {len(updates)} connections adjusted, delta={weight_delta:.4f}",
            source="micro_evolution"
        )
        record.applied = True
        record.effect = {"weight_adjustments": len(updates), "direction": "positive" if success else "negative"}
        self.micro_records.append(record)

        return {
            "tier": "micro",
            "status": "applied",
            "updates": len(updates),
            "direction": "positive" if success else "negative"
        }

    def process_meso_evolution(self, execution_history, cycle_count):
        if not self.is_tier_enabled(EvolutionTier.MESO):
            return {"tier": "meso", "status": "disabled"}

        if cycle_count % self.config.review_periods[EvolutionTier.MESO] != 0:
            return {"tier": "meso", "status": "waiting", "next_review": self.config.review_periods[EvolutionTier.MESO] - (cycle_count % self.config.review_periods[EvolutionTier.MESO])}

        rule_updates_made = 0
        kg_updates_made = 0

        action_frequency = {}
        for record in execution_history[-100:]:
            action_name = record.get("node_name", "")
            if action_name:
                action_frequency[action_name] = action_frequency.get(action_name, 0) + 1

        for action_name, count in action_frequency.items():
            if count >= 5:
                rule_update = RuleUpdate(
                    rule_id=f"rule_{action_name}",
                    update_type="priority_increase",
                    content=f"Automatic priority increase for frequent action: {action_name}",
                    priority_change=0.1
                )
                self.rule_updates.append(rule_update)
                rule_updates_made += 1

                if self.rule_engine_interface and hasattr(self.rule_engine_interface, 'update_rule_priority'):
                    self.rule_engine_interface.update_rule_priority(action_name, 0.1)

        success_rates = {}
        for record in execution_history[-50:]:
            action_name = record.get("node_name", "")
            if action_name:
                if action_name not in success_rates:
                    success_rates[action_name] = {"success": 0, "total": 0}
                success_rates[action_name]["total"] += 1
                if record.get("status") == "completed":
                    success_rates[action_name]["success"] += 1

        for action_name, stats in success_rates.items():
            rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.5
            if rate < 0.3:
                rule_update = RuleUpdate(
                    rule_id=f"rule_{action_name}",
                    update_type="flag_for_review",
                    content=f"Low success rate ({rate:.2f}) detected for action: {action_name}",
                    priority_change=-0.2
                )
                self.rule_updates.append(rule_update)
                rule_updates_made += 1

        if self.knowledge_graph_interface and hasattr(self.knowledge_graph_interface, 'update_node'):
            for action_name, stats in success_rates.items():
                rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.5
                self.knowledge_graph_interface.update_node(action_name, {"success_rate": rate})
                kg_updates_made += 1

        record = EvolutionRecord(
            record_id=f"meso_{uuid.uuid4().hex[:8]}",
            tier=EvolutionTier.MESO,
            trigger_type=EvolutionTrigger.MANUAL,
            content=f"Meso evolution: {rule_updates_made} rule updates, {kg_updates_made} KG updates",
            source="meso_evolution"
        )
        record.applied = True
        record.effect = {"rule_updates": rule_updates_made, "kg_updates": kg_updates_made}
        self.meso_records.append(record)

        return {
            "tier": "meso",
            "status": "applied",
            "rule_updates": rule_updates_made,
            "knowledge_graph_updates": kg_updates_made
        }

    def propose_macro_evolution(self, analysis_data):
        if not self.is_tier_enabled(EvolutionTier.MACRO):
            return {"tier": "macro", "status": "disabled"}

        mutation_id = f"macro_{uuid.uuid4().hex[:8]}"

        performance_trend = analysis_data.get("performance_trend", 0)
        if performance_trend < -0.1:
            mutation_type = "structure_optimization"
        elif analysis_data.get("novelty_rate", 0) > 0.6:
            mutation_type = "expansion"
        else:
            mutation_type = "refinement"

        mutation = ArchitectureMutation(
            mutation_id=mutation_id,
            target_component=analysis_data.get("target", "thinking_framework"),
            mutation_type=mutation_type,
            parameters=analysis_data.get("parameters", {})
        )
        self.architecture_mutations[mutation_id] = mutation

        record = EvolutionRecord(
            record_id=f"macro_{uuid.uuid4().hex[:8]}",
            tier=EvolutionTier.MACRO,
            trigger_type=EvolutionTrigger.STAGNATION,
            content=f"Macro mutation proposed: {mutation_type} on {mutation.target_component}",
            source="macro_evolution"
        )
        self.macro_records.append(record)

        return {
            "tier": "macro",
            "status": "proposed",
            "mutation_id": mutation_id,
            "mutation_type": mutation_type,
            "target": mutation.target_component
        }

    def approve_macro_mutation(self, mutation_id):
        if mutation_id in self.architecture_mutations:
            mutation = self.architecture_mutations[mutation_id]
            mutation.approve()
            mutation.apply()

            record = EvolutionRecord(
                record_id=f"macro_applied_{uuid.uuid4().hex[:8]}",
                tier=EvolutionTier.MACRO,
                trigger_type=EvolutionTrigger.MANUAL,
                content=f"Macro mutation applied: {mutation.mutation_type}",
                source="macro_evolution"
            )
            record.applied = True
            self.macro_records.append(record)

            return {"mutation_id": mutation_id, "status": "approved"}
        return {"mutation_id": mutation_id, "status": "not_found"}

    def process_meta_evolution(self, long_term_performance):
        if not self.is_tier_enabled(EvolutionTier.META):
            return {"tier": "meta", "status": "disabled"}

        avg_performance = np.mean(long_term_performance) if long_term_performance else 0.5

        if avg_performance > self.config.validation_standards[EvolutionTier.META]:
            self.config.trigger_thresholds[EvolutionTier.MICRO] = max(0.02, self.config.trigger_thresholds[EvolutionTier.MICRO] - 0.01)
            self.config.review_periods[EvolutionTier.MESO] = max(20, self.config.review_periods[EvolutionTier.MESO] - 10)
        elif avg_performance < 0.5:
            self.config.trigger_thresholds[EvolutionTier.MICRO] = min(0.1, self.config.trigger_thresholds[EvolutionTier.MICRO] + 0.02)
            self.config.review_periods[EvolutionTier.MESO] = min(100, self.config.review_periods[EvolutionTier.MESO] + 20)

        self.meta_config_history.append({
            "timestamp": time.time(),
            "avg_performance": avg_performance,
            "config_snapshot": self.config.to_dict()
        })

        record = EvolutionRecord(
            record_id=f"meta_{uuid.uuid4().hex[:8]}",
            tier=EvolutionTier.META,
            trigger_type=EvolutionTrigger.STAGNATION,
            content=f"Meta evolution: config adjusted based on avg performance {avg_performance:.4f}",
            source="meta_evolution"
        )
        record.applied = True
        record.effect = {"avg_performance": avg_performance}
        self.meta_records.append(record)

        return {
            "tier": "meta",
            "status": "applied",
            "avg_performance": avg_performance,
            "config_adjusted": True
        }

    def run_evolution_cycle(self, action_result, execution_history, long_term_performance=None):
        self.cycle_counter += 1

        results = {}

        results["micro"] = self.process_micro_evolution(action_result, action_result.get("confidence", 0.5))

        results["meso"] = self.process_meso_evolution(execution_history, self.cycle_counter)

        if self.cycle_counter % self.config.review_periods[EvolutionTier.MACRO] == 0:
            analysis_data = {
                "performance_trend": self._calculate_trend(long_term_performance),
                "novelty_rate": np.random.uniform(0, 1),
                "target": "thinking_framework"
            }
            results["macro"] = self.propose_macro_evolution(analysis_data)
        else:
            results["macro"] = {"tier": "macro", "status": "waiting"}

        if self.cycle_counter % self.config.review_periods[EvolutionTier.META] == 0:
            results["meta"] = self.process_meta_evolution(long_term_performance or [])
        else:
            results["meta"] = {"tier": "meta", "status": "waiting"}

        current_performance = action_result.get("confidence", 0.5)
        if abs(current_performance - self.last_performance_score) < 0.02:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_performance_score = current_performance

        return {
            "cycle": self.cycle_counter,
            "results": results,
            "stagnation_counter": self.stagnation_counter,
            "last_performance": current_performance
        }

    def _calculate_trend(self, performance_data):
        if not performance_data or len(performance_data) < 10:
            return 0.0
        recent = performance_data[-10:]
        return np.polyfit(range(len(recent)), recent, 1)[0]

    def get_macro_proposals(self):
        return [m for m in self.architecture_mutations.values() if m.status == "proposed"]

    def get_all_stats(self):
        return {
            "cycle": self.cycle_counter,
            "config": self.config.to_dict(),
            "micro": {
                "records": len(self.micro_records),
                "synaptic_updates": len(self.synaptic_updates)
            },
            "meso": {
                "records": len(self.meso_records),
                "rule_updates": len(self.rule_updates)
            },
            "macro": {
                "records": len(self.macro_records),
                "mutations": {
                    "proposed": len([m for m in self.architecture_mutations.values() if m.status == "proposed"]),
                    "approved": len([m for m in self.architecture_mutations.values() if m.status == "approved"]),
                    "applied": len([m for m in self.architecture_mutations.values() if m.status == "applied"])
                }
            },
            "meta": {
                "records": len(self.meta_records),
                "config_changes": len(self.meta_config_history)
            },
            "stagnation_counter": self.stagnation_counter,
            "last_performance": self.last_performance_score
        }