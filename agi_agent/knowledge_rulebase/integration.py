"""
跨学科知识强化规则体系 - 集成模块

提供规则体系与其他系统模块的集成接口：
- 知识图谱集成
- 语义搜索集成
- 自动化联动引擎集成
- 决策引擎集成
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger("agi_agent.knowledge_rulebase.integration")


class RuleIntegrationManager:
    """规则体系集成管理器"""

    def __init__(self, rule_registry=None, knowledge_graph=None, 
                 semantic_search=None, automation_engine=None,
                 logical_deductor=None):
        self._rule_registry = rule_registry
        self._knowledge_graph = knowledge_graph
        self._semantic_search = semantic_search
        self._automation_engine = automation_engine
        self._logical_deductor = logical_deductor

    def set_components(self, **components):
        """设置集成组件"""
        if "rule_registry" in components:
            self._rule_registry = components["rule_registry"]
        if "knowledge_graph" in components:
            self._knowledge_graph = components["knowledge_graph"]
        if "semantic_search" in components:
            self._semantic_search = components["semantic_search"]
        if "automation_engine" in components:
            self._automation_engine = components["automation_engine"]
        if "logical_deductor" in components:
            self._logical_deductor = components["logical_deductor"]

    def sync_rules_to_knowledge_graph(self) -> int:
        """同步规则到知识图谱"""
        if self._rule_registry is None or self._knowledge_graph is None:
            logger.warning("Rule registry or knowledge graph not set")
            return 0

        return self._rule_registry.export_rules_to_knowledge_graph(self._knowledge_graph)

    def search_rules_semantically(self, query: str, scene_tags: List[str] = None) -> List[Dict[str, Any]]:
        """语义搜索规则"""
        results = []
        
        if self._semantic_search and self._rule_registry:
            normalized_query = self._semantic_search.normalize(query)
            
            by_description = self._rule_registry.search_by_description(query)
            by_concept = self._rule_registry.search_by_concept(query)
            by_formula = self._rule_registry.search_by_formula(query)
            
            all_results = []
            seen = set()
            
            for rule in by_description:
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    all_results.append({"rule": rule, "score": 0.8, "match_type": "description"})
            
            for rule in by_concept:
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    all_results.append({"rule": rule, "score": 0.6, "match_type": "concept"})
            
            for rule in by_formula:
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    all_results.append({"rule": rule, "score": 0.7, "match_type": "formula"})
            
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            for item in all_results[:10]:
                rule = item["rule"]
                results.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "formula": rule.formula,
                    "description": rule.description,
                    "discipline": rule.discipline.value,
                    "rule_type": rule.rule_type.value,
                    "difficulty": rule.difficulty.value,
                    "confidence": rule.confidence,
                    "match_type": item["match_type"],
                    "score": item["score"],
                })

        return results

    def register_knowledge_trigger_rules(self):
        """注册知识触发联动规则"""
        if self._automation_engine is None or self._rule_registry is None:
            logger.warning("Automation engine or rule registry not set")
            return

        from agi_agent.orchestration.automation_linkage import TriggerPriority, LinkageRule, LinkageRuleType

        rule_high_confidence = LinkageRule(
            rule_id="knowledge_high_confidence",
            name="高置信度规则触发",
            description="当规则匹配置信度超过阈值时，自动触发深度推理",
            condition=lambda state: state.confidence > 0.8,
            actions=[self._action_trigger_deep_reasoning],
            priority=TriggerPriority.HIGH,
            cooldown_steps=30,
            rule_type=LinkageRuleType.THRESHOLD
        )

        rule_knowledge_gap = LinkageRule(
            rule_id="knowledge_gap_detection",
            name="知识缺口检测",
            description="当推理过程中发现缺少关键规则时，触发知识学习",
            condition=lambda state: state.novelty > 0.7,
            actions=[self._action_identify_knowledge_gap],
            priority=TriggerPriority.HIGH,
            cooldown_steps=50,
            rule_type=LinkageRuleType.THRESHOLD
        )

        rule_rule_conflict = LinkageRule(
            rule_id="rule_conflict_detection",
            name="规则冲突检测",
            description="当发现规则间存在冲突时，触发规则验证流程",
            condition=lambda state: state.entropy > 0.6,
            actions=[self._action_resolve_rule_conflict],
            priority=TriggerPriority.CRITICAL,
            cooldown_steps=100,
            rule_type=LinkageRuleType.PATTERN
        )

        self._automation_engine.register_rule(rule_high_confidence)
        self._automation_engine.register_rule(rule_knowledge_gap)
        self._automation_engine.register_rule(rule_rule_conflict)

        logger.info("Registered knowledge trigger rules to automation engine")

    def _action_trigger_deep_reasoning(self, agent, state):
        """触发深度推理"""
        if self._logical_deductor is None:
            return {"status": "failed", "reason": "Logical deductor not set"}

        results = self._logical_deductor.get_deduction_stats()
        return {
            "status": "success",
            "action": "deep_reasoning",
            "deduction_stats": results,
            "step": state.step
        }

    def _action_identify_knowledge_gap(self, agent, state):
        """识别知识缺口"""
        if self._rule_registry is None:
            return {"status": "failed", "reason": "Rule registry not set"}

        stats = self._rule_registry.get_stats()
        return {
            "status": "success",
            "action": "knowledge_gap_identification",
            "knowledge_stats": stats,
            "novelty": state.novelty,
            "step": state.step
        }

    def _action_resolve_rule_conflict(self, agent, state):
        """解决规则冲突"""
        if self._rule_registry is None:
            return {"status": "failed", "reason": "Rule registry not set"}

        return {
            "status": "success",
            "action": "rule_conflict_resolution",
            "entropy": state.entropy,
            "step": state.step
        }

    def query_with_rules(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """使用规则体系回答问题"""
        result = {
            "question": question,
            "context": context or {},
            "matched_rules": [],
            "reasoning_steps": [],
            "final_answer": None,
            "confidence": 0.0,
            "sources": []
        }

        if self._rule_registry is None:
            result["error"] = "Rule registry not available"
            return result

        matched_rules = self.search_rules_semantically(question)
        result["matched_rules"] = matched_rules

        if matched_rules and self._logical_deductor:
            facts = context.copy()
            for rule in matched_rules[:5]:
                rule_obj = self._rule_registry.get_rule(rule["rule_id"])
                if rule_obj:
                    for var in rule_obj.variables:
                        if var.symbol not in facts and var.default_value is not None:
                            facts[var.symbol] = var.default_value

            problem = {
                "question": question,
                "facts": facts,
                "goal": question
            }

            reasoning_result = self._logical_deductor.solve_complex_problem(problem)
            result["reasoning_steps"] = reasoning_result["solution_path"]
            result["final_answer"] = reasoning_result["final_answer"]
            result["confidence"] = reasoning_result["confidence"]

        else:
            if matched_rules:
                rule = matched_rules[0]
                result["final_answer"] = f"根据{rule['name']}，{rule['description']}"
                result["confidence"] = rule["confidence"]

        for rule in matched_rules[:3]:
            result["sources"].append({
                "rule_id": rule["rule_id"],
                "rule_name": rule["name"],
                "formula": rule["formula"]
            })

        return result

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识规则体系摘要"""
        summary = {
            "rule_registry": {},
            "knowledge_graph": {},
            "semantic_search": {},
            "logical_deductor": {},
            "automation_engine": {}
        }

        if self._rule_registry:
            summary["rule_registry"] = self._rule_registry.get_stats()

        if self._knowledge_graph:
            summary["knowledge_graph"] = {
                "nodes": len(self._knowledge_graph._nodes),
                "edges": len(self._knowledge_graph._edges),
                "entity_types": len(self._knowledge_graph._nodes_by_type)
            }

        if self._logical_deductor:
            summary["logical_deductor"] = self._logical_deductor.get_deduction_stats()

        if self._automation_engine:
            summary["automation_engine"] = {
                "rules_count": len(self._automation_engine._rules),
                "total_triggered": self._automation_engine._stats["total_triggered"]
            }

        return summary

    def add_rule_from_knowledge(self, knowledge_entry: Dict[str, Any]) -> bool:
        """从知识条目创建规则"""
        if self._rule_registry is None:
            return False

        from .disciplinary_rule import DisciplinaryRule, Discipline, RuleType, RuleDifficulty, RuleVariable

        discipline = Discipline(knowledge_entry.get("discipline", "PHYSICS"))
        rule_type = RuleType(knowledge_entry.get("rule_type", "FORMULA"))
        difficulty = RuleDifficulty(knowledge_entry.get("difficulty", "INTERMEDIATE"))

        variables = []
        for var_data in knowledge_entry.get("variables", []):
            variables.append(RuleVariable(
                name=var_data.get("name", ""),
                symbol=var_data.get("symbol", ""),
                unit=var_data.get("unit", ""),
                description=var_data.get("description", "")
            ))

        rule = DisciplinaryRule(
            rule_id=knowledge_entry.get("rule_id", ""),
            discipline=discipline,
            rule_type=rule_type,
            name=knowledge_entry.get("name", ""),
            formula=knowledge_entry.get("formula", ""),
            description=knowledge_entry.get("description", ""),
            variables=variables,
            conditions=knowledge_entry.get("conditions", []),
            units=knowledge_entry.get("units", {}),
            prerequisite_rules=knowledge_entry.get("prerequisite_rules", []),
            dependent_rules=knowledge_entry.get("dependent_rules", []),
            related_concepts=knowledge_entry.get("related_concepts", []),
            difficulty=difficulty,
            confidence=knowledge_entry.get("confidence", 0.9),
            real_world_examples=knowledge_entry.get("real_world_examples", []),
        )

        return self._rule_registry.register_rule(rule)
