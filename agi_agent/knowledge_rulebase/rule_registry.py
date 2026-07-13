"""
跨学科知识强化规则体系 - 规则注册中心

提供插件式规则注册机制，支持：
- 多学科规则注册与管理
- 规则间依赖关系分析
- 规则检索与匹配
- 动态扩展新学科
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .disciplinary_rule import DisciplinaryRule, Discipline, RuleType, RuleDifficulty

logger = logging.getLogger("agi_agent.knowledge_rulebase")


class DisciplinaryRuleRegistry:
    """规则注册中心"""

    def __init__(self):
        self._rules: Dict[str, DisciplinaryRule] = {}
        self._rules_by_discipline: Dict[str, List[DisciplinaryRule]] = defaultdict(list)
        self._rules_by_type: Dict[str, List[DisciplinaryRule]] = defaultdict(list)
        self._rules_by_difficulty: Dict[int, List[DisciplinaryRule]] = defaultdict(list)
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._reverse_dependency: Dict[str, List[str]] = defaultdict(list)
        self._concept_index: Dict[str, List[str]] = defaultdict(list)
        self._formula_index: Dict[str, List[str]] = defaultdict(list)
        self._registered_disciplines: Dict[str, Any] = {}

    def register_rule(self, rule: DisciplinaryRule) -> bool:
        """注册规则"""
        if rule.rule_id in self._rules:
            logger.warning(f"Rule {rule.rule_id} already exists, overwriting")

        self._rules[rule.rule_id] = rule
        self._rules_by_discipline[rule.discipline.value].append(rule)
        self._rules_by_type[rule.rule_type.value].append(rule)
        self._rules_by_difficulty[rule.difficulty.value].append(rule)

        for prereq in rule.prerequisite_rules:
            self._dependency_graph[rule.rule_id].append(prereq)
            self._reverse_dependency[prereq].append(rule.rule_id)

        for concept in rule.related_concepts:
            self._concept_index[concept.lower()].append(rule.rule_id)

        normalized_formula = rule.normalize_formula()
        for token in normalized_formula.split():
            if len(token) >= 2:
                self._formula_index[token.lower()].append(rule.rule_id)

        logger.info(f"Registered rule: [{rule.discipline.value}] {rule.name} ({rule.rule_id})")
        return True

    def register_rules(self, rules: List[DisciplinaryRule]) -> int:
        """批量注册规则"""
        count = 0
        for rule in rules:
            if self.register_rule(rule):
                count += 1
        logger.info(f"Registered {count} rules")
        return count

    def register_discipline(self, discipline: Discipline, config: Dict[str, Any] = None):
        """注册学科配置"""
        self._registered_disciplines[discipline.value] = config or {}

    def get_rule(self, rule_id: str) -> Optional[DisciplinaryRule]:
        """获取规则"""
        return self._rules.get(rule_id)

    def get_rules_by_discipline(self, discipline: Discipline) -> List[DisciplinaryRule]:
        """按学科获取规则"""
        return self._rules_by_discipline.get(discipline.value, [])

    def get_rules_by_type(self, rule_type: RuleType) -> List[DisciplinaryRule]:
        """按类型获取规则"""
        return self._rules_by_type.get(rule_type.value, [])

    def get_rules_by_difficulty(self, difficulty: RuleDifficulty) -> List[DisciplinaryRule]:
        """按难度获取规则"""
        return self._rules_by_difficulty.get(difficulty.value, [])

    def search_by_concept(self, concept: str) -> List[DisciplinaryRule]:
        """按概念搜索规则"""
        rule_ids = self._concept_index.get(concept.lower(), [])
        return [self._rules[rid] for rid in rule_ids if rid in self._rules]

    def search_by_formula(self, formula_query: str) -> List[DisciplinaryRule]:
        """按公式搜索规则"""
        from agi_agent.memory.semantic_search import TextNormalizer
        normalizer = TextNormalizer()
        normalized = normalizer.normalize(formula_query).lower()

        matched = set()
        for token in normalized.split():
            if len(token) >= 2:
                matched.update(self._formula_index.get(token, []))

        return [self._rules[rid] for rid in matched if rid in self._rules]

    def search_by_description(self, query: str) -> List[DisciplinaryRule]:
        """按描述搜索规则"""
        query_lower = query.lower()
        results = []
        for rule in self._rules.values():
            if query_lower in rule.description.lower() or query_lower in rule.name.lower():
                results.append(rule)
        return results

    def get_dependencies(self, rule_id: str) -> List[DisciplinaryRule]:
        """获取规则的依赖规则"""
        dep_ids = self._dependency_graph.get(rule_id, [])
        return [self._rules[rid] for rid in dep_ids if rid in self._rules]

    def get_dependents(self, rule_id: str) -> List[DisciplinaryRule]:
        """获取依赖于某规则的规则"""
        dep_ids = self._reverse_dependency.get(rule_id, [])
        return [self._rules[rid] for rid in dep_ids if rid in self._rules]

    def get_rule_hierarchy(self, rule_id: str, depth: int = 3) -> Dict[str, Any]:
        """获取规则的层级依赖结构"""
        rule = self._rules.get(rule_id)
        if not rule:
            return {}

        result = {
            "rule_id": rule_id,
            "name": rule.name,
            "formula": rule.formula,
            "prerequisites": [],
        }

        if depth > 0:
            for prereq_id in rule.prerequisite_rules:
                prereq_rule = self._rules.get(prereq_id)
                if prereq_rule:
                    result["prerequisites"].append(
                        self.get_rule_hierarchy(prereq_id, depth - 1)
                    )

        return result

    def get_all_disciplines(self) -> List[str]:
        """获取所有已注册学科"""
        return list(self._rules_by_discipline.keys())

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_rules": len(self._rules),
            "disciplines": {
                disc: len(rules) for disc, rules in self._rules_by_discipline.items()
            },
            "rule_types": {
                rt: len(rules) for rt, rules in self._rules_by_type.items()
            },
            "difficulty_distribution": {
                str(d): len(rules) for d, rules in self._rules_by_difficulty.items()
            },
            "dependency_count": len(self._dependency_graph),
        }

    def infer_from_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从上下文推断适用规则并应用"""
        applicable_rules = []
        for rule in self._rules.values():
            if rule.is_applicable(context):
                applicable_rules.append(rule)

        results = []
        for rule in applicable_rules:
            inputs = {}
            for var in rule.variables:
                if var.symbol in context:
                    inputs[var.symbol] = context[var.symbol]
            if inputs:
                result = rule.apply(inputs)
                results.append(result)

        return results

    def chain_reasoning(self, start_rule_id: str, max_steps: int = 5) -> List[Dict[str, Any]]:
        """链式推理：从起始规则出发推导相关规则"""
        visited = set()
        chain = []
        queue = [(start_rule_id, 0)]

        while queue and len(chain) < max_steps:
            rule_id, step = queue.pop(0)
            if rule_id in visited:
                continue

            rule = self._rules.get(rule_id)
            if not rule:
                continue

            visited.add(rule_id)
            chain.append({
                "step": step,
                "rule_id": rule_id,
                "rule_name": rule.name,
                "formula": rule.formula,
                "description": rule.description,
                "dependencies": rule.prerequisite_rules,
                "dependents": rule.dependent_rules,
            })

            for dep_id in rule.dependent_rules:
                if dep_id not in visited:
                    queue.append((dep_id, step + 1))

        return chain

    def export_rules_to_knowledge_graph(self, kg) -> int:
        """导出规则到知识图谱"""
        from agi_agent.learning.enhanced_knowledge_graph import EntityType, RelationType

        count = 0
        for rule in self._rules.values():
            node_id = f"rule_{rule.rule_id}"
            kg.add_node(
                label=rule.name,
                entity_type=EntityType.RULE,
                description=rule.description,
                properties={
                    "rule_id": rule.rule_id,
                    "discipline": rule.discipline.value,
                    "rule_type": rule.rule_type.value,
                    "formula": rule.formula,
                    "difficulty": rule.difficulty.value,
                    "variables": [v.symbol for v in rule.variables],
                }
            )

            for prereq_id in rule.prerequisite_rules:
                prereq_node_id = f"rule_{prereq_id}"
                kg.add_edge(
                    source=node_id,
                    target=prereq_node_id,
                    relation_type=RelationType.DEPENDS_ON,
                    weight=0.8,
                    description=f"{rule.name} depends on {prereq_id}"
                )

            for concept in rule.related_concepts:
                concept_node_id = f"concept_{concept.lower().replace(' ', '_')}"
                kg.add_edge(
                    source=node_id,
                    target=concept_node_id,
                    relation_type=RelationType.RELATED_TO,
                    weight=0.5,
                    description=f"{rule.name} relates to {concept}"
                )

            count += 1

        logger.info(f"Exported {count} rules to knowledge graph")
        return count


def register_default_disciplines(registry: DisciplinaryRuleRegistry) -> int:
    """注册默认学科规则"""
    from .physics_rules import PhysicsRules
    from .math_rules import MathRules
    from .chemistry_rules import ChemistryRules
    from .biology_rules import BiologyRules
    from .chinese_rules import ChineseRules

    total_count = 0

    physics_rules = PhysicsRules()
    count = registry.register_rules(physics_rules.get_all_rules())
    registry.register_discipline(Discipline.PHYSICS, {"base_rules": len(physics_rules.get_all_rules())})
    total_count += count

    math_rules = MathRules()
    count = registry.register_rules(math_rules.get_all_rules())
    registry.register_discipline(Discipline.MATHEMATICS, {"base_rules": len(math_rules.get_all_rules())})
    total_count += count

    chemistry_rules = ChemistryRules()
    count = registry.register_rules(chemistry_rules.get_all_rules())
    registry.register_discipline(Discipline.CHEMISTRY, {"base_rules": len(chemistry_rules.get_all_rules())})
    total_count += count

    biology_rules = BiologyRules()
    count = registry.register_rules(biology_rules.get_all_rules())
    registry.register_discipline(Discipline.BIOLOGY, {"base_rules": len(biology_rules.get_all_rules())})
    total_count += count

    chinese_rules = ChineseRules()
    count = registry.register_rules(chinese_rules.get_all_rules())
    registry.register_discipline(Discipline.CHINESE, {"base_rules": len(chinese_rules.get_all_rules())})
    total_count += count

    return total_count
