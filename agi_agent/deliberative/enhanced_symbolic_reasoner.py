"""
enhanced_symbolic_reasoner.py - 增强符号推理器

扩展现有神经符号推理器，添加：
1. 一阶谓词逻辑表示与推理规则库
2. 归纳推理算法
3. 类比推理映射机制
4. 符号推理与自然语言处理的接口转换
"""
import time
import uuid
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple, Set

from .neuro_symbolic_reasoner import (
    NeuroSymbolicReasoner, SymbolicExpression, NeuralSymbol, SymbolType, InferenceRule
)
from ..meta_orchestration.data_contract import ReasoningTrace, ReasoningStep, ProblemFeature


class PredicateType(Enum):
    """谓词类型"""
    ATOMIC = "atomic"
    COMPOUND = "compound"
    RELATIONAL = "relational"
    QUANTIFIED = "quantified"


class Quantifier(Enum):
    """量词"""
    UNIVERSAL = "forall"
    EXISTENTIAL = "exists"
    UNIQUE = "exists_unique"


class LogicalOperator(Enum):
    """逻辑运算符"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    EQUIVALENT = "EQUIVALENT"


class RelationType(Enum):
    """关系类型"""
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    PART_OF = "part_of"
    CAUSES = "causes"
    FOLLOWS = "follows"
    SIMILAR_TO = "similar_to"
    ANALOGOUS_TO = "analogous_to"


class Predicate:
    """一阶谓词逻辑谓词"""

    def __init__(self, name: str, arguments: List[str], predicate_type: PredicateType = PredicateType.ATOMIC,
                 quantifier: Optional[Quantifier] = None):
        self.name = name
        self.arguments = arguments
        self.predicate_type = predicate_type
        self.quantifier = quantifier
        self.confidence = 0.5
        self.truth_value = None

    def evaluate(self, facts: Dict[str, Any]) -> float:
        """评估谓词真值"""
        if self.predicate_type == PredicateType.ATOMIC:
            return self._evaluate_atomic(facts)
        elif self.predicate_type == PredicateType.COMPOUND:
            return self._evaluate_compound(facts)
        elif self.predicate_type == PredicateType.RELATIONAL:
            return self._evaluate_relational(facts)
        elif self.predicate_type == PredicateType.QUANTIFIED:
            return self._evaluate_quantified(facts)
        return 0.5

    def _evaluate_atomic(self, facts: Dict[str, Any]) -> float:
        key = f"{self.name}({', '.join(self.arguments)})"
        if key in facts:
            return float(facts[key])
        for arg in self.arguments:
            if arg in facts:
                return 0.8
        return 0.3

    def _evaluate_compound(self, facts: Dict[str, Any]) -> float:
        values = []
        for arg in self.arguments:
            if arg in facts:
                values.append(float(facts[arg]))
            else:
                values.append(0.5)
        if values:
            return sum(values) / len(values)
        return 0.5

    def _evaluate_relational(self, facts: Dict[str, Any]) -> float:
        if len(self.arguments) != 2:
            return 0.5
        arg1, arg2 = self.arguments
        val1 = facts.get(arg1, 0)
        val2 = facts.get(arg2, 0)
        try:
            val1 = float(val1)
            val2 = float(val2)
            if val1 == val2:
                return 1.0
            return max(0.0, 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1))
        except (ValueError, TypeError):
            return 0.5

    def _evaluate_quantified(self, facts: Dict[str, Any]) -> float:
        if not self.quantifier:
            return 0.5
        matches = sum(1 for key in facts if any(arg in key for arg in self.arguments))
        total = len(facts) if facts else 1
        ratio = matches / total
        if self.quantifier == Quantifier.UNIVERSAL:
            return ratio
        elif self.quantifier == Quantifier.EXISTENTIAL:
            return 1.0 if matches > 0 else 0.0
        elif self.quantifier == Quantifier.UNIQUE:
            return 1.0 if matches == 1 else 0.0
        return ratio

    def to_string(self) -> str:
        """转换为字符串表示"""
        args_str = ", ".join(self.arguments)
        if self.quantifier:
            return f"{self.quantifier.value} {args_str}: {self.name}({args_str})"
        return f"{self.name}({args_str})"

    def __repr__(self):
        return self.to_string()


class EnhancedSymbolicReasoner:
    """增强符号推理器

    扩展能力：
    1. 一阶谓词逻辑表示与推理规则库
    2. 归纳推理算法
    3. 类比推理映射机制
    4. 符号推理与自然语言处理的接口转换
    """

    def __init__(self, symbol_dim: int = 64, hidden_dim: int = 128):
        self._base_reasoner = NeuroSymbolicReasoner(symbol_dim, hidden_dim)
        self._predicates: Dict[str, Predicate] = {}
        self._inference_rules: Dict[str, 'SymbolicRule'] = {}
        self._analogy_mappings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._inductive_patterns: List[Dict[str, Any]] = []
        self._reasoning_history = deque(maxlen=100)

        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认推理规则"""
        self.add_rule("modus_ponens", [Predicate("P", ["x"])], [Predicate("Q", ["x"])],
                      description="如果P(x)成立且P(x)蕴含Q(x)，则Q(x)成立")

        self.add_rule("modus_tollens", [Predicate("NOT", ["Q(x)"])], [Predicate("NOT", ["P(x)"])],
                      description="如果Q(x)不成立且P(x)蕴含Q(x)，则P(x)不成立")

        self.add_rule("universal_instantiation",
                      [Predicate("forall", ["x", "P(x)"])],
                      [Predicate("P", ["a"])],
                      description="从全称命题推出特称命题")

        self.add_rule("existential_generalization",
                      [Predicate("P", ["a"])],
                      [Predicate("exists", ["x", "P(x)"])],
                      description="从特称命题推广到存在命题")

        self.add_rule("conjunction_elimination",
                      [Predicate("AND", ["P", "Q"])],
                      [Predicate("P", []), Predicate("Q", [])],
                      description="合取消去")

        self.add_rule("conjunction_introduction",
                      [Predicate("P", []), Predicate("Q", [])],
                      [Predicate("AND", ["P", "Q"])],
                      description="合取引入")

        self.add_rule("disjunction_elimination",
                      [Predicate("OR", ["P", "Q"]), Predicate("NOT", ["P"])],
                      [Predicate("Q", [])],
                      description="析取消去")

    def add_predicate(self, name: str, arguments: List[str],
                      predicate_type: PredicateType = PredicateType.ATOMIC,
                      quantifier: Optional[Quantifier] = None) -> Predicate:
        """添加谓词"""
        predicate = Predicate(name, arguments, predicate_type, quantifier)
        key = f"{name}({','.join(arguments)})"
        self._predicates[key] = predicate
        return predicate

    def get_predicate(self, name: str, arguments: List[str]) -> Optional[Predicate]:
        """获取谓词"""
        key = f"{name}({','.join(arguments)})"
        return self._predicates.get(key)

    class SymbolicRule:
        """符号推理规则"""

        def __init__(self, rule_id: str, conditions: List[Predicate],
                     conclusions: List[Predicate], confidence: float = 0.9,
                     priority: int = 1, description: str = ""):
            self.rule_id = rule_id
            self.conditions = conditions
            self.conclusions = conclusions
            self.confidence = confidence
            self.priority = priority
            self.description = description
            self.usage_count = 0
            self.last_used = 0

        def evaluate_conditions(self, facts: Dict[str, Any]) -> float:
            """评估规则条件"""
            if not self.conditions:
                return 1.0
            values = [cond.evaluate(facts) for cond in self.conditions]
            return min(values) if values else 1.0

        def apply(self, facts: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
            """应用规则"""
            condition_value = self.evaluate_conditions(facts)
            if condition_value > 0.5:
                results = {}
                for conclusion in self.conclusions:
                    conclusion.truth_value = condition_value * self.confidence
                    conclusion.confidence = condition_value * self.confidence
                    key = conclusion.to_string()
                    results[key] = conclusion.truth_value
                self.usage_count += 1
                self.last_used = time.time()
                return True, results
            return False, {}

    def add_rule(self, rule_id: str, conditions: List[Predicate], conclusions: List[Predicate],
                 confidence: float = 0.9, priority: int = 1, description: str = "") -> None:
        """添加推理规则"""
        rule = self.SymbolicRule(rule_id, conditions, conclusions, confidence, priority, description)
        self._inference_rules[rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """移除推理规则"""
        self._inference_rules.pop(rule_id, None)

    def get_rule(self, rule_id: str) -> Optional[SymbolicRule]:
        """获取推理规则"""
        return self._inference_rules.get(rule_id)

    def deductive_reasoning(self, facts: Dict[str, Any], goal: Optional[str] = None,
                            max_steps: int = 20) -> Dict[str, Any]:
        """演绎推理

        从已知事实出发，应用推理规则得出结论
        """
        results = {
            "steps": [],
            "derived_facts": {},
            "goal_achieved": False,
            "final_confidence": 0.0,
            "explanation": ""
        }

        current_facts = facts.copy()
        visited_rules = set()
        step_count = 0

        while step_count < max_steps:
            applicable_rules = []
            for rule_id, rule in self._inference_rules.items():
                if rule_id in visited_rules:
                    continue
                condition_value = rule.evaluate_conditions(current_facts)
                if condition_value > 0.5:
                    applicable_rules.append((rule, condition_value))

            if not applicable_rules:
                break

            applicable_rules.sort(key=lambda x: (-x[0].priority, -x[1], -x[0].confidence))
            best_rule, condition_value = applicable_rules[0]

            applied, derived = best_rule.apply(current_facts)

            if applied:
                step = {
                    "step": step_count + 1,
                    "rule_id": best_rule.rule_id,
                    "rule_name": best_rule.description,
                    "condition_value": condition_value,
                    "confidence": best_rule.confidence,
                    "derived": derived
                }
                results["steps"].append(step)

                for key, value in derived.items():
                    if key not in current_facts:
                        current_facts[key] = value
                        results["derived_facts"][key] = value

                if goal and self._check_goal(goal, current_facts):
                    results["goal_achieved"] = True
                    break

            visited_rules.add(best_rule.rule_id)
            step_count += 1

        if results["steps"]:
            results["final_confidence"] = float(sum(s.get("confidence", 0.5) for s in results["steps"]) / len(results["steps"]))

        results["explanation"] = self._generate_deductive_explanation(results["steps"])

        return results

    def _check_goal(self, goal: str, facts: Dict[str, Any]) -> bool:
        """检查目标是否达成"""
        goal_lower = goal.lower()
        for key, value in facts.items():
            if key.lower() in goal_lower or str(value).lower() in goal_lower:
                return True
        return False

    def _generate_deductive_explanation(self, steps: List[Dict]) -> str:
        """生成演绎推理解释"""
        if not steps:
            return "未应用任何推理规则"

        lines = ["演绎推理过程："]
        for i, step in enumerate(steps, 1):
            rule_name = step.get("rule_name", step.get("rule_id", "未知规则"))
            condition_value = step.get("condition_value", "")
            derived = step.get("derived", {})
            line = f"步骤{i}：应用'{rule_name}'规则"
            if condition_value:
                line += f" (条件满足度: {condition_value:.2f})"
            if derived:
                derived_str = ", ".join([f"{k}={v:.2f}" for k, v in derived.items()])
                line += f"，得出：{derived_str}"
            lines.append(line)

        return "\n".join(lines)

    def inductive_reasoning(self, observations: List[Dict[str, Any]],
                            generalize: bool = True) -> Dict[str, Any]:
        """归纳推理

        从多个观察中归纳出一般规律
        """
        results = {
            "patterns": [],
            "generalizations": [],
            "confidence": 0.0,
            "explanation": ""
        }

        if len(observations) < 2:
            results["explanation"] = "需要至少两个观察才能进行归纳"
            return results

        patterns = self._find_common_patterns(observations)
        results["patterns"] = patterns

        if generalize and patterns:
            generalizations = self._generalize_patterns(patterns, observations)
            results["generalizations"] = generalizations

        results["confidence"] = self._calculate_inductive_confidence(patterns, observations)
        results["explanation"] = self._generate_inductive_explanation(patterns, results["generalizations"], results["confidence"])

        self._inductive_patterns.extend(patterns)

        return results

    def _find_common_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """发现共同模式"""
        patterns = []

        if not observations:
            return patterns

        first_obs = observations[0]
        for key in first_obs:
            values = [obs.get(key) for obs in observations if key in obs]
            if len(values) == len(observations) and len(set(values)) <= len(values) // 2:
                common_value = max(set(values), key=values.count)
                support = sum(1 for v in values if v == common_value) / len(values)
                patterns.append({
                    "property": key,
                    "common_value": common_value,
                    "support": support,
                    "observation_count": len(observations)
                })

        for i, obs1 in enumerate(observations):
            for j, obs2 in enumerate(observations):
                if i >= j:
                    continue
                for key1 in obs1:
                    for key2 in obs2:
                        if obs1.get(key1) == obs2.get(key2):
                            patterns.append({
                                "property_pair": (key1, key2),
                                "common_value": obs1.get(key1),
                                "support": 1.0,
                                "observation_count": 2
                            })

        return patterns

    def _generalize_patterns(self, patterns: List[Dict[str, Any]],
                             observations: List[Dict[str, Any]]) -> List[str]:
        """推广模式为一般规律"""
        generalizations = []

        for pattern in patterns:
            if pattern["support"] >= 0.7:
                if "property" in pattern:
                    gen = f"所有观察对象都具有属性 '{pattern['property']}' = '{pattern['common_value']}'"
                    generalizations.append(gen)
                elif "property_pair" in pattern:
                    gen = f"属性 '{pattern['property_pair'][0]}' 和 '{pattern['property_pair'][1]}' 相关联，值为 '{pattern['common_value']}'"
                    generalizations.append(gen)

        return generalizations

    def _calculate_inductive_confidence(self, patterns: List[Dict[str, Any]],
                                        observations: List[Dict[str, Any]]) -> float:
        """计算归纳置信度"""
        if not patterns:
            return 0.0

        avg_support = sum(p.get("support", 0.5) for p in patterns) / len(patterns)
        observation_diversity = min(1.0, len(set(str(o) for o in observations)) / len(observations))
        pattern_count_bonus = min(0.2, len(patterns) * 0.05)

        return min(1.0, avg_support * 0.6 + observation_diversity * 0.2 + pattern_count_bonus)

    def _generate_inductive_explanation(self, patterns: List[Dict[str, Any]],
                                        generalizations: List[str], confidence: float) -> str:
        """生成归纳推理解释"""
        lines = ["归纳推理过程："]
        lines.append(f"观察数量：{len(patterns)} 个模式")

        for i, pattern in enumerate(patterns, 1):
            if "property" in pattern:
                lines.append(f"模式{i}：属性 '{pattern['property']}' 的值 '{pattern['common_value']}' 支持度 {pattern['support']:.2f}")
            elif "property_pair" in pattern:
                lines.append(f"模式{i}：属性对 ({pattern['property_pair'][0]}, {pattern['property_pair'][1]}) 关联值 '{pattern['common_value']}'")

        if generalizations:
            lines.append("")
            lines.append("归纳结论：")
            for gen in generalizations:
                lines.append(f"- {gen}")

        lines.append(f"")
        lines.append(f"归纳置信度：{confidence:.2f}")

        return "\n".join(lines)

    def analogical_reasoning(self, source_domain: Dict[str, Any], target_domain: Dict[str, Any],
                             similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """类比推理

        基于源域和目标域之间的相似性进行推理
        """
        results = {
            "similarities": [],
            "mappings": [],
            "inferences": [],
            "confidence": 0.0,
            "explanation": ""
        }

        similarities = self._calculate_similarities(source_domain, target_domain)
        results["similarities"] = similarities

        mappings = []
        if any(s["score"] >= similarity_threshold for s in similarities):
            mappings = self._create_analogical_mappings(source_domain, target_domain, similarities)
            results["mappings"] = mappings

            inferences = self._transfer_knowledge(source_domain, target_domain, mappings)
            results["inferences"] = inferences

        results["confidence"] = self._calculate_analogical_confidence(similarities, mappings)
        results["explanation"] = self._generate_analogical_explanation(similarities, mappings, results["inferences"], results["confidence"])

        for mapping in mappings:
            key = f"{mapping['source']}->{mapping['target']}"
            self._analogy_mappings[key].append((mapping['target'], results["confidence"]))

        return results

    def _calculate_similarities(self, source: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算源域和目标域之间的相似性"""
        similarities = []

        for source_key, source_value in source.items():
            for target_key, target_value in target.items():
                score = self._compute_similarity(source_value, target_value)
                if score > 0.3:
                    similarities.append({
                        "source_key": source_key,
                        "target_key": target_key,
                        "source_value": source_value,
                        "target_value": target_value,
                        "score": score
                    })

        similarities.sort(key=lambda x: -x["score"])

        return similarities

    def _compute_similarity(self, value1: Any, value2: Any) -> float:
        """计算两个值之间的相似度"""
        if value1 == value2:
            return 1.0

        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            max_val = max(abs(value1), abs(value2), 1)
            return max(0.0, 1.0 - abs(value1 - value2) / max_val)

        str1 = str(value1).lower()
        str2 = str(value2).lower()

        if str1 in str2 or str2 in str1:
            return 0.7

        common_words = set(str1.split()) & set(str2.split())
        total_words = set(str1.split()) | set(str2.split())
        if total_words:
            return len(common_words) / len(total_words)

        return 0.0

    def _create_analogical_mappings(self, source: Dict[str, Any], target: Dict[str, Any],
                                    similarities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建类比映射"""
        mappings = []

        used_source = set()
        used_target = set()

        for similarity in similarities:
            if similarity["source_key"] in used_source or similarity["target_key"] in used_target:
                continue

            mappings.append({
                "source": similarity["source_key"],
                "target": similarity["target_key"],
                "similarity_score": similarity["score"],
                "source_value": similarity["source_value"],
                "target_value": similarity["target_value"]
            })

            used_source.add(similarity["source_key"])
            used_target.add(similarity["target_key"])

        return mappings

    def _transfer_knowledge(self, source: Dict[str, Any], target: Dict[str, Any],
                            mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于映射进行知识转移"""
        inferences = []

        for mapping in mappings:
            source_key = mapping["source"]
            target_key = mapping["target"]

            if source_key in source and target_key not in target:
                inferences.append({
                    "target_key": target_key,
                    "inferred_value": source[source_key],
                    "confidence": mapping["similarity_score"] * 0.8,
                    "based_on": source_key,
                    "reasoning": f"因为 {source_key} 在源域中是 {source[source_key]}，且 {source_key} 与 {target_key} 相似（相似度 {mapping['similarity_score']:.2f}），所以推断 {target_key} 也是 {source[source_key]}"
                })

        return inferences

    def _calculate_analogical_confidence(self, similarities: List[Dict[str, Any]],
                                         mappings: List[Dict[str, Any]]) -> float:
        """计算类比推理置信度"""
        if not similarities or not mappings:
            return 0.0

        avg_similarity = sum(s["score"] for s in similarities) / len(similarities)
        mapping_count_bonus = min(0.3, len(mappings) * 0.1)

        return min(1.0, avg_similarity * 0.7 + mapping_count_bonus)

    def _generate_analogical_explanation(self, similarities: List[Dict[str, Any]],
                                         mappings: List[Dict[str, Any]],
                                         inferences: List[Dict[str, Any]], confidence: float) -> str:
        """生成类比推理解释"""
        lines = ["类比推理过程："]

        lines.append("相似性发现：")
        for i, sim in enumerate(similarities[:5], 1):
            lines.append(f"  {i}. '{sim['source_key']}' ({sim['source_value']}) 与 '{sim['target_key']}' ({sim['target_value']}) 相似度 {sim['score']:.2f}")

        if mappings:
            lines.append("")
            lines.append("类比映射：")
            for mapping in mappings:
                lines.append(f"  - {mapping['source']} -> {mapping['target']} (相似度 {mapping['similarity_score']:.2f})")

        if inferences:
            lines.append("")
            lines.append("知识转移：")
            for inference in inferences:
                lines.append(f"  - {inference['reasoning']} (置信度 {inference['confidence']:.2f})")

        lines.append(f"")
        lines.append(f"类比置信度：{confidence:.2f}")

        return "\n".join(lines)

    def natural_language_to_symbolic(self, text: str) -> Dict[str, Any]:
        """自然语言到符号表示的转换"""
        predicates = []
        relations = []
        entities = []

        text_lower = text.lower()

        entity_patterns = ["is a", "are", "has", "have", "contains", "consists of"]
        for pattern in entity_patterns:
            idx = text_lower.find(pattern)
            if idx != -1:
                parts = text[:idx].strip().split()
                if parts:
                    entities.append(" ".join(parts))

        if "if" in text_lower and "then" in text_lower:
            if_idx = text_lower.find("if")
            then_idx = text_lower.find("then")
            condition = text[if_idx + 2:then_idx].strip()
            conclusion = text[then_idx + 4:].strip()

            predicates.append(Predicate("condition", [condition], PredicateType.ATOMIC))
            predicates.append(Predicate("conclusion", [conclusion], PredicateType.ATOMIC))

            implies_pred = Predicate("IMPLIES", [condition, conclusion], PredicateType.COMPOUND)
            predicates.append(implies_pred)

        if "all" in text_lower or "every" in text_lower:
            predicates.append(Predicate("universal", ["x"], PredicateType.QUANTIFIED, Quantifier.UNIVERSAL))

        if "some" in text_lower or "there exists" in text_lower:
            predicates.append(Predicate("existential", ["x"], PredicateType.QUANTIFIED, Quantifier.EXISTENTIAL))

        return {
            "predicates": [p.to_string() for p in predicates],
            "relations": relations,
            "entities": entities,
            "raw_text": text
        }

    def symbolic_to_natural_language(self, predicates: List[Predicate]) -> str:
        """符号表示到自然语言的转换"""
        sentences = []

        for pred in predicates:
            if pred.predicate_type == PredicateType.ATOMIC:
                sentences.append(f"{pred.name}({', '.join(pred.arguments)}) 为真")
            elif pred.predicate_type == PredicateType.COMPOUND:
                sentences.append(f"复合谓词 {pred.name} 应用于 ({', '.join(pred.arguments)})")
            elif pred.predicate_type == PredicateType.RELATIONAL:
                sentences.append(f"{pred.arguments[0]} 和 {pred.arguments[1]} 之间存在关系 {pred.name}")
            elif pred.predicate_type == PredicateType.QUANTIFIED:
                quant_str = {"forall": "对于所有", "exists": "存在", "exists_unique": "存在唯一"}.get(pred.quantifier.value, "")
                sentences.append(f"{quant_str} {', '.join(pred.arguments)}，{pred.name}成立")

        return "；".join(sentences)

    def reason(self, problem_text: str, facts: Optional[Dict[str, Any]] = None,
               mode: str = "deductive", goal: Optional[str] = None) -> ReasoningTrace:
        """统一推理接口"""
        start_time = time.time()

        trace = ReasoningTrace(
            trace_id=f"symbolic_trace_{uuid.uuid4().hex[:12]}",
            problem_text=problem_text,
            strategy={"deductive": "deductive_reasoning", "inductive": "inductive_reasoning", "analogical": "analogical_reasoning"}.get(mode, "deductive_reasoning"),
            strategy_confidence=0.8
        )

        if mode == "deductive":
            result = self.deductive_reasoning(facts or {}, goal)
            for i, step_data in enumerate(result.get("steps", []), 1):
                step = ReasoningStep(
                    step_id=f"symbolic_step_{i:04d}",
                    step_number=i,
                    type="deductive_inference",
                    description=step_data.get("rule_name", step_data.get("rule_id", "推理")),
                    natural_language=f"应用规则 '{step_data.get('rule_name', '未知')}'，得出 {step_data.get('derived', {})}",
                    premise=str(facts),
                    result=str(step_data.get("derived", {})),
                    confidence=step_data.get("confidence", 0.5),
                    used_rules=[step_data.get("rule_id", "")]
                )
                trace.add_step(step)

            trace.final_confidence = result.get("final_confidence", 0.0)
            trace.explanation = result.get("explanation", "")

        elif mode == "inductive":
            observations = facts.get("observations", []) if isinstance(facts, dict) else []
            result = self.inductive_reasoning(observations)

            step = ReasoningStep(
                step_id="inductive_step_0001",
                step_number=1,
                type="inductive_generalization",
                description="归纳推理",
                natural_language=result.get("explanation", ""),
                premise=f"观察到 {len(observations)} 个实例",
                result=str(result.get("generalizations", [])),
                confidence=result.get("confidence", 0.5)
            )
            trace.add_step(step)

            trace.final_confidence = result.get("confidence", 0.0)
            trace.explanation = result.get("explanation", "")

        elif mode == "analogical":
            source = facts.get("source", {}) if isinstance(facts, dict) else {}
            target = facts.get("target", {}) if isinstance(facts, dict) else {}
            result = self.analogical_reasoning(source, target)

            step = ReasoningStep(
                step_id="analogical_step_0001",
                step_number=1,
                type="analogical_transfer",
                description="类比推理",
                natural_language=result.get("explanation", ""),
                premise=f"源域: {source}, 目标域: {target}",
                result=str(result.get("inferences", [])),
                confidence=result.get("confidence", 0.5)
            )
            trace.add_step(step)

            trace.final_confidence = result.get("confidence", 0.0)
            trace.explanation = result.get("explanation", "")

        trace.is_complete = True
        trace.is_success = trace.final_confidence >= 0.5
        trace.execution_time = time.time() - start_time

        self._reasoning_history.append(trace)

        return trace

    def get_stats(self) -> Dict[str, Any]:
        """获取推理器统计信息"""
        return {
            "predicate_count": len(self._predicates),
            "rule_count": len(self._inference_rules),
            "analogy_mapping_count": sum(len(v) for v in self._analogy_mappings.values()),
            "inductive_pattern_count": len(self._inductive_patterns),
            "reasoning_history_length": len(self._reasoning_history)
        }

    def get_base_reasoner(self) -> NeuroSymbolicReasoner:
        """获取基础神经符号推理器"""
        return self._base_reasoner