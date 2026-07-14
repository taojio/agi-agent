import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import DEVICE


class LogicType(Enum):
    PREDICATE = "predicate"
    MODAL = "modal"
    TEMPORAL = "temporal"
    PROBABILISTIC = "probabilistic"


class Predicate:
    def __init__(self, name: str, arguments: List[str], truth_value: Optional[float] = None):
        self.name = name
        self.arguments = arguments
        self.truth_value = truth_value
        self.confidence = 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "arguments": self.arguments,
            "truth_value": self.truth_value,
            "confidence": self.confidence
        }


class LogicalStatement:
    def __init__(self, predicates: List[Predicate], operator: str = "AND", confidence: float = 0.5):
        self.predicates = predicates
        self.operator = operator
        self.confidence = confidence
        self.negated = False

    def negate(self):
        self.negated = not self.negated
        self.confidence = 1.0 - self.confidence

    def evaluate(self, knowledge_base) -> float:
        if not self.predicates:
            return 0.5

        values = []
        for pred in self.predicates:
            value = knowledge_base.query_predicate(pred)
            if value is not None:
                values.append(value)

        if not values:
            return self.confidence

        if self.operator == "AND":
            result = np.prod(values)
        elif self.operator == "OR":
            result = 1.0 - np.prod(1.0 - np.array(values))
        elif self.operator == "IMPLIES":
            if len(values) >= 2:
                result = 1.0 - values[0] + values[0] * values[1]
            else:
                result = np.mean(values)
        elif self.operator == "XOR":
            if len(values) >= 2:
                result = abs(values[0] - values[1])
            else:
                result = np.mean(values)
        else:
            result = np.mean(values)

        if self.negated:
            result = 1.0 - result

        return min(1.0, max(0.0, result))

    def to_dict(self):
        return {
            "predicates": [p.to_dict() for p in self.predicates],
            "operator": self.operator,
            "confidence": self.confidence,
            "negated": self.negated
        }


class KnowledgeBase:
    def __init__(self):
        self.predicates: Dict[str, Predicate] = {}
        self.relations: Dict[str, List[Tuple[str, str, float]]] = {}
        self.inference_rules = []

    def add_predicate(self, predicate: Predicate):
        key = f"{predicate.name}_{tuple(predicate.arguments)}"
        self.predicates[key] = predicate

    def query_predicate(self, predicate: Predicate) -> Optional[float]:
        key = f"{predicate.name}_{tuple(predicate.arguments)}"
        if key in self.predicates:
            return self.predicates[key].truth_value
        return None

    def add_relation(self, relation_type: str, source: str, target: str, strength: float = 1.0):
        if relation_type not in self.relations:
            self.relations[relation_type] = []
        self.relations[relation_type].append((source, target, strength))

    def query_relation(self, relation_type: str, source: str = None, target: str = None) -> List[Tuple[str, str, float]]:
        results = []
        for rel in self.relations.get(relation_type, []):
            if (source is None or rel[0] == source) and (target is None or rel[1] == target):
                results.append(rel)
        return results

    def add_inference_rule(self, condition: LogicalStatement, conclusion: LogicalStatement, confidence: float = 0.9):
        self.inference_rules.append({
            "condition": condition,
            "conclusion": conclusion,
            "confidence": confidence
        })

    def apply_rules(self) -> List[LogicalStatement]:
        new_conclusions = []
        for rule in self.inference_rules:
            condition_value = rule["condition"].evaluate(self)
            if condition_value > 0.5:
                conclusion = rule["conclusion"]
                conclusion.confidence = condition_value * rule["confidence"]
                new_conclusions.append(conclusion)
                for pred in conclusion.predicates:
                    if pred.truth_value is None:
                        pred.truth_value = conclusion.confidence
                    else:
                        pred.truth_value = max(pred.truth_value, conclusion.confidence)
                    self.add_predicate(pred)
        return new_conclusions


class AdvancedReasoner:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.knowledge_base = KnowledgeBase()
        self.reasoning_history = deque(maxlen=200)
        self.heuristic_rules = []
        self.confidence_threshold = 0.7

        self._init_heuristic_rules()

    def _init_heuristic_rules(self):
        self.heuristic_rules = [
            {
                "name": "similarity_transfer",
                "description": "相似性传递",
                "apply": self._heuristic_similarity_transfer
            },
            {
                "name": "analogical_reasoning",
                "description": "类比推理",
                "apply": self._heuristic_analogical_reasoning
            },
            {
                "name": "abduction",
                "description": "溯因推理",
                "apply": self._heuristic_abduction
            },
            {
                "name": "induction",
                "description": "归纳推理",
                "apply": self._heuristic_induction
            },
            {
                "name": "default_reasoning",
                "description": "缺省推理",
                "apply": self._heuristic_default_reasoning
            }
        ]

    def add_predicate(self, name: str, arguments: List[str], truth_value: float = None):
        predicate = Predicate(name, arguments, truth_value)
        self.knowledge_base.add_predicate(predicate)

    def add_relation(self, relation_type: str, source: str, target: str, strength: float = 1.0):
        self.knowledge_base.add_relation(relation_type, source, target, strength)

    def add_inference_rule(self, condition_predicates: List[dict], condition_operator: str,
                          conclusion_predicates: List[dict], conclusion_operator: str = "AND",
                          confidence: float = 0.9):
        condition = LogicalStatement(
            [Predicate(p["name"], p.get("arguments", []), p.get("truth_value")) for p in condition_predicates],
            operator=condition_operator
        )
        conclusion = LogicalStatement(
            [Predicate(p["name"], p.get("arguments", [])) for p in conclusion_predicates],
            operator=conclusion_operator,
            confidence=confidence
        )
        self.knowledge_base.add_inference_rule(condition, conclusion, confidence)

    def reason(self, query: LogicalStatement) -> Dict[str, Any]:
        result = {
            "query": query.to_dict(),
            "truth_value": query.evaluate(self.knowledge_base),
            "confidence": query.confidence,
            "steps": [],
            "heuristics_applied": [],
            "deduced_facts": []
        }

        steps = []

        direct_evaluation = query.evaluate(self.knowledge_base)
        steps.append({
            "step": "direct_evaluation",
            "value": direct_evaluation,
            "description": "直接评估查询语句"
        })

        if direct_evaluation < self.confidence_threshold:
            deduced = self.knowledge_base.apply_rules()
            if deduced:
                steps.append({
                    "step": "rule_inference",
                    "deduced_count": len(deduced),
                    "description": "应用推理规则"
                })
                result["deduced_facts"] = [d.to_dict() for d in deduced]

            for heuristic in self.heuristic_rules:
                heuristic_result = heuristic["apply"](query)
                if heuristic_result:
                    steps.append({
                        "step": heuristic["name"],
                        "result": heuristic_result,
                        "description": heuristic["description"]
                    })
                    result["heuristics_applied"].append(heuristic["name"])

        final_value = query.evaluate(self.knowledge_base)
        steps.append({
            "step": "final_evaluation",
            "value": final_value,
            "description": "最终评估"
        })

        result["steps"] = steps
        result["truth_value"] = final_value
        result["confidence"] = final_value

        self.reasoning_history.append({
            "query": query.to_dict(),
            "truth_value": final_value,
            "steps_count": len(steps),
            "heuristics_applied": result["heuristics_applied"],
            "timestamp": np.random.randint(1000000)
        })

        return result

    def _heuristic_similarity_transfer(self, query: LogicalStatement) -> Optional[Dict]:
        for predicate in query.predicates:
            similar_preds = []
            for key, pred in self.knowledge_base.predicates.items():
                if pred.name == predicate.name and pred.truth_value is not None:
                    arg_overlap = len(set(predicate.arguments) & set(pred.arguments)) / max(
                        len(predicate.arguments), len(pred.arguments), 1)
                    if arg_overlap > 0.5:
                        similar_preds.append((pred, arg_overlap))

            if similar_preds:
                similar_preds.sort(key=lambda x: -x[1])
                best_match, overlap = similar_preds[0]
                if predicate.truth_value is None:
                    predicate.truth_value = best_match.truth_value * overlap
                    self.knowledge_base.add_predicate(predicate)
                return {
                    "method": "similarity_transfer",
                    "source_predicate": best_match.name,
                    "overlap": overlap,
                    "transferred_value": predicate.truth_value
                }
        return None

    def _heuristic_analogical_reasoning(self, query: LogicalStatement) -> Optional[Dict]:
        for predicate in query.predicates:
            for rel_type, relations in self.knowledge_base.relations.items():
                for source, target, strength in relations:
                    if source in predicate.arguments:
                        analogous_args = [target if arg == source else arg for arg in predicate.arguments]
                        analogous_pred = Predicate(predicate.name, analogous_args)
                        analogous_key = f"{analogous_pred.name}_{tuple(analogous_pred.arguments)}"
                        if analogous_key in self.knowledge_base.predicates:
                            analogous_value = self.knowledge_base.predicates[analogous_key].truth_value
                            if analogous_value is not None and predicate.truth_value is None:
                                predicate.truth_value = analogous_value * strength
                                self.knowledge_base.add_predicate(predicate)
                                return {
                                    "method": "analogical_reasoning",
                                    "relation_type": rel_type,
                                    "source": source,
                                    "target": target,
                                    "strength": strength,
                                    "analogous_value": predicate.truth_value
                                }
        return None

    def _heuristic_abduction(self, query: LogicalStatement) -> Optional[Dict]:
        for predicate in query.predicates:
            if predicate.truth_value is None:
                for rule in self.knowledge_base.inference_rules:
                    conclusion = rule["conclusion"]
                    for conc_pred in conclusion.predicates:
                        if conc_pred.name == predicate.name:
                            condition_value = rule["condition"].evaluate(self.knowledge_base)
                            if condition_value > 0.5:
                                predicate.truth_value = condition_value * rule["confidence"]
                                self.knowledge_base.add_predicate(predicate)
                                return {
                                    "method": "abduction",
                                    "rule_condition": rule["condition"].operator,
                                    "condition_value": condition_value,
                                    "deduced_value": predicate.truth_value
                                }
        return None

    def _heuristic_induction(self, query: LogicalStatement) -> Optional[Dict]:
        for predicate in query.predicates:
            if predicate.truth_value is None:
                related_preds = []
                for key, pred in self.knowledge_base.predicates.items():
                    if pred.name == predicate.name and pred.truth_value is not None:
                        related_preds.append(pred.truth_value)

                if len(related_preds) >= 3:
                    avg_value = float(np.mean(related_preds))
                    std_value = float(np.std(related_preds))
                    predicate.truth_value = avg_value if std_value < 0.3 else None
                    if predicate.truth_value is not None:
                        self.knowledge_base.add_predicate(predicate)
                        return {
                            "method": "induction",
                            "sample_count": len(related_preds),
                            "mean_value": avg_value,
                            "std_value": std_value,
                            "generalized_value": predicate.truth_value
                        }
        return None

    def _heuristic_default_reasoning(self, query: LogicalStatement) -> Optional[Dict]:
        for predicate in query.predicates:
            if predicate.truth_value is None:
                predicate.truth_value = 0.5
                predicate.confidence = 0.3
                self.knowledge_base.add_predicate(predicate)
                return {
                    "method": "default_reasoning",
                    "default_value": 0.5,
                    "confidence": 0.3
                }
        return None

    def create_query(self, predicate_name: str, arguments: List[str],
                     operator: str = "AND") -> LogicalStatement:
        predicate = Predicate(predicate_name, arguments)
        return LogicalStatement([predicate], operator=operator)

    def get_reasoning_stats(self) -> Dict[str, Any]:
        stats = {
            "total_reasoning_cycles": len(self.reasoning_history),
            "predicate_count": len(self.knowledge_base.predicates),
            "relation_count": sum(len(v) for v in self.knowledge_base.relations.values()),
            "rule_count": len(self.knowledge_base.inference_rules),
            "avg_confidence": 0.0,
            "heuristics_usage": {}
        }

        if self.reasoning_history:
            recent = list(self.reasoning_history)[-20:]
            stats["avg_confidence"] = float(np.mean([r["truth_value"] for r in recent]))

            for heuristic in self.heuristic_rules:
                stats["heuristics_usage"][heuristic["name"]] = sum(
                    1 for r in recent if heuristic["name"] in r["heuristics_applied"]
                )

        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim