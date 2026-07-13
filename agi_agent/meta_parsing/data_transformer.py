import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class TransformationType(Enum):
    FILTER = "filter"
    MAP = "map"
    REDUCE = "reduce"
    AGGREGATE = "aggregate"
    PIVOT = "pivot"
    NORMALIZE = "normalize"
    ENRICH = "enrich"
    VALIDATE = "validate"
    CLEAN = "clean"


class TransformationRule:
    def __init__(self, rule_id: str, transformation_type: TransformationType,
                 condition: str = None, action: Callable = None, params: Dict[str, Any] = None):
        self.rule_id = rule_id
        self.transformation_type = transformation_type
        self.condition = condition
        self.action = action
        self.params = params or {}
        self.applied_count: int = 0
        self.success_count: int = 0
        self.created_at = np.random.randint(1000000)

    def apply(self, data: Any, context: Dict[str, Any] = None) -> Tuple[Any, bool]:
        self.applied_count += 1
        
        if self.condition and not self._evaluate_condition(data, context):
            return data, False
        
        try:
            if self.action:
                result = self.action(data, context, self.params)
            else:
                result = self._default_transform(data, context)
            
            self.success_count += 1
            return result, True
        except Exception:
            return data, False

    def _evaluate_condition(self, data: Any, context: Dict[str, Any] = None) -> bool:
        if not self.condition:
            return True
        
        try:
            context = context or {}
            local_vars = {"data": data, **context}
            return eval(self.condition, {}, local_vars)
        except Exception:
            return False

    def _default_transform(self, data: Any, context: Dict[str, Any] = None) -> Any:
        if self.transformation_type == TransformationType.FILTER:
            key = self.params.get("key")
            value = self.params.get("value")
            if isinstance(data, list):
                return [item for item in data if item.get(key) == value]
        
        elif self.transformation_type == TransformationType.MAP:
            key = self.params.get("key")
            if isinstance(data, list):
                return [item.get(key) for item in data if key in item]
        
        elif self.transformation_type == TransformationType.NORMALIZE:
            if isinstance(data, dict):
                return {k.lower(): v for k, v in data.items()}
        
        return data

    def get_effectiveness(self) -> float:
        if self.applied_count == 0:
            return 0.5
        return self.success_count / self.applied_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "type": self.transformation_type.value,
            "condition": self.condition,
            "params": self.params,
            "applied_count": self.applied_count,
            "success_count": self.success_count,
            "effectiveness": self.get_effectiveness(),
            "created_at": self.created_at
        }


class TransformationContext:
    def __init__(self):
        self.input_schema: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings
        }


class TransformationResult:
    def __init__(self):
        self.success: bool = False
        self.transformed_data: Any = None
        self.context: TransformationContext = TransformationContext()
        self.applied_rules: List[str] = []
        self.skipped_rules: List[str] = []
        self.confidence: float = 0.0
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "applied_rules": self.applied_rules,
            "skipped_rules": self.skipped_rules,
            "context": self.context.to_dict(),
            "timestamp": self.timestamp
        }


class DataTransformer:
    def __init__(self):
        self.rules: Dict[str, TransformationRule] = {}
        self.rule_chains: Dict[str, List[str]] = {}
        self.transformation_history: deque = deque(maxlen=200)

    def add_rule(self, rule: TransformationRule):
        self.rules[rule.rule_id] = rule

    def create_rule(self, rule_id: str, transformation_type: TransformationType,
                    condition: str = None, action: Callable = None,
                    params: Dict[str, Any] = None) -> TransformationRule:
        rule = TransformationRule(rule_id, transformation_type, condition, action, params)
        self.add_rule(rule)
        return rule

    def create_chain(self, chain_id: str, rule_ids: List[str]):
        self.rule_chains[chain_id] = rule_ids

    def transform(self, data: Any, rules: List[str] = None,
                 chain_id: str = None, context: Dict[str, Any] = None) -> TransformationResult:
        result = TransformationResult()
        result.context = TransformationContext()
        result.context.input_schema = self._infer_schema(data)
        
        rule_list = []
        if chain_id and chain_id in self.rule_chains:
            rule_list = self.rule_chains[chain_id]
        elif rules:
            rule_list = rules
        else:
            rule_list = list(self.rules.keys())
        
        current_data = data
        applied = []
        skipped = []
        
        for rule_id in rule_list:
            if rule_id not in self.rules:
                skipped.append(rule_id)
                continue
            
            rule = self.rules[rule_id]
            transformed, success = rule.apply(current_data, context)
            
            if success:
                current_data = transformed
                applied.append(rule_id)
            else:
                skipped.append(rule_id)
        
        result.transformed_data = current_data
        result.applied_rules = applied
        result.skipped_rules = skipped
        result.success = len(applied) > 0
        result.confidence = self._calculate_confidence(applied, skipped)
        result.context.output_schema = self._infer_schema(current_data)
        
        self.transformation_history.append(result)
        
        return result

    def _infer_schema(self, data: Any) -> Dict[str, Any]:
        if data is None:
            return {"type": "null"}
        
        if isinstance(data, dict):
            schema = {"type": "object", "properties": {}}
            for key, value in data.items():
                schema["properties"][key] = self._infer_schema(value)
            return schema
        
        if isinstance(data, list):
            if not data:
                return {"type": "array", "items": {"type": "unknown"}}
            return {"type": "array", "items": self._infer_schema(data[0])}
        
        if isinstance(data, str):
            return {"type": "string", "length": len(data)}
        
        if isinstance(data, int):
            return {"type": "integer"}
        
        if isinstance(data, float):
            return {"type": "number"}
        
        return {"type": str(type(data).__name__)}

    def _calculate_confidence(self, applied: List[str], skipped: List[str]) -> float:
        if not applied and not skipped:
            return 0.5
        
        total = len(applied) + len(skipped)
        if total == 0:
            return 0.5
        
        return len(applied) / total

    def auto_transform(self, data: Any, target_format: str = "") -> TransformationResult:
        rules_to_apply = []
        
        if isinstance(data, list):
            rules_to_apply.append("normalize_list")
            rules_to_apply.append("clean_nulls")
        elif isinstance(data, dict):
            rules_to_apply.append("normalize_dict")
        
        if target_format == "tabular":
            rules_to_apply.append("convert_to_table")
        
        return self.transform(data, rules_to_apply)

    def batch_transform(self, data_list: List[Any], rules: List[str] = None,
                       chain_id: str = None) -> List[TransformationResult]:
        results = []
        for data in data_list:
            result = self.transform(data, rules, chain_id)
            results.append(result)
        return results

    def get_rule_summary(self) -> Dict[str, Any]:
        summary = {}
        for rule_id, rule in self.rules.items():
            summary[rule_id] = {
                "type": rule.transformation_type.value,
                "applied_count": rule.applied_count,
                "success_count": rule.success_count,
                "effectiveness": rule.get_effectiveness(),
                "params": rule.params
            }
        return summary

    def get_transformation_summary(self) -> Dict[str, Any]:
        if not self.transformation_history:
            return {"total_transformations": 0}
        
        results = list(self.transformation_history)
        success_rate = len([r for r in results if r.success]) / len(results)
        avg_confidence = np.mean([r.confidence for r in results])
        
        type_dist = {}
        for r in results:
            for rule_id in r.applied_rules:
                if rule_id in self.rules:
                    rule_type = self.rules[rule_id].transformation_type.value
                    type_dist[rule_type] = type_dist.get(rule_type, 0) + 1
        
        return {
            "total_transformations": len(results),
            "success_rate": success_rate,
            "avg_confidence": float(avg_confidence),
            "rule_stats": self.get_rule_summary(),
            "chain_stats": {chain_id: len(rule_ids) for chain_id, rule_ids in self.rule_chains.items()},
            "transformation_type_distribution": type_dist
        }

    def optimize_rules(self):
        for rule in self.rules.values():
            if rule.applied_count > 0 and rule.get_effectiveness() < 0.3:
                rule.params = self._improve_params(rule)

    def _improve_params(self, rule: TransformationRule) -> Dict[str, Any]:
        new_params = rule.params.copy()
        
        if rule.transformation_type == TransformationType.FILTER:
            if "threshold" in new_params:
                new_params["threshold"] = new_params["threshold"] * 0.9
        
        return new_params