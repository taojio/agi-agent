import numpy as np
from collections import deque
from typing import Dict


class ProductionRule:
    def __init__(self, condition, action, priority=1.0, confidence=0.9, rule_id=None):
        self.rule_id = rule_id or f"rule_{hash(str(condition))}"
        self.condition = condition
        self.action = action
        self.priority = priority
        self.confidence = confidence
        self.usage_count = 0
        self.success_count = 0
        self.last_used = 0

    def match(self, input_data):
        if callable(self.condition):
            return self.condition(input_data)
        elif isinstance(self.condition, dict):
            for key, value in self.condition.items():
                if key not in input_data:
                    return False
                if isinstance(value, (int, float)):
                    if not (input_data[key] >= value[0] and input_data[key] <= value[1]):
                        return False
                elif input_data[key] != value:
                    return False
            return True
        return False


class RuleEngine:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.rules = []
        self.rule_index = {}
        self.execution_history = deque(maxlen=200)
        self.rule_performance = {}
        self.max_rules = 100

    def add_rule(self, rule):
        if len(self.rules) >= self.max_rules:
            self._prune_rules()
        
        self.rules.append(rule)
        self.rule_index[rule.rule_id] = rule
        self.rule_performance[rule.rule_id] = {"executions": 0, "successes": 0}

    def add_rules(self, rules):
        for rule in rules:
            self.add_rule(rule)

    def _prune_rules(self):
        sorted_rules = sorted(self.rules, key=lambda r: r.usage_count, reverse=True)
        self.rules = sorted_rules[:self.max_rules]
        self.rule_index = {r.rule_id: r for r in self.rules}

    def match_rules(self, input_data):
        matched = []
        for rule in self.rules:
            if rule.match(input_data):
                matched.append((rule, rule.priority * rule.confidence))
        
        matched.sort(key=lambda x: -x[1])
        return matched

    def execute_best_rule(self, input_data):
        matched = self.match_rules(input_data)
        
        if not matched:
            return None, None, 0.0
        
        best_rule, score = matched[0]
        best_rule.usage_count += 1
        self.rule_performance[best_rule.rule_id]["executions"] += 1
        
        if callable(best_rule.action):
            result = best_rule.action(input_data)
        else:
            result = best_rule.action
        
        execution_record = {
            "rule_id": best_rule.rule_id,
            "input": str(input_data)[:200],
            "result": str(result)[:200],
            "score": score,
            "confidence": best_rule.confidence,
            "timestamp": np.random.randint(1000000)
        }
        self.execution_history.append(execution_record)
        
        return best_rule.rule_id, result, score

    def execute_all_matched(self, input_data):
        matched = self.match_rules(input_data)
        results = []
        
        for rule, score in matched:
            rule.usage_count += 1
            self.rule_performance[rule.rule_id]["executions"] += 1
            
            if callable(rule.action):
                result = rule.action(input_data)
            else:
                result = rule.action
            
            results.append({
                "rule_id": rule.rule_id,
                "action": rule.action,
                "result": result,
                "score": score,
                "confidence": rule.confidence
            })
        
        return results

    def update_rule_performance(self, rule_id, success):
        if rule_id in self.rule_performance:
            self.rule_performance[rule_id]["successes"] += 1 if success else 0
        
        if rule_id in self.rule_index:
            rule = self.rule_index[rule_id]
            executions = self.rule_performance[rule_id]["executions"]
            if executions > 0:
                rule.confidence = self.rule_performance[rule_id]["successes"] / executions

    def get_rule_stats(self):
        stats = {}
        for rule in self.rules:
            perf = self.rule_performance.get(rule.rule_id, {"executions": 0, "successes": 0})
            success_rate = perf["successes"] / perf["executions"] if perf["executions"] > 0 else 0.0
            stats[rule.rule_id] = {
                "priority": rule.priority,
                "confidence": rule.confidence,
                "usage_count": rule.usage_count,
                "success_rate": success_rate,
                "executions": perf["executions"]
            }
        return stats

    def get_execution_history(self, limit=20):
        return list(self.execution_history)[-limit:]

    def clear_rules(self):
        self.rules = []
        self.rule_index = {}
        self.rule_performance = {}

    def resize(self, new_dim):
        self.feature_dim = new_dim

    def update_rule_priority(self, rule_id: str, delta: float):
        """
        调整规则的优先级

        Args:
            rule_id: 规则ID
            delta: 优先级增量（正值提升，负值降低）
        """
        rule = self.rule_index.get(rule_id)
        if rule is None:
            return False
        rule.priority = max(0.0, rule.priority + delta)
        return True