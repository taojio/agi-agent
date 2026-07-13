import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional


class DeductionStep:
    def __init__(self, step_id, premise, operation, result, confidence=0.5):
        self.step_id = step_id
        self.premise = premise
        self.operation = operation
        self.result = result
        self.confidence = confidence
        self.valid = True

    def to_dict(self):
        return {
            "step_id": self.step_id,
            "premise": self.premise,
            "operation": self.operation,
            "result": self.result,
            "confidence": self.confidence,
            "valid": self.valid
        }


class LogicalDeductor:
    def __init__(self, feature_dim=16, rule_registry=None):
        self.feature_dim = feature_dim
        self.deduction_rules = []
        self.deduction_history = deque(maxlen=200)
        self._rule_registry = rule_registry
        
        self._init_default_rules()

    def _init_default_rules(self):
        self.deduction_rules = [
            {"name": "modus_ponens", "description": "肯定前件", "pattern": lambda p: "if" in p.lower() and "then" in p.lower()},
            {"name": "modus_tollens", "description": "否定后件", "pattern": lambda p: "if" in p.lower() and "not" in p.lower()},
            {"name": "disjunction_elimination", "description": "析取消去", "pattern": lambda p: "or" in p.lower()},
            {"name": "conjunction_introduction", "description": "合取引入", "pattern": lambda p: "and" in p.lower()},
            {"name": "hypothetical_syllogism", "description": "假言三段论", "pattern": lambda p: "if" in p.lower() and "," in p},
            {"name": "reductio_ad_absurdum", "description": "归谬法", "pattern": lambda p: "contradiction" in p.lower() or "impossible" in p.lower()}
        ]

    def set_rule_registry(self, rule_registry):
        self._rule_registry = rule_registry

    def _apply_rule(self, premise, rule):
        if rule["pattern"](premise):
            return {
                "rule_applied": rule["name"],
                "result": f"Deducted from: {premise}",
                "confidence": 0.9
            }
        return None

    def deduce(self, hypothesis, problem):
        steps = []
        current_confidence = hypothesis.confidence
        all_valid = True
        
        premises = hypothesis.premises + [str(problem.goal)]
        
        for i, premise in enumerate(premises):
            for rule in self.deduction_rules:
                result = self._apply_rule(premise, rule)
                if result:
                    step = DeductionStep(
                        step_id=f"step_{i}_{rule['name']}",
                        premise=premise,
                        operation=rule["name"],
                        result=result["result"],
                        confidence=result["confidence"]
                    )
                    steps.append(step)
                    current_confidence = min(1.0, current_confidence * 0.95 + result["confidence"] * 0.05)
        
        if not steps:
            default_rule = {"name": "default_reasoning", "description": "默认推理", "pattern": lambda p: True}
            for i, premise in enumerate(premises):
                step = DeductionStep(
                    step_id=f"step_{i}_default",
                    premise=premise,
                    operation="default_reasoning",
                    result=f"Accepted based on premise: {premise[:50]}...",
                    confidence=current_confidence
                )
                steps.append(step)
        
        if current_confidence < 0.3:
            all_valid = False
        
        hypothesis.deduction_path = steps
        hypothesis.confidence = current_confidence
        
        self.deduction_history.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "steps_count": len(steps),
            "final_confidence": current_confidence,
            "all_valid": all_valid,
            "timestamp": np.random.randint(1000000)
        })
        
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "steps": [s.to_dict() for s in steps],
            "final_confidence": current_confidence,
            "valid": all_valid
        }

    def check_consistency(self, hypothesis, problem):
        conflicts = []
        
        forbidden = problem.boundaries.get("forbidden_actions", [])
        
        solution_action = ""
        if isinstance(hypothesis.solution, dict):
            solution_action = hypothesis.solution.get("action", "")
        elif hasattr(hypothesis.solution, 'action'):
            solution_action = hypothesis.solution.action
        
        if solution_action in forbidden:
            conflicts.append(f"Action '{solution_action}' is forbidden")
        
        risk = getattr(hypothesis, 'potential_risk', 0.0)
        tolerance = problem.constraints.get("risk_tolerance", 0.5)
        if risk > tolerance:
            conflicts.append(f"Risk {risk:.2f} exceeds tolerance {tolerance}")
        
        return {
            "consistent": len(conflicts) == 0,
            "conflicts": conflicts,
            "risk_acceptable": risk <= tolerance
        }

    def get_deduction_stats(self):
        stats = {
            "total_deductions": len(self.deduction_history),
            "avg_steps": 0.0,
            "avg_confidence": 0.0,
            "valid_rate": 0.0
        }
        
        if self.deduction_history:
            recent = list(self.deduction_history)[-20:]
            stats["avg_steps"] = float(np.mean([d["steps_count"] for d in recent]))
            stats["avg_confidence"] = float(np.mean([d["final_confidence"] for d in recent]))
            stats["valid_rate"] = float(np.mean([1 if d["all_valid"] else 0 for d in recent]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim

    def forward_chain_reasoning(self, facts: Dict[str, Any], goal: str = None, max_steps: int = 10) -> Dict[str, Any]:
        results = {
            "steps": [],
            "derived_facts": {},
            "goal_achieved": False,
            "final_confidence": 0.0
        }

        if self._rule_registry is None:
            results["error"] = "Rule registry not set"
            return results

        current_facts = facts.copy()
        visited = set()
        step_count = 0

        while step_count < max_steps:
            applicable_rules = []
            for rule in self._rule_registry._rules.values():
                if rule.rule_id in visited:
                    continue
                if rule.is_applicable(current_facts):
                    applicable_rules.append(rule)

            if not applicable_rules:
                break

            applicable_rules.sort(key=lambda r: r.confidence, reverse=True)
            best_rule = applicable_rules[0]

            inputs = {}
            for var in best_rule.variables:
                if var.symbol in current_facts:
                    inputs[var.symbol] = current_facts[var.symbol]

            if inputs:
                try:
                    rule_result = best_rule.apply(inputs)
                    
                    step = {
                        "step": step_count + 1,
                        "rule_id": best_rule.rule_id,
                        "rule_name": best_rule.name,
                        "rule_type": best_rule.rule_type.value,
                        "formula": best_rule.formula,
                        "inputs": inputs,
                        "output": rule_result,
                        "confidence": best_rule.confidence,
                        "prerequisites": best_rule.prerequisite_rules
                    }
                    results["steps"].append(step)

                    for key, value in rule_result.items():
                        if key not in current_facts:
                            current_facts[key] = value
                            results["derived_facts"][key] = value

                    if goal and self._check_goal_achieved(goal, current_facts):
                        results["goal_achieved"] = True
                        break

                except Exception as e:
                    step = {
                        "step": step_count + 1,
                        "rule_id": best_rule.rule_id,
                        "rule_name": best_rule.name,
                        "error": str(e),
                        "status": "failed"
                    }
                    results["steps"].append(step)

            visited.add(best_rule.rule_id)
            step_count += 1

        if results["steps"]:
            results["final_confidence"] = float(np.mean([s.get("confidence", 0.5) for s in results["steps"] if "confidence" in s]))

        results["total_steps"] = step_count
        return results

    def _check_goal_achieved(self, goal: str, facts: Dict[str, Any]) -> bool:
        goal_lower = goal.lower()
        for key, value in facts.items():
            if key.lower() in goal_lower or str(value).lower() in goal_lower:
                return True
        return False

    def solve_complex_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "analysis": [],
            "solution_path": [],
            "final_answer": None,
            "confidence": 0.0
        }

        if self._rule_registry is None:
            results["error"] = "Rule registry not set"
            return results

        facts = problem.get("facts", {})
        goal = problem.get("goal", "")
        question = problem.get("question", "")

        results["analysis"].append(f"Problem: {question}")
        results["analysis"].append(f"Initial facts: {facts}")
        results["analysis"].append(f"Goal: {goal}")

        reasoning_result = self.forward_chain_reasoning(facts, goal)
        
        results["solution_path"] = reasoning_result["steps"]
        results["derived_facts"] = reasoning_result["derived_facts"]
        results["confidence"] = reasoning_result["final_confidence"]

        if reasoning_result["goal_achieved"]:
            results["final_answer"] = self._generate_final_answer(problem, reasoning_result)
            results["analysis"].append(f"Goal achieved! Answer: {results['final_answer']}")
        else:
            results["analysis"].append(f"Goal not fully achieved. Derived facts: {reasoning_result['derived_facts']}")

        return results

    def _generate_final_answer(self, problem: Dict[str, Any], reasoning_result: Dict[str, Any]) -> str:
        question = problem.get("question", "")
        derived = reasoning_result.get("derived_facts", {})
        
        if not derived:
            return "No solution found"

        answer_parts = []
        for key, value in derived.items():
            answer_parts.append(f"{key} = {value}")

        return f"根据物理定律推导，{', '.join(answer_parts)}"