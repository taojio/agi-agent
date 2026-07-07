import numpy as np
from enum import Enum
from collections import deque


class OptimizationCriteria(Enum):
    EFFICIENCY = "efficiency"
    COST = "cost"
    STABILITY = "stability"
    BALANCED = "balanced"


class Solution:
    def __init__(self, solution_id, hypothesis_id, action_plan, priority=0.0, confidence=0.5, risk_level="low", execution_steps=None):
        self.solution_id = solution_id
        self.hypothesis_id = hypothesis_id
        self.action_plan = action_plan
        self.priority = priority
        self.confidence = confidence
        self.risk_level = risk_level
        self.execution_steps = execution_steps or []
        self.validation_nodes = []

    def to_dict(self):
        return {
            "solution_id": self.solution_id,
            "hypothesis_id": self.hypothesis_id,
            "action_plan": self.action_plan,
            "priority": self.priority,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "execution_steps": self.execution_steps,
            "validation_nodes": self.validation_nodes
        }


class DecisionOptimizer:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.solutions = {}
        self.optimization_history = deque(maxlen=200)
        self.current_criteria = OptimizationCriteria.BALANCED

    def _calculate_score(self, hypothesis, simulation_result, criteria):
        weights = {
            OptimizationCriteria.EFFICIENCY: {"success_rate": 0.5, "time": 0.3, "resource": 0.2},
            OptimizationCriteria.COST: {"success_rate": 0.3, "time": 0.2, "resource": 0.5},
            OptimizationCriteria.STABILITY: {"success_rate": 0.6, "time": 0.2, "resource": 0.2},
            OptimizationCriteria.BALANCED: {"success_rate": 0.4, "time": 0.3, "resource": 0.3}
        }
        
        w = weights[criteria]
        
        success_score = simulation_result.success_rate
        time_score = max(0, 1 - simulation_result.estimated_time / 100)
        resource_score = max(0, 1 - simulation_result.resource_consumption / 10)
        
        score = w["success_rate"] * success_score + w["time"] * time_score + w["resource"] * resource_score
        
        score *= (1 - hypothesis.potential_risk * 0.5)
        
        return score

    def _determine_risk_level(self, risk_score):
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.5:
            return "medium"
        else:
            return "high"

    def optimize(self, hypotheses, simulation_results, criteria=None):
        criteria = criteria or self.current_criteria
        
        scored_hypotheses = []
        for hyp, sim_result in zip(hypotheses, simulation_results):
            score = self._calculate_score(hyp, sim_result, criteria)
            scored_hypotheses.append((hyp, sim_result, score))
        
        scored_hypotheses.sort(key=lambda x: -x[2])
        
        if not scored_hypotheses:
            return None, []
        
        best_hyp, best_sim, best_score = scored_hypotheses[0]
        
        solution_id = f"sol_{len(self.solutions) + 1}"
        risk_level = self._determine_risk_level(best_sim.risk_score)
        
        execution_steps = self._generate_execution_steps(best_hyp)
        validation_nodes = self._generate_validation_nodes(best_hyp)
        
        solution = Solution(
            solution_id=solution_id,
            hypothesis_id=best_hyp.hypothesis_id,
            action_plan=best_hyp.solution,
            priority=best_score,
            confidence=best_sim.success_rate,
            risk_level=risk_level,
            execution_steps=execution_steps
        )
        solution.validation_nodes = validation_nodes
        
        self.solutions[solution_id] = solution
        
        alternatives = []
        for hyp, sim_result, score in scored_hypotheses[1:3]:
            alt_solution = Solution(
                solution_id=f"sol_{len(self.solutions) + 1}",
                hypothesis_id=hyp.hypothesis_id,
                action_plan=hyp.solution,
                priority=score,
                confidence=sim_result.success_rate,
                risk_level=self._determine_risk_level(sim_result.risk_score)
            )
            alternatives.append(alt_solution)
        
        self.optimization_history.append({
            "solution_id": solution_id,
            "best_score": best_score,
            "criteria": criteria.value,
            "alternatives_count": len(alternatives),
            "timestamp": np.random.randint(1000000)
        })
        
        return solution, alternatives

    def _generate_execution_steps(self, hypothesis):
        action = hypothesis.solution.get("action", "unknown")
        steps = []
        
        if action == "correct":
            steps = [
                {"step": 1, "action": "assess_current_state"},
                {"step": 2, "action": "apply_correction"},
                {"step": 3, "action": "verify_result"}
            ]
        elif action == "approach":
            num_steps = hypothesis.solution.get("steps", 3)
            steps = [{"step": i + 1, "action": f"approach_step_{i + 1}"} for i in range(num_steps)]
            steps.append({"step": num_steps + 1, "action": "finalize"})
        elif action == "explore":
            depth = hypothesis.solution.get("depth", 2)
            steps = [{"step": i + 1, "action": f"explore_level_{i + 1}"} for i in range(depth)]
            steps.append({"step": depth + 1, "action": "summarize_findings"})
        else:
            steps = [{"step": 1, "action": "execute"}, {"step": 2, "action": "verify"}]
        
        return steps

    def _generate_validation_nodes(self, hypothesis):
        return [
            {"node": "pre_execution", "check": "resources_available"},
            {"node": "mid_execution", "check": "progress_on_track"},
            {"node": "post_execution", "check": "goal_achieved"}
        ]

    def set_criteria(self, criteria):
        if isinstance(criteria, OptimizationCriteria):
            self.current_criteria = criteria
        elif isinstance(criteria, str):
            for c in OptimizationCriteria:
                if c.value == criteria.lower():
                    self.current_criteria = c
                    break

    def get_optimization_stats(self):
        stats = {
            "total_optimizations": len(self.optimization_history),
            "current_criteria": self.current_criteria.value,
            "avg_best_score": 0.0,
            "solutions_count": len(self.solutions)
        }
        
        if self.optimization_history:
            recent = list(self.optimization_history)[-20:]
            stats["avg_best_score"] = float(np.mean([o["best_score"] for o in recent]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim