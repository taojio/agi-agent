import numpy as np
from enum import Enum
from collections import deque


class TriggerType(Enum):
    DEVIATION = "deviation"
    OPPORTUNITY = "opportunity"
    UNKNOWN = "unknown"


class ProblemDefinition:
    def __init__(self, problem_id, trigger_type, goal, current_state, constraints=None, boundaries=None):
        self.problem_id = problem_id
        self.trigger_type = trigger_type
        self.goal = goal
        self.current_state = current_state
        self.constraints = constraints or {}
        self.boundaries = boundaries or {}
        self.priority = self._calculate_priority()
        self.created_at = np.random.randint(1000000)

    def _calculate_priority(self):
        severity = self.constraints.get("deviation_severity", 0.5)
        sensitivity = self.constraints.get("time_sensitivity", 0.5)
        resource_limit = self.constraints.get("resource_limit", 1.0)
        
        base_priority = {
            TriggerType.DEVIATION: 0.7,
            TriggerType.OPPORTUNITY: 0.5,
            TriggerType.UNKNOWN: 0.6
        }.get(self.trigger_type, 0.5)
        
        priority = base_priority * (0.6 * severity + 0.3 * sensitivity + 0.1 * resource_limit)
        
        return min(1.0, max(0.0, priority))

    def to_dict(self):
        return {
            "problem_id": self.problem_id,
            "trigger_type": self.trigger_type.value,
            "goal": self.goal,
            "current_state": self.current_state if isinstance(self.current_state, list) else list(self.current_state),
            "constraints": self.constraints,
            "boundaries": self.boundaries,
            "priority": self.priority,
            "created_at": self.created_at
        }


class ProblemFormulator:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.problems = {}
        self.problem_history = deque(maxlen=200)
        
        self.deviation_threshold = 0.3
        self.opportunity_threshold = 0.4

    def _detect_deviation(self, input_vector, target_state):
        input_vec = np.array(input_vector).flatten()
        target_vec = np.array(target_state).flatten()
        
        if len(input_vec) != len(target_vec):
            min_len = min(len(input_vec), len(target_vec))
            input_vec = input_vec[:min_len]
            target_vec = target_vec[:min_len]
        
        deviation = np.mean(np.abs(input_vec - target_vec))
        
        return min(1.0, deviation)

    def _detect_opportunity(self, input_vector, baseline_state):
        input_vec = np.array(input_vector).flatten()
        baseline_vec = np.array(baseline_state).flatten()
        
        if len(input_vec) != len(baseline_vec):
            min_len = min(len(input_vec), len(baseline_vec))
            input_vec = input_vec[:min_len]
            baseline_vec = baseline_vec[:min_len]
        
        improvement = np.mean(np.maximum(0, baseline_vec - input_vec))
        
        return min(1.0, improvement)

    def _detect_unknown(self, input_vector, known_patterns):
        input_vec = np.array(input_vector).flatten()
        
        if not known_patterns:
            return 1.0
        
        max_similarity = 0.0
        for pattern in known_patterns:
            pattern_vec = np.array(pattern).flatten()
            
            if len(input_vec) != len(pattern_vec):
                min_len = min(len(input_vec), len(pattern_vec))
                input_vec_norm = input_vec[:min_len]
                pattern_vec_norm = pattern_vec[:min_len]
            else:
                input_vec_norm = input_vec
                pattern_vec_norm = pattern_vec
            
            similarity = np.dot(input_vec_norm, pattern_vec_norm) / (
                np.linalg.norm(input_vec_norm) * np.linalg.norm(pattern_vec_norm) + 1e-8
            )
            max_similarity = max(max_similarity, similarity)
        
        unknown_score = 1.0 - max_similarity
        
        return min(1.0, max(0.0, unknown_score))

    def formulate_problem(self, input_vector, context=None):
        context = context or {}
        target_state = context.get("target_state", np.zeros(self.feature_dim))
        baseline_state = context.get("baseline_state", np.zeros(self.feature_dim))
        known_patterns = context.get("known_patterns", [])
        
        deviation_score = self._detect_deviation(input_vector, target_state)
        opportunity_score = self._detect_opportunity(input_vector, baseline_state)
        unknown_score = self._detect_unknown(input_vector, known_patterns)
        
        trigger_type = None
        if deviation_score > self.deviation_threshold:
            trigger_type = TriggerType.DEVIATION
        elif opportunity_score > self.opportunity_threshold:
            trigger_type = TriggerType.OPPORTUNITY
        elif unknown_score > 0.5:
            trigger_type = TriggerType.UNKNOWN
        
        if trigger_type is None:
            return None
        
        problem_id = f"problem_{len(self.problems) + 1}"
        problem = ProblemDefinition(
            problem_id=problem_id,
            trigger_type=trigger_type,
            goal=context.get("goal", "resolve situation"),
            current_state=input_vector,
            constraints={
                "deviation_severity": deviation_score,
                "time_sensitivity": context.get("time_sensitivity", 0.5),
                "resource_limit": context.get("resource_limit", 1.0),
                "risk_tolerance": context.get("risk_tolerance", 0.5)
            },
            boundaries={
                "min_confidence": context.get("min_confidence", 0.7),
                "max_execution_time": context.get("max_execution_time", 300),
                "allowed_actions": context.get("allowed_actions", []),
                "forbidden_actions": context.get("forbidden_actions", [])
            }
        )
        
        self.problems[problem_id] = problem
        self.problem_history.append({
            "problem_id": problem_id,
            "trigger_type": trigger_type.value,
            "deviation_score": deviation_score,
            "opportunity_score": opportunity_score,
            "unknown_score": unknown_score,
            "priority": problem.priority,
            "timestamp": problem.created_at
        })
        
        return problem

    def get_problem_stats(self):
        stats = {
            "total_problems": len(self.problems),
            "deviation_count": len([p for p in self.problems.values() if p.trigger_type == TriggerType.DEVIATION]),
            "opportunity_count": len([p for p in self.problems.values() if p.trigger_type == TriggerType.OPPORTUNITY]),
            "unknown_count": len([p for p in self.problems.values() if p.trigger_type == TriggerType.UNKNOWN]),
            "avg_priority": 0.0,
            "recent_problems": []
        }
        
        if self.problems:
            stats["avg_priority"] = float(np.mean([p.priority for p in self.problems.values()]))
        
        if self.problem_history:
            stats["recent_problems"] = list(self.problem_history)[-5:]
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim