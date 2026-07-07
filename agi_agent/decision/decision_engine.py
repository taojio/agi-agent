import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class DecisionPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class GoalType(Enum):
    HOMEOSTATIC = "homeostatic"
    EXPLORATORY = "exploratory"
    KNOWLEDGE_SEEKING = "knowledge_seeking"
    SKILL_ACQUISITION = "skill_acquisition"
    SOCIAL = "social"
    CUSTOM = "custom"


@dataclass
class Goal:
    goal_id: str
    goal_type: GoalType
    description: str
    priority: DecisionPriority
    target_state: Dict[str, float]
    deadline_step: Optional[int] = None
    created_step: int = 0
    progress: float = 0.0
    status: str = "pending"
    sub_goals: List["Goal"] = field(default_factory=list)
    parent_goal_id: Optional[str] = None

    def get_priority_value(self) -> int:
        return self.priority.value


@dataclass
class DecisionOption:
    option_id: str
    name: str
    description: str
    estimated_cost: float = 1.0
    estimated_reward: float = 0.0
    success_probability: float = 0.5
    action_sequence: List[Any] = field(default_factory=list)
    required_resources: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "low"


class AutonomousDecisionEngine:
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim

        self.active_goals: List[Goal] = []
        self.completed_goals: deque = deque(maxlen=100)
        self.failed_goals: deque = deque(maxlen=50)
        self.decision_history: deque = deque(maxlen=200)

        self.option_generation_net = None
        self.prediction_model = None

        self.goal_counter = 0
        self.option_counter = 0

        self.current_best_option: Optional[DecisionOption] = None
        self._decision_count = 0

        self._goal_generation_templates = {
            GoalType.HOMEOSTATIC: self._generate_homeostatic_goals,
            GoalType.EXPLORATORY: self._generate_exploratory_goals,
            GoalType.KNOWLEDGE_SEEKING: self._generate_knowledge_goals,
        }

    def generate_goals(self, internal_state: Dict[str, Any],
                       external_state: Dict[str, Any],
                       current_step: int) -> List[Goal]:
        new_goals = []

        for goal_type, generator in self._goal_generation_templates.items():
            goals = generator(internal_state, external_state, current_step)
            new_goals.extend(goals)

        for goal in new_goals:
            goal.created_step = current_step

        self.active_goals.extend(new_goals)
        self._prioritize_goals()

        return new_goals

    def _flatten_state(self, state: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        flat = {}
        for key, value in state.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, (int, float, np.integer, np.floating)):
                flat[full_key] = float(value)
            elif isinstance(value, dict):
                nested = self._flatten_state(value, full_key)
                flat.update(nested)
        return flat

    def _generate_homeostatic_goals(self, internal_state: Dict[str, Any],
                                     external_state: Dict[str, Any],
                                     current_step: int) -> List[Goal]:
        goals = []

        flat_state = self._flatten_state(internal_state)

        for need_name, need_value in flat_state.items():
            baseline = 0.5
            deviation = abs(need_value - baseline)

            if deviation > 0.2:
                priority = DecisionPriority.CRITICAL if deviation > 0.4 else \
                    DecisionPriority.HIGH if deviation > 0.3 else DecisionPriority.MEDIUM

                if need_value < baseline:
                    target = min(1.0, baseline + 0.2)
                    desc = f"提升{need_name}至{target:.2f}"
                else:
                    target = max(0.0, baseline - 0.1)
                    desc = f"降低{need_name}至{target:.2f}"

                self.goal_counter += 1
                goal = Goal(
                    goal_id=f"goal_homeo_{self.goal_counter}",
                    goal_type=GoalType.HOMEOSTATIC,
                    description=desc,
                    priority=priority,
                    target_state={need_name: target}
                )
                goals.append(goal)

        return goals

    def _generate_exploratory_goals(self, internal_state: Dict[str, float],
                                     external_state: Dict[str, Any],
                                     current_step: int) -> List[Goal]:
        goals = []

        novelty = external_state.get("novelty", 0.5)
        curiosity = internal_state.get("curiosity", 0.5)

        if curiosity > 0.6 and novelty < 0.3:
            self.goal_counter += 1
            goal = Goal(
                goal_id=f"goal_explore_{self.goal_counter}",
                goal_type=GoalType.EXPLORATORY,
                description="探索新环境状态",
                priority=DecisionPriority.MEDIUM,
                target_state={"novelty": 0.7}
            )
            goals.append(goal)

        return goals

    def _generate_knowledge_goals(self, internal_state: Dict[str, float],
                                   external_state: Dict[str, Any],
                                   current_step: int) -> List[Goal]:
        goals = []

        confidence = external_state.get("confidence", 0.5)
        free_energy = external_state.get("free_energy", 1.0)

        if confidence < 0.4 and free_energy > 0.5:
            self.goal_counter += 1
            goal = Goal(
                goal_id=f"goal_knowledge_{self.goal_counter}",
                goal_type=GoalType.KNOWLEDGE_SEEKING,
                description="提升认知置信度",
                priority=DecisionPriority.LOW,
                target_state={"confidence": 0.7}
            )
            goals.append(goal)

        return goals

    def _prioritize_goals(self):
        self.active_goals.sort(key=lambda g: (g.get_priority_value(), -g.progress))

    def get_highest_priority_goal(self) -> Optional[Goal]:
        if not self.active_goals:
            return None
        return self.active_goals[0]

    def generate_options(self, goal: Goal, current_state: Dict[str, Any],
                         available_actions: List[str]) -> List[DecisionOption]:
        options = []

        for action_name in available_actions:
            self.option_counter += 1

            cost, reward, success_prob, risk_level = self._estimate_option_parameters(
                action_name, goal, current_state
            )

            option = DecisionOption(
                option_id=f"opt_{self.option_counter}",
                name=f"{action_name}_策略",
                description=f"通过{action_name}达成目标: {goal.description}",
                estimated_cost=cost,
                estimated_reward=reward,
                success_probability=success_prob,
                action_sequence=[action_name],
                required_resources=self._estimate_required_resources(action_name),
                risk_level=risk_level
            )
            options.append(option)

        self._evaluate_options(options, goal, current_state)
        self._predict_consequences(options, goal, current_state)

        return options

    def _estimate_option_parameters(self, action_name: str, goal: Goal,
                                     current_state: Dict[str, Any]) -> Tuple[float, float, float, str]:
        base_cost = 0.3
        base_reward = 0.5
        base_success = 0.6

        if goal.goal_type.value == "homeostatic":
            base_cost = 0.2
            base_reward = 0.7
            base_success = 0.8
        elif goal.goal_type.value == "exploratory":
            base_cost = 0.6
            base_reward = 0.4
            base_success = 0.5
        elif goal.goal_type.value == "knowledge_seeking":
            base_cost = 0.5
            base_reward = 0.8
            base_success = 0.6

        if "learn" in action_name.lower():
            base_cost += 0.2
            base_reward += 0.2
            base_success -= 0.1
        elif "action" in action_name.lower():
            base_cost += 0.1
            base_reward += 0.1
        elif "monitor" in action_name.lower():
            base_cost -= 0.1
            base_success += 0.1

        free_energy = current_state.get("free_energy", 0.5)
        confidence = current_state.get("confidence", 0.5)

        cost = base_cost + (1.0 - free_energy) * 0.3
        reward = base_reward + confidence * 0.2
        success_prob = base_success + confidence * 0.2 - (1.0 - free_energy) * 0.1

        risk_level = "low"
        if cost > 0.7:
            risk_level = "high"
        elif cost > 0.5:
            risk_level = "medium"

        return (
            max(0.1, min(1.0, cost)),
            max(0.1, min(1.0, reward)),
            max(0.1, min(1.0, success_prob)),
            risk_level
        )

    def _estimate_required_resources(self, action_name: str) -> Dict[str, float]:
        resources = {"computation": 0.3, "memory": 0.2, "attention": 0.4}

        if "learn" in action_name.lower():
            resources["computation"] = 0.6
            resources["memory"] = 0.5
        elif "action" in action_name.lower():
            resources["attention"] = 0.7
        elif "monitor" in action_name.lower():
            resources["computation"] = 0.2
            resources["attention"] = 0.3

        return resources

    def _evaluate_options(self, options: List[DecisionOption], goal: Goal,
                           current_state: Dict[str, Any]):
        for option in options:
            priority_weight = 1.0 - (goal.get_priority_value() / 4.0)
            cost_efficiency = option.estimated_reward / max(option.estimated_cost, 0.01)

            risk_penalty = 0.0
            if option.risk_level == "high":
                risk_penalty = 0.2
            elif option.risk_level == "medium":
                risk_penalty = 0.1

            consequence_bonus = getattr(option, 'consequence_score', 0.0) * 0.1

            option.score = (
                option.estimated_reward * 0.35 +
                option.success_probability * 0.25 +
                cost_efficiency * 0.2 +
                priority_weight * 0.1 +
                consequence_bonus -
                risk_penalty
            )

        options.sort(key=lambda o: o.score, reverse=True)

    def _predict_consequences(self, options: List[DecisionOption], goal: Goal,
                               current_state: Dict[str, Any]):
        for option in options:
            consequence_score = 0.0
            long_term_effect = 0.0

            if option.risk_level == "low":
                long_term_effect = 0.1
            elif option.risk_level == "high":
                long_term_effect = -0.1

            if goal.goal_type.value == "homeostatic":
                long_term_effect += 0.1

            if "learn" in option.description.lower():
                long_term_effect += 0.2
                consequence_score += 0.3

            if option.success_probability > 0.7:
                consequence_score += 0.2

            option.consequence_score = consequence_score
            option.long_term_effect = long_term_effect

    def select_best_option(self, options: List[DecisionOption]) -> Optional[DecisionOption]:
        if not options:
            return None

        best = options[0]
        self.current_best_option = best
        self._decision_count += 1

        self.decision_history.append({
            "option_id": best.option_id,
            "name": best.name,
            "score": best.score,
            "timestamp_step": self._decision_count
        })

        return best

    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal_description = context.get("goal", "unknown")
        current_state = context.get("current_state", {})
        available_options = context.get("available_options", [])
        expected_utility = context.get("expected_utility", 0.5)
        resource_estimate = context.get("resource_estimate", {})
        risk_level = context.get("risk_level", "low")

        if not available_options:
            return {"decision": "no_action", "reason": "No options available", "confidence": 0.3}

        goal = Goal(
            goal_id=f"goal_decision_{self._decision_count}",
            goal_type=GoalType.CUSTOM,
            description=goal_description,
            priority=DecisionPriority.HIGH if risk_level == "high" else DecisionPriority.MEDIUM,
            target_state={}
        )

        action_names = [opt.get("action", "execute") for opt in available_options]
        options = self.generate_options(goal, current_state, action_names)

        if not options:
            return {"decision": "no_action", "reason": "Failed to generate options", "confidence": 0.2}

        best_option = self.select_best_option(options)

        if best_option and best_option.score > 0.4:
            decision = "execute"
            confidence = min(1.0, best_option.score + expected_utility * 0.3)
        else:
            decision = "reconsider"
            confidence = 0.3

        return {
            "decision": decision,
            "option": best_option.name if best_option else "none",
            "score": best_option.score if best_option else 0.0,
            "confidence": confidence,
            "risk_level": risk_level,
            "resource_estimate": resource_estimate,
            "reason": f"Selected {decision} based on score {best_option.score:.2f}" if best_option else "No valid option"
        }

    def execute_decision(self, option: DecisionOption) -> Dict[str, Any]:
        if not option:
            return {"success": False, "reason": "No option provided"}

        execution_result = {
            "success": True,
            "option_id": option.option_id,
            "option_name": option.name,
            "start_time": time.time(),
            "resources_used": {},
            "consequence_summary": {}
        }

        execution_result["resources_used"] = {k: v * 0.5 for k, v in option.required_resources.items()}

        if option.risk_level == "high":
            success_chance = option.success_probability * 0.8
        else:
            success_chance = option.success_probability

        if np.random.random() < success_chance:
            execution_result["success"] = True
            execution_result["outcome"] = "completed"
            execution_result["consequence_summary"] = {
                "immediate_reward": option.estimated_reward,
                "long_term_effect": getattr(option, 'long_term_effect', 0.0),
                "learning_value": 0.3 if "learn" in option.description.lower() else 0.1
            }
        else:
            execution_result["success"] = False
            execution_result["outcome"] = "failed"
            execution_result["consequence_summary"] = {
                "immediate_reward": 0.0,
                "long_term_effect": -0.1,
                "learning_value": 0.2
            }

        execution_result["end_time"] = time.time()
        execution_result["duration"] = execution_result["end_time"] - execution_result["start_time"]

        return execution_result

    def update_goal_progress(self, goal_id: str, progress_delta: float,
                              current_step: int) -> Dict[str, Any]:
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.progress = max(0.0, min(1.0, goal.progress + progress_delta))

                if goal.progress >= 1.0:
                    goal.status = "completed"
                    self.completed_goals.append(goal)
                    self.active_goals.remove(goal)
                    return {"status": "completed", "goal": goal}

                return {"status": "in_progress", "goal": goal}

        return {"status": "not_found"}

    def fail_goal(self, goal_id: str, reason: str = "", current_step: int = 0):
        for goal in list(self.active_goals):
            if goal.goal_id == goal_id:
                goal.status = "failed"
                self.failed_goals.append({"goal": goal, "reason": reason, "step": current_step})
                self.active_goals.remove(goal)
                return True
        return False

    def get_decision_stats(self) -> Dict[str, Any]:
        return {
            "total_decisions": self._decision_count,
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "failed_goals": len(self.failed_goals),
            "highest_priority_goal": self.get_highest_priority_goal().description if self.get_highest_priority_goal() else None,
            "current_option": self.current_best_option.name if self.current_best_option else None
        }

    def resize(self, new_feature_dim: int):
        self.feature_dim = new_feature_dim
