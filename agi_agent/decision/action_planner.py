import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PathNode:
    state: Any
    action: Optional[str] = None
    parent: Optional["PathNode"] = None
    cost: float = 0.0
    heuristic: float = 0.0
    depth: int = 0

    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic


@dataclass
class ActionPlan:
    plan_id: str
    goal_description: str
    action_sequence: List[Dict[str, Any]]
    estimated_total_cost: float = 0.0
    estimated_duration_steps: int = 0
    success_probability: float = 0.5
    status: str = "planned"
    current_step_index: int = 0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)


class ActionPlanner:
    def __init__(self, feature_dim: int = 16, max_plan_length: int = 20):
        self.feature_dim = feature_dim
        self.max_plan_length = max_plan_length

        self.action_primitives: Dict[str, Dict[str, Any]] = {}
        self.plans: deque = deque(maxlen=50)
        self.current_plan: Optional[ActionPlan] = None
        self.plan_counter = 0

        self._init_default_primitives()

    def _init_default_primitives(self):
        primitives = [
            ("explore", {"cost": 1.0, "duration": 3, "type": "exploration"}),
            ("exploit", {"cost": 0.5, "duration": 1, "type": "exploitation"}),
            ("learn", {"cost": 1.5, "duration": 5, "type": "learning"}),
            ("rest", {"cost": 0.2, "duration": 2, "type": "recovery"}),
            ("adapt", {"cost": 2.0, "duration": 4, "type": "adaptation"}),
            ("predict", {"cost": 0.8, "duration": 2, "type": "cognition"}),
            ("focus", {"cost": 0.6, "duration": 1, "type": "attention"}),
            ("evolve", {"cost": 3.0, "duration": 10, "type": "evolution"}),
        ]

        for name, props in primitives:
            self.action_primitives[name] = props

    def register_primitive(self, name: str, properties: Dict[str, Any]):
        self.action_primitives[name] = properties

    def plan_actions(self, goal_description: str,
                     current_state: Dict[str, Any],
                     target_state: Dict[str, Any],
                     available_primitives: Optional[List[str]] = None) -> ActionPlan:
        primitives = available_primitives or list(self.action_primitives.keys())

        plan_steps = self._greedy_plan(current_state, target_state, primitives)

        self.plan_counter += 1
        plan = ActionPlan(
            plan_id=f"plan_{self.plan_counter}",
            goal_description=goal_description,
            action_sequence=plan_steps,
            estimated_total_cost=sum(s.get("cost", 1.0) for s in plan_steps),
            estimated_duration_steps=sum(s.get("duration", 1) for s in plan_steps),
            success_probability=self._estimate_success_probability(plan_steps, current_state)
        )

        self.current_plan = plan
        self.plans.append(plan)

        return plan

    def _greedy_plan(self, current_state: Dict[str, Any],
                      target_state: Dict[str, Any],
                      primitives: List[str]) -> List[Dict[str, Any]]:
        plan = []
        remaining_primitives = list(primitives)
        state_estimate = dict(current_state)

        for _ in range(self.max_plan_length):
            if self._goal_reached(state_estimate, target_state):
                break

            best_primitive = None
            best_progress = -float('inf')

            for prim_name in remaining_primitives:
                props = self.action_primitives.get(prim_name, {})
                progress = self._estimate_progress(state_estimate, target_state, prim_name, props)

                if progress > best_progress:
                    best_progress = progress
                    best_primitive = prim_name

            if best_primitive is None or best_progress <= 0:
                break

            props = self.action_primitives[best_primitive]
            plan.append({
                "action": best_primitive,
                "cost": props.get("cost", 1.0),
                "duration": props.get("duration", 1),
                "type": props.get("type", "general"),
                "estimated_progress": best_progress
            })

            self._simulate_step(state_estimate, best_primitive, props)

        return plan

    def _estimate_progress(self, current_state: Dict[str, Any],
                            target_state: Dict[str, Any],
                            action_name: str,
                            action_props: Dict[str, Any]) -> float:
        action_type = action_props.get("type", "general")

        progress = 0.0

        for target_key, target_val in target_state.items():
            current_val = current_state.get(target_key, 0.5)

            if action_type == "exploration" and target_key == "novelty":
                progress += abs(target_val - current_val) * 0.5
            elif action_type == "learning" and target_key == "confidence":
                progress += abs(target_val - current_val) * 0.6
            elif action_type == "recovery" and target_key == "energy":
                progress += abs(target_val - current_val) * 0.7
            elif action_type == "adaptation" and target_key == "competence":
                progress += abs(target_val - current_val) * 0.4
            else:
                progress += abs(target_val - current_val) * 0.1

        return progress - action_props.get("cost", 1.0) * 0.1

    def _simulate_step(self, state: Dict[str, Any], action_name: str,
                        action_props: Dict[str, Any]):
        action_type = action_props.get("type", "general")

        if action_type == "exploration":
            state["novelty"] = min(1.0, state.get("novelty", 0.5) + 0.1)
        elif action_type == "learning":
            state["confidence"] = min(1.0, state.get("confidence", 0.5) + 0.15)
        elif action_type == "recovery":
            state["energy"] = min(1.0, state.get("energy", 0.5) + 0.2)
        elif action_type == "adaptation":
            state["competence"] = min(1.0, state.get("competence", 0.5) + 0.1)

    def _goal_reached(self, current_state: Dict[str, Any],
                       target_state: Dict[str, Any],
                       tolerance: float = 0.1) -> bool:
        for key, target_val in target_state.items():
            current_val = current_state.get(key, 0.0)
            if abs(current_val - target_val) > tolerance:
                return False
        return True

    def _estimate_success_probability(self, plan_steps: List[Dict[str, Any]],
                                       current_state: Dict[str, Any]) -> float:
        if not plan_steps:
            return 0.0

        base_prob = 0.6
        complexity_penalty = len(plan_steps) * 0.02
        cost_penalty = sum(s.get("cost", 1.0) for s in plan_steps) * 0.01

        return max(0.1, min(0.95, base_prob - complexity_penalty - cost_penalty))

    def get_next_action(self) -> Optional[Dict[str, Any]]:
        if not self.current_plan:
            return None

        if self.current_plan.current_step_index >= len(self.current_plan.action_sequence):
            return None

        action = self.current_plan.action_sequence[self.current_plan.current_step_index]
        return action

    def advance_plan(self, success: bool = True) -> Dict[str, Any]:
        if not self.current_plan:
            return {"status": "no_plan"}

        if success:
            self.current_plan.current_step_index += 1

            if self.current_plan.current_step_index >= len(self.current_plan.action_sequence):
                self.current_plan.status = "completed"
                return {"status": "completed", "plan": self.current_plan}

            return {"status": "in_progress", "step_index": self.current_plan.current_step_index}

        return {"status": "failed_step"}

    def replan(self, current_state: Dict[str, Any],
                target_state: Dict[str, Any]) -> Optional[ActionPlan]:
        if not self.current_plan:
            return None

        new_plan = self.plan_actions(
            goal_description=self.current_plan.goal_description + "(重规划)",
            current_state=current_state,
            target_state=target_state
        )

        return new_plan

    def get_plan_stats(self) -> Dict[str, Any]:
        return {
            "total_plans": self.plan_counter,
            "current_plan_id": self.current_plan.plan_id if self.current_plan else None,
            "current_step": self.current_plan.current_step_index if self.current_plan else 0,
            "total_steps": len(self.current_plan.action_sequence) if self.current_plan else 0,
            "registered_primitives": list(self.action_primitives.keys())
        }

    def resize(self, new_feature_dim: int):
        self.feature_dim = new_feature_dim
