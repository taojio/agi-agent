import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LearningPhase(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    CONSOLIDATE = "consolidate"
    TRANSFER = "transfer"
    ADAPT = "adapt"


class LearningStyle(Enum):
    ACTIVE = "active"
    REFLECTIVE = "reflective"
    THEORETICAL = "theoretical"
    PRAGMATIC = "pragmatic"


class LearningObjective:
    def __init__(self, id: str, name: str, description: str, 
                 priority: float = 0.5, target_proficiency: float = 0.9):
        self.id = id
        self.name = name
        self.description = description
        self.priority = priority
        self.target_proficiency = target_proficiency
        self.current_proficiency = 0.0
        self.progress_history: deque = deque(maxlen=50)
        self.created_at = np.random.randint(1000000)
        self.completed_at: Optional[int] = None

    def update_progress(self, proficiency: float):
        self.progress_history.append(self.current_proficiency)
        self.current_proficiency = min(proficiency, self.target_proficiency)
        
        if self.current_proficiency >= self.target_proficiency and self.completed_at is None:
            self.completed_at = np.random.randint(1000000)

    def is_completed(self) -> bool:
        return self.current_proficiency >= self.target_proficiency

    def get_progress_rate(self) -> float:
        if len(self.progress_history) < 2:
            return 0.0
        
        recent = list(self.progress_history)[-10:]
        if len(recent) < 2:
            return 0.0
        
        x = np.arange(len(recent))
        y = np.array(recent)
        return np.polyfit(x, y, 1)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "current_proficiency": self.current_proficiency,
            "target_proficiency": self.target_proficiency,
            "progress_rate": self.get_progress_rate(),
            "is_completed": self.is_completed(),
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }


class LearningSelfRegulator:
    def __init__(self):
        self.objectives: Dict[str, LearningObjective] = {}
        self.current_phase = LearningPhase.EXPLORE
        self.learning_style = LearningStyle.ACTIVE
        self.strategy_history: deque = deque(maxlen=200)
        self._performance_history: deque = deque(maxlen=100)
        
        self._exploration_ratio = 0.3
        self._learning_rate = 0.01
        self._consolidation_threshold = 0.8
        self._transfer_threshold = 0.9

    def add_objective(self, objective: LearningObjective):
        self.objectives[objective.id] = objective

    def create_objective(self, name: str, description: str, 
                        priority: float = 0.5, target_proficiency: float = 0.9) -> LearningObjective:
        obj_id = f"obj_{len(self.objectives) + 1}"
        objective = LearningObjective(obj_id, name, description, priority, target_proficiency)
        self.add_objective(objective)
        return objective

    def update_objective_progress(self, obj_id: str, proficiency: float):
        if obj_id in self.objectives:
            self.objectives[obj_id].update_progress(proficiency)

    def select_learning_strategy(self) -> Dict[str, Any]:
        active_objectives = [o for o in self.objectives.values() if not o.is_completed()]
        
        if not active_objectives:
            return {"strategy": "explore_new_topics", "phase": LearningPhase.EXPLORE.value}
        
        avg_proficiency = np.mean([o.current_proficiency for o in active_objectives])
        avg_progress_rate = np.mean([o.get_progress_rate() for o in active_objectives])
        
        if avg_proficiency < 0.3:
            self.current_phase = LearningPhase.EXPLORE
            strategy = {"strategy": "explore_fundamentals", "phase": LearningPhase.EXPLORE.value}
        elif avg_proficiency < self._consolidation_threshold:
            self.current_phase = LearningPhase.EXPLOIT
            strategy = {"strategy": "deepen_understanding", "phase": LearningPhase.EXPLOIT.value}
        elif avg_proficiency < self._transfer_threshold:
            self.current_phase = LearningPhase.CONSOLIDATE
            strategy = {"strategy": "consolidate_knowledge", "phase": LearningPhase.CONSOLIDATE.value}
        else:
            self.current_phase = LearningPhase.TRANSFER
            strategy = {"strategy": "transfer_knowledge", "phase": LearningPhase.TRANSFER.value}
        
        if avg_progress_rate < 0.001 and avg_proficiency > 0.5:
            self.current_phase = LearningPhase.ADAPT
            strategy = {"strategy": "adapt_learning_method", "phase": LearningPhase.ADAPT.value}
        
        strategy.update({
            "avg_proficiency": avg_proficiency,
            "avg_progress_rate": avg_progress_rate,
            "active_objectives": len(active_objectives),
            "exploration_ratio": self._exploration_ratio,
            "learning_rate": self._learning_rate
        })
        
        self.strategy_history.append(strategy)
        return strategy

    def adjust_exploration_ratio(self, performance: float):
        if performance < 0.5:
            self._exploration_ratio = min(0.5, self._exploration_ratio + 0.05)
        elif performance > 0.8:
            self._exploration_ratio = max(0.1, self._exploration_ratio - 0.05)

    def adjust_learning_rate(self, progress_rate: float):
        if progress_rate < 0.001:
            self._learning_rate = min(0.05, self._learning_rate * 1.2)
        elif progress_rate > 0.01:
            self._learning_rate = max(0.001, self._learning_rate * 0.8)

    def record_performance(self, metrics: Dict[str, float]):
        self._performance_history.append({
            "timestamp": np.random.randint(1000000),
            **metrics
        })

    def evaluate_learning_effectiveness(self) -> Dict[str, Any]:
        completed = [o for o in self.objectives.values() if o.is_completed()]
        active = [o for o in self.objectives.values() if not o.is_completed()]
        
        avg_completion_time = 0.0
        if completed:
            avg_completion_time = np.mean([o.completed_at - o.created_at for o in completed])
        
        avg_progress = np.mean([o.current_proficiency for o in self.objectives.values()]) if self.objectives else 0.0
        
        return {
            "completed_objectives": len(completed),
            "active_objectives": len(active),
            "total_objectives": len(self.objectives),
            "avg_proficiency": avg_progress,
            "avg_completion_time": avg_completion_time,
            "current_phase": self.current_phase.value,
            "learning_style": self.learning_style.value,
            "exploration_ratio": self._exploration_ratio,
            "learning_rate": self._learning_rate,
            "effectiveness_score": min(1.0, avg_progress * (1 + len(completed) * 0.1))
        }

    def generate_learning_plan(self) -> List[Dict[str, Any]]:
        plan = []
        
        sorted_objectives = sorted(
            self.objectives.values(),
            key=lambda o: (-o.priority, o.current_proficiency)
        )
        
        for obj in sorted_objectives[:5]:
            gap = obj.target_proficiency - obj.current_proficiency
            estimated_steps = gap / self._learning_rate
            
            plan.append({
                "objective_id": obj.id,
                "objective_name": obj.name,
                "current_proficiency": obj.current_proficiency,
                "target_proficiency": obj.target_proficiency,
                "priority": obj.priority,
                "estimated_steps": max(1, int(estimated_steps)),
                "phase": self.current_phase.value,
                "recommended_action": self._get_recommended_action(obj)
            })
        
        return plan

    def _get_recommended_action(self, objective: LearningObjective) -> str:
        gap = objective.target_proficiency - objective.current_proficiency
        
        if gap > 0.5:
            return "deep_dive_study"
        elif gap > 0.3:
            return "focused_practice"
        elif gap > 0.1:
            return "review_and_reinforce"
        else:
            return "consolidate_and_transfer"

    def get_objective_summary(self, obj_id: str) -> Optional[Dict[str, Any]]:
        if obj_id in self.objectives:
            return self.objectives[obj_id].to_dict()
        return None

    def get_all_objectives(self) -> List[Dict[str, Any]]:
        return [o.to_dict() for o in self.objectives.values()]

    def get_strategy_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.strategy_history)[-limit:]

    def set_learning_style(self, style: LearningStyle):
        self.learning_style = style