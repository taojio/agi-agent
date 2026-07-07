from enum import Enum
import numpy as np
from collections import deque


class InstinctType(Enum):
    SURVIVAL = "survival"
    EXPLORE = "explore"
    AVOID = "avoid"
    APPROACH = "approach"
    REST = "rest"
    LEARN = "learn"


class InstinctAction:
    def __init__(self, instinct_type, action_func, priority=1.0, condition=None):
        self.instinct_type = instinct_type
        self.action_func = action_func
        self.priority = priority
        self.condition = condition
        self.trigger_count = 0
        self.success_count = 0

    def can_trigger(self, input_data):
        if self.condition is None:
            return True
        return self.condition(input_data)

    def execute(self, input_data):
        self.trigger_count += 1
        result = self.action_func(input_data)
        if result:
            self.success_count += 1
        return result

    def get_success_rate(self):
        if self.trigger_count == 0:
            return 0.5
        return self.success_count / self.trigger_count


class InstinctActions:
    def __init__(self):
        self.instincts = {}
        self.trigger_history = deque(maxlen=100)
        self.active_instinct = None
        
        self._register_default_instincts()

    def _register_default_instincts(self):
        self.register_instinct(
            InstinctType.SURVIVAL,
            lambda data: {"action": "preserve_state", "priority": 10.0},
            priority=10.0,
            condition=lambda d: d.get("risk_level", 0) > 0.7
        )
        
        self.register_instinct(
            InstinctType.AVOID,
            lambda data: {"action": "avoid", "target": data.get("threat_source")},
            priority=8.0,
            condition=lambda d: d.get("threat_detected", False)
        )
        
        self.register_instinct(
            InstinctType.APPROACH,
            lambda data: {"action": "approach", "target": data.get("goal_source")},
            priority=5.0,
            condition=lambda d: d.get("goal_detected", False)
        )
        
        self.register_instinct(
            InstinctType.EXPLORE,
            lambda data: {"action": "explore", "direction": data.get("novel_direction")},
            priority=3.0,
            condition=lambda d: d.get("novelty", 0) > 0.5
        )
        
        self.register_instinct(
            InstinctType.LEARN,
            lambda data: {"action": "learn", "topic": data.get("unknown_feature")},
            priority=4.0,
            condition=lambda d: d.get("unknown_detected", False)
        )
        
        self.register_instinct(
            InstinctType.REST,
            lambda data: {"action": "rest", "duration": 10},
            priority=2.0,
            condition=lambda d: d.get("fatigue", 0) > 0.8
        )

    def register_instinct(self, instinct_type, action_func, priority=1.0, condition=None):
        self.instincts[instinct_type.value] = InstinctAction(
            instinct_type=instinct_type,
            action_func=action_func,
            priority=priority,
            condition=condition
        )

    def get_triggerable_instincts(self, input_data):
        triggerable = []
        for instinct_key, instinct in self.instincts.items():
            if instinct.can_trigger(input_data):
                triggerable.append((instinct, instinct.priority))
        
        triggerable.sort(key=lambda x: -x[1])
        return triggerable

    def trigger_highest_priority(self, input_data):
        triggerable = self.get_triggerable_instincts(input_data)
        
        if not triggerable:
            return None, 0.0
        
        best_instinct, priority = triggerable[0]
        self.active_instinct = best_instinct.instinct_type
        
        result = best_instinct.execute(input_data)
        
        self.trigger_history.append({
            "instinct_type": best_instinct.instinct_type.value,
            "priority": priority,
            "success": result is not None,
            "input_summary": str(input_data)[:100],
            "timestamp": np.random.randint(1000000)
        })
        
        return result, priority

    def trigger_all(self, input_data):
        results = []
        triggerable = self.get_triggerable_instincts(input_data)
        
        for instinct, priority in triggerable:
            result = instinct.execute(input_data)
            results.append({
                "instinct_type": instinct.instinct_type.value,
                "priority": priority,
                "result": result,
                "success": result is not None
            })
        
        return results

    def get_instinct_stats(self):
        stats = {}
        for instinct_key, instinct in self.instincts.items():
            stats[instinct_key] = {
                "type": instinct.instinct_type.value,
                "priority": instinct.priority,
                "trigger_count": instinct.trigger_count,
                "success_count": instinct.success_count,
                "success_rate": instinct.get_success_rate()
            }
        return stats

    def get_recent_triggers(self, limit=10):
        return list(self.trigger_history)[-limit:]

    def clear_history(self):
        self.trigger_history.clear()