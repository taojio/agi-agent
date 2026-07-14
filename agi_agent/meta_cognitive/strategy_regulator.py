import numpy as np
from collections import deque
from enum import Enum


class ThinkingStrategy(Enum):
    FAST_ONLY = "fast_only"
    SLOW_ONLY = "slow_only"
    AUTO = "auto"
    BALANCED = "balanced"


class ActionStrategy(Enum):
    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class ResourceAllocation:
    def __init__(self, perception=0.2, cognitive=0.3, execution=0.3, evolve=0.2):
        self.perception = perception
        self.cognitive = cognitive
        self.execution = execution
        self.evolve = evolve

    def to_dict(self):
        return {
            "perception": self.perception,
            "cognitive": self.cognitive,
            "execution": self.execution,
            "evolve": self.evolve
        }


class StrategyRegulator:
    def __init__(self):
        self.thinking_strategy = ThinkingStrategy.AUTO
        self.action_strategy = ActionStrategy.STANDARD
        self.resource_allocation = ResourceAllocation()
        self.strategy_history = deque(maxlen=200)
        self.confidence_threshold = 0.85
        self.thinking_depth = 5

    def adjust_thinking_strategy(self, context):
        novelty = context.get("novelty", 0.0)
        confidence = context.get("confidence", 0.5)
        resource_load = context.get("resource_load", 0.0)

        if resource_load > 0.8:
            self.thinking_strategy = ThinkingStrategy.FAST_ONLY
            self.thinking_depth = 3
        elif novelty > 0.6 or confidence < 0.5:
            self.thinking_strategy = ThinkingStrategy.SLOW_ONLY
            self.thinking_depth = 10
        elif confidence < 0.7:
            self.thinking_strategy = ThinkingStrategy.BALANCED
            self.thinking_depth = 7
        else:
            self.thinking_strategy = ThinkingStrategy.AUTO
            self.thinking_depth = 5

        self.confidence_threshold = max(0.5, min(0.95, 0.85 - novelty * 0.2))

        self.strategy_history.append({
            "thinking_strategy": self.thinking_strategy.value,
            "thinking_depth": self.thinking_depth,
            "confidence_threshold": self.confidence_threshold,
            "context": {"novelty": novelty, "confidence": confidence, "resource_load": resource_load},
            "timestamp": np.random.randint(1000000)
        })

    def adjust_action_strategy(self, context):
        risk_level = context.get("risk_level", "low")
        confidence = context.get("confidence", 0.5)

        if risk_level == "high" or confidence < 0.5:
            self.action_strategy = ActionStrategy.CONSERVATIVE
        elif risk_level == "low" and confidence > 0.8:
            self.action_strategy = ActionStrategy.AGGRESSIVE
        else:
            self.action_strategy = ActionStrategy.STANDARD

    def adjust_resource_allocation(self, context):
        novelty = context.get("novelty", 0.0)
        resource_load = context.get("resource_load", 0.0)

        if novelty > 0.6:
            self.resource_allocation = ResourceAllocation(perception=0.3, cognitive=0.4, execution=0.2, evolve=0.1)
        elif resource_load > 0.7:
            self.resource_allocation = ResourceAllocation(perception=0.1, cognitive=0.2, execution=0.5, evolve=0.2)
        else:
            self.resource_allocation = ResourceAllocation(perception=0.2, cognitive=0.3, execution=0.3, evolve=0.2)

    def get_current_strategy(self):
        return {
            "thinking_strategy": self.thinking_strategy.value,
            "action_strategy": self.action_strategy.value,
            "thinking_depth": self.thinking_depth,
            "confidence_threshold": self.confidence_threshold,
            "resource_allocation": self.resource_allocation.to_dict()
        }

    def get_strategy_history(self, limit=20):
        return list(self.strategy_history)[-limit:]

    def get_strategy_stats(self):
        stats = {
            "strategy_changes": len(self.strategy_history),
            "current_thinking_strategy": self.thinking_strategy.value,
            "current_action_strategy": self.action_strategy.value,
            "avg_thinking_depth": 0.0
        }
        
        if self.strategy_history:
            stats["avg_thinking_depth"] = float(np.mean([s["thinking_depth"] for s in self.strategy_history]))
        
        return stats