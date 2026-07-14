import numpy as np
from collections import deque
from enum import Enum


class ExplorationType(Enum):
    INSPECTION = "inspection"
    OPTIMIZATION = "optimization"
    PRECIPITATION = "precipitation"
    COMPLETION = "completion"


class ProactiveAction:
    def __init__(self, action_id, action_type, target, priority=0.5, estimated_value=0.0):
        self.action_id = action_id
        self.action_type = action_type
        self.target = target
        self.priority = priority
        self.estimated_value = estimated_value
        self.status = "pending"
        self.result = None
        self.execution_time = 0.0

    def to_dict(self):
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": self.target,
            "priority": self.priority,
            "estimated_value": self.estimated_value,
            "status": self.status,
            "result": self.result,
            "execution_time": self.execution_time
        }


class ActiveExplorer:
    def __init__(self):
        self.exploration_history = deque(maxlen=200)
        self.proactive_actions = deque(maxlen=50)
        self.inspection_interval = 300
        self.last_inspection_time = 0
        self.action_counter = 0

    def _generate_action_id(self):
        self.action_counter += 1
        return f"proactive_{self.action_counter:04d}"

    def _calculate_priority(self, action_type, target, context):
        base_priority = {
            ExplorationType.INSPECTION: 0.3,
            ExplorationType.OPTIMIZATION: 0.6,
            ExplorationType.PRECIPITATION: 0.4,
            ExplorationType.COMPLETION: 0.7
        }.get(action_type, 0.5)

        if "critical" in str(target).lower():
            base_priority += 0.3
        if "high" in context.get("risk_level", "normal"):
            base_priority += 0.2

        return min(1.0, base_priority)

    def schedule_inspection(self, scope="all", context=None):
        context = context or {}
        action = ProactiveAction(
            action_id=self._generate_action_id(),
            action_type=ExplorationType.INSPECTION,
            target=scope,
            priority=self._calculate_priority(ExplorationType.INSPECTION, scope, context),
            estimated_value=0.2
        )
        self.proactive_actions.append(action)

        self.exploration_history.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "target": action.target,
            "priority": action.priority,
            "status": "scheduled",
            "timestamp": np.random.randint(1000000)
        })

        return action

    def identify_optimization_opportunities(self, execution_history):
        opportunities = []
        repeated_actions = {}

        for record in execution_history:
            action_name = record.get("node_name", "")
            if action_name:
                repeated_actions[action_name] = repeated_actions.get(action_name, 0) + 1

        for action_name, count in repeated_actions.items():
            if count > 3:
                opportunities.append({
                    "action_name": action_name,
                    "frequency": count,
                    "type": "repetitive"
                })

        for record in execution_history:
            if record.get("execution_time", 0) > 1000:
                opportunities.append({
                    "action_name": record.get("node_name", ""),
                    "latency": record["execution_time"],
                    "type": "slow"
                })

        return opportunities

    def schedule_optimization(self, opportunity, context=None):
        context = context or {}
        target = opportunity.get("action_name", "unknown")

        action = ProactiveAction(
            action_id=self._generate_action_id(),
            action_type=ExplorationType.OPTIMIZATION,
            target=target,
            priority=self._calculate_priority(ExplorationType.OPTIMIZATION, target, context),
            estimated_value=opportunity.get("frequency", 1) * 0.1
        )
        self.proactive_actions.append(action)

        self.exploration_history.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "target": action.target,
            "priority": action.priority,
            "status": "scheduled",
            "opportunity": opportunity,
            "timestamp": np.random.randint(1000000)
        })

        return action

    def schedule_precipitation(self, task_result, context=None):
        context = context or {}

        action = ProactiveAction(
            action_id=self._generate_action_id(),
            action_type=ExplorationType.PRECIPITATION,
            target=task_result.get("node_name", "unknown"),
            priority=self._calculate_priority(ExplorationType.PRECIPITATION, task_result, context),
            estimated_value=0.3
        )
        self.proactive_actions.append(action)

        self.exploration_history.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "target": action.target,
            "priority": action.priority,
            "status": "scheduled",
            "timestamp": np.random.randint(1000000)
        })

        return action

    def schedule_completion(self, gap_type, context=None):
        context = context or {}

        action = ProactiveAction(
            action_id=self._generate_action_id(),
            action_type=ExplorationType.COMPLETION,
            target=gap_type,
            priority=self._calculate_priority(ExplorationType.COMPLETION, gap_type, context),
            estimated_value=0.5
        )
        self.proactive_actions.append(action)

        self.exploration_history.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "target": action.target,
            "priority": action.priority,
            "status": "scheduled",
            "timestamp": np.random.randint(1000000)
        })

        return action

    def execute_proactive_action(self, action, executor):
        action.status = "running"

        mock_results = {
            ExplorationType.INSPECTION: lambda: {"success": True, "output": "Inspection completed", "issues": []},
            ExplorationType.OPTIMIZATION: lambda: {"success": True, "output": "Optimization applied", "improvement": 0.2},
            ExplorationType.PRECIPITATION: lambda: {"success": True, "output": "Knowledge archived"},
            ExplorationType.COMPLETION: lambda: {"success": True, "output": "Gap filled"}
        }

        result_func = mock_results.get(action.action_type, lambda: {"success": True, "output": "Executed"})
        result = result_func()

        action.result = result
        action.status = "completed"
        action.execution_time = np.random.uniform(50, 500)

        self.exploration_history.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "target": action.target,
            "priority": action.priority,
            "status": "completed",
            "result": result,
            "execution_time": action.execution_time,
            "timestamp": np.random.randint(1000000)
        })

        return result

    def get_pending_actions(self, limit=10):
        pending = [a for a in self.proactive_actions if a.status == "pending"]
        pending.sort(key=lambda x: -x.priority)
        return pending[:limit]

    def get_exploration_history(self, limit=20):
        return list(self.exploration_history)[-limit:]

    def get_exploration_stats(self):
        if not self.exploration_history:
            return {"total_actions": 0, "by_type": {}, "completion_rate": 0.0}

        total = len(self.exploration_history)
        by_type = {}
        completed_count = 0

        for record in self.exploration_history:
            action_type = record.get("action_type", "unknown")
            by_type[action_type] = by_type.get(action_type, 0) + 1
            if record.get("status") == "completed":
                completed_count += 1

        return {
            "total_actions": total,
            "by_type": by_type,
            "completion_rate": completed_count / total,
            "pending_count": len([a for a in self.proactive_actions if a.status == "pending"])
        }