import numpy as np
from collections import deque
from .target_decomposer import TargetDecomposer, DecompositionLevel
from .path_planner import PathPlanner, PriorityLevel
from .action_executor import ActionExecutor, ExecutionStatus
from .active_explorer import ActiveExplorer, ExplorationType


class ActionOrchestrator:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.decomposer = TargetDecomposer()
        self.planner = PathPlanner()
        self.executor = ActionExecutor()
        self.explorer = ActiveExplorer()

        self.action_history = deque(maxlen=200)
        self.active_goals = {}
        self.goal_counter = 0

    def resize(self, new_dim):
        """适配新的特征维度，更新内部组件的 feature_dim。

        参考 ThinkingOrchestrator.resize 实现。由于内部 decomposer、planner、
        executor、explorer 不一定都具备 feature_dim 属性或 resize 方法，
        这里采用安全的 hasattr 检查，仅在属性/方法存在时进行更新。
        """
        self.feature_dim = new_dim
        for component in (self.decomposer, self.planner, self.executor, self.explorer):
            if hasattr(component, "feature_dim"):
                component.feature_dim = new_dim
            if hasattr(component, "resize"):
                try:
                    component.resize(new_dim)
                except TypeError:
                    # 某些组件 resize 方法签名可能不同，忽略以保持健壮性
                    pass

    def _generate_goal_id(self):
        self.goal_counter += 1
        return f"goal_{self.goal_counter:04d}"

    def execute_goal(self, goal_definition):
        goal_id = self._generate_goal_id()

        decomposition = self.decomposer.decompose(goal_definition)
        if not decomposition.validate():
            return {
                "goal_id": goal_id,
                "status": "failed",
                "error": f"Decomposition validation failed: {decomposition.validation_errors}"
            }

        execution_path = self.planner.plan_path(decomposition)
        if execution_path is None:
            return {
                "goal_id": goal_id,
                "status": "failed",
                "error": "Path planning failed"
            }

        optimization_goal = goal_definition.get("optimization", "efficiency")
        execution_path = self.planner.optimize_path(execution_path, optimization_goal)

        self.active_goals[goal_id] = {
            "goal_id": goal_id,
            "decomposition": decomposition,
            "execution_path": execution_path,
            "status": "running",
            "start_time": np.random.randint(1000000)
        }

        execution_result = self.executor.execute_node(decomposition.root_node)

        self.active_goals[goal_id]["status"] = execution_result["status"]
        self.active_goals[goal_id]["end_time"] = np.random.randint(1000000)

        report = self._generate_execution_report(goal_id, decomposition, execution_result)

        self.action_history.append({
            "goal_id": goal_id,
            "goal_name": goal_definition.get("name", "unknown"),
            "status": execution_result["status"],
            "total_nodes": len(decomposition.all_nodes),
            "execution_time": execution_result.get("execution_time", 0),
            "report": report,
            "timestamp": np.random.randint(1000000)
        })

        if execution_result["status"] == ExecutionStatus.COMPLETED.value:
            self.explorer.schedule_precipitation(execution_result)

        return {
            "goal_id": goal_id,
            "status": execution_result["status"],
            "result": execution_result["result"],
            "report": report,
            "execution_path": execution_path.to_dict()
        }

    def _generate_execution_report(self, goal_id, decomposition, execution_result):
        nodes = decomposition.all_nodes
        completed_count = sum(1 for n in nodes.values() if n.status == ExecutionStatus.COMPLETED.value)
        failed_count = sum(1 for n in nodes.values() if n.status == ExecutionStatus.FAILED.value)

        return {
            "goal_id": goal_id,
            "goal_name": decomposition.root_node.name,
            "total_nodes": len(nodes),
            "completed_nodes": completed_count,
            "failed_nodes": failed_count,
            "success_rate": completed_count / len(nodes) if nodes else 0.0,
            "execution_time": execution_result.get("execution_time", 0),
            "nodes_by_level": {
                level.value: len(decomposition.get_nodes_by_level(level))
                for level in DecompositionLevel
            },
            "errors": [n.error for n in nodes.values() if n.error]
        }

    def run_proactive_actions(self):
        pending_actions = self.explorer.get_pending_actions()
        results = []

        for action in pending_actions:
            result = self.explorer.execute_proactive_action(action, self.executor)
            results.append({
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "target": action.target,
                "result": result
            })

        return results

    def schedule_inspection(self, scope="all"):
        return self.explorer.schedule_inspection(scope)

    def schedule_optimization(self, opportunity):
        return self.explorer.schedule_optimization(opportunity)

    def schedule_completion(self, gap_type):
        return self.explorer.schedule_completion(gap_type)

    def identify_optimization_opportunities(self):
        history = self.executor.get_execution_history()
        return self.explorer.identify_optimization_opportunities(history)

    def get_active_goals(self):
        return list(self.active_goals.values())

    def get_goal_status(self, goal_id):
        return self.active_goals.get(goal_id)

    def cancel_goal(self, goal_id):
        if goal_id in self.active_goals:
            self.active_goals[goal_id]["status"] = ExecutionStatus.CANCELLED.value
            return True
        return False

    def pause_goal(self, goal_id):
        if goal_id in self.active_goals:
            self.active_goals[goal_id]["status"] = ExecutionStatus.PAUSED.value
            return True
        return False

    def resume_goal(self, goal_id):
        if goal_id in self.active_goals:
            self.active_goals[goal_id]["status"] = ExecutionStatus.RUNNING.value
            return True
        return False

    def get_action_history(self, limit=20):
        return list(self.action_history)[-limit:]

    def get_all_stats(self):
        return {
            "decomposition": self.decomposer.get_decomposition_stats(),
            "path_planning": self.planner.get_path_stats(),
            "execution": self.executor.get_execution_stats(),
            "exploration": self.explorer.get_exploration_stats(),
            "active_goals": len(self.active_goals),
            "total_actions": len(self.action_history)
        }

    def get_action_stats(self):
        return self.get_all_stats()