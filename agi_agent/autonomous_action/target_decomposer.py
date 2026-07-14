import numpy as np
from collections import deque
from enum import Enum


class DecompositionLevel(Enum):
    GOAL = "goal"
    TASK = "task"
    STEP = "step"
    ACTION = "action"


class TaskNode:
    def __init__(self, node_id, level, name, description="", parent_id=None,
                 acceptance_criteria=None, constraints=None, dependencies=None):
        self.node_id = node_id
        self.level = level
        self.name = name
        self.description = description
        self.parent_id = parent_id
        self.acceptance_criteria = acceptance_criteria or {}
        self.constraints = constraints or {}
        self.dependencies = dependencies or []
        self.children = []
        self.status = "pending"
        self.result = None
        self.error = None
        self.execution_time = 0.0
        self.resource_cost = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "level": self.level.value,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "acceptance_criteria": self.acceptance_criteria,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "children_count": len(self.children),
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "resource_cost": self.resource_cost
        }


class DecompositionResult:
    def __init__(self, root_node, all_nodes, decomposition_depth):
        self.root_node = root_node
        self.all_nodes = all_nodes
        self.decomposition_depth = decomposition_depth
        self.validation_errors = []

    def validate(self):
        self.validation_errors = []
        for node in self.all_nodes.values():
            if node.level == DecompositionLevel.ACTION and not node.children:
                if not node.acceptance_criteria:
                    self.validation_errors.append(
                        f"Action node {node.node_id} missing acceptance criteria"
                    )
            if node.dependencies:
                for dep in node.dependencies:
                    if dep not in self.all_nodes:
                        self.validation_errors.append(
                            f"Node {node.node_id} has missing dependency {dep}"
                        )
        return len(self.validation_errors) == 0

    def get_nodes_by_level(self, level):
        return [n for n in self.all_nodes.values() if n.level == level]

    def to_dict(self):
        return {
            "root_node": self.root_node.to_dict() if self.root_node else None,
            "total_nodes": len(self.all_nodes),
            "decomposition_depth": self.decomposition_depth,
            "validation_errors": self.validation_errors,
            "is_valid": self.validate()
        }


class TargetDecomposer:
    def __init__(self):
        self.decomposition_history = deque(maxlen=200)
        self.decomposition_rules = {
            DecompositionLevel.GOAL: self._decompose_goal,
            DecompositionLevel.TASK: self._decompose_task,
            DecompositionLevel.STEP: self._decompose_step,
            DecompositionLevel.ACTION: self._decompose_action
        }
        self.max_depth = 4
        self.node_counter = 0

    def _generate_node_id(self):
        self.node_counter += 1
        return f"node_{self.node_counter:04d}"

    def _decompose_goal(self, goal, parent_id=None):
        goal_node = TaskNode(
            node_id=self._generate_node_id(),
            level=DecompositionLevel.GOAL,
            name=goal.get("name", "Unnamed Goal"),
            description=goal.get("description", ""),
            parent_id=parent_id,
            acceptance_criteria=goal.get("acceptance_criteria", {}),
            constraints=goal.get("constraints", {}),
            dependencies=goal.get("dependencies", [])
        )

        tasks = goal.get("tasks", [])
        for task_def in tasks:
            task_node = self._decompose_task(task_def, goal_node.node_id)
            goal_node.add_child(task_node)

        return goal_node

    def _decompose_task(self, task, parent_id=None):
        task_node = TaskNode(
            node_id=self._generate_node_id(),
            level=DecompositionLevel.TASK,
            name=task.get("name", "Unnamed Task"),
            description=task.get("description", ""),
            parent_id=parent_id,
            acceptance_criteria=task.get("acceptance_criteria", {}),
            constraints=task.get("constraints", {}),
            dependencies=task.get("dependencies", [])
        )

        steps = task.get("steps", [])
        if not steps:
            steps = self._auto_generate_steps(task)

        for step_def in steps:
            step_node = self._decompose_step(step_def, task_node.node_id)
            task_node.add_child(step_node)

        return task_node

    def _decompose_step(self, step, parent_id=None):
        step_node = TaskNode(
            node_id=self._generate_node_id(),
            level=DecompositionLevel.STEP,
            name=step.get("name", "Unnamed Step"),
            description=step.get("description", ""),
            parent_id=parent_id,
            acceptance_criteria=step.get("acceptance_criteria", {}),
            constraints=step.get("constraints", {}),
            dependencies=step.get("dependencies", [])
        )

        actions = step.get("actions", [])
        if not actions:
            actions = self._auto_generate_actions(step)

        for action_def in actions:
            action_node = self._decompose_action(action_def, step_node.node_id)
            step_node.add_child(action_node)

        return step_node

    def _decompose_action(self, action, parent_id=None):
        action_node = TaskNode(
            node_id=self._generate_node_id(),
            level=DecompositionLevel.ACTION,
            name=action.get("name", "Unnamed Action"),
            description=action.get("description", ""),
            parent_id=parent_id,
            acceptance_criteria=action.get("acceptance_criteria", {
                "success": True,
                "output_type": "any"
            }),
            constraints=action.get("constraints", {}),
            dependencies=action.get("dependencies", [])
        )

        return action_node

    def _auto_generate_steps(self, task):
        task_name = task.get("name", "")
        steps = []

        if "clean" in task_name.lower() or "clear" in task_name.lower():
            steps = [
                {"name": "Scan and identify target items", "description": "Scan the environment for items to clean"},
                {"name": "Process identified items", "description": "Process each identified item"},
                {"name": "Verify completion", "description": "Verify all items have been cleaned"}
            ]
        elif "optimize" in task_name.lower():
            steps = [
                {"name": "Analyze current state", "description": "Analyze current performance and identify bottlenecks"},
                {"name": "Generate optimization plan", "description": "Create optimization strategies"},
                {"name": "Apply optimizations", "description": "Implement optimization measures"},
                {"name": "Verify improvements", "description": "Verify optimization results"}
            ]
        elif "backup" in task_name.lower() or "save" in task_name.lower():
            steps = [
                {"name": "Prepare backup target", "description": "Prepare storage location for backup"},
                {"name": "Execute backup", "description": "Perform the backup operation"},
                {"name": "Verify backup integrity", "description": "Verify backup file integrity"}
            ]
        else:
            steps = [
                {"name": "Preparation", "description": "Prepare resources and environment"},
                {"name": "Execution", "description": "Execute the main operation"},
                {"name": "Verification", "description": "Verify execution results"}
            ]

        return steps

    def _auto_generate_actions(self, step):
        step_name = step.get("name", "")
        actions = []

        if "scan" in step_name.lower():
            actions = [
                {"name": "execute_scan", "description": "Execute scanning operation"},
                {"name": "collect_results", "description": "Collect scan results"}
            ]
        elif "process" in step_name.lower():
            actions = [
                {"name": "validate_input", "description": "Validate input data"},
                {"name": "execute_process", "description": "Execute processing logic"},
                {"name": "output_result", "description": "Output processed result"}
            ]
        elif "verify" in step_name.lower():
            actions = [
                {"name": "check_criteria", "description": "Check against acceptance criteria"},
                {"name": "generate_report", "description": "Generate verification report"}
            ]
        else:
            actions = [
                {"name": "execute_action", "description": "Execute the action"},
                {"name": "check_result", "description": "Check execution result"}
            ]

        return actions

    def decompose(self, goal_definition, depth=None):
        depth = depth or self.max_depth

        root_node = self._decompose_goal(goal_definition)

        all_nodes = {}
        queue = deque([root_node])
        while queue:
            node = queue.popleft()
            all_nodes[node.node_id] = node
            queue.extend(node.children)

        result = DecompositionResult(root_node, all_nodes, depth)

        self.decomposition_history.append({
            "goal_name": goal_definition.get("name", "unknown"),
            "total_nodes": len(all_nodes),
            "depth": depth,
            "is_valid": result.validate(),
            "timestamp": np.random.randint(1000000)
        })

        return result

    def refine_decomposition(self, result, node_id, new_definition):
        if node_id not in result.all_nodes:
            return None

        node = result.all_nodes[node_id]
        old_status = node.status

        if node.level == DecompositionLevel.TASK:
            new_node = self._decompose_task(new_definition, node.parent_id)
        elif node.level == DecompositionLevel.STEP:
            new_node = self._decompose_step(new_definition, node.parent_id)
        elif node.level == DecompositionLevel.ACTION:
            new_node = self._decompose_action(new_definition, node.parent_id)
        else:
            return None

        parent_id = node.parent_id
        if parent_id in result.all_nodes:
            parent_node = result.all_nodes[parent_id]
            parent_node.children = [
                c for c in parent_node.children if c.node_id != node_id
            ]
            parent_node.add_child(new_node)

        for old_child_id in [c.node_id for c in node.children]:
            if old_child_id in result.all_nodes:
                del result.all_nodes[old_child_id]

        result.all_nodes[new_node.node_id] = new_node
        queue = deque([new_node])
        while queue:
            child = queue.popleft()
            result.all_nodes[child.node_id] = child
            queue.extend(child.children)

        new_node.status = old_status

        return result

    def get_decomposition_history(self, limit=20):
        return list(self.decomposition_history)[-limit:]

    def get_decomposition_stats(self):
        if not self.decomposition_history:
            return {"total_decompositions": 0, "avg_nodes": 0, "valid_rate": 0.0}

        total = len(self.decomposition_history)
        avg_nodes = np.mean([h["total_nodes"] for h in self.decomposition_history])
        valid_count = sum(1 for h in self.decomposition_history if h["is_valid"])

        return {
            "total_decompositions": total,
            "avg_nodes": avg_nodes,
            "valid_rate": valid_count / total
        }