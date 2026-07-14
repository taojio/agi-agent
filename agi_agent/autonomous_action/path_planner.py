import numpy as np
from collections import deque
from enum import Enum
from .target_decomposer import TaskNode, DecompositionLevel


class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ResourceAllocation:
    def __init__(self, cpu=0.2, memory=0.2, network=0.1, gpu=0.0):
        self.cpu = cpu
        self.memory = memory
        self.network = network
        self.gpu = gpu

    def total(self):
        return self.cpu + self.memory + self.network + self.gpu

    def to_dict(self):
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "network": self.network,
            "gpu": self.gpu,
            "total": self.total()
        }


class ExecutionPath:
    def __init__(self, path_id):
        self.path_id = path_id
        self.steps = []
        self.dependencies = {}
        self.parallel_groups = []
        self.estimated_time = 0.0
        self.resource_requirement = ResourceAllocation()
        self.priority = PriorityLevel.NORMAL
        self.risk_score = 0.0

    def add_step(self, node_id, position, requires=None):
        self.steps.append({
            "node_id": node_id,
            "position": position,
            "requires": requires or []
        })
        if requires:
            self.dependencies[node_id] = requires

    def to_dict(self):
        return {
            "path_id": self.path_id,
            "steps": self.steps,
            "parallel_groups": self.parallel_groups,
            "estimated_time": self.estimated_time,
            "resource_requirement": self.resource_requirement.to_dict(),
            "priority": self.priority.value,
            "risk_score": self.risk_score
        }


class PathPlanner:
    def __init__(self):
        self.path_history = deque(maxlen=200)
        self.path_counter = 0

    def _generate_path_id(self):
        self.path_counter += 1
        return f"path_{self.path_counter:04d}"

    def _calculate_priority(self, goal_definition):
        urgency = goal_definition.get("urgency", 0.5)
        importance = goal_definition.get("importance", 0.5)

        priority_score = (urgency + importance) / 2

        if priority_score >= 0.8:
            return PriorityLevel.CRITICAL
        elif priority_score >= 0.6:
            return PriorityLevel.HIGH
        elif priority_score >= 0.4:
            return PriorityLevel.NORMAL
        else:
            return PriorityLevel.LOW

    def _estimate_resource_requirement(self, nodes):
        action_count = len([n for n in nodes.values() if n.level == DecompositionLevel.ACTION])
        step_count = len([n for n in nodes.values() if n.level == DecompositionLevel.STEP])

        base_cpu = 0.1 + min(0.4, action_count * 0.05)
        base_memory = 0.1 + min(0.3, step_count * 0.03)

        return ResourceAllocation(cpu=base_cpu, memory=base_memory, network=0.1)

    def _build_dag(self, nodes):
        in_degree = {node_id: 0 for node_id in nodes}
        adjacency = {node_id: [] for node_id in nodes}

        for node_id, node in nodes.items():
            for dep in node.dependencies:
                if dep in nodes:
                    adjacency[dep].append(node_id)
                    in_degree[node_id] += 1

            if node.parent_id and node.parent_id in nodes:
                parent_node = nodes[node.parent_id]
                for sibling in parent_node.children:
                    if sibling.node_id != node_id:
                        if sibling.node_id not in node.dependencies:
                            adjacency[node_id].append(sibling.node_id)
                            in_degree[sibling.node_id] += 1

        return in_degree, adjacency

    def _topological_sort(self, nodes):
        in_degree, adjacency = self._build_dag(nodes)

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        sorted_order = []
        parallel_groups = []
        current_group = []

        while queue:
            level_size = len(queue)
            current_group = []

            for _ in range(level_size):
                node_id = queue.popleft()
                current_group.append(node_id)
                sorted_order.append(node_id)

                for neighbor in adjacency[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if current_group:
                parallel_groups.append(current_group)

        if len(sorted_order) != len(nodes):
            raise ValueError("DAG contains cycles")

        return sorted_order, parallel_groups

    def _calculate_risk_score(self, nodes):
        risk_score = 0.0
        action_nodes = [n for n in nodes.values() if n.level == DecompositionLevel.ACTION]

        for node in action_nodes:
            if "delete" in node.name.lower() or "remove" in node.name.lower():
                risk_score += 0.3
            if "system" in node.name.lower() or "root" in node.name.lower():
                risk_score += 0.2
            if node.constraints.get("requires_confirmation", False):
                risk_score += 0.1

        return min(1.0, risk_score)

    def plan_path(self, decomposition_result):
        if not decomposition_result.validate():
            return None

        nodes = decomposition_result.all_nodes
        sorted_order, parallel_groups = self._topological_sort(nodes)

        execution_path = ExecutionPath(self._generate_path_id())
        execution_path.parallel_groups = parallel_groups
        execution_path.resource_requirement = self._estimate_resource_requirement(nodes)
        execution_path.priority = self._calculate_priority(decomposition_result.root_node.constraints)
        execution_path.risk_score = self._calculate_risk_score(nodes)

        for idx, node_id in enumerate(sorted_order):
            node = nodes[node_id]
            execution_path.add_step(node_id, idx, node.dependencies)

        base_time_per_action = 0.5
        action_count = len([n for n in nodes.values() if n.level == DecompositionLevel.ACTION])
        parallel_factor = min(3, len(parallel_groups))

        execution_path.estimated_time = (action_count * base_time_per_action) / parallel_factor

        self.path_history.append({
            "path_id": execution_path.path_id,
            "goal_name": decomposition_result.root_node.name,
            "steps_count": len(execution_path.steps),
            "parallel_groups": len(parallel_groups),
            "estimated_time": execution_path.estimated_time,
            "priority": execution_path.priority.value,
            "risk_score": execution_path.risk_score,
            "timestamp": np.random.randint(1000000)
        })

        return execution_path

    def optimize_path(self, execution_path, optimization_goal="efficiency"):
        if optimization_goal == "efficiency":
            self._optimize_for_efficiency(execution_path)
        elif optimization_goal == "stability":
            self._optimize_for_stability(execution_path)
        elif optimization_goal == "cost":
            self._optimize_for_cost(execution_path)

        return execution_path

    def _optimize_for_efficiency(self, path):
        if path.parallel_groups:
            path.estimated_time *= 0.85

    def _optimize_for_stability(self, path):
        path.risk_score = max(0.0, path.risk_score - 0.1)
        path.estimated_time *= 1.2

    def _optimize_for_cost(self, path):
        resources = path.resource_requirement
        resources.cpu = max(0.05, resources.cpu * 0.7)
        resources.memory = max(0.05, resources.memory * 0.7)
        path.resource_requirement = resources

    def get_path_history(self, limit=20):
        return list(self.path_history)[-limit:]

    def get_path_stats(self):
        if not self.path_history:
            return {"total_paths": 0, "avg_steps": 0, "avg_time": 0.0}

        total = len(self.path_history)
        avg_steps = np.mean([h["steps_count"] for h in self.path_history])
        avg_time = np.mean([h["estimated_time"] for h in self.path_history])

        return {
            "total_paths": total,
            "avg_steps": avg_steps,
            "avg_time": avg_time
        }