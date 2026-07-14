"""
autonomous_action/__init__.py - 自主行动体系

负责目标导向的全流程自管控，无需人工分步指令，自主完成任务拆解、路径规划、
资源调度、执行纠错、结果验收、归档沉淀的完整链路
"""
from .target_decomposer import TargetDecomposer, DecompositionLevel, TaskNode, DecompositionResult
from .path_planner import PathPlanner, ExecutionPath, ResourceAllocation, PriorityLevel
from .action_executor import ActionExecutor, ExecutionStatus, ErrorLevel, CorrectionResult
from .active_explorer import ActiveExplorer, ExplorationType, ProactiveAction
from .action_orchestrator import ActionOrchestrator

__all__ = ["TargetDecomposer", "DecompositionLevel", "TaskNode", "DecompositionResult",
           "PathPlanner", "ExecutionPath", "ResourceAllocation", "PriorityLevel",
           "ActionExecutor", "ExecutionStatus", "ErrorLevel", "CorrectionResult",
           "ActiveExplorer", "ExplorationType", "ProactiveAction",
           "ActionOrchestrator"]