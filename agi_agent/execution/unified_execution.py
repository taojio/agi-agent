"""
execution/unified_execution.py - 统一执行接口

整合底层动作执行与高层任务规划，提供统一的执行入口。

职责划分：
- ActionExecutionLayer: 底层神经网络动作执行（连续动作输出）
- ActionOrchestrator: 高层任务规划与自主行动（任务分解、路径规划、执行纠错）

统一接口提供：
- execute_action: 执行单个动作（底层）
- execute_task: 执行完整任务（高层，含分解与纠错）
- get_execution_stats: 获取执行统计
"""
from typing import Any, Dict, Optional, Tuple, List
import logging

from ..config.settings import DEVICE
from .action_executor import ActionExecutionLayer
from ..autonomous_action import (
    ActionOrchestrator, TargetDecomposer, TaskNode, DecompositionLevel,
    ActionExecutor, ExecutionStatus,
)

logger = logging.getLogger(__name__)


class UnifiedExecution:
    """
    统一执行管理器
    
    整合底层神经网络动作执行与高层任务规划，提供统一的执行接口。
    
    Attributes:
        action_layer: 底层动作执行层（神经网络）
        action_orchestrator: 高层任务编排器
        action_executor: 动作执行器（任务节点级）
        target_decomposer: 目标分解器
    """

    def __init__(self, action_dim: int = 8, feature_dim: int = 16):
        self.action_layer = ActionExecutionLayer(action_dim=action_dim, feature_dim=feature_dim)
        self.action_orchestrator = ActionOrchestrator()
        self.action_executor = ActionExecutor()
        self.target_decomposer = TargetDecomposer()
        self._feature_dim = feature_dim

    def execute_action(self, feature: Any, pred_feature: Any) -> Dict[str, Any]:
        """
        执行底层神经网络动作
        
        Args:
            feature: 当前特征
            pred_feature: 预测特征
            
        Returns:
            包含动作输出和统计信息的字典
        """
        action = self.action_layer.autonomous_action(feature, pred_feature)
        stats = self.action_layer.get_action_stats()
        return {
            "action": action,
            "stats": stats,
        }

    def update_action_net(self, feature: Any, pred_feature: Any, reward: float) -> float:
        """
        更新动作网络（强化学习）
        
        Args:
            feature: 当前特征
            pred_feature: 预测特征
            reward: 奖励信号
            
        Returns:
            损失值
        """
        return self.action_layer.update_action_net(feature, pred_feature, reward)

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行完整任务（高层任务规划）
        
        Args:
            task_description: 任务描述
            context: 执行上下文
            
        Returns:
            任务执行结果
        """
        context = context or {}
        
        decomposition = self.target_decomposer.decompose(task_description)
        
        if decomposition is None:
            return {
                "success": False,
                "error": "Failed to decompose task",
            }
        
        root_node = decomposition.root_node
        
        result = self.action_executor.execute_node(root_node, context)
        
        return {
            "success": result["status"] == ExecutionStatus.COMPLETED.value,
            "task_description": task_description,
            "node_id": result["node_id"],
            "status": result["status"],
            "result": result.get("result"),
            "error": result.get("error"),
            "execution_time": result.get("execution_time"),
            "decomposition_level": root_node.level.value if root_node else "unknown",
        }

    def plan_and_execute(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        规划并执行目标
        
        Args:
            goal: 目标描述
            context: 执行上下文
            
        Returns:
            执行结果
        """
        context = context or {}
        
        plan_result = self.action_orchestrator.plan(goal, context)
        
        if not plan_result.get("success", False):
            return {
                "success": False,
                "error": plan_result.get("error", "Planning failed"),
            }
        
        execution_result = self.action_orchestrator.execute_plan(plan_result["plan_id"])
        
        return {
            "success": execution_result.get("success", False),
            "goal": goal,
            "plan_id": plan_result.get("plan_id"),
            "execution": execution_result,
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            包含动作层和任务执行统计的字典
        """
        return {
            "action_layer": self.action_layer.get_action_stats(),
            "task_executor": self.action_executor.get_execution_stats(),
            "orchestrator": self.action_orchestrator.get_stats() if hasattr(self.action_orchestrator, 'get_stats') else {},
        }

    def hardware_adapt(self, new_feature_dim: int) -> None:
        """
        硬件自适应（调整特征维度）
        
        Args:
            new_feature_dim: 新的特征维度
        """
        self.action_layer.hardware_adapt(new_feature_dim)
        self._feature_dim = new_feature_dim
        logger.info(f"UnifiedExecution hardware adapted to feature_dim={new_feature_dim}")

    def set_lr(self, lr: float) -> None:
        """
        设置学习率
        
        Args:
            lr: 学习率
        """
        self.action_layer.set_lr(lr)

    def set_exploration_noise(self, noise: float) -> None:
        """
        设置探索噪声
        
        Args:
            noise: 噪声强度 (0-0.5)
        """
        self.action_layer.set_exploration_noise(noise)

    def get_active_executions(self) -> List[Dict]:
        """
        获取当前活跃的执行任务
        
        Returns:
            活跃执行列表
        """
        return self.action_executor.get_active_executions()

    def get_execution_history(self, limit: int = 20) -> List[Dict]:
        """
        获取执行历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            执行历史列表
        """
        return self.action_executor.get_execution_history(limit)

    def compose_task_tree(self, tasks: List[Dict]) -> Optional[TaskNode]:
        """
        组合任务树
        
        Args:
            tasks: 任务列表，每个任务包含name和可选的children
            
        Returns:
            任务树根节点
        """
        return self.target_decomposer.compose(tasks)


_global_unified_execution: Optional[UnifiedExecution] = None


def get_unified_execution(action_dim: int = 8, feature_dim: int = 16) -> UnifiedExecution:
    """
    获取全局统一执行管理器实例
    
    Args:
        action_dim: 动作维度
        feature_dim: 特征维度
        
    Returns:
        UnifiedExecution 实例
    """
    global _global_unified_execution
    if _global_unified_execution is None:
        _global_unified_execution = UnifiedExecution(action_dim=action_dim, feature_dim=feature_dim)
    return _global_unified_execution