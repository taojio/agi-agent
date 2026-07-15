"""
unified_reasoning.py - 统一推理引擎

整合 cognitive/ 和 deliberative/ 的推理功能，提供统一的推理接口：
- 快速推理（系统1）：基于脉冲神经网络和预测编码
- 深度推理（系统2）：基于逻辑演绎、因果推理、假设生成
- 思维链推理：多步推理链
- 神经符号推理：神经网络与符号逻辑的结合

使用方式：
    from agi_agent.cognitive import UnifiedReasoningEngine
    
    reasoner = UnifiedReasoningEngine(feature_dim=32)
    result = reasoner.reason(input_vector, goal="solve problem")
"""
import time
import numpy as np
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from .dual_system import System1, System2, DualSystemCognition
from .inference_engine import CognitiveInferenceLayer
from .causal_reasoning import CausalReasoningEngine
from .orchestrator import UnifiedCognitiveOrchestrator
from ..deliberative.thinking_orchestrator import ThinkingOrchestrator, ThinkingMode
from ..deliberative.autonomous_thinking_engine import AutonomousThinkingEngine
from ..deliberative.neuro_symbolic_reasoner import NeuroSymbolicReasoner
from ..deliberative.problem_formulator import ProblemFormulator
from ..deliberative.hypothesis_generator import HypothesisGenerator
from ..deliberative.logical_deductor import LogicalDeductor
from ..deliberative.simulation_engine import SimulationEngine


class ReasoningMode(Enum):
    FAST = "fast"
    SLOW = "slow"
    AUTO = "auto"
    CHAIN = "chain"
    SYMBOLIC = "symbolic"


class ReasoningResult:
    """
    推理结果封装
    
    Attributes:
        mode: 推理模式
        response: 推理响应
        confidence: 置信度
        steps: 推理步数
        elapsed_time: 耗时（秒）
        converged: 是否收敛
        thought_chain: 思维链（如果是CHAIN模式）
        symbolic_trace: 符号推理轨迹（如果是SYMBOLIC模式）
    """
    
    def __init__(self):
        self.mode: str = ""
        self.response: Dict[str, Any] = {}
        self.confidence: float = 0.0
        self.steps: int = 0
        self.elapsed_time: float = 0.0
        self.converged: bool = False
        self.thought_chain: List[Dict[str, Any]] = []
        self.symbolic_trace: List[Dict[str, Any]] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "response": self.response,
            "confidence": self.confidence,
            "steps": self.steps,
            "elapsed_time": self.elapsed_time,
            "converged": self.converged,
            "thought_chain": self.thought_chain,
            "symbolic_trace": self.symbolic_trace,
        }


class UnifiedReasoningEngine:
    """
    统一推理引擎
    
    整合快速推理、深度推理、思维链推理和神经符号推理，
    根据任务复杂度和置信度自动选择最优推理策略。
    
    Attributes:
        feature_dim: 特征维度
        mode: 当前推理模式
        confidence_threshold: 置信度阈值
        dual_system: 双系统认知（系统1+系统2）
        inference_layer: 认知推理层
        causal_engine: 因果推理引擎
        thinking_orchestrator: 思考编排器
        autonomous_engine: 自主思考引擎
        symbolic_reasoner: 神经符号推理器
        problem_formulator: 问题公式化器
        hypothesis_generator: 假设生成器
        logical_deductor: 逻辑演绎器
        simulation_engine: 模拟引擎
    """
    
    def __init__(self, feature_dim: int = 16, mode: str = "auto"):
        self.feature_dim = feature_dim
        self.mode = ReasoningMode.AUTO
        self.confidence_threshold = 0.85
        
        self.dual_system = DualSystemCognition(feature_dim)
        self.inference_layer = CognitiveInferenceLayer(feat_dim=feature_dim)
        self.causal_engine = CausalReasoningEngine()
        
        self.thinking_orchestrator = ThinkingOrchestrator(feature_dim=feature_dim)
        self.autonomous_engine = AutonomousThinkingEngine(feature_dim=feature_dim)
        self.symbolic_reasoner = NeuroSymbolicReasoner()
        self.problem_formulator = ProblemFormulator()
        self.hypothesis_generator = HypothesisGenerator()
        self.logical_deductor = LogicalDeductor()
        self.simulation_engine = SimulationEngine()
        
        self.set_mode(mode)
        
        self.stats = {
            "total_inferences": 0,
            "fast_inferences": 0,
            "slow_inferences": 0,
            "chain_inferences": 0,
            "symbolic_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "success_rate": 0.0,
            "converged_count": 0,
        }
    
    def set_mode(self, mode: Union[str, ReasoningMode]) -> None:
        """
        设置推理模式
        
        Args:
            mode: 推理模式 (fast | slow | auto | chain | symbolic)
        """
        if isinstance(mode, ReasoningMode):
            self.mode = mode
        elif isinstance(mode, str):
            mode_map = {
                "fast": ReasoningMode.FAST,
                "slow": ReasoningMode.SLOW,
                "auto": ReasoningMode.AUTO,
                "chain": ReasoningMode.CHAIN,
                "symbolic": ReasoningMode.SYMBOLIC,
            }
            self.mode = mode_map.get(mode.lower(), ReasoningMode.AUTO)
        
        if self.mode == ReasoningMode.FAST:
            self.thinking_orchestrator.set_mode("fast")
        elif self.mode == ReasoningMode.SLOW:
            self.thinking_orchestrator.set_mode("slow")
        else:
            self.thinking_orchestrator.set_mode("auto")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        设置置信度阈值
        
        Args:
            threshold: 置信度阈值 (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.thinking_orchestrator.set_confidence_threshold(self.confidence_threshold)
    
    def reason(self, input_vector: Any, goal: Optional[str] = None, 
               context: Optional[Dict[str, Any]] = None, 
               max_steps: int = 5) -> ReasoningResult:
        """
        执行推理
        
        Args:
            input_vector: 输入向量
            goal: 推理目标
            context: 上下文信息
            max_steps: 最大推理步数（仅CHAIN模式有效）
        
        Returns:
            ReasoningResult: 推理结果
        """
        start_time = time.perf_counter()
        result = ReasoningResult()
        context = context or {}
        
        if self.mode == ReasoningMode.FAST:
            response = self._fast_reason(input_vector, context)
            result.mode = "system1"
            result.response = response
            result.confidence = response.get("confidence", 0.0)
            result.converged = result.confidence >= self.confidence_threshold
            self.stats["fast_inferences"] += 1
        
        elif self.mode == ReasoningMode.SLOW:
            response = self._slow_reason(input_vector, goal, context)
            result.mode = "system2"
            result.response = response
            result.confidence = response.get("confidence", 0.0)
            result.converged = result.confidence >= self.confidence_threshold
            self.stats["slow_inferences"] += 1
        
        elif self.mode == ReasoningMode.CHAIN:
            chain_result = self._chain_reason(input_vector, goal, context, max_steps)
            result.mode = "chain"
            result.response = chain_result.get("response", {})
            result.confidence = chain_result.get("final_confidence", 0.0)
            result.steps = chain_result.get("steps_taken", 0)
            result.converged = chain_result.get("converged", False)
            result.thought_chain = chain_result.get("thought_chain", [])
            self.stats["chain_inferences"] += 1
        
        elif self.mode == ReasoningMode.SYMBOLIC:
            symbolic_result = self._symbolic_reason(input_vector, goal, context)
            result.mode = "symbolic"
            result.response = symbolic_result.get("response", {})
            result.confidence = symbolic_result.get("confidence", 0.0)
            result.converged = symbolic_result.get("converged", False)
            result.symbolic_trace = symbolic_result.get("trace", [])
            self.stats["symbolic_inferences"] += 1
        
        else:
            response = self.thinking_orchestrator.process(input_vector, context)
            result.mode = response.get("mode", "unknown")
            result.response = response.get("response", {})
            result.confidence = response.get("confidence", 0.0)
            result.converged = result.confidence >= self.confidence_threshold
            
            if result.mode == "system1" or result.mode == "system1_fallback":
                self.stats["fast_inferences"] += 1
            else:
                self.stats["slow_inferences"] += 1
        
        elapsed = time.perf_counter() - start_time
        result.elapsed_time = elapsed
        
        self.stats["total_inferences"] += 1
        self.stats["total_time"] += elapsed
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_inferences"]
        
        if result.converged:
            self.stats["converged_count"] += 1
        self.stats["success_rate"] = self.stats["converged_count"] / self.stats["total_inferences"]
        
        return result
    
    def _fast_reason(self, input_vector: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """快速推理（系统1）"""
        return self.dual_system.fast_process(input_vector, context)
    
    def _slow_reason(self, input_vector: Any, goal: Optional[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """深度推理（系统2）"""
        thinking_result = self.autonomous_engine.think(input_vector, context)
        return thinking_result.to_dict()
    
    def _chain_reason(self, input_vector: Any, goal: Optional[str], 
                      context: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
        """思维链推理"""
        if goal:
            goal_dict = {"goal_type": goal, "progress": 0.0}
        else:
            goal_dict = {"goal_type": "general", "progress": 0.0}
        
        return self.thinking_orchestrator.chain_of_thought(
            input_vector, goal_dict, max_steps, context
        )
    
    def _symbolic_reason(self, input_vector: Any, goal: Optional[str], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """神经符号推理"""
        if isinstance(input_vector, np.ndarray):
            input_str = str(input_vector[:10])
        else:
            input_str = str(input_vector)
        
        result = self.symbolic_reasoner.reason(input_str, context)
        return {
            "response": result,
            "confidence": result.get("confidence", 0.5),
            "converged": result.get("confidence", 0.5) >= 0.7,
            "trace": result.get("trace", []),
        }
    
    def decompose_problem(self, problem_description: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        问题分解
        
        Args:
            problem_description: 问题描述
            context: 上下文信息
        
        Returns:
            分解后的子问题列表
        """
        return self.thinking_orchestrator.decompose_problem(problem_description, context)
    
    def synthesize_knowledge(self, knowledge_fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        知识综合
        
        Args:
            knowledge_fragments: 知识片段列表
        
        Returns:
            综合后的知识
        """
        return self.thinking_orchestrator.synthesize_knowledge(knowledge_fragments)
    
    def critical_analysis(self, idea: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        批判性分析
        
        Args:
            idea: 待分析的想法
            context: 上下文信息
        
        Returns:
            分析结果
        """
        return self.thinking_orchestrator.critical_analysis(idea, context)
    
    def causal_inference(self, observation: Any, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        因果推理
        
        Args:
            observation: 观测数据
            context: 上下文信息
        
        Returns:
            因果推理结果
        """
        return self.causal_engine.infer(observation, context)
    
    def predict_next(self, current_feature: Any) -> Any:
        """
        预测下一个状态
        
        Args:
            current_feature: 当前特征
        
        Returns:
            预测的下一个状态
        """
        return self.inference_layer.autonomous_thinking(current_feature)
    
    def learn_from_outcome(self, input_vector: Any, outcome: Any, success: bool) -> None:
        """
        从结果中学习
        
        Args:
            input_vector: 输入向量
            outcome: 结果
            success: 是否成功
        """
        self.thinking_orchestrator.learn_from_outcome(input_vector, outcome, success)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        获取知识摘要
        
        Returns:
            知识摘要统计
        """
        return self.inference_layer.get_knowledge_summary()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取推理统计信息
        
        Returns:
            统计信息字典
        """
        orchestrator_stats = self.thinking_orchestrator.get_stats()
        return {
            "mode": self.mode.value,
            "confidence_threshold": self.confidence_threshold,
            "feature_dim": self.feature_dim,
            **self.stats,
            "orchestrator_stats": orchestrator_stats,
        }
    
    def resize(self, new_dim: int) -> None:
        """
        调整特征维度
        
        Args:
            new_dim: 新的特征维度
        """
        self.feature_dim = new_dim
        self.inference_layer.resize(new_dim)
        self.thinking_orchestrator.resize(new_dim)
        self.autonomous_engine.resize(new_dim)
    
    def enter_sleep_mode(self) -> None:
        """进入睡眠模式"""
        self.thinking_orchestrator.enter_sleep_mode()
    
    def wake_up(self) -> None:
        """唤醒"""
        self.thinking_orchestrator.wake_up()


def get_unified_reasoner(feature_dim: int = 16, mode: str = "auto") -> UnifiedReasoningEngine:
    """
    获取统一推理引擎实例
    
    Args:
        feature_dim: 特征维度
        mode: 推理模式
    
    Returns:
        UnifiedReasoningEngine 实例
    """
    return UnifiedReasoningEngine(feature_dim=feature_dim, mode=mode)