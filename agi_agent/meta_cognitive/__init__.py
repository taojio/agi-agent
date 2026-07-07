"""
meta_cognitive/__init__.py - 元认知层（监管系统）

负责审视思考过程、校准行动方向、管控能力边界，对应人类"自我意识与反思"
"""
from .self_model import SelfModel, IdentityRepresentation, CapabilityProfile, StateRepresentation, HistoryRepresentation
from .cognitive_monitor import CognitiveMonitor, ThinkingTracer, ExecutionTracker, ResourceMonitor
from .strategy_regulator import StrategyRegulator, ThinkingStrategy, ActionStrategy, ResourceAllocation
from .boundary_guardian import BoundaryGuardian, SafetyBoundary, PermissionBoundary, EthicalBoundary
from .meta_learning_engine import MetaLearningEngine, LearningStrategy, StrategyOptimizer
from .meta_cognitive_orchestrator import MetaCognitiveOrchestrator

__all__ = ["SelfModel", "IdentityRepresentation", "CapabilityProfile", "StateRepresentation", "HistoryRepresentation",
           "CognitiveMonitor", "ThinkingTracer", "ExecutionTracker", "ResourceMonitor",
           "StrategyRegulator", "ThinkingStrategy", "ActionStrategy", "ResourceAllocation",
           "BoundaryGuardian", "SafetyBoundary", "PermissionBoundary", "EthicalBoundary",
           "MetaLearningEngine", "LearningStrategy", "StrategyOptimizer",
           "MetaCognitiveOrchestrator"]