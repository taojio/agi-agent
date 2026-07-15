"""
deliberative/__init__.py - 慎思层（思考系统）

负责深度分析、复杂决策、路径规划，对应人类"系统2"理性思考
"""
from .problem_formulator import ProblemFormulator, ProblemDefinition
from .hypothesis_generator import HypothesisGenerator, Hypothesis
from .logical_deductor import LogicalDeductor, DeductionStep
from .causal_reasoner import CausalReasoner, CausalChain
from .simulation_engine import SimulationEngine, SimulationResult
from .decision_optimizer import DecisionOptimizer, Solution
from .autonomous_thinking_engine import AutonomousThinkingEngine
from .thinking_orchestrator import ThinkingOrchestrator
from .advanced_reasoner import AdvancedReasoner
from .abstract_thinking import AbstractionEngine
from .neuro_symbolic_reasoner import NeuroSymbolicReasoner, SymbolType, NeuralSymbol, SymbolicExpression, InferenceRule
from .neuro_symbolic_world_coordinator import NeuroSymbolicWorldCoordinator, InteractionProtocol, CoordinationMessage
from .chain_of_thought_engine import ChainOfThoughtEngine, ReasoningMode
from .enhanced_symbolic_reasoner import (
    EnhancedSymbolicReasoner, Predicate, PredicateType, Quantifier,
    LogicalOperator, RelationType
)
from .reasoning_strategy_selector import (
    StrategySelector, StrategyPerformanceMetric, StrategySelectionMode
)

__all__ = ["ProblemFormulator", "ProblemDefinition",
           "HypothesisGenerator", "Hypothesis",
           "LogicalDeductor", "DeductionStep",
           "CausalReasoner", "CausalChain",
           "SimulationEngine", "SimulationResult",
           "DecisionOptimizer", "Solution",
           "AutonomousThinkingEngine", "ThinkingOrchestrator",
           "AdvancedReasoner", "AbstractionEngine",
           "NeuroSymbolicReasoner", "SymbolType", "NeuralSymbol", "SymbolicExpression", "InferenceRule",
           "NeuroSymbolicWorldCoordinator", "InteractionProtocol", "CoordinationMessage",
           "ChainOfThoughtEngine", "ReasoningMode",
           "EnhancedSymbolicReasoner", "Predicate", "PredicateType", "Quantifier",
           "LogicalOperator", "RelationType",
           "StrategySelector", "StrategyPerformanceMetric", "StrategySelectionMode"]