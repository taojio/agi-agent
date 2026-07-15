"""
decision/__init__.py - 决策模块

提供完整的决策能力：
- 自主决策引擎 v3.0
- 多种决策策略
- 风险评估
- 决策质量追踪
- 行动规划与执行监控
- 元学习增强决策
- 元决策闭环优化系统
"""
from .decision_engine import (
    AutonomousDecisionEngine,
    Goal,
    DecisionOption,
    DecisionPriority,
    GoalType,
)
from .action_planner import ActionPlanner
from .execution_monitor import ExecutionMonitor
from .world_model_bridge import WorldModelDecisionBridge, BridgeMode, PredictionResult, SimulationScenario, DecisionSupportInfo
from .decision_strategies import (
    DecisionStrategy,
    StrategyType,
    DecisionContext,
    StrategyResult,
    UtilityMaximizationStrategy,
    BayesianStrategy,
    MultiObjectiveStrategy,
    FuzzyStrategy,
    CaseBasedStrategy,
    StrategyRegistry,
)
from .risk_assessment import (
    RiskAssessmentEngine,
    RiskProfile,
    RiskFactor,
    RiskLevel,
    RiskFactorType,
)
from .decision_tracker import (
    DecisionQualityTracker,
    DecisionRecord,
    DecisionStatus,
    BiasPattern,
    BiasType,
)
from .meta_integration import (
    MetaEnhancedDecisionStrategy,
    MetaEnhancedRiskAssessor,
    MetaEnhancedActionPlanner,
    DecisionMetaIntegration,
)
from .decision_quality_metrics import (
    MetricType,
    MetricCategory,
    QualityMetric,
    DecisionQualityReport,
    DecisionQualityMetrics,
)
from .strategy_adjustment_engine import (
    StrategyStatus,
    AdjustmentType,
    StrategyVersion,
    DecisionStrategy as EngineDecisionStrategy,
    AdjustmentRequest,
    AdjustmentResult,
    StrategyAdjustmentEngine,
)
from .multi_objective_tradeoff import (
    ObjectiveType,
    ObjectivePriority,
    ConflictSeverity,
    DecisionObjective,
    ObjectiveConflict,
    TradeoffStrategy,
    MultiObjectiveTradeoffSystem,
)
from .decision_feedback_loop import (
    FeedbackTriggerType,
    FeedbackStatus,
    AdjustmentActionType,
    ExecutionDataPoint,
    FeedbackEvent,
    QualityAdjustmentMapping,
    DecisionFeedbackLoop,
)
from .weight_adjustment import (
    LearningMode,
    AdjustmentMode,
    BoundaryType,
    WeightBound,
    WeightUpdate,
    LearningSample,
    StabilityConstraint,
    WeightAdjustmentSystem,
)

__all__ = [
    # 基础决策
    "AutonomousDecisionEngine",
    "Goal",
    "DecisionOption",
    "DecisionPriority",
    "GoalType",
    "ActionPlanner",
    "ExecutionMonitor",
    # 世界模型桥接
    "WorldModelDecisionBridge",
    "BridgeMode",
    "PredictionResult",
    "SimulationScenario",
    "DecisionSupportInfo",
    # 决策策略
    "DecisionStrategy",
    "StrategyType",
    "DecisionContext",
    "StrategyResult",
    "UtilityMaximizationStrategy",
    "BayesianStrategy",
    "MultiObjectiveStrategy",
    "FuzzyStrategy",
    "CaseBasedStrategy",
    "StrategyRegistry",
    # 风险评估
    "RiskAssessmentEngine",
    "RiskProfile",
    "RiskFactor",
    "RiskLevel",
    "RiskFactorType",
    # 质量追踪
    "DecisionQualityTracker",
    "DecisionRecord",
    "DecisionStatus",
    "BiasPattern",
    "BiasType",
    # 元学习增强
    "MetaEnhancedDecisionStrategy",
    "MetaEnhancedRiskAssessor",
    "MetaEnhancedActionPlanner",
    "DecisionMetaIntegration",
    # 决策质量评估指标
    "MetricType",
    "MetricCategory",
    "QualityMetric",
    "DecisionQualityReport",
    "DecisionQualityMetrics",
    # 策略动态调整引擎
    "StrategyStatus",
    "AdjustmentType",
    "StrategyVersion",
    "EngineDecisionStrategy",
    "AdjustmentRequest",
    "AdjustmentResult",
    "StrategyAdjustmentEngine",
    # 多目标决策权衡
    "ObjectiveType",
    "ObjectivePriority",
    "ConflictSeverity",
    "DecisionObjective",
    "ObjectiveConflict",
    "TradeoffStrategy",
    "MultiObjectiveTradeoffSystem",
    # 反馈闭环
    "FeedbackTriggerType",
    "FeedbackStatus",
    "AdjustmentActionType",
    "ExecutionDataPoint",
    "FeedbackEvent",
    "QualityAdjustmentMapping",
    "DecisionFeedbackLoop",
    # 权重动态调整
    "LearningMode",
    "AdjustmentMode",
    "BoundaryType",
    "WeightBound",
    "WeightUpdate",
    "LearningSample",
    "StabilityConstraint",
    "WeightAdjustmentSystem",
]
