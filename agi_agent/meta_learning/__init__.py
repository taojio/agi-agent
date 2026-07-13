"""
meta_learning/__init__.py - 元学习模块

构建学习策略优化系统，支持快速适应新任务和环境的学习能力

核心组件：
- MetaLearner: 元学习引擎
- LearningStrategyOptimizer: 学习策略优化器
- TaskAdaptationEngine: 任务自适应引擎
- MetaKnowledgeBase: 元知识库
"""
from .meta_learner import (
    MetaLearner, MetaLearningTask, MetaLearningResult,
    TaskEmbedding, TaskSimilarity, MetaLearningMode,
)
from .strategy_optimizer import (
    LearningStrategyOptimizer, LearningStrategy, StrategyPerformance,
    HyperparameterSpace, OptimizationResult,
)
from .task_adaptation import (
    TaskAdaptationEngine, TaskDescriptor, AdaptationStrategy,
    AdaptationResult, TransferLearningConfig,
)
from .meta_knowledge_base import (
    MetaKnowledgeBase, MetaRule, RuleType, MetaPattern,
    KnowledgeTransfer, KnowledgeConsolidation,
)
from .orchestrator import MetaLearningOrchestrator

__all__ = [
    # 元学习引擎
    "MetaLearner", "MetaLearningTask", "MetaLearningResult",
    "TaskEmbedding", "TaskSimilarity", "MetaLearningMode",
    # 策略优化器
    "LearningStrategyOptimizer", "LearningStrategy", "StrategyPerformance",
    "HyperparameterSpace", "OptimizationResult",
    # 任务自适应
    "TaskAdaptationEngine", "TaskDescriptor", "AdaptationStrategy",
    "AdaptationResult", "TransferLearningConfig",
    # 元知识库
    "MetaKnowledgeBase", "MetaRule", "RuleType", "MetaPattern",
    "KnowledgeTransfer", "KnowledgeConsolidation",
    # 编排器
    "MetaLearningOrchestrator",
]