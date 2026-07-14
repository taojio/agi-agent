"""
learning/__init__.py - 学习模块

v3.0 升级：新增在线学习器与经验沉淀器
"""
from .meta_learning import MetaLearningLayer
from .knowledge_graph import KnowledgeGraph
from .knowledge_ingestor import StructuredKnowledgeIngestor
from .online_learner import (
    OnlineLearner, LearningMode, DriftType,
    LearningSample, LearningUpdate, DriftDetection,
)
from .experience_distiller import (
    ExperienceDistiller, PatternType, PatternStatus,
    Pattern, StrategyPrototype,
)

__all__ = [
    "MetaLearningLayer", "KnowledgeGraph", "StructuredKnowledgeIngestor",
    # v3.0 新增
    "OnlineLearner", "LearningMode", "DriftType",
    "LearningSample", "LearningUpdate", "DriftDetection",
    "ExperienceDistiller", "PatternType", "PatternStatus",
    "Pattern", "StrategyPrototype",
]