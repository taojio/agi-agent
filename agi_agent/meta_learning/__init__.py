"""
meta_learning/__init__.py - 元学习模块

构建学习策略优化系统，支持快速适应新任务和环境的学习能力

核心组件：
- MetaLearner: 元学习引擎
- LearningStrategyOptimizer: 学习策略优化器
- TaskAdaptationEngine: 任务自适应引擎
- MetaKnowledgeBase: 元知识库
- HyperparameterController: 超参数智能调控系统
- MAMLAlgorithm: MAML核心算法框架
- TaskStrategyKnowledgeBase: 任务-策略匹配智能知识库
- MetaLearningInterface: 元学习策略接口规范
- MetaLearningIntegration: 元学习模块深度集成控制器
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
from .hyperparameter_controller import (
    HyperparameterController, ParameterType, AdjustmentStrategy, PerformanceMetric,
)
from .bayesian_optimizer import (
    BayesianOptimizer, RandomSearchOptimizer, GridSearchOptimizer,
    GeneticAlgorithmOptimizer, ParticleSwarmOptimizer, HyperparameterOptimizer,
    OptimizationAlgorithm, AcquisitionFunction, HyperparameterConfig, OptimizationResult,
)
from .maml_algorithm import (
    MAMLAlgorithm, MAMLMode, MAMLModel, MAMLTask, MAMLResult,
    cross_entropy_loss, mse_loss, compute_accuracy, compute_gradients,
)
from .task_strategy_knowledge import (
    TaskStrategyKnowledgeBase, TaskType, TaskComplexity, DataDistribution,
    StrategyType, TaskFeatureVector, StrategyVector, StrategyPerformanceRecord,
)
from .meta_learning_interface import (
    MetaLearningInterface, InterfaceProtocol, DataFormat, MessageType,
    MetaLearningMessage, TaskSpecification, StrategySpecification,
    AdaptationRequest, AdaptationResponse, ParameterUpdateRequest,
    PerformanceReport, KnowledgeTransferRequest, KnowledgeTransferResponse,
    MessageValidator, DataConverter, ProtocolHandler,
)
from .integration import (
    MetaLearningIntegration, LearningIntegrationMode, LearningAdaptationTrigger,
    get_meta_learning_integration,
)

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
    # 超参数控制器
    "HyperparameterController", "ParameterType", "AdjustmentStrategy", "PerformanceMetric",
    # 贝叶斯优化器
    "BayesianOptimizer", "RandomSearchOptimizer", "GridSearchOptimizer",
    "GeneticAlgorithmOptimizer", "ParticleSwarmOptimizer", "HyperparameterOptimizer",
    "OptimizationAlgorithm", "AcquisitionFunction", "HyperparameterConfig", "OptimizationResult",
    # MAML算法
    "MAMLAlgorithm", "MAMLMode", "MAMLModel", "MAMLTask", "MAMLResult",
    "cross_entropy_loss", "mse_loss", "compute_accuracy", "compute_gradients",
    # 任务-策略知识库
    "TaskStrategyKnowledgeBase", "TaskType", "TaskComplexity", "DataDistribution",
    "StrategyType", "TaskFeatureVector", "StrategyVector", "StrategyPerformanceRecord",
    # 接口规范
    "MetaLearningInterface", "InterfaceProtocol", "DataFormat", "MessageType",
    "MetaLearningMessage", "TaskSpecification", "StrategySpecification",
    "AdaptationRequest", "AdaptationResponse", "ParameterUpdateRequest",
    "PerformanceReport", "KnowledgeTransferRequest", "KnowledgeTransferResponse",
    "MessageValidator", "DataConverter", "ProtocolHandler",
    # 深度集成
    "MetaLearningIntegration", "LearningIntegrationMode", "LearningAdaptationTrigger",
    "get_meta_learning_integration",
]