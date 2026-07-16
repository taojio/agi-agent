"""
self_improvement/__init__.py - 自我改进模块

v3.0 升级：新增性能基准、反馈闭环、回归验证
"""
from .self_improver import RecursiveSelfImprover
from ..evaluation.evaluator import PerformanceEvaluator
from .self_diagnostic import SelfDiagnosticEngine
from .safety_verifier import ImprovementSafetyVerifier
from .tiered_modification import (
    TieredSelfModifier, SelfModificationTier, ModifiableParamSpec,
    ParamType, RuleSpec, ModuleInterfaceSpec, TieredModificationRequest
)
from .symbolic_self_model import SymbolicSelfModel
from .symbolic_verifier import SymbolicFormalVerifier
from .bootstrapped_improver import BootstrappedSelfImprover, ImprovementStage
from .automated_improvement_loop import AutomatedSelfImprovementLoop, IterationPhase, ImprovementOutcome, IterationRecord, PerformanceThreshold
from .performance_baseline import (
    PerformanceBaseline, BaselineType, TrendDirection,
    PerformanceMetric, MetricSample,
)
from .feedback_loop import (
    FeedbackLoop, FeedbackPhase, IssueSeverity, IssueType,
    Issue, ImprovementProposal, FeedbackCycle,
)
from .regression_validator import (
    RegressionValidator, RegressionSeverity, ValidationStatus,
    RegressionItem, RegressionReport, ValidationResult,
)
from .improvement_orchestrator import (
    ImprovementOrchestrator,
    ImprovementWorkflow,
    WorkflowInstance,
    WorkflowNode,
    WorkflowNodeType,
    WorkflowStatus,
    WorkflowEvent,
    RollbackConfig,
    RollbackTrigger,
    get_improvement_orchestrator,
)
from .version_manager import (
    ImprovementVersionManager,
    Version,
    VersionChange,
    VersionComparison,
    VersionStatus,
    VersionType,
    get_version_manager,
)

__all__ = [
    "RecursiveSelfImprover", "PerformanceEvaluator",
    "SelfDiagnosticEngine", "ImprovementSafetyVerifier",
    "TieredSelfModifier", "SelfModificationTier", "ModifiableParamSpec",
    "ParamType", "RuleSpec", "ModuleInterfaceSpec", "TieredModificationRequest",
    "SymbolicSelfModel", "SymbolicFormalVerifier",
    "BootstrappedSelfImprover", "ImprovementStage",
    # 自动化改进循环
    "AutomatedSelfImprovementLoop", "IterationPhase", "ImprovementOutcome",
    "IterationRecord", "PerformanceThreshold",
    # v3.0 新增
    "PerformanceBaseline", "BaselineType", "TrendDirection",
    "PerformanceMetric", "MetricSample",
    "FeedbackLoop", "FeedbackPhase", "IssueSeverity", "IssueType",
    "Issue", "ImprovementProposal", "FeedbackCycle",
    "RegressionValidator", "RegressionSeverity", "ValidationStatus",
    "RegressionItem", "RegressionReport", "ValidationResult",
    # 改进编排引擎
    "ImprovementOrchestrator",
    "ImprovementWorkflow",
    "WorkflowInstance",
    "WorkflowNode",
    "WorkflowNodeType",
    "WorkflowStatus",
    "WorkflowEvent",
    "RollbackConfig",
    "RollbackTrigger",
    "get_improvement_orchestrator",
    # 版本管理系统
    "ImprovementVersionManager",
    "Version",
    "VersionChange",
    "VersionComparison",
    "VersionStatus",
    "VersionType",
    "get_version_manager",
]
