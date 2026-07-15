"""
meta_programming/__init__.py - 元编程模块

提供代码生成、代码分析和动态代码执行能力，支持系统自我修改和扩展

核心组件：
- CodeGenerator: 代码生成引擎
- CodeAnalyzer: 代码分析器
- DynamicExecutor: 动态代码执行器
- MetaProgrammingOrchestrator: 元编程编排器
- CrossLanguageCodeAnalyzer: 跨语言代码分析器
- SelfModifyingSandbox: 自修改安全沙箱
- PerformanceDashboard: 性能监控仪表盘
- AutomatedTestSuite: 自动化测试套件
- LanguageRules: 语言规则基础类
- PythonLanguageRules: Python语言规则
- CLanguageRules: C语言规则
- AssemblyLanguageRules: 汇编语言规则
- SemanticAnalyzer: 语义分析器
- RuleValidator: 规则验证器
- RuleQueryEngine: 规则查询引擎
- RuleDocumentationGenerator: 规则文档生成器
"""
from .code_generator import (
    CodeGenerator, CodeTemplate, GenerationContext,
    CodeGenerationResult, GenerationMode, CodeQuality,
)
from .code_analyzer import (
    CodeAnalyzer, CodeAnalysisResult, CodeStructure,
    ComplexityMetrics, DependencyGraph, CodeIssue, IssueSeverity,
)
from .dynamic_executor import (
    DynamicExecutor, ExecutionContext, ExecutionResult,
    SandboxConfig, ExecutionStatus,
)
from .orchestrator import MetaProgrammingOrchestrator
from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SyntaxRule, SemanticRule, ValidationError, ValidationResult,
    SymbolTable, TypeDefinition,
)
from .python_rules import PythonLanguageRules
from .c_rules import CLanguageRules
from .assembly_rules import AssemblyLanguageRules
from .semantic_analyzer import SemanticAnalyzer
from .rule_validator import RuleValidator, RuleQueryEngine
from .documentation_generator import RuleDocumentationGenerator
from .cross_language_analyzer import (
    CrossLanguageCodeAnalyzer, LanguageType as CrossLangLanguageType,
    CodeAnalysisResult as CrossLangAnalysisResult,
    OptimizationSuggestion as CrossLangOptimizationSuggestion,
    ComplexityMetrics as CrossLangComplexityMetrics,
    CodeQualityLevel as CrossLangCodeQualityLevel,
)
from .self_modifying_sandbox import (
    SelfModifyingSandbox, SandboxMode, PermissionLevel,
    ModificationRequest, ModificationType, ApprovalStatus,
    SecurityBoundary, PermissionManager, VersionControl,
    AuditLogger, ResourceQuota,
)
from .performance_detector import (
    PerformanceDashboard, PerformanceCollector,
    BottleneckAnalyzer, OptimizationSuggestionGenerator,
    BottleneckType, SeverityLevel, MetricType,
    BottleneckDetection, OptimizationSuggestion,
)
from .test_case_generator import (
    AutomatedTestSuite, TestCaseGenerator, TestExecutor,
    TestReportGenerator, InputGenerator, CodeAnalyzer as TestCodeAnalyzer,
    TestType, TestStrategy, TestResultStatus,
    TestCase, TestResult, CoverageReport, TestSuite,
)

__all__ = [
    # 代码生成
    "CodeGenerator", "CodeTemplate", "GenerationContext",
    "CodeGenerationResult", "GenerationMode", "CodeQuality",
    # 代码分析
    "CodeAnalyzer", "CodeAnalysisResult", "CodeStructure",
    "ComplexityMetrics", "DependencyGraph", "CodeIssue", "IssueSeverity",
    # 动态执行
    "DynamicExecutor", "ExecutionContext", "ExecutionResult",
    "SandboxConfig", "ExecutionStatus",
    # 编排器
    "MetaProgrammingOrchestrator",
    # 跨语言分析
    "CrossLanguageCodeAnalyzer",
    # 自修改沙箱
    "SelfModifyingSandbox", "SandboxMode", "PermissionLevel",
    "ModificationRequest", "ModificationType", "ApprovalStatus",
    # 性能检测
    "PerformanceDashboard", "BottleneckType", "SeverityLevel",
    # 测试生成
    "AutomatedTestSuite", "TestType", "TestStrategy", "TestResultStatus",
    # 语言规则基础类
    "LanguageType", "GrammarCategory", "RuleSeverity",
    "SyntaxRule", "SemanticRule", "ValidationError", "ValidationResult",
    "SymbolTable", "TypeDefinition",
    # 语言规则实现
    "PythonLanguageRules", "CLanguageRules", "AssemblyLanguageRules",
    # 语义分析
    "SemanticAnalyzer",
    # 规则验证和查询
    "RuleValidator", "RuleQueryEngine",
    # 文档生成
    "RuleDocumentationGenerator",
]