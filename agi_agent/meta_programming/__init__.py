"""
meta_programming/__init__.py - 元编程模块

提供代码生成、代码分析和动态代码执行能力，支持系统自我修改和扩展

核心组件：
- CodeGenerator: 代码生成引擎
- CodeAnalyzer: 代码分析器
- DynamicExecutor: 动态代码执行器
- MetaProgrammingOrchestrator: 元编程编排器
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
]