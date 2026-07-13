"""
meta_parsing/__init__.py - 元解析模块

开发数据解析与处理的元层次控制机制，增强系统对复杂数据的理解能力

核心组件：
- MetaParser: 元解析器，支持多种解析策略和自适应解析
- DataTransformer: 数据转换器，实现元层次的数据转换
- ComplexDataProcessor: 复杂数据处理器
- ParsingOrchestrator: 解析编排器
"""
from .meta_parser import (
    MetaParser, ParsingStrategy, ParsingResult,
    ParserRegistry, ParserSelector,
)
from .data_transformer import (
    DataTransformer, TransformationRule, TransformationContext,
    TransformationResult, TransformationType,
)
from .complex_data_processor import (
    ComplexDataProcessor, DataComplexity, ComplexityAnalysis,
    DataUnderstanding, SchemaInference,
)
from .orchestrator import ParsingOrchestrator

__all__ = [
    # 元解析器
    "MetaParser", "ParsingStrategy", "ParsingResult",
    "ParserRegistry", "ParserSelector",
    # 数据转换器
    "DataTransformer", "TransformationRule", "TransformationContext",
    "TransformationResult", "TransformationType",
    # 复杂数据处理
    "ComplexDataProcessor", "DataComplexity", "ComplexityAnalysis",
    "DataUnderstanding", "SchemaInference",
    # 编排器
    "ParsingOrchestrator",
]