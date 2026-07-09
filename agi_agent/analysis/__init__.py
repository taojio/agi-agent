"""
analysis/__init__.py - 多维度数据分析引擎

提供 7 个分析维度的数据分析能力，作为决策引擎的核心数据输入
"""
from .engine import (
    AnalysisEngine,
    AnalysisResult,
    AnalysisDimension,
)
from .dimensions import (
    TrendAnalyzer,
    AnomalyDetector,
    CorrelationAnalyzer,
    FrequencyAnalyzer,
    DistributionAnalyzer,
    ClusterAnalyzer,
    ForecastAnalyzer,
)
from .metrics import (
    TimeSeriesData,
    DataPoint,
    AnalysisReport,
)

__all__ = [
    "AnalysisEngine",
    "AnalysisResult",
    "AnalysisDimension",
    "TrendAnalyzer",
    "AnomalyDetector",
    "CorrelationAnalyzer",
    "FrequencyAnalyzer",
    "DistributionAnalyzer",
    "ClusterAnalyzer",
    "ForecastAnalyzer",
    "TimeSeriesData",
    "DataPoint",
    "AnalysisReport",
]
