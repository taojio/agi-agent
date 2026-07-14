"""
analysis/metrics.py - 分析数据模型

定义分析引擎使用的数据结构
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AnalysisDimension(Enum):
    """分析维度"""
    TREND = "trend"                  # 时序趋势分析
    ANOMALY = "anomaly"              # 异常检测
    CORRELATION = "correlation"      # 相关性分析
    FREQUENCY = "frequency"          # 频率分析
    DISTRIBUTION = "distribution"    # 分布分析
    CLUSTERING = "clustering"        # 聚类分析
    FORECAST = "forecast"            # 预测分析


@dataclass
class DataPoint:
    """数据点

    Attributes:
        timestamp: 时间戳
        value: 数值
        label: 标签（可选）
        metadata: 元数据
    """

    timestamp: float
    value: float
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesData:
    """时序数据

    Attributes:
        name: 数据名称
        points: 数据点列表
        unit: 单位
        source: 数据来源
    """

    name: str
    points: List[DataPoint] = field(default_factory=list)
    unit: str = ""
    source: str = ""

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def values(self) -> List[float]:
        """获取所有值"""
        return [p.value for p in self.points]

    @property
    def timestamps(self) -> List[float]:
        """获取所有时间戳"""
        return [p.timestamp for p in self.points]

    @property
    def time_range(self) -> Tuple[float, float]:
        """时间范围"""
        if not self.points:
            return (0.0, 0.0)
        return (self.points[0].timestamp, self.points[-1].timestamp)

    def add_point(self, value: float, timestamp: float = None,
                  label: str = "", **metadata) -> None:
        """添加数据点"""
        self.points.append(DataPoint(
            timestamp=timestamp if timestamp is not None else time.time(),
            value=value,
            label=label,
            metadata=metadata,
        ))

    def slice(self, start: int = None, end: int = None) -> "TimeSeriesData":
        """切片"""
        return TimeSeriesData(
            name=self.name,
            points=self.points[start:end],
            unit=self.unit,
            source=self.source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
            "points": [
                {"timestamp": p.timestamp, "value": p.value,
                 "label": p.label, "metadata": p.metadata}
                for p in self.points
            ],
        }


@dataclass
class AnalysisResult:
    """分析结果

    Attributes:
        dimension: 分析维度
        success: 是否成功
        data: 分析结果数据
        insights: 关键洞察
        confidence: 置信度 (0-1)
        timestamp: 分析时间
        error: 错误信息（如果失败）
    """

    dimension: AnalysisDimension
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def add_insight(self, insight: str) -> None:
        """添加洞察"""
        self.insights.append(insight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "success": self.success,
            "data": self.data,
            "insights": self.insights,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class AnalysisReport:
    """综合分析报告

    包含多个维度的分析结果
    """

    target: str = ""
    results: Dict[AnalysisDimension, AnalysisResult] = field(default_factory=dict)
    summary: str = ""
    created_at: float = field(default_factory=time.time)

    def add_result(self, result: AnalysisResult) -> None:
        """添加分析结果"""
        self.results[result.dimension] = result

    def get_result(self, dimension: AnalysisDimension) -> Optional[AnalysisResult]:
        """获取指定维度的结果"""
        return self.results.get(dimension)

    @property
    def is_complete(self) -> bool:
        """是否所有维度都已完成"""
        return len(self.results) == len(AnalysisDimension)

    @property
    def success_count(self) -> int:
        """成功分析的数量"""
        return sum(1 for r in self.results.values() if r.success)

    def generate_summary(self) -> str:
        """生成综合摘要"""
        lines = [f"Analysis Report for '{self.target}'"]
        lines.append(f"  Dimensions: {len(self.results)}/{len(AnalysisDimension)}")
        lines.append(f"  Successful: {self.success_count}")

        for dim, result in self.results.items():
            status = "OK" if result.success else "FAIL"
            lines.append(f"  [{status}] {dim.value}: "
                        f"confidence={result.confidence:.2f}")

        if self.summary:
            lines.append(f"\nSummary: {self.summary}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "results": {dim.value: r.to_dict()
                       for dim, r in self.results.items()},
            "summary": self.summary,
            "created_at": self.created_at,
        }
