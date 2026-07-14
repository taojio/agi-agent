"""
analysis/engine.py - 多维度分析引擎

整合 7 个分析维度，提供统一的分析入口
"""
import time
from typing import Any, Dict, List, Optional, Union

from .metrics import (
    AnalysisResult,
    AnalysisDimension,
    TimeSeriesData,
    AnalysisReport,
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


class AnalysisEngine:
    """多维度分析引擎

    整合 7 个分析维度，提供：
    - 单维度分析
    - 多维度组合分析
    - 综合分析报告
    - 智能洞察生成
    """

    def __init__(self,
                 trend_window: int = 10,
                 anomaly_z_threshold: float = 3.0,
                 forecast_horizon: int = 5):
        self.trend_analyzer = TrendAnalyzer(window_size=trend_window)
        self.anomaly_detector = AnomalyDetector(z_threshold=anomaly_z_threshold)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.distribution_analyzer = DistributionAnalyzer()
        self.cluster_analyzer = ClusterAnalyzer()
        self.forecast_analyzer = ForecastAnalyzer(forecast_horizon=forecast_horizon)

        self._stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "by_dimension": {dim.value: 0 for dim in AnalysisDimension},
        }

    def analyze_trend(self, data: TimeSeriesData) -> AnalysisResult:
        """趋势分析"""
        result = self.trend_analyzer.analyze(data)
        self._record_result(result)
        return result

    def analyze_anomaly(self, data: TimeSeriesData) -> AnalysisResult:
        """异常检测"""
        result = self.anomaly_detector.analyze(data)
        self._record_result(result)
        return result

    def analyze_correlation(self, data: Dict[str, List[float]]) -> AnalysisResult:
        """相关性分析"""
        result = self.correlation_analyzer.analyze(data)
        self._record_result(result)
        return result

    def analyze_frequency(self, data: TimeSeriesData) -> AnalysisResult:
        """频率分析"""
        result = self.frequency_analyzer.analyze(data)
        self._record_result(result)
        return result

    def analyze_distribution(self, data: List[float]) -> AnalysisResult:
        """分布分析"""
        result = self.distribution_analyzer.analyze(data)
        self._record_result(result)
        return result

    def analyze_clustering(self, data: List[List[float]],
                            k: int = None) -> AnalysisResult:
        """聚类分析"""
        result = self.cluster_analyzer.analyze(data, k=k)
        self._record_result(result)
        return result

    def analyze_forecast(self, data: TimeSeriesData,
                         horizon: int = None) -> AnalysisResult:
        """预测分析"""
        result = self.forecast_analyzer.analyze(data, horizon=horizon)
        self._record_result(result)
        return result

    def analyze_all(self, data: TimeSeriesData,
                    extra_metrics: Dict[str, List[float]] = None,
                    cluster_data: List[List[float]] = None) -> AnalysisReport:
        """全维度分析

        对时序数据执行所有维度的分析

        Args:
            data: 主时序数据
            extra_metrics: 额外指标（用于相关性分析）
            cluster_data: 聚类数据（用于聚类分析）

        Returns:
            综合分析报告
        """
        report = AnalysisReport(target=data.name)

        report.add_result(self.analyze_trend(data))
        report.add_result(self.analyze_anomaly(data))
        report.add_result(self.analyze_frequency(data))
        report.add_result(self.analyze_forecast(data))
        report.add_result(self.analyze_distribution(data.values))

        if extra_metrics:
            metrics_data = {data.name: data.values, **extra_metrics}
            report.add_result(self.analyze_correlation(metrics_data))
        else:
            empty_result = AnalysisResult(
                dimension=AnalysisDimension.CORRELATION,
                success=False,
                error="No extra metrics provided for correlation analysis",
            )
            report.add_result(empty_result)

        if cluster_data:
            report.add_result(self.analyze_clustering(cluster_data))
        else:
            empty_result = AnalysisResult(
                dimension=AnalysisDimension.CLUSTERING,
                success=False,
                error="No cluster data provided",
            )
            report.add_result(empty_result)

        report.summary = self._generate_summary(report)

        return report

    def analyze_dimension(self, dimension: AnalysisDimension,
                           *args, **kwargs) -> AnalysisResult:
        """按维度执行分析"""
        if dimension == AnalysisDimension.TREND:
            return self.analyze_trend(*args, **kwargs)
        elif dimension == AnalysisDimension.ANOMALY:
            return self.analyze_anomaly(*args, **kwargs)
        elif dimension == AnalysisDimension.CORRELATION:
            return self.analyze_correlation(*args, **kwargs)
        elif dimension == AnalysisDimension.FREQUENCY:
            return self.analyze_frequency(*args, **kwargs)
        elif dimension == AnalysisDimension.DISTRIBUTION:
            return self.analyze_distribution(*args, **kwargs)
        elif dimension == AnalysisDimension.CLUSTERING:
            return self.analyze_clustering(*args, **kwargs)
        elif dimension == AnalysisDimension.FORECAST:
            return self.analyze_forecast(*args, **kwargs)
        else:
            return AnalysisResult(
                dimension=dimension,
                success=False,
                error=f"Unknown dimension: {dimension}",
            )

    def get_available_dimensions(self) -> List[AnalysisDimension]:
        """获取可用分析维度"""
        return list(AnalysisDimension)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return dict(self._stats)

    def _record_result(self, result: AnalysisResult) -> None:
        """记录分析结果"""
        self._stats["total_analyses"] += 1
        self._stats["by_dimension"][result.dimension.value] += 1

        if result.success:
            self._stats["successful_analyses"] += 1
        else:
            self._stats["failed_analyses"] += 1

    def _generate_summary(self, report: AnalysisReport) -> str:
        """生成综合摘要"""
        success_dims = [r for r in report.results.values() if r.success]
        total_insights = sum(len(r.insights) for r in success_dims)

        lines = [
            f"完成 {len(success_dims)}/{len(report.results)} 个维度分析",
            f"生成 {total_insights} 条洞察",
        ]

        high_confidence = [
            r for r in success_dims if r.confidence > 0.7
        ]
        if high_confidence:
            lines.append(
                f"高置信度分析: {', '.join(r.dimension.value for r in high_confidence)}"
            )

        return "；".join(lines)
