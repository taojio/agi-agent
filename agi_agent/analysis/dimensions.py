"""
analysis/dimensions.py - 7 个分析维度实现

每个维度独立实现，支持单独调用或组合分析
"""
import math
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import AnalysisResult, AnalysisDimension, TimeSeriesData, DataPoint


class TrendAnalyzer:
    """时序趋势分析

    分析指标随时间的变化趋势，包括：
    - 移动平均
    - 线性回归
    - 指数平滑
    - 趋势方向和变化率
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """执行趋势分析"""
        result = AnalysisResult(
            dimension=AnalysisDimension.TREND,
            timestamp=time.time(),
        )

        if data.is_empty or data.length < 2:
            result.success = False
            result.error = "Insufficient data for trend analysis (need >= 2 points)"
            return result

        values = np.array(data.values)
        timestamps = np.array(data.timestamps)

        try:
            moving_avg = self._moving_average(values)
            linear_trend = self._linear_regression(timestamps, values)
            exponential_smoothed = self._exponential_smoothing(values)
            change_rate = self._calculate_change_rate(values)

            trend_direction = self._determine_trend_direction(linear_trend, change_rate)

            result.data = {
                "moving_average": moving_avg.tolist(),
                "linear_trend": linear_trend,
                "exponential_smoothed": exponential_smoothed.tolist(),
                "change_rate": change_rate,
                "trend_direction": trend_direction,
                "current_value": float(values[-1]),
                "average_value": float(np.mean(values)),
                "min_value": float(np.min(values)),
                "max_value": float(np.max(values)),
                "volatility": float(np.std(values)),
            }

            result.confidence = self._calculate_confidence(values, linear_trend)

            if trend_direction == "increasing":
                result.add_insight(f"指标呈上升趋势，变化率 {change_rate:.4f}")
            elif trend_direction == "decreasing":
                result.add_insight(f"指标呈下降趋势，变化率 {change_rate:.4f}")
            else:
                result.add_insight("指标保持稳定")

            if np.std(values) > np.mean(values) * 0.5:
                result.add_insight("指标波动较大")

        except Exception as e:
            result.success = False
            result.error = f"Trend analysis failed: {e}"

        return result

    def _moving_average(self, values: np.ndarray) -> np.ndarray:
        """计算移动平均"""
        if len(values) < self.window_size:
            return np.mean(values) * np.ones_like(values)
        kernel = np.ones(self.window_size) / self.window_size
        return np.convolve(values, kernel, mode='same')

    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """线性回归"""
        x_norm = x - x[0] if len(x) > 0 else x
        n = len(x)
        if n < 2:
            return {"slope": 0.0, "intercept": float(y[0]) if len(y) > 0 else 0.0,
                    "r_squared": 0.0}

        x_mean = np.mean(x_norm)
        y_mean = np.mean(y)

        numerator = np.sum((x_norm - x_mean) * (y - y_mean))
        denominator = np.sum((x_norm - x_mean) ** 2)

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        intercept = y_mean - slope * x_mean

        y_pred = slope * x_norm + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        }

    def _exponential_smoothing(self, values: np.ndarray,
                                alpha: float = 0.3) -> np.ndarray:
        """指数平滑"""
        smoothed = np.zeros_like(values, dtype=float)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    def _calculate_change_rate(self, values: np.ndarray) -> float:
        """计算变化率"""
        if len(values) < 2:
            return 0.0
        return float((values[-1] - values[0]) / max(abs(values[0]), 1e-10))

    def _determine_trend_direction(self, linear_trend: Dict[str, float],
                                    change_rate: float) -> str:
        """判断趋势方向"""
        slope = linear_trend.get("slope", 0.0)
        threshold = 0.01

        if slope > threshold and change_rate > 0:
            return "increasing"
        elif slope < -threshold and change_rate < 0:
            return "decreasing"
        else:
            return "stable"

    def _calculate_confidence(self, values: np.ndarray,
                               linear_trend: Dict[str, float]) -> float:
        """计算分析置信度"""
        r_squared = linear_trend.get("r_squared", 0.0)
        data_factor = min(1.0, len(values) / 20.0)
        return float(0.5 * r_squared + 0.5 * data_factor)


class AnomalyDetector:
    """异常检测

    识别数据中的异常点，支持多种方法：
    - Z-score
    - IQR (四分位距)
    - 孤立森林 (简化版)
    """

    def __init__(self, z_threshold: float = 3.0,
                 iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """执行异常检测"""
        result = AnalysisResult(
            dimension=AnalysisDimension.ANOMALY,
            timestamp=time.time(),
        )

        if data.is_empty:
            result.success = False
            result.error = "No data for anomaly detection"
            return result

        values = np.array(data.values)

        try:
            z_anomalies = self._z_score_detection(values)
            iqr_anomalies = self._iqr_detection(values)

            all_anomalies = sorted(set(z_anomalies + iqr_anomalies))

            anomaly_details = []
            for idx in all_anomalies:
                if 0 <= idx < len(values):
                    anomaly_details.append({
                        "index": idx,
                        "value": float(values[idx]),
                        "timestamp": data.points[idx].timestamp if idx < len(data.points) else 0,
                        "z_score": float((values[idx] - np.mean(values)) / max(np.std(values), 1e-10)),
                    })

            result.data = {
                "anomaly_count": len(all_anomalies),
                "anomaly_indices": all_anomalies,
                "anomaly_details": anomaly_details,
                "z_score_anomalies": z_anomalies,
                "iqr_anomalies": iqr_anomalies,
                "threshold_z": self.z_threshold,
            }

            result.confidence = 1.0 - min(1.0, len(all_anomalies) / max(len(values), 1))

            if all_anomalies:
                result.add_insight(
                    f"检测到 {len(all_anomalies)} 个异常点"
                )
                if len(all_anomalies) > len(values) * 0.1:
                    result.add_insight("异常点比例超过 10%，数据质量可能有问题")
            else:
                result.add_insight("未检测到异常点")

        except Exception as e:
            result.success = False
            result.error = f"Anomaly detection failed: {e}"

        return result

    def _z_score_detection(self, values: np.ndarray) -> List[int]:
        """Z-score 检测"""
        if len(values) < 2:
            return []

        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return []

        anomalies = []
        for i, v in enumerate(values):
            z = abs((v - mean) / std)
            if z > self.z_threshold:
                anomalies.append(i)
        return anomalies

    def _iqr_detection(self, values: np.ndarray) -> List[int]:
        """IQR 检测"""
        if len(values) < 4:
            return []

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        anomalies = []
        for i, v in enumerate(values):
            if v < lower or v > upper:
                anomalies.append(i)
        return anomalies


class CorrelationAnalyzer:
    """相关性分析

    分析多个指标间的关联关系
    """

    def analyze(self, data: Dict[str, List[float]]) -> AnalysisResult:
        """执行相关性分析

        Args:
            data: 指标名到值列表的映射
        """
        result = AnalysisResult(
            dimension=AnalysisDimension.CORRELATION,
            timestamp=time.time(),
        )

        if len(data) < 2:
            result.success = False
            result.error = "Need at least 2 metrics for correlation analysis"
            return result

        try:
            metrics = list(data.keys())
            n = len(metrics)

            correlation_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        corr = self._pearson_correlation(
                            data[metrics[i]], data[metrics[j]]
                        )
                        correlation_matrix[i, j] = corr

            strong_correlations = []
            for i in range(n):
                for j in range(i + 1, n):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.7:
                        strong_correlations.append({
                            "metric_a": metrics[i],
                            "metric_b": metrics[j],
                            "correlation": float(corr),
                            "type": "positive" if corr > 0 else "negative",
                            "strength": "strong" if abs(corr) > 0.8 else "moderate",
                        })

            result.data = {
                "metrics": metrics,
                "correlation_matrix": correlation_matrix.tolist(),
                "strong_correlations": strong_correlations,
                "metric_count": n,
            }

            result.confidence = 0.7 if strong_correlations else 0.4

            for sc in strong_correlations[:3]:
                result.add_insight(
                    f"{sc['metric_a']} 与 {sc['metric_b']} 存在"
                    f"{sc['strength']}的{sc['type']}相关 (r={sc['correlation']:.3f})"
                )

        except Exception as e:
            result.success = False
            result.error = f"Correlation analysis failed: {e}"

        return result

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """皮尔逊相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        x_arr = np.array(x)
        y_arr = np.array(y)

        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sqrt(
            np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2)
        )

        if denominator < 1e-10:
            return 0.0
        return float(numerator / denominator)


class FrequencyAnalyzer:
    """频率分析

    分析事件发生频率和周期性
    """

    def __init__(self, min_freq: float = 0.01,
                 max_freq: float = 0.5):
        self.min_freq = min_freq
        self.max_freq = max_freq

    def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """执行频率分析"""
        result = AnalysisResult(
            dimension=AnalysisDimension.FREQUENCY,
            timestamp=time.time(),
        )

        if data.is_empty or data.length < 4:
            result.success = False
            result.error = "Insufficient data for frequency analysis (need >= 4 points)"
            return result

        values = np.array(data.values)

        try:
            fft_result = self._fft_analysis(values)
            autocorrelation = self._autocorrelation(values)
            periodicity = self._detect_periodicity(autocorrelation)

            result.data = {
                "dominant_frequency": fft_result["dominant_freq"],
                "frequency_spectrum": fft_result["spectrum"],
                "autocorrelation": autocorrelation.tolist(),
                "periodicity_detected": periodicity["detected"],
                "period_length": periodicity["period"],
                "event_count": len(values),
                "average_interval": float(np.mean(np.diff(data.timestamps)))
                                    if len(data.timestamps) > 1 else 0.0,
            }

            result.confidence = 0.8 if periodicity["detected"] else 0.3

            if periodicity["detected"]:
                result.add_insight(
                    f"检测到周期性，周期长度约 {periodicity['period']}"
                )
            else:
                result.add_insight("未检测到明显周期性")

        except Exception as e:
            result.success = False
            result.error = f"Frequency analysis failed: {e}"

        return result

    def _fft_analysis(self, values: np.ndarray) -> Dict[str, Any]:
        """FFT 分析"""
        n = len(values)
        if n < 4:
            return {"dominant_freq": 0.0, "spectrum": []}

        fft_vals = np.fft.fft(values)
        magnitudes = np.abs(fft_vals)[:n // 2]
        frequencies = np.fft.fftfreq(n)[:n // 2]

        if len(magnitudes) > 1:
            dominant_idx = np.argmax(magnitudes[1:]) + 1
            dominant_freq = float(frequencies[dominant_idx])
        else:
            dominant_freq = 0.0

        return {
            "dominant_freq": dominant_freq,
            "spectrum": magnitudes.tolist(),
        }

    def _autocorrelation(self, values: np.ndarray) -> np.ndarray:
        """自相关"""
        n = len(values)
        if n < 2:
            return np.array([1.0])

        values_centered = values - np.mean(values)
        result = np.correlate(values_centered, values_centered, mode='full')
        result = result[n - 1:]
        if result[0] > 0:
            result = result / result[0]
        return result[:min(n, 50)]

    def _detect_periodicity(self, autocorrelation: np.ndarray) -> Dict[str, Any]:
        """检测周期性"""
        if len(autocorrelation) < 4:
            return {"detected": False, "period": 0}

        threshold = 0.5
        peaks = []

        for i in range(1, len(autocorrelation) - 1):
            if (autocorrelation[i] > threshold and
                autocorrelation[i] > autocorrelation[i - 1] and
                autocorrelation[i] > autocorrelation[i + 1]):
                peaks.append(i)

        if peaks:
            return {"detected": True, "period": peaks[0]}
        return {"detected": False, "period": 0}


class DistributionAnalyzer:
    """分布分析

    分析数据分布特征和统计量
    """

    def analyze(self, data: List[float]) -> AnalysisResult:
        """执行分布分析"""
        result = AnalysisResult(
            dimension=AnalysisDimension.DISTRIBUTION,
            timestamp=time.time(),
        )

        if not data or len(data) < 2:
            result.success = False
            result.error = "Insufficient data for distribution analysis"
            return result

        values = np.array(data)

        try:
            histogram = self._compute_histogram(values)
            percentiles = self._compute_percentiles(values)
            distribution_type = self._classify_distribution(values)
            moments = self._compute_moments(values)

            result.data = {
                "histogram": histogram,
                "percentiles": percentiles,
                "distribution_type": distribution_type,
                "moments": moments,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
            }

            result.confidence = 0.7

            result.add_insight(
                f"数据分布类型: {distribution_type}，均值 {np.mean(values):.4f}，"
                f"标准差 {np.std(values):.4f}"
            )

            skewness = moments["skewness"]
            if abs(skewness) > 1.0:
                direction = "右" if skewness > 0 else "左"
                result.add_insight(f"分布显著{direction}偏（偏度={skewness:.3f}）")

        except Exception as e:
            result.success = False
            result.error = f"Distribution analysis failed: {e}"

        return result

    def _compute_histogram(self, values: np.ndarray,
                            bins: int = 10) -> Dict[str, Any]:
        """计算直方图"""
        hist, edges = np.histogram(values, bins=bins)
        return {
            "counts": hist.tolist(),
            "bin_edges": edges.tolist(),
            "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        }

    def _compute_percentiles(self, values: np.ndarray) -> Dict[str, float]:
        """计算分位数"""
        return {
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    def _classify_distribution(self, values: np.ndarray) -> str:
        """分类分布类型"""
        if len(values) < 4:
            return "unknown"

        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return "constant"

        skewness = self._skewness(values, mean, std)
        kurtosis = self._kurtosis(values, mean, std)

        if abs(skewness) < 0.5 and abs(kurtosis - 3.0) < 1.0:
            return "normal"
        elif skewness > 1.0:
            return "right_skewed"
        elif skewness < -1.0:
            return "left_skewed"
        elif kurtosis > 5.0:
            return "heavy_tailed"
        else:
            return "uniform"

    def _compute_moments(self, values: np.ndarray) -> Dict[str, float]:
        """计算矩"""
        mean = np.mean(values)
        std = np.std(values)
        return {
            "mean": float(mean),
            "variance": float(np.var(values)),
            "std": float(std),
            "skewness": float(self._skewness(values, mean, std)),
            "kurtosis": float(self._kurtosis(values, mean, std)),
        }

    def _skewness(self, values: np.ndarray, mean: float, std: float) -> float:
        """偏度"""
        if std < 1e-10:
            return 0.0
        n = len(values)
        return float(np.sum(((values - mean) / std) ** 3) / n)

    def _kurtosis(self, values: np.ndarray, mean: float, std: float) -> float:
        """峰度"""
        if std < 1e-10:
            return 0.0
        n = len(values)
        return float(np.sum(((values - mean) / std) ** 4) / n)


class ClusterAnalyzer:
    """聚类分析

    数据自动分组，使用 K-Means 算法
    """

    def __init__(self, k: int = 3, max_iterations: int = 100):
        self.k = k
        self.max_iterations = max_iterations

    def analyze(self, data: List[List[float]],
                k: int = None) -> AnalysisResult:
        """执行聚类分析

        Args:
            data: 数据点列表（每个点是特征向量）
            k: 聚类数（可选，覆盖默认值）
        """
        result = AnalysisResult(
            dimension=AnalysisDimension.CLUSTERING,
            timestamp=time.time(),
        )

        if not data or len(data) < 2:
            result.success = False
            result.error = "Insufficient data for clustering"
            return result

        try:
            k = k or self.k
            k = min(k, len(data))

            data_arr = np.array(data)
            labels, centroids, silhouette = self._kmeans(data_arr, k)

            cluster_info = []
            for i in range(k):
                mask = labels == i
                cluster_points = data_arr[mask]
                if len(cluster_points) > 0:
                    cluster_info.append({
                        "cluster_id": i,
                        "size": int(np.sum(mask)),
                        "centroid": centroids[i].tolist(),
                        "mean": cluster_points.mean(axis=0).tolist(),
                        "std": cluster_points.std(axis=0).tolist(),
                    })

            result.data = {
                "k": k,
                "labels": labels.tolist(),
                "centroids": centroids.tolist(),
                "clusters": cluster_info,
                "silhouette_score": float(silhouette),
                "total_points": len(data),
            }

            result.confidence = float(max(0.0, min(1.0, silhouette)))

            result.add_insight(
                f"将 {len(data)} 个数据点分为 {k} 个聚类，"
                f"轮廓系数 {silhouette:.3f}"
            )

            largest = max(cluster_info, key=lambda c: c["size"])
            result.add_insight(
                f"最大聚类 #{largest['cluster_id']} 包含 {largest['size']} 个点"
            )

        except Exception as e:
            result.success = False
            result.error = f"Cluster analysis failed: {e}"

        return result

    def _kmeans(self, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """K-Means 算法"""
        n = len(data)
        k = min(k, n)

        np.random.seed(42)
        indices = np.random.choice(n, k, replace=False)
        centroids = data[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(self.max_iterations):
            distances = np.array([
                np.linalg.norm(data - c, axis=1) for c in centroids
            ])
            new_labels = np.argmin(distances, axis=0)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    centroids[i] = data[mask].mean(axis=0)

        silhouette = self._silhouette_score(data, labels, centroids)

        return labels, centroids, silhouette

    def _silhouette_score(self, data: np.ndarray, labels: np.ndarray,
                           centroids: np.ndarray) -> float:
        """轮廓系数（简化版）"""
        n = len(data)
        if n < 2 or len(np.unique(labels)) < 2:
            return 0.0

        total = 0.0
        for i in range(n):
            same_cluster = labels == labels[i]
            same_cluster[i] = False
            if np.sum(same_cluster) == 0:
                continue

            a = np.mean(np.linalg.norm(data[i] - data[same_cluster], axis=1))

            other_clusters = [c for c in np.unique(labels) if c != labels[i]]
            if not other_clusters:
                continue

            b_values = []
            for c in other_clusters:
                mask = labels == c
                b_values.append(np.mean(np.linalg.norm(data[i] - data[mask], axis=1)))
            b = min(b_values)

            if max(a, b) > 0:
                total += (b - a) / max(a, b)

        return total / n if n > 0 else 0.0


class ForecastAnalyzer:
    """预测分析

    预测未来值，使用简单方法（移动平均 + 线性外推）
    """

    def __init__(self, forecast_horizon: int = 5,
                 confidence_level: float = 0.95):
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level

    def analyze(self, data: TimeSeriesData,
                horizon: int = None) -> AnalysisResult:
        """执行预测分析"""
        result = AnalysisResult(
            dimension=AnalysisDimension.FORECAST,
            timestamp=time.time(),
        )

        if data.is_empty or data.length < 3:
            result.success = False
            result.error = "Insufficient data for forecasting (need >= 3 points)"
            return result

        try:
            horizon = horizon or self.forecast_horizon
            values = np.array(data.values)
            timestamps = np.array(data.timestamps)

            ma_forecast = self._moving_average_forecast(values, horizon)
            linear_forecast = self._linear_forecast(timestamps, values, horizon)

            combined_forecast = [(ma + lin) / 2 for ma, lin in
                                  zip(ma_forecast, linear_forecast)]

            confidence_intervals = self._compute_confidence_intervals(
                values, combined_forecast
            )

            result.data = {
                "horizon": horizon,
                "ma_forecast": ma_forecast,
                "linear_forecast": linear_forecast,
                "combined_forecast": combined_forecast,
                "confidence_intervals": confidence_intervals,
                "last_value": float(values[-1]),
                "trend_direction": "up" if combined_forecast[-1] > values[-1] else "down",
                "expected_change": float(combined_forecast[-1] - values[-1]),
            }

            result.confidence = 0.6

            direction = "上升" if combined_forecast[-1] > values[-1] else "下降"
            change_pct = abs(combined_forecast[-1] - values[-1]) / max(abs(values[-1]), 1e-10) * 100
            result.add_insight(
                f"预测未来 {horizon} 步将{direction}，"
                f"预期变化 {change_pct:.2f}%"
            )

        except Exception as e:
            result.success = False
            result.error = f"Forecast analysis failed: {e}"

        return result

    def _moving_average_forecast(self, values: np.ndarray,
                                  horizon: int) -> List[float]:
        """移动平均预测"""
        window = min(5, len(values))
        last_ma = float(np.mean(values[-window:]))
        return [last_ma] * horizon

    def _linear_forecast(self, timestamps: np.ndarray,
                          values: np.ndarray,
                          horizon: int) -> List[float]:
        """线性外推预测"""
        if len(values) < 2:
            return [float(values[-1])] * horizon

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)

        future_x = np.arange(len(values), len(values) + horizon)
        return np.polyval(coeffs, future_x).tolist()

    def _compute_confidence_intervals(self, values: np.ndarray,
                                       forecast: List[float]) -> List[Dict[str, float]]:
        """计算置信区间"""
        std = float(np.std(values))
        z = 1.96  # 95% 置信度

        intervals = []
        for i, f in enumerate(forecast):
            margin = std * z * math.sqrt(1 + (i + 1) / len(values))
            intervals.append({
                "lower": float(f - margin),
                "upper": float(f + margin),
                "center": float(f),
            })
        return intervals
