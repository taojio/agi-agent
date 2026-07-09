"""
ai_algorithms/forecasting.py - 时间序列预测组件

支持多种预测算法：
- ARIMA（自回归积分滑动平均，简化实现）
- 指数平滑（Holt-Winters 三次指数平滑）
- 线性回归预测
- 移动平均预测
- 自回归（AR）预测
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


@dataclass
class ForecastResult:
    """预测结果"""
    values: np.ndarray                    # 预测值
    lower_bound: np.ndarray               # 置信区间下界
    upper_bound: np.ndarray               # 置信区间上界
    horizon: int                          # 预测步长
    method: str                           # 使用的算法
    confidence_level: float = 0.95        # 置信水平
    residuals: Optional[np.ndarray] = None  # 残差

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist(),
            "horizon": self.horizon,
            "method": self.method,
            "confidence_level": self.confidence_level,
            "residuals": self.residuals.tolist() if self.residuals is not None else None,
        }


class ExponentialSmoothingForecaster(AIAlgorithmComponent):
    """三次指数平滑预测器（Holt-Winters）

    适用于具有趋势和季节性的时间序列。

    参数:
        alpha: 水平平滑系数 (0-1)
        beta: 趋势平滑系数 (0-1)
        gamma: 季节平滑系数 (0-1)
        seasonal_period: 季节周期长度
    """

    def __init__(self, name: str = "exp_smoothing",
                 alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1,
                 seasonal_period: int = 0):
        super().__init__(name, alpha=alpha, beta=beta, gamma=gamma,
                         seasonal_period=seasonal_period)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self._level = 0.0
        self._trend = 0.0
        self._seasonals: List[float] = []
        self._fitted_values: Optional[np.ndarray] = None
        self._last_values: List[float] = []

    @property
    def component_type(self) -> str:
        return "forecasting"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ExponentialSmoothingForecaster":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64).flatten()
            if len(data) < 2:
                raise ValueError("Need at least 2 data points")

            # 初始化
            if self.seasonal_period > 0 and len(data) >= 2 * self.seasonal_period:
                # 季节性初始化
                self._level = np.mean(data[:self.seasonal_period])
                self._trend = (np.mean(data[self.seasonal_period:2 * self.seasonal_period])
                               - self._level) / self.seasonal_period
                self._seasonals = list(data[:self.seasonal_period] - self._level)
            else:
                # 无季节性
                self._level = data[0]
                self._trend = data[1] - data[0] if len(data) > 1 else 0.0
                self._seasonals = []

            fitted = np.zeros(len(data))
            for i in range(len(data)):
                if i == 0:
                    fitted[i] = self._level + self._trend + \
                               (self._seasonals[i % len(self._seasonals)]
                                if self._seasonals else 0)
                    continue
                # Holt-Winters 更新
                if self._seasonals:
                    season_idx = (i - 1) % len(self._seasonals)
                    forecast = self._level + self._trend + self._seasonals[season_idx]
                    new_level = self.alpha * (data[i] - self._seasonals[season_idx]) + \
                                (1 - self.alpha) * (self._level + self._trend)
                    new_trend = self.beta * (new_level - self._level) + \
                                (1 - self.beta) * self._trend
                    new_seasonal = self.gamma * (data[i] - new_level) + \
                                   (1 - self.gamma) * self._seasonals[season_idx]
                    self._level = new_level
                    self._trend = new_trend
                    self._seasonals[season_idx] = new_seasonal
                else:
                    # 二次指数平滑
                    forecast = self._level + self._trend
                    new_level = self.alpha * data[i] + (1 - self.alpha) * (self._level + self._trend)
                    new_trend = self.beta * (new_level - self._level) + (1 - self.beta) * self._trend
                    self._level = new_level
                    self._trend = new_trend
                fitted[i] = forecast

            self._fitted_values = fitted
            self._last_values = list(data)

            # 计算残差
            residuals = data - fitted
            self.metrics.rmse = float(np.sqrt(np.mean(residuals ** 2)))
            self.metrics.mae = float(np.mean(np.abs(residuals)))
            self.metrics.sample_count = len(data)

            self._end_training(start)
            return self
        except Exception as e:
            self.status = AlgorithmStatus.ERROR  # 错误状态
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            horizon = int(np.asarray(X).flatten()[0]) if np.asarray(X).size > 0 else 1
            predictions = np.zeros(horizon)
            level, trend = self._level, self._trend
            seasonals = list(self._seasonals) if self._seasonals else []

            for h in range(horizon):
                if seasonals:
                    season_idx = h % len(seasonals)
                    predictions[h] = level + (h + 1) * trend + seasonals[season_idx]
                else:
                    predictions[h] = level + (h + 1) * trend

            self._end_prediction(start)
            return predictions
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def forecast(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """预测未来 horizon 步"""
        predictions = self.predict(np.array([horizon]))
        # 置信区间基于残差标准差
        if self._fitted_values is not None and len(self._last_values) > 0:
            residuals = np.array(self._last_values) - self._fitted_values
            std = float(np.std(residuals)) if len(residuals) > 1 else 0.1
        else:
            std = 0.1
        # 置信区间随预测步长扩大
        z_value = 1.96 if confidence_level == 0.95 else 2.58
        margins = np.array([std * z_value * np.sqrt(h + 1) for h in range(horizon)])
        lower = predictions - margins
        upper = predictions + margins

        return ForecastResult(
            values=predictions,
            lower_bound=lower,
            upper_bound=upper,
            horizon=horizon,
            method="exponential_smoothing",
            confidence_level=confidence_level,
            residuals=(np.array(self._last_values) - self._fitted_values
                      if self._fitted_values is not None else None),
        )


class ARIMAForecaster(AIAlgorithmComponent):
    """ARIMA 预测器（简化实现）

    实现 AR(p) + 差分(d) + MA(q) 的简化版本。
    使用最小二乘法估计 AR 系数，MA 部分使用残差滑动平均。

    参数:
        p: 自回归阶数
        d: 差分阶数
        q: 滑动平均阶数
    """

    def __init__(self, name: str = "arima",
                 p: int = 2, d: int = 1, q: int = 1):
        super().__init__(name, p=p, d=d, q=q)
        self.p = p
        self.d = d
        self.q = q
        self._ar_coeffs: Optional[np.ndarray] = None
        self._ma_coeffs: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._original_data: Optional[np.ndarray] = None
        self._diffed_data: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "forecasting"

    def _difference(self, data: np.ndarray, d: int) -> np.ndarray:
        """d 阶差分"""
        diffed = data.copy()
        for _ in range(d):
            diffed = np.diff(diffed)
        return diffed

    def _integrate(self, diffed: np.ndarray, d: int,
                   original: np.ndarray) -> np.ndarray:
        """逆差分（积分）"""
        result = diffed.copy()
        for i in range(d):
            if i == 0:
                # 使用最后一个原始值作为起点
                result = np.concatenate([[original[-1]], result]).cumsum()[1:]
            else:
                result = np.concatenate([[result[0]], result]).cumsum()[1:]
        return result

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ARIMAForecaster":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64).flatten()
            if len(data) < self.p + self.d + 2:
                raise ValueError(f"Need at least {self.p + self.d + 2} data points")

            self._original_data = data
            # 差分
            diffed = self._difference(data, self.d)
            self._diffed_data = diffed

            # AR(p) 参数估计：使用最小二乘法
            if self.p > 0 and len(diffed) > self.p:
                # 构建设计矩阵
                n = len(diffed)
                X_ar = np.zeros((n - self.p, self.p))
                y_ar = np.zeros(n - self.p)
                for i in range(self.p, n):
                    X_ar[i - self.p] = diffed[i - self.p:i][::-1]
                    y_ar[i - self.p] = diffed[i]
                # 最小二乘求解
                self._ar_coeffs = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
            else:
                self._ar_coeffs = np.zeros(self.p)

            # 计算残差
            predictions = np.zeros(len(diffed))
            for i in range(self.p, len(diffed)):
                if self.p > 0:
                    predictions[i] = np.dot(self._ar_coeffs, diffed[i - self.p:i][::-1])

            self._residuals = diffed - predictions

            # MA(q) 参数：使用残差的自相关
            if self.q > 0 and len(self._residuals) > self.q:
                # 简化：MA 系数为残差的滑动平均权重
                ma_window = self._residuals[-self.q:]
                self._ma_coeffs = ma_window / (np.sum(np.abs(ma_window)) + 1e-8)
            else:
                self._ma_coeffs = np.zeros(self.q)

            # 拟合值
            fitted = predictions
            # 逆差分得到原始空间的拟合值
            if self.d > 0:
                # 简化：直接使用差分后的拟合值
                pass

            self.metrics.rmse = float(np.sqrt(np.mean(self._residuals ** 2)))
            self.metrics.mae = float(np.mean(np.abs(self._residuals)))
            self.metrics.sample_count = len(data)

            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            horizon = int(np.asarray(X).flatten()[0]) if np.asarray(X).size > 0 else 1
            predictions = np.zeros(horizon)
            # 使用差分后的数据进行 AR 预测
            recent = list(self._diffed_data[-self.p:]) if self.p > 0 else []
            recent_residuals = list(self._residuals[-self.q:]) if self.q > 0 else []

            for h in range(horizon):
                # AR 部分
                ar_pred = 0.0
                if self.p > 0 and len(recent) >= self.p:
                    ar_pred = np.dot(self._ar_coeffs, recent[-self.p:][::-1])
                # MA 部分
                ma_pred = 0.0
                if self.q > 0 and len(recent_residuals) >= self.q:
                    ma_pred = np.dot(self._ma_coeffs, recent_residuals[-self.q:][::-1])
                pred = ar_pred + ma_pred
                predictions[h] = pred
                recent.append(pred)
                recent_residuals.append(0.0)  # 假设未来残差为 0

            # 逆差分
            if self.d > 0:
                # 累积求和，从最后一个原始值开始
                last_value = self._original_data[-1]
                for h in range(horizon):
                    last_value += predictions[h]
                    predictions[h] = last_value

            self._end_prediction(start)
            return predictions
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def forecast(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        predictions = self.predict(np.array([horizon]))
        std = float(np.std(self._residuals)) if self._residuals is not None else 0.1
        z_value = 1.96 if confidence_level == 0.95 else 2.58
        margins = np.array([std * z_value * np.sqrt(h + 1) for h in range(horizon)])
        return ForecastResult(
            values=predictions,
            lower_bound=predictions - margins,
            upper_bound=predictions + margins,
            horizon=horizon,
            method=f"ARIMA({self.p},{self.d},{self.q})",
            confidence_level=confidence_level,
            residuals=self._residuals,
        )


class LinearRegressionForecaster(AIAlgorithmComponent):
    """线性回归预测器

    使用线性回归拟合时间序列的趋势。
    """

    def __init__(self, name: str = "linear_regression"):
        super().__init__(name)
        self._slope = 0.0
        self._intercept = 0.0
        self._residuals: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "forecasting"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LinearRegressionForecaster":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64).flatten()
            n = len(data)
            if n < 2:
                raise ValueError("Need at least 2 data points")
            x = np.arange(n, dtype=np.float64)
            # 最小二乘
            A = np.vstack([x, np.ones(n)]).T
            self._slope, self._intercept = np.linalg.lstsq(A, data, rcond=None)[0]
            fitted = self._slope * x + self._intercept
            self._residuals = data - fitted
            self.metrics.rmse = float(np.sqrt(np.mean(self._residuals ** 2)))
            self.metrics.mae = float(np.mean(np.abs(self._residuals)))
            self.metrics.sample_count = n
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            horizon = int(np.asarray(X).flatten()[0]) if np.asarray(X).size > 0 else 1
            n = self.metrics.sample_count
            future_x = np.arange(n, n + horizon, dtype=np.float64)
            predictions = self._slope * future_x + self._intercept
            self._end_prediction(start)
            return predictions
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def forecast(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        predictions = self.predict(np.array([horizon]))
        std = float(np.std(self._residuals)) if self._residuals is not None else 0.1
        z_value = 1.96 if confidence_level == 0.95 else 2.58
        margins = np.array([std * z_value * np.sqrt(h + 1) for h in range(horizon)])
        return ForecastResult(
            values=predictions,
            lower_bound=predictions - margins,
            upper_bound=predictions + margins,
            horizon=horizon,
            method="linear_regression",
            confidence_level=confidence_level,
            residuals=self._residuals,
        )


class MovingAverageForecaster(AIAlgorithmComponent):
    """移动平均预测器"""

    def __init__(self, name: str = "moving_average", window: int = 5):
        super().__init__(name, window=window)
        self.window = window
        self._data: Optional[np.ndarray] = None
        self._last_ma = 0.0

    @property
    def component_type(self) -> str:
        return "forecasting"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MovingAverageForecaster":
        start = self._start_training()
        try:
            data = np.asarray(X, dtype=np.float64).flatten()
            self._data = data
            self._last_ma = float(np.mean(data[-self.window:])) if len(data) >= self.window \
                           else float(np.mean(data))
            # 拟合值
            fitted = np.zeros(len(data))
            for i in range(len(data)):
                start_idx = max(0, i - self.window + 1)
                fitted[i] = float(np.mean(data[start_idx:i + 1]))
            residuals = data - fitted
            self.metrics.rmse = float(np.sqrt(np.mean(residuals ** 2)))
            self.metrics.mae = float(np.mean(np.abs(residuals)))
            self.metrics.sample_count = len(data)
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            horizon = int(np.asarray(X).flatten()[0]) if np.asarray(X).size > 0 else 1
            # 简单地使用最后一个移动平均值
            predictions = np.full(horizon, self._last_ma)
            self._end_prediction(start)
            return predictions
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def forecast(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        predictions = self.predict(np.array([horizon]))
        if self._data is not None:
            residuals = self._data[-self.window:] - self._last_ma if \
                       len(self._data) >= self.window else self._data - self._last_ma
            std = float(np.std(residuals)) if len(residuals) > 1 else 0.1
        else:
            std = 0.1
        z_value = 1.96 if confidence_level == 0.95 else 2.58
        margins = np.full(horizon, std * z_value)
        return ForecastResult(
            values=predictions,
            lower_bound=predictions - margins,
            upper_bound=predictions + margins,
            horizon=horizon,
            method=f"moving_average(window={self.window})",
            confidence_level=confidence_level,
        )


class ForecastingEnsemble(AIAlgorithmComponent):
    """预测集成器

    组合多个预测器，通过加权平均提升预测稳定性。
    """

    def __init__(self, name: str = "ensemble",
                 forecasters: Optional[List[AIAlgorithmComponent]] = None,
                 weights: Optional[List[float]] = None):
        super().__init__(name)
        self._forecasters: List[AIAlgorithmComponent] = forecasters or []
        self._weights: List[float] = weights or []
        self._residuals: Optional[np.ndarray] = None

    @property
    def component_type(self) -> str:
        return "forecasting"

    def add_forecaster(self, forecaster: AIAlgorithmComponent,
                        weight: float = 1.0) -> "ForecastingEnsemble":
        self._forecasters.append(forecaster)
        self._weights.append(weight)
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ForecastingEnsemble":
        start = self._start_training()
        try:
            if not self._forecasters:
                raise ValueError("No forecasters in ensemble")
            data = np.asarray(X, dtype=np.float64).flatten()
            # 训练所有预测器
            all_predictions = []
            for f in self._forecasters:
                f.fit(data)
                fitted = f.predict(np.array([1]))  # 单步预测作为拟合
                all_predictions.append(fitted)

            # 如果权重未设置，根据 RMSE 自动加权
            if not self._weights or sum(self._weights) == 0:
                self._weights = []
                for f in self._forecasters:
                    rmse = f.metrics.rmse if f.metrics.rmse > 0 else 1.0
                    self._weights.append(1.0 / rmse)

            # 归一化权重
            total_weight = sum(self._weights)
            self._weights = [w / total_weight for w in self._weights]

            self.metrics.sample_count = len(data)
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        start = self._start_prediction()
        try:
            horizon = int(np.asarray(X).flatten()[0]) if np.asarray(X).size > 0 else 1
            predictions = np.zeros(horizon)
            for f, w in zip(self._forecasters, self._weights):
                pred = f.predict(np.array([horizon]))
                predictions += w * pred
            self._end_prediction(start)
            return predictions
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def forecast(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        predictions = self.predict(np.array([horizon]))
        # 收集各预测器的预测，计算标准差
        all_preds = []
        for f in self._forecasters:
            all_preds.append(f.predict(np.array([horizon])))
        if len(all_preds) > 1:
            pred_matrix = np.array(all_preds)
            std = float(np.mean(np.std(pred_matrix, axis=0)))
        else:
            std = 0.1
        z_value = 1.96 if confidence_level == 0.95 else 2.58
        margins = np.full(horizon, std * z_value)
        return ForecastResult(
            values=predictions,
            lower_bound=predictions - margins,
            upper_bound=predictions + margins,
            horizon=horizon,
            method="ensemble",
            confidence_level=confidence_level,
        )


class TimeSeriesForecaster:
    """时间序列预测统一接口

    提供多种预测算法的统一访问入口。

    Usage:
        forecaster = TimeSeriesForecaster()
        result = forecaster.forecast(data, horizon=10, method="auto")
    """

    METHODS = ["arima", "exp_smoothing", "linear", "moving_average", "ensemble", "auto"]

    def __init__(self):
        self._forecasters: Dict[str, AIAlgorithmComponent] = {
            "arima": ARIMAForecaster(),
            "exp_smoothing": ExponentialSmoothingForecaster(),
            "linear": LinearRegressionForecaster(),
            "moving_average": MovingAverageForecaster(),
        }

    def forecast(self, data: np.ndarray, horizon: int = 10,
                  method: str = "auto", **kwargs) -> ForecastResult:
        """预测时间序列

        Args:
            data: 时间序列数据
            horizon: 预测步长
            method: 预测方法 ("auto" 自动选择)
            **kwargs: 算法参数

        Returns:
            ForecastResult: 预测结果
        """
        data = np.asarray(data, dtype=np.float64).flatten()
        if len(data) < 2:
            raise ValueError("Need at least 2 data points")

        if method == "auto":
            method = self._select_best_method(data)

        if method == "ensemble":
            ensemble = ForecastingEnsemble()
            for f in self._forecasters.values():
                ensemble.add_forecaster(f)
            ensemble.fit(data)
            return ensemble.forecast(horizon)
        elif method in self._forecasters:
            forecaster = self._forecasters[method]
            # 应用参数
            if kwargs:
                forecaster.set_params(**kwargs)
            forecaster.fit(data)
            return forecaster.forecast(horizon)
        else:
            raise ValueError(f"Unknown method: {method}. Available: {self.METHODS}")

    def _select_best_method(self, data: np.ndarray) -> str:
        """自动选择最佳预测方法"""
        if len(data) < 10:
            return "linear"
        # 检测趋势
        x = np.arange(len(data))
        try:
            slope = np.polyfit(x, data, 1)[0]
            if abs(slope) > 0.1 * np.std(data):
                return "linear"  # 强趋势用线性回归
        except (ValueError, np.linalg.LinAlgError):
            pass
        # 默认用 ARIMA
        return "arima"

    def compare_methods(self, data: np.ndarray,
                          horizon: int = 5) -> Dict[str, Any]:
        """对比所有方法的预测效果"""
        data = np.asarray(data, dtype=np.float64).flatten()
        results = {}
        for method_name, forecaster in self._forecasters.items():
            try:
                forecaster.fit(data)
                result = forecaster.forecast(horizon)
                results[method_name] = {
                    "rmse": forecaster.metrics.rmse,
                    "mae": forecaster.metrics.mae,
                    "training_time": forecaster.metrics.training_time,
                    "prediction": result.values.tolist(),
                }
            except Exception as e:
                results[method_name] = {"error": str(e)}
        return results
