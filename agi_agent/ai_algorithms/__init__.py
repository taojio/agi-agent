"""
ai_algorithms/__init__.py - AI 算法组件库

Phase 4: 提供 6 类 AI 算法组件：
- 时间序列预测 (forecasting)
- 异常检测 (anomaly_detection)
- 自动聚类 (clustering)
- 特征工程 (feature_engineering)
- 模式识别 (pattern_recognition)
- 智能调度 (scheduling)
"""
from .base import (
    AIAlgorithmComponent, AlgorithmStatus, AlgorithmMetrics,
)
from .forecasting import (
    TimeSeriesForecaster, ForecastResult,
    ExponentialSmoothingForecaster, ARIMAForecaster,
    LinearRegressionForecaster, MovingAverageForecaster,
    ForecastingEnsemble,
)
from .anomaly_detection import (
    AnomalyDetector, AnomalyResult, AnomalyReport,
    AnomalyType, AnomalySeverity,
    ZScoreDetector, IQRDetector, IsolationForestDetector, AutoEncoderDetector,
)
from .clustering import (
    AutoClusterer, ClusterResult,
    KMeansClusterer, DBSCANClusterer, HierarchicalClusterer,
)
from .feature_engineering import (
    FeatureEngineer, FeatureResult,
    PCAComponent, MutualInfoSelector, FeatureImportanceEvaluator,
)
from .pattern_recognition import (
    PatternRecognizer, PatternResult,
    AprioriMiner, SequencePatternMiner,
    FrequentItemset, AssociationRule, SequencePattern,
)
from .scheduling import (
    SmartScheduler, ScheduleResult, Task,
    GeneticScheduler, SimulatedAnnealingScheduler, GreedyScheduler,
)

__all__ = [
    # 基类
    "AIAlgorithmComponent", "AlgorithmStatus", "AlgorithmMetrics",
    # 时间序列预测
    "TimeSeriesForecaster", "ForecastResult",
    "ExponentialSmoothingForecaster", "ARIMAForecaster",
    "LinearRegressionForecaster", "MovingAverageForecaster",
    "ForecastingEnsemble",
    # 异常检测
    "AnomalyDetector", "AnomalyResult", "AnomalyReport",
    "AnomalyType", "AnomalySeverity",
    "ZScoreDetector", "IQRDetector", "IsolationForestDetector", "AutoEncoderDetector",
    # 聚类
    "AutoClusterer", "ClusterResult",
    "KMeansClusterer", "DBSCANClusterer", "HierarchicalClusterer",
    # 特征工程
    "FeatureEngineer", "FeatureResult",
    "PCAComponent", "MutualInfoSelector", "FeatureImportanceEvaluator",
    # 模式识别
    "PatternRecognizer", "PatternResult",
    "AprioriMiner", "SequencePatternMiner",
    "FrequentItemset", "AssociationRule", "SequencePattern",
    # 调度
    "SmartScheduler", "ScheduleResult", "Task",
    "GeneticScheduler", "SimulatedAnnealingScheduler", "GreedyScheduler",
]
