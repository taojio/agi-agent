import numpy as np
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque


class MetricCategory(Enum):
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    COMPLEXITY = "complexity"
    FAIRNESS = "fairness"
    STABILITY = "stability"


class MetricType(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MAPE = "mape"
    R2_SCORE = "r2_score"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    PARAMETER_COUNT = "parameter_count"
    FLOPS = "flops"
    DROPOUT_RESILIENCE = "dropout_resilience"
    NOISE_ROBUSTNESS = "noise_robustness"
    ADVERSARIAL_ROBUSTNESS = "adversarial_robustness"
    DATA_SHIFT_RESILIENCE = "data_shift_resilience"
    TRAIN_TIME = "train_time"
    INFERENCE_TIME = "inference_time"
    MODEL_SIZE = "model_size"
    CONVERGENCE_SPEED = "convergence_speed"
    GENERALIZATION_GAP = "generalization_gap"
    CALIBRATION_ERROR = "calibration_error"
    COVERAGE = "coverage"
    DIVERSITY = "diversity"


@dataclass
class MetricDefinition:
    metric_type: MetricType
    category: MetricCategory
    name: str
    description: str
    direction: str
    unit: str
    threshold: Optional[float] = None
    weight: float = 1.0


@dataclass
class MetricValue:
    metric_type: MetricType
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class PerformanceSnapshot:
    snapshot_id: str
    timestamp: float
    metrics: Dict[str, MetricValue]
    model_version: str = ""
    dataset_name: str = ""
    environment: str = "default"


@dataclass
class EvaluationResult:
    overall_score: float
    category_scores: Dict[str, float]
    metric_scores: Dict[str, float]
    snapshot: PerformanceSnapshot
    timestamp: float
    success: bool


class MetricRegistry:
    def __init__(self):
        self.metrics: Dict[str, MetricDefinition] = {}
        self._init_default_metrics()

    def _init_default_metrics(self):
        self.register(MetricDefinition(
            metric_type=MetricType.ACCURACY,
            category=MetricCategory.ACCURACY,
            name="准确率",
            description="模型预测正确的比例",
            direction="higher",
            unit="%",
            threshold=0.8,
            weight=1.0
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.PRECISION,
            category=MetricCategory.ACCURACY,
            name="精确率",
            description="预测为正例的样本中实际为正例的比例",
            direction="higher",
            unit="%",
            threshold=0.7,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.RECALL,
            category=MetricCategory.ACCURACY,
            name="召回率",
            description="实际为正例的样本中被预测为正例的比例",
            direction="higher",
            unit="%",
            threshold=0.7,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.F1_SCORE,
            category=MetricCategory.ACCURACY,
            name="F1分数",
            description="精确率和召回率的调和平均数",
            direction="higher",
            unit="",
            threshold=0.7,
            weight=1.0
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.AUC,
            category=MetricCategory.ACCURACY,
            name="AUC",
            description="ROC曲线下面积",
            direction="higher",
            unit="",
            threshold=0.8,
            weight=0.9
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.MSE,
            category=MetricCategory.ACCURACY,
            name="均方误差",
            description="预测值与真实值差的平方的均值",
            direction="lower",
            unit="",
            threshold=0.1,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.R2_SCORE,
            category=MetricCategory.ACCURACY,
            name="R²分数",
            description="决定系数，衡量模型拟合优度",
            direction="higher",
            unit="",
            threshold=0.7,
            weight=0.9
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.THROUGHPUT,
            category=MetricCategory.EFFICIENCY,
            name="吞吐量",
            description="单位时间内处理的数据量",
            direction="higher",
            unit="samples/sec",
            threshold=100,
            weight=0.7
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.LATENCY,
            category=MetricCategory.EFFICIENCY,
            name="延迟",
            description="单次推理的平均时间",
            direction="lower",
            unit="ms",
            threshold=100,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.MEMORY_USAGE,
            category=MetricCategory.EFFICIENCY,
            name="内存使用",
            description="模型运行时占用的内存",
            direction="lower",
            unit="MB",
            threshold=512,
            weight=0.6
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.PARAMETER_COUNT,
            category=MetricCategory.COMPLEXITY,
            name="参数数量",
            description="模型的可训练参数总数",
            direction="lower",
            unit="",
            weight=0.5
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.TRAIN_TIME,
            category=MetricCategory.EFFICIENCY,
            name="训练时间",
            description="完成一轮训练所需的时间",
            direction="lower",
            unit="min",
            weight=0.6
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.INFERENCE_TIME,
            category=MetricCategory.EFFICIENCY,
            name="推理时间",
            description="单次推理的时间",
            direction="lower",
            unit="ms",
            threshold=50,
            weight=0.7
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.DROPOUT_RESILIENCE,
            category=MetricCategory.ROBUSTNESS,
            name="Dropout韧性",
            description="在Dropout下的性能保持程度",
            direction="higher",
            unit="%",
            threshold=0.9,
            weight=0.7
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.NOISE_ROBUSTNESS,
            category=MetricCategory.ROBUSTNESS,
            name="噪声鲁棒性",
            description="对输入噪声的抵抗能力",
            direction="higher",
            unit="%",
            threshold=0.85,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.ADVERSARIAL_ROBUSTNESS,
            category=MetricCategory.ROBUSTNESS,
            name="对抗鲁棒性",
            description="对对抗样本的抵抗能力",
            direction="higher",
            unit="%",
            threshold=0.7,
            weight=0.9
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.DATA_SHIFT_RESILIENCE,
            category=MetricCategory.ROBUSTNESS,
            name="数据偏移韧性",
            description="对数据分布变化的适应能力",
            direction="higher",
            unit="%",
            threshold=0.8,
            weight=0.7
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.CONVERGENCE_SPEED,
            category=MetricCategory.STABILITY,
            name="收敛速度",
            description="模型达到收敛所需的迭代次数",
            direction="lower",
            unit="iterations",
            weight=0.6
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.GENERALIZATION_GAP,
            category=MetricCategory.STABILITY,
            name="泛化差距",
            description="训练集和测试集性能差距",
            direction="lower",
            unit="%",
            threshold=5,
            weight=0.8
        ))
        self.register(MetricDefinition(
            metric_type=MetricType.CALIBRATION_ERROR,
            category=MetricCategory.STABILITY,
            name="校准误差",
            description="模型预测概率与实际频率的偏差",
            direction="lower",
            unit="",
            threshold=0.1,
            weight=0.7
        ))

    def register(self, definition: MetricDefinition):
        self.metrics[definition.metric_type.value] = definition

    def get(self, metric_type: str) -> Optional[MetricDefinition]:
        return self.metrics.get(metric_type)

    def get_by_category(self, category: MetricCategory) -> List[MetricDefinition]:
        return [m for m in self.metrics.values() if m.category == category]

    def list_all(self) -> List[MetricDefinition]:
        return list(self.metrics.values())


class PerformanceEvaluator:
    def __init__(self):
        self.registry = MetricRegistry()
        self.history: deque = deque(maxlen=1000)
        self.baselines: Dict[str, float] = {}
        self._metric_computers: Dict[str, Callable] = {}
        self._init_computers()

    def _init_computers(self):
        self._metric_computers[MetricType.ACCURACY.value] = self._compute_accuracy
        self._metric_computers[MetricType.PRECISION.value] = self._compute_precision
        self._metric_computers[MetricType.RECALL.value] = self._compute_recall
        self._metric_computers[MetricType.F1_SCORE.value] = self._compute_f1
        self._metric_computers[MetricType.MSE.value] = self._compute_mse
        self._metric_computers[MetricType.R2_SCORE.value] = self._compute_r2
        self._metric_computers[MetricType.MAE.value] = self._compute_mae

    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def _compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def _compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def _compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = self._compute_precision(y_true, y_pred)
        recall = self._compute_recall(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    def _compute_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def _compute_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def compute_metric(self, metric_type: str, y_true: np.ndarray,
                       y_pred: np.ndarray) -> Optional[float]:
        computer = self._metric_computers.get(metric_type)
        if computer:
            return computer(y_true, y_pred)
        return None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 additional_metrics: Dict[str, float] = None) -> EvaluationResult:
        metric_scores = {}

        for metric_type in [MetricType.ACCURACY, MetricType.PRECISION,
                           MetricType.RECALL, MetricType.F1_SCORE,
                           MetricType.MSE, MetricType.R2_SCORE]:
            value = self.compute_metric(metric_type.value, y_true, y_pred)
            if value is not None:
                metric_scores[metric_type.value] = value

        if additional_metrics:
            metric_scores.update(additional_metrics)

        category_scores = self._compute_category_scores(metric_scores)
        overall_score = self._compute_overall_score(category_scores, metric_scores)

        snapshot = PerformanceSnapshot(
            snapshot_id=f"perf_{int(time.time() * 1000)}",
            timestamp=time.time(),
            metrics={k: MetricValue(metric_type=MetricType(k), value=v, timestamp=time.time())
                     for k, v in metric_scores.items()}
        )

        self.history.append(snapshot)

        return EvaluationResult(
            overall_score=overall_score,
            category_scores=category_scores,
            metric_scores=metric_scores,
            snapshot=snapshot,
            timestamp=time.time(),
            success=True
        )

    def _compute_category_scores(self, metric_scores: Dict[str, float]) -> Dict[str, float]:
        categories = {}
        for metric_name, score in metric_scores.items():
            definition = self.registry.get(metric_name)
            if definition:
                if definition.category.value not in categories:
                    categories[definition.category.value] = []
                normalized = self._normalize_score(score, definition)
                categories[definition.category.value].append((normalized, definition.weight))

        result = {}
        for cat, items in categories.items():
            total_weight = sum(w for _, w in items)
            if total_weight > 0:
                result[cat] = sum(s * w for s, w in items) / total_weight
            else:
                result[cat] = 0.0

        return result

    def _normalize_score(self, score: float, definition: MetricDefinition) -> float:
        if definition.direction == "higher":
            if definition.threshold:
                return min(1.0, score / definition.threshold)
            return min(1.0, score)
        else:
            if definition.threshold and definition.threshold > 0:
                return max(0.0, 1.0 - score / definition.threshold)
            return max(0.0, 1.0 - score)

    def _compute_overall_score(self, category_scores: Dict[str, float],
                               metric_scores: Dict[str, float]) -> float:
        category_weights = {
            MetricCategory.ACCURACY.value: 0.35,
            MetricCategory.EFFICIENCY.value: 0.25,
            MetricCategory.ROBUSTNESS.value: 0.20,
            MetricCategory.STABILITY.value: 0.15,
            MetricCategory.COMPLEXITY.value: 0.05
        }

        total = 0.0
        total_weight = 0.0

        for cat, score in category_scores.items():
            weight = category_weights.get(cat, 0.1)
            total += score * weight
            total_weight += weight

        if total_weight > 0:
            total /= total_weight

        return total

    def set_baseline(self, metric_type: str, value: float):
        self.baselines[metric_type] = value

    def compare_with_baseline(self, metric_type: str, value: float) -> Dict[str, Any]:
        baseline = self.baselines.get(metric_type)
        if baseline is None:
            return {"value": value, "baseline": None, "improvement": None}

        definition = self.registry.get(metric_type)
        if definition and definition.direction == "higher":
            improvement = (value - baseline) / max(baseline, 0.01)
        else:
            improvement = (baseline - value) / max(baseline, 0.01)

        return {
            "value": value,
            "baseline": baseline,
            "improvement": improvement,
            "improved": improvement > 0
        }

    def get_recent_trend(self, metric_type: str, window_size: int = 10) -> Dict[str, Any]:
        recent = list(self.history)[-window_size:]
        values = []
        for snapshot in recent:
            if metric_type in snapshot.metrics:
                values.append(snapshot.metrics[metric_type].value)

        if len(values) < 2:
            return {"trend": "stable", "slope": 0.0, "values": values}

        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        definition = self.registry.get(metric_type)
        if definition:
            if definition.direction == "lower" and trend == "increasing":
                trend = "deteriorating"
            elif definition.direction == "higher" and trend == "decreasing":
                trend = "deteriorating"

        return {
            "trend": trend,
            "slope": float(slope),
            "values": values,
            "avg_value": float(np.mean(values)),
            "std_value": float(np.std(values))
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"error": "No evaluation history available"}

        latest = self.history[-1]
        return {
            "total_evaluations": len(self.history),
            "latest_timestamp": latest.timestamp,
            "metrics": {k: v.value for k, v in latest.metrics.items()},
            "baselines": self.baselines
        }

    def compute_robustness_metrics(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        results = {}

        try:
            predictions = model.predict(X_test)
            base_accuracy = self._compute_accuracy(y_test, predictions)

            noise = np.random.normal(0, 0.1, X_test.shape)
            noisy_predictions = model.predict(X_test + noise)
            noise_accuracy = self._compute_accuracy(y_test, noisy_predictions)
            results[MetricType.NOISE_ROBUSTNESS.value] = noise_accuracy / base_accuracy if base_accuracy > 0 else 0.0

            dropout_predictions = model.predict(X_test)
            dropout_accuracy = self._compute_accuracy(y_test, dropout_predictions)
            results[MetricType.DROPOUT_RESILIENCE.value] = dropout_accuracy / base_accuracy if base_accuracy > 0 else 0.0

        except Exception:
            pass

        return results


class BenchmarkRunner:
    def __init__(self, evaluator: PerformanceEvaluator = None):
        self.evaluator = evaluator or PerformanceEvaluator()
        self.benchmark_results: Dict[str, EvaluationResult] = {}

    def run_benchmark(self, model, dataset_name: str, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()

        try:
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_time = time.time() - start_time

            train_eval = self.evaluator.evaluate(y_train, train_preds)
            test_eval = self.evaluator.evaluate(y_test, test_preds)

            robustness_metrics = self.evaluator.compute_robustness_metrics(model, X_test, y_test)

            generalization_gap = abs(train_eval.metric_scores.get("accuracy", 0) -
                                     test_eval.metric_scores.get("accuracy", 0))

            benchmark_id = f"bench_{dataset_name}_{int(time.time())}"
            self.benchmark_results[benchmark_id] = test_eval

            return {
                "benchmark_id": benchmark_id,
                "dataset": dataset_name,
                "train_time": train_time,
                "train_metrics": train_eval.metric_scores,
                "test_metrics": test_eval.metric_scores,
                "robustness_metrics": robustness_metrics,
                "generalization_gap": generalization_gap,
                "overall_score": test_eval.overall_score,
                "category_scores": test_eval.category_scores
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def compare_models(self, models: Dict[str, Any], dataset_name: str,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        results = {}
        for name, model in models.items():
            result = self.run_benchmark(model, dataset_name, X_train, y_train, X_test, y_test)
            if result.get("success") is not False:
                results[name] = result

        if not results:
            return {"error": "No models completed benchmark"}

        best_model = max(results.keys(), key=lambda k: results[k]["overall_score"])

        return {
            "comparison": results,
            "best_model": best_model,
            "best_score": results[best_model]["overall_score"],
            "dataset": dataset_name
        }