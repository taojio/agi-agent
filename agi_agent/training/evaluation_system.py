import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class MetricTier(Enum):
    CORE = "core"
    CAPABILITY = "capability"
    PROCESS = "process"


class ConvergenceState(Enum):
    DIVERGING = "diverging"
    STABLE = "stable"
    CONVERGING = "converging"
    CONVERGED = "converged"
    OVERFITTING = "overfitting"


@dataclass
class MetricSpec:
    name: str
    tier: MetricTier
    description: str
    higher_is_better: bool = True
    target_value: Optional[float] = None
    weight: float = 1.0
    unit: str = ""
    current_value: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=500))

    def update(self, value: float, step: int = None):
        self.current_value = value
        self.history.append({"step": step, "value": value, "timestamp": time.time()})

    def trend(self, window: int = 50) -> float:
        if len(self.history) < window:
            return 0.0
        values = [h["value"] for h in list(self.history)[-window:]]
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)

    def std(self, window: int = 50) -> float:
        if len(self.history) < 2:
            return 0.0
        values = [h["value"] for h in list(self.history)[-window:]]
        return float(np.std(values))


@dataclass
class EvaluationResult:
    overall_score: float
    tier_scores: Dict[str, float]
    metric_scores: Dict[str, Dict[str, float]]
    trends: Dict[str, float]
    convergence_state: ConvergenceState
    timestamp: float = field(default_factory=time.time)
    step: int = 0


class TrainingEvaluator:
    def __init__(self):
        self.metrics: Dict[str, MetricSpec] = {}
        self.evaluation_history: deque = deque(maxlen=1000)
        self.current_step = 0

        self._init_core_metrics()
        self._init_capability_metrics()
        self._init_process_metrics()

    def _init_core_metrics(self):
        self.metrics["overall_score"] = MetricSpec(
            name="overall_score",
            tier=MetricTier.CORE,
            description="综合性能得分",
            higher_is_better=True,
            target_value=0.8,
            weight=1.0
        )
        self.metrics["free_energy"] = MetricSpec(
            name="free_energy",
            tier=MetricTier.CORE,
            description="自由能水平",
            higher_is_better=False,
            target_value=0.2,
            weight=0.3
        )
        self.metrics["confidence"] = MetricSpec(
            name="confidence",
            tier=MetricTier.CORE,
            description="置信度水平",
            higher_is_better=True,
            target_value=0.7,
            weight=0.25
        )
        self.metrics["decision_accuracy"] = MetricSpec(
            name="decision_accuracy",
            tier=MetricTier.CORE,
            description="决策准确率",
            higher_is_better=True,
            target_value=0.85,
            weight=0.2
        )
        self.metrics["latency_ms"] = MetricSpec(
            name="latency_ms",
            tier=MetricTier.CORE,
            description="单步响应延迟(ms)",
            higher_is_better=False,
            target_value=500,
            weight=0.15,
            unit="ms"
        )
        self.metrics["safety_compliance_rate"] = MetricSpec(
            name="safety_compliance_rate",
            tier=MetricTier.CORE,
            description="安全合规率",
            higher_is_better=True,
            target_value=0.99,
            weight=0.1
        )

    def _init_capability_metrics(self):
        self.metrics["feature_discrimination"] = MetricSpec(
            name="feature_discrimination",
            tier=MetricTier.CAPABILITY,
            description="感知特征区分度",
            higher_is_better=True,
            target_value=0.7,
            weight=0.2
        )
        self.metrics["noise_robustness"] = MetricSpec(
            name="noise_robustness",
            tier=MetricTier.CAPABILITY,
            description="噪声鲁棒性",
            higher_is_better=True,
            target_value=0.8,
            weight=0.15
        )
        self.metrics["reasoning_accuracy"] = MetricSpec(
            name="reasoning_accuracy",
            tier=MetricTier.CAPABILITY,
            description="推理准确率",
            higher_is_better=True,
            target_value=0.75,
            weight=0.2
        )
        self.metrics["kg_node_count"] = MetricSpec(
            name="kg_node_count",
            tier=MetricTier.CAPABILITY,
            description="知识图谱节点数",
            higher_is_better=True,
            target_value=500,
            weight=0.1
        )
        self.metrics["learning_efficiency"] = MetricSpec(
            name="learning_efficiency",
            tier=MetricTier.CAPABILITY,
            description="学习效率(收敛速度)",
            higher_is_better=True,
            target_value=0.6,
            weight=0.2
        )
        self.metrics["knowledge_retention"] = MetricSpec(
            name="knowledge_retention",
            tier=MetricTier.CAPABILITY,
            description="知识保留率",
            higher_is_better=True,
            target_value=0.8,
            weight=0.15
        )

    def _init_process_metrics(self):
        self.metrics["gradient_norm"] = MetricSpec(
            name="gradient_norm",
            tier=MetricTier.PROCESS,
            description="梯度范数",
            higher_is_better=False,
            target_value=1.0,
            weight=0.2
        )
        self.metrics["weight_update_rate"] = MetricSpec(
            name="weight_update_rate",
            tier=MetricTier.PROCESS,
            description="权重更新率",
            higher_is_better=True,
            target_value=0.01,
            weight=0.2
        )
        self.metrics["activation_sparsity"] = MetricSpec(
            name="activation_sparsity",
            tier=MetricTier.PROCESS,
            description="激活稀疏度",
            higher_is_better=True,
            target_value=0.5,
            weight=0.2
        )
        self.metrics["loss_smoothness"] = MetricSpec(
            name="loss_smoothness",
            tier=MetricTier.PROCESS,
            description="损失曲线平滑度",
            higher_is_better=True,
            target_value=0.8,
            weight=0.2
        )
        self.metrics["memory_usage_gb"] = MetricSpec(
            name="memory_usage_gb",
            tier=MetricTier.PROCESS,
            description="内存使用量(GB)",
            higher_is_better=False,
            target_value=2.0,
            weight=0.2,
            unit="GB"
        )

    def update_metric(self, name: str, value: float, step: int = None):
        if name not in self.metrics:
            return
        self.metrics[name].update(float(value), step)
        self.current_step = step or self.current_step + 1

    def batch_update(self, metrics: Dict[str, float], step: int = None):
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.update_metric(name, float(value), step)

    def evaluate(self, step: int = None) -> EvaluationResult:
        step = step or self.current_step

        tier_scores: Dict[str, float] = {}
        metric_scores: Dict[str, Dict[str, float]] = {}

        for tier in MetricTier:
            tier_metrics = [m for m in self.metrics.values() if m.tier == tier]
            if not tier_metrics:
                tier_scores[tier.value] = 0.0
                continue

            total_weight = sum(m.weight for m in tier_metrics)
            if total_weight == 0:
                tier_scores[tier.value] = 0.0
                continue

            weighted_score = 0.0
            for metric in tier_metrics:
                score = self._compute_metric_score(metric)
                metric_scores[metric.name] = {
                    "current": metric.current_value,
                    "target": metric.target_value or 0.0,
                    "score": score,
                    "weight": metric.weight
                }
                weighted_score += score * metric.weight

            tier_scores[tier.value] = weighted_score / total_weight

        core_weight = 0.5
        capability_weight = 0.35
        process_weight = 0.15

        overall_score = (
            tier_scores.get(MetricTier.CORE.value, 0) * core_weight +
            tier_scores.get(MetricTier.CAPABILITY.value, 0) * capability_weight +
            tier_scores.get(MetricTier.PROCESS.value, 0) * process_weight
        )

        trends = {
            name: metric.trend()
            for name, metric in self.metrics.items()
            if len(metric.history) >= 10
        }

        convergence_state = self._detect_convergence()

        result = EvaluationResult(
            overall_score=overall_score,
            tier_scores=tier_scores,
            metric_scores=metric_scores,
            trends=trends,
            convergence_state=convergence_state,
            step=step
        )

        self.evaluation_history.append(result)

        return result

    def _compute_metric_score(self, metric: MetricSpec) -> float:
        if metric.target_value is None:
            return 0.5

        current = metric.current_value
        target = metric.target_value

        if target == 0:
            if metric.higher_is_better:
                return min(1.0, max(0.0, current))
            else:
                return min(1.0, max(0.0, 1.0 - abs(current)))

        ratio = current / target

        if metric.higher_is_better:
            score = min(1.0, max(0.0, ratio))
        else:
            if ratio <= 1.0:
                score = 1.0
            else:
                score = max(0.0, 2.0 - ratio)

        return score

    def _detect_convergence(self) -> ConvergenceState:
        fe_metric = self.metrics.get("free_energy")
        conf_metric = self.metrics.get("confidence")

        if fe_metric is None or len(fe_metric.history) < 20:
            return ConvergenceState.STABLE

        fe_recent = [h["value"] for h in list(fe_metric.history)[-20:]]
        fe_avg = np.mean(fe_recent)
        fe_std = np.std(fe_recent)
        fe_trend = fe_metric.trend(20)

        if fe_avg < 0.01 and abs(fe_trend) < 0.001 and fe_std < 0.005:
            return ConvergenceState.CONVERGED

        if fe_trend < -0.001:
            return ConvergenceState.CONVERGING

        if fe_trend > 0.005:
            return ConvergenceState.DIVERGING

        if fe_std > 0.1:
            return ConvergenceState.STABLE

        if conf_metric and len(conf_metric.history) >= 20:
            conf_trend = conf_metric.trend(20)
            if fe_trend < 0 and conf_trend < -0.005:
                return ConvergenceState.OVERFITTING

        return ConvergenceState.STABLE

    def get_metric_summary(self, tier: Optional[MetricTier] = None) -> Dict[str, Any]:
        metrics = self.metrics.values()
        if tier:
            metrics = [m for m in metrics if m.tier == tier]

        summary = {}
        for metric in metrics:
            summary[metric.name] = {
                "current": metric.current_value,
                "target": metric.target_value,
                "trend": metric.trend(),
                "std": metric.std(),
                "higher_is_better": metric.higher_is_better,
                "description": metric.description
            }
        return summary

    def get_evaluation_report(self) -> Dict[str, Any]:
        latest = self.evaluate()

        return {
            "step": latest.step,
            "overall_score": latest.overall_score,
            "convergence_state": latest.convergence_state.value,
            "tier_scores": latest.tier_scores,
            "metric_count": len(self.metrics),
            "total_evaluations": len(self.evaluation_history),
            "top_metrics": {
                name: data["score"]
                for name, data in sorted(
                    latest.metric_scores.items(),
                    key=lambda x: x[1]["weight"],
                    reverse=True
                )[:5]
            }
        }


class ConvergenceDetector:
    def __init__(self, window_size: int = 500, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.metric_history: deque = deque(maxlen=window_size * 2)
        self.consecutive_stable = 0
        self.required_stable_steps = 100

    def update(self, metric_value: float, step: int = None) -> ConvergenceState:
        self.metric_history.append({"step": step, "value": metric_value})

        if len(self.metric_history) < self.window_size:
            return ConvergenceState.STABLE

        recent = [h["value"] for h in list(self.metric_history)[-self.window_size:]]
        earlier = [h["value"] for h in list(self.metric_history)[-2*self.window_size:-self.window_size]]

        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier) if earlier else recent_avg

        recent_std = np.std(recent)
        improvement_rate = abs(earlier_avg - recent_avg) / max(abs(earlier_avg), 1e-8)

        if recent_std < self.threshold and improvement_rate < self.threshold:
            self.consecutive_stable += 1
            if self.consecutive_stable >= self.required_stable_steps:
                return ConvergenceState.CONVERGED
            return ConvergenceState.STABLE
        elif improvement_rate > self.threshold * 2:
            self.consecutive_stable = 0
            if recent_avg > earlier_avg:
                return ConvergenceState.DIVERGING
            else:
                return ConvergenceState.CONVERGING
        else:
            self.consecutive_stable = 0
            return ConvergenceState.STABLE

    def reset(self):
        self.metric_history.clear()
        self.consecutive_stable = 0
