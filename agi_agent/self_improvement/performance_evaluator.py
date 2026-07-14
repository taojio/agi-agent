import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from enum import Enum
from dataclasses import dataclass, field


class MetricCategory(Enum):
    COGNITIVE = "cognitive"
    PERFORMANCE = "performance"
    LEARNING = "learning"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"


@dataclass
class PerformanceMetric:
    name: str
    category: MetricCategory
    value: float
    baseline: float
    target: float
    weight: float = 1.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    trend: str = "stable"

    def get_score(self) -> float:
        if self.target == self.baseline:
            return 1.0
        progress = (self.value - self.baseline) / (self.target - self.baseline)
        return max(0.0, min(1.0, progress))


class PerformanceEvaluator:
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.evaluation_history: deque = deque(maxlen=500)
        self._eval_count = 0

        self._init_default_metrics()

    def _init_default_metrics(self):
        default_metrics = [
            ("free_energy", MetricCategory.COGNITIVE, 1.0, 0.3, 1.5),
            ("confidence", MetricCategory.COGNITIVE, 0.5, 0.9, 1.0),
            ("prediction_accuracy", MetricCategory.COGNITIVE, 0.5, 0.85, 1.2),
            ("action_success_rate", MetricCategory.PERFORMANCE, 0.5, 0.8, 1.0),
            ("goal_completion_rate", MetricCategory.PERFORMANCE, 0.3, 0.7, 1.3),
            ("learning_rate_efficiency", MetricCategory.LEARNING, 0.5, 0.8, 0.8),
            ("knowledge_growth_rate", MetricCategory.LEARNING, 0.2, 0.6, 0.7),
            ("stability_score", MetricCategory.STABILITY, 0.5, 0.9, 1.0),
            ("error_rate", MetricCategory.STABILITY, 0.1, 0.02, 1.2),
            ("throughput_steps_per_sec", MetricCategory.EFFICIENCY, 10.0, 100.0, 0.9),
            ("memory_efficiency", MetricCategory.EFFICIENCY, 0.5, 0.8, 0.6),
            ("safety_compliance_rate", MetricCategory.SAFETY, 0.9, 0.99, 2.0),
            ("risk_level", MetricCategory.SAFETY, 0.5, 0.1, 1.5),
        ]

        for name, category, baseline, target, weight in default_metrics:
            self.metrics[name] = PerformanceMetric(
                name=name,
                category=category,
                value=baseline,
                baseline=baseline,
                target=target,
                weight=weight
            )

    def update_metric(self, name: str, value: float) -> bool:
        if name not in self.metrics:
            return False

        metric = self.metrics[name]
        old_value = metric.value
        metric.value = value
        metric.history.append(value)

        if len(metric.history) >= 10:
            recent = list(metric.history)[-5:]
            earlier = list(metric.history)[:5]
            avg_recent = float(np.mean(recent))
            avg_earlier = float(np.mean(earlier))
            if avg_recent > avg_earlier * 1.05:
                metric.trend = "improving"
            elif avg_recent < avg_earlier * 0.95:
                metric.trend = "declining"
            else:
                metric.trend = "stable"

        return True

    def batch_update(self, metrics_dict: Dict[str, float]):
        for name, value in metrics_dict.items():
            self.update_metric(name, value)

    def evaluate(self) -> Dict[str, Any]:
        self._eval_count += 1

        category_scores: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for metric in self.metrics.values():
            score = metric.get_score()
            cat = metric.category.value

            if cat not in category_scores:
                category_scores[cat] = {"total_weight": 0.0, "weighted_sum": 0.0}

            category_scores[cat]["total_weight"] += metric.weight
            category_scores[cat]["weighted_sum"] += score * metric.weight

            total_weight += metric.weight
            weighted_sum += score * metric.weight

        for cat in category_scores:
            tw = category_scores[cat]["total_weight"]
            category_scores[cat] = category_scores[cat]["weighted_sum"] / max(tw, 0.01)

        overall_score = weighted_sum / max(total_weight, 0.01)

        result = {
            "evaluation_id": self._eval_count,
            "timestamp": time.time(),
            "overall_score": overall_score,
            "category_scores": category_scores,
            "metric_details": {
                name: {
                    "value": m.value,
                    "baseline": m.baseline,
                    "target": m.target,
                    "score": m.get_score(),
                    "trend": m.trend,
                    "weight": m.weight
                }
                for name, m in self.metrics.items()
            },
            "grade": self._get_grade(overall_score)
        }

        self.evaluation_history.append(result)

        return result

    def _get_grade(self, score: float) -> str:
        if score >= 0.9:
            return "S"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"

    def get_weakest_metrics(self, count: int = 5) -> List[Dict[str, Any]]:
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda m: m.get_score()
        )
        return [
            {
                "name": m.name,
                "score": m.get_score(),
                "value": m.value,
                "target": m.target,
                "category": m.category.value
            }
            for m in sorted_metrics[:count]
        ]

    def get_improving_metrics(self) -> List[str]:
        return [m.name for m in self.metrics.values() if m.trend == "improving"]

    def get_declining_metrics(self) -> List[str]:
        return [m.name for m in self.metrics.values() if m.trend == "declining"]

    def get_evaluation_stats(self) -> Dict[str, Any]:
        latest = self.evaluation_history[-1] if self.evaluation_history else None
        return {
            "total_evaluations": self._eval_count,
            "total_metrics": len(self.metrics),
            "latest_score": latest["overall_score"] if latest else 0.0,
            "latest_grade": latest["grade"] if latest else "N/A",
            "improving_count": len(self.get_improving_metrics()),
            "declining_count": len(self.get_declining_metrics())
        }
