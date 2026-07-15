"""
decision/decision_quality_metrics.py - 决策质量评估指标体系

设计完整的决策质量评估指标，覆盖准确性、效率、鲁棒性、成本等维度
支持定量评估和数据采集
"""
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple


class MetricType(Enum):
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    COST = "cost"
    CONFIDENCE = "confidence"
    COMPLIANCE = "compliance"


class MetricCategory(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DERIVED = "derived"


@dataclass
class QualityMetric:
    name: str
    metric_type: MetricType
    category: MetricCategory
    description: str
    calculation_method: str
    unit: str = ""
    target_value: float = 0.0
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    current_value: float = 0.0
    historical_values: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def normalize(self, value: float) -> float:
        if self.max_value == self.min_value:
            return 0.0
        return min(1.0, max(0.0, (value - self.min_value) / (self.max_value - self.min_value)))

    def update(self, value: float):
        self.current_value = value
        self.historical_values.append(value)
        if len(self.historical_values) > 100:
            self.historical_values = self.historical_values[-100:]
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "category": self.category.value,
            "description": self.description,
            "unit": self.unit,
            "target_value": self.target_value,
            "weight": self.weight,
            "current_value": self.current_value,
            "normalized_value": self.normalize(self.current_value),
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionQualityReport:
    decision_id: str
    overall_quality_score: float
    metrics: Dict[str, QualityMetric]
    dimension_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "overall_quality_score": self.overall_quality_score,
            "dimension_scores": self.dimension_scores,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_suggestions": self.improvement_suggestions,
            "timestamp": self.timestamp,
        }


class DecisionQualityMetrics:
    def __init__(self):
        self.metrics: Dict[str, QualityMetric] = self._initialize_metrics()
        self._metric_history: Dict[str, List[Dict[str, Any]]] = {}
        self._dimension_weights: Dict[str, float] = {
            "accuracy": 0.3,
            "efficiency": 0.2,
            "robustness": 0.25,
            "cost": 0.15,
            "confidence": 0.1,
        }

    def _initialize_metrics(self) -> Dict[str, QualityMetric]:
        return {
            "prediction_accuracy": QualityMetric(
                name="预测准确度",
                metric_type=MetricType.ACCURACY,
                category=MetricCategory.PRIMARY,
                description="决策预期结果与实际结果的匹配程度",
                calculation_method="1 - 平均绝对误差/期望值",
                target_value=0.9,
                weight=0.4,
            ),
            "success_rate": QualityMetric(
                name="成功率",
                metric_type=MetricType.ACCURACY,
                category=MetricCategory.PRIMARY,
                description="决策执行成功的比例",
                calculation_method="成功次数/总次数",
                target_value=0.85,
                weight=0.3,
            ),
            "timeliness": QualityMetric(
                name="时效性",
                metric_type=MetricType.EFFICIENCY,
                category=MetricCategory.PRIMARY,
                description="决策从制定到执行完成的时间",
                calculation_method="执行时间/预期时间",
                unit="秒",
                target_value=1.0,
                weight=0.3,
                min_value=0.0,
                max_value=300.0,
            ),
            "resource_efficiency": QualityMetric(
                name="资源效率",
                metric_type=MetricType.EFFICIENCY,
                category=MetricCategory.SECONDARY,
                description="决策执行消耗资源的效率",
                calculation_method="实际资源消耗/预期资源消耗",
                target_value=1.0,
                weight=0.2,
            ),
            "adaptability": QualityMetric(
                name="适应性",
                metric_type=MetricType.ROBUSTNESS,
                category=MetricCategory.PRIMARY,
                description="决策在变化环境中的适应能力",
                calculation_method="环境变化后决策效果保持度",
                target_value=0.8,
                weight=0.3,
            ),
            "error_resilience": QualityMetric(
                name="容错性",
                metric_type=MetricType.ROBUSTNESS,
                category=MetricCategory.SECONDARY,
                description="决策应对错误和异常的能力",
                calculation_method="异常情况下决策成功率",
                target_value=0.7,
                weight=0.25,
            ),
            "cost_effectiveness": QualityMetric(
                name="成本效益",
                metric_type=MetricType.COST,
                category=MetricCategory.PRIMARY,
                description="决策收益与成本的比值",
                calculation_method="收益/成本",
                target_value=2.0,
                weight=0.3,
                min_value=0.0,
                max_value=10.0,
            ),
            "resource_utilization": QualityMetric(
                name="资源利用率",
                metric_type=MetricType.COST,
                category=MetricCategory.SECONDARY,
                description="资源使用效率",
                calculation_method="有效资源使用/总资源投入",
                target_value=0.7,
                weight=0.2,
            ),
            "decision_confidence": QualityMetric(
                name="决策置信度",
                metric_type=MetricType.CONFIDENCE,
                category=MetricCategory.PRIMARY,
                description="决策结果的置信程度",
                calculation_method="基于模型输出的置信度评分",
                target_value=0.75,
                weight=0.25,
            ),
            "compliance_rate": QualityMetric(
                name="合规率",
                metric_type=MetricType.COMPLIANCE,
                category=MetricCategory.SECONDARY,
                description="决策符合规则和约束的程度",
                calculation_method="合规项数/总约束项数",
                target_value=0.95,
                weight=0.15,
            ),
        }

    def calculate_dimension_score(self, dimension: str) -> float:
        dimension_metrics = {
            "accuracy": ["prediction_accuracy", "success_rate"],
            "efficiency": ["timeliness", "resource_efficiency"],
            "robustness": ["adaptability", "error_resilience"],
            "cost": ["cost_effectiveness", "resource_utilization"],
            "confidence": ["decision_confidence", "compliance_rate"],
        }

        metrics_to_evaluate = dimension_metrics.get(dimension, [])
        if not metrics_to_evaluate:
            return 0.0

        total_weight = sum(self.metrics[m].weight for m in metrics_to_evaluate)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            self.metrics[m].weight * self.metrics[m].normalize(self.metrics[m].current_value)
            for m in metrics_to_evaluate
        )

        return float(min(1.0, weighted_sum / total_weight))

    def calculate_overall_quality(self) -> float:
        dimension_scores = {}
        total_weight = sum(self._dimension_weights.values())

        for dimension, weight in self._dimension_weights.items():
            score = self.calculate_dimension_score(dimension)
            dimension_scores[dimension] = score

        overall = sum(
            weight * dimension_scores[dimension]
            for dimension, weight in self._dimension_weights.items()
        ) / total_weight

        return float(min(1.0, max(0.0, overall)))

    def evaluate_decision(self, decision_id: str,
                          execution_data: Dict[str, Any]) -> DecisionQualityReport:
        self._update_metrics_from_execution(execution_data)
        overall_score = self.calculate_overall_quality()
        dimension_scores = {
            dim: self.calculate_dimension_score(dim)
            for dim in self._dimension_weights.keys()
        }

        strengths, weaknesses = self._identify_strengths_weaknesses(dimension_scores)
        suggestions = self._generate_improvement_suggestions(dimension_scores)

        report = DecisionQualityReport(
            decision_id=decision_id,
            overall_quality_score=overall_score,
            metrics=self.metrics,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
        )

        self._record_history(decision_id, report)
        return report

    def _update_metrics_from_execution(self, execution_data: Dict[str, Any]):
        actual_outcome = execution_data.get("actual_outcome", {})
        expected_outcome = execution_data.get("expected_outcome", {})

        if actual_outcome and expected_outcome:
            accuracy = self._calculate_prediction_accuracy(expected_outcome, actual_outcome)
            self.metrics["prediction_accuracy"].update(accuracy)

        success = execution_data.get("success", False)
        self.metrics["success_rate"].update(1.0 if success else 0.0)

        execution_time = execution_data.get("execution_time", 0.0)
        expected_time = execution_data.get("expected_time", 1.0)
        timeliness = min(1.0, expected_time / max(1e-10, execution_time)) if expected_time > 0 else 0.0
        self.metrics["timeliness"].update(timeliness)

        actual_cost = execution_data.get("actual_cost", 1.0)
        expected_cost = execution_data.get("expected_cost", 1.0)
        resource_efficiency = min(1.0, expected_cost / max(1e-10, actual_cost)) if expected_cost > 0 else 0.0
        self.metrics["resource_efficiency"].update(resource_efficiency)

        adaptability = execution_data.get("adaptability", 0.5)
        self.metrics["adaptability"].update(adaptability)

        error_resilience = execution_data.get("error_resilience", 0.5)
        self.metrics["error_resilience"].update(error_resilience)

        actual_reward = execution_data.get("actual_reward", 0.0)
        cost_effectiveness = actual_reward / max(1e-10, actual_cost) if actual_cost > 0 else 0.0
        self.metrics["cost_effectiveness"].update(min(10.0, cost_effectiveness))

        confidence = execution_data.get("confidence", 0.5)
        self.metrics["decision_confidence"].update(confidence)

        compliance = execution_data.get("compliance", 1.0)
        self.metrics["compliance_rate"].update(compliance)

    def _calculate_prediction_accuracy(self, expected: Dict[str, Any],
                                        actual: Dict[str, Any]) -> float:
        common_keys = set(expected.keys()) & set(actual.keys())
        if not common_keys:
            return 0.5

        errors = []
        for key in common_keys:
            exp_val = expected[key]
            act_val = actual[key]
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                if abs(exp_val) > 1e-10:
                    error = abs(act_val - exp_val) / abs(exp_val)
                    errors.append(error)

        if not errors:
            return 0.5

        return float(1.0 - np.mean(errors))

    def _identify_strengths_weaknesses(self, dimension_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        strengths = []
        weaknesses = []

        dimension_labels = {
            "accuracy": "准确性",
            "efficiency": "效率",
            "robustness": "鲁棒性",
            "cost": "成本效益",
            "confidence": "置信度",
        }

        for dim, score in dimension_scores.items():
            if score >= 0.7:
                strengths.append(f"{dimension_labels.get(dim, dim)} ({score:.2f})")
            elif score < 0.5:
                weaknesses.append(f"{dimension_labels.get(dim, dim)} ({score:.2f})")

        return strengths, weaknesses

    def _generate_improvement_suggestions(self, dimension_scores: Dict[str, float]) -> List[str]:
        suggestions = []

        if dimension_scores.get("accuracy", 0) < 0.6:
            suggestions.append("提升预测模型精度，增加验证数据")
            suggestions.append("引入更精确的结果评估指标")

        if dimension_scores.get("efficiency", 0) < 0.6:
            suggestions.append("优化决策流程，减少不必要的步骤")
            suggestions.append("引入并行处理机制")

        if dimension_scores.get("robustness", 0) < 0.6:
            suggestions.append("增加异常处理和容错机制")
            suggestions.append("进行更多场景的测试验证")

        if dimension_scores.get("cost", 0) < 0.6:
            suggestions.append("优化资源分配策略")
            suggestions.append("评估替代方案的成本效益")

        if dimension_scores.get("confidence", 0) < 0.6:
            suggestions.append("增加决策置信度评估机制")
            suggestions.append("引入多模型集成提高可信度")

        return suggestions[:5]

    def _record_history(self, decision_id: str, report: DecisionQualityReport):
        if decision_id not in self._metric_history:
            self._metric_history[decision_id] = []
        self._metric_history[decision_id].append(report.to_dict())
        if len(self._metric_history[decision_id]) > 50:
            self._metric_history[decision_id] = self._metric_history[decision_id][-50:]

    def get_metric_history(self, metric_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        history = []
        for decision_history in self._metric_history.values():
            for record in decision_history:
                if metric_name in record.get("metrics", {}):
                    history.append({
                        "timestamp": record["timestamp"],
                        "value": record["metrics"][metric_name]["current_value"],
                    })
        return sorted(history, key=lambda x: x["timestamp"])[-limit:]

    def get_quality_trends(self) -> Dict[str, Any]:
        all_reports = []
        for decision_history in self._metric_history.values():
            all_reports.extend(decision_history)

        if not all_reports:
            return {"trend": "no_data", "average_quality": 0.0}

        all_reports.sort(key=lambda x: x["timestamp"])
        recent_reports = all_reports[-10:]
        quality_scores = [r["overall_quality_score"] for r in recent_reports]

        trend = "stable"
        if len(quality_scores) >= 2:
            slope = (quality_scores[-1] - quality_scores[0]) / max(1, len(quality_scores) - 1)
            if slope > 0.02:
                trend = "increasing"
            elif slope < -0.02:
                trend = "decreasing"

        return {
            "trend": trend,
            "average_quality": float(np.mean(quality_scores)),
            "latest_quality": quality_scores[-1],
            "report_count": len(all_reports),
            "dimension_averages": {
                dim: float(np.mean([r["dimension_scores"].get(dim, 0) for r in all_reports]))
                for dim in self._dimension_weights.keys()
            },
        }

    def set_dimension_weights(self, weights: Dict[str, float]):
        total = sum(weights.values())
        if total > 0:
            self._dimension_weights = {k: v / total for k, v in weights.items()}

    def get_metric_definitions(self) -> List[Dict[str, Any]]:
        return [metric.to_dict() for metric in self.metrics.values()]