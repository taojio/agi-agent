import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class QualityDimension(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"
    ROBUSTNESS = "robustness"
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"


class BiasType(Enum):
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    STATUS_QUO = "status_quo"
    GROUPTHINK = "groupthink"
    HINDSIGHT = "hindsight"
    FRAMING = "framing"


class QualityScore:
    def __init__(self):
        self.scores: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.overall_score: float = 0.0

    def set_score(self, dimension: QualityDimension, score: float, weight: float = 1.0):
        self.scores[dimension.value] = score
        self.weights[dimension.value] = weight

    def calculate_overall(self) -> float:
        if not self.scores:
            return 0.0
        
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return float(np.mean(list(self.scores.values())))
        
        self.overall_score = sum(
            self.scores[d] * self.weights[d] for d in self.scores
        ) / total_weight
        
        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_scores": self.scores,
            "weights": self.weights,
            "overall_score": self.overall_score
        }


class BiasDetector:
    def __init__(self):
        self.bias_patterns: Dict[str, List[Callable]] = {}
        self._init_bias_patterns()

    def _init_bias_patterns(self):
        self.bias_patterns[BiasType.CONFIRMATION.value] = [
            self._detect_confirmation_bias,
            self._detect_selective_evidence,
        ]
        self.bias_patterns[BiasType.ANCHORING.value] = [
            self._detect_anchoring_bias,
        ]
        self.bias_patterns[BiasType.OVERCONFIDENCE.value] = [
            self._detect_overconfidence,
        ]

    def _detect_confirmation_bias(self, decision_data: Dict[str, Any]) -> float:
        evidence_count = len(decision_data.get("factors_considered", []))
        supporting_count = sum(1 for f in decision_data.get("factors_considered", []) 
                              if "support" in f.lower())
        
        if evidence_count == 0:
            return 0.0
        
        ratio = supporting_count / evidence_count
        return float(min(1.0, (ratio - 0.5) * 2))

    def _detect_selective_evidence(self, decision_data: Dict[str, Any]) -> float:
        options = decision_data.get("options_considered", [])
        if len(options) < 2:
            return 0.0
        
        return float(np.random.uniform(0, 0.3))

    def _detect_anchoring_bias(self, decision_data: Dict[str, Any]) -> float:
        first_evidence = decision_data.get("first_evidence", "")
        if first_evidence:
            return float(np.random.uniform(0.1, 0.4))
        return 0.0

    def _detect_overconfidence(self, decision_data: Dict[str, Any]) -> float:
        confidence = decision_data.get("confidence", 0.5)
        quality = decision_data.get("quality_score", 0.5)
        
        diff = confidence - quality
        return float(max(0.0, min(1.0, diff * 2)))

    def detect_biases(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        detected = []
        
        for bias_type, detectors in self.bias_patterns.items():
            for detector in detectors:
                score = detector(decision_data)
                if score > 0.2:
                    detected.append({
                        "bias_type": bias_type,
                        "severity": score,
                        "detector": detector.__name__
                    })
        
        detected.sort(key=lambda x: x["severity"], reverse=True)
        return detected

    def get_bias_summary(self, decisions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_biases = []
        for data in decisions_data:
            all_biases.extend(self.detect_biases(data))
        
        bias_counts = {}
        for bias in all_biases:
            bias_counts[bias["bias_type"]] = bias_counts.get(bias["bias_type"], 0) + 1
        
        return {
            "total_biases_detected": len(all_biases),
            "bias_distribution": bias_counts,
            "avg_severity": float(np.mean([b["severity"] for b in all_biases])) if all_biases else 0.0
        }


class DecisionQualityAnalyzer:
    def __init__(self):
        self.bias_detector = BiasDetector()
        self.quality_history: deque = deque(maxlen=200)
        self.dimension_weights: Dict[str, float] = {
            QualityDimension.ACCURACY.value: 0.2,
            QualityDimension.COMPLETENESS.value: 0.15,
            QualityDimension.CONSISTENCY.value: 0.15,
            QualityDimension.TIMELINESS.value: 0.1,
            QualityDimension.RELEVANCE.value: 0.15,
            QualityDimension.ROBUSTNESS.value: 0.1,
            QualityDimension.TRANSPARENCY.value: 0.1,
            QualityDimension.FAIRNESS.value: 0.05,
        }

    def analyze_quality(self, decision_data: Dict[str, Any]) -> QualityScore:
        score = QualityScore()
        
        score.set_score(QualityDimension.ACCURACY, 
                       self._assess_accuracy(decision_data),
                       self.dimension_weights[QualityDimension.ACCURACY.value])
        
        score.set_score(QualityDimension.COMPLETENESS,
                       self._assess_completeness(decision_data),
                       self.dimension_weights[QualityDimension.COMPLETENESS.value])
        
        score.set_score(QualityDimension.CONSISTENCY,
                       self._assess_consistency(decision_data),
                       self.dimension_weights[QualityDimension.CONSISTENCY.value])
        
        score.set_score(QualityDimension.TIMELINESS,
                       self._assess_timeliness(decision_data),
                       self.dimension_weights[QualityDimension.TIMELINESS.value])
        
        score.set_score(QualityDimension.RELEVANCE,
                       self._assess_relevance(decision_data),
                       self.dimension_weights[QualityDimension.RELEVANCE.value])
        
        score.calculate_overall()
        
        decision_data["quality_score"] = score.overall_score
        self.quality_history.append(decision_data)
        
        return score

    def _assess_accuracy(self, data: Dict[str, Any]) -> float:
        confidence = data.get("confidence", 0.5)
        outcome = data.get("outcome", "pending")
        
        if outcome == "success":
            return float(min(1.0, confidence + 0.2))
        elif outcome == "failure":
            return float(max(0.0, confidence - 0.3))
        return confidence

    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        factors = data.get("factors_considered", [])
        options = data.get("options_considered", [])
        
        factor_score = min(1.0, len(factors) / 10)
        option_score = min(1.0, len(options) / 5)
        
        return float((factor_score + option_score) / 2)

    def _assess_consistency(self, data: Dict[str, Any]) -> float:
        patterns = data.get("patterns", [])
        if not patterns:
            return float(np.random.uniform(0.6, 0.9))
        return float(min(1.0, 0.7 + len(patterns) * 0.1))

    def _assess_timeliness(self, data: Dict[str, Any]) -> float:
        duration = data.get("duration_ms", 0)
        if duration == 0:
            return 0.5
        return float(max(0.0, min(1.0, 1.0 - duration / 2000)))

    def _assess_relevance(self, data: Dict[str, Any]) -> float:
        factors = data.get("factors_considered", [])
        if not factors:
            return 0.5
        return float(min(1.0, 0.5 + len(factors) * 0.1))

    def detect_biases(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.bias_detector.detect_biases(decision_data)

    def get_quality_trend(self, window_size: int = 50) -> Dict[str, Any]:
        if len(self.quality_history) < window_size:
            return {"insufficient_data": True}
        
        recent = list(self.quality_history)[-window_size:]
        scores = [d.get("quality_score", 0.5) for d in recent]
        
        return {
            "avg_score": float(np.mean(scores)),
            "score_trend": float(np.polyfit(np.arange(len(scores)), scores, 1)[0]),
            "score_variance": float(np.var(scores)),
            "min_score": float(min(scores)),
            "max_score": float(max(scores))
        }

    def get_quality_summary(self) -> Dict[str, Any]:
        if not self.quality_history:
            return {"total_analyzed": 0, "avg_quality": 0.0}
        
        scores = [d.get("quality_score", 0.5) for d in self.quality_history]
        
        return {
            "total_analyzed": len(self.quality_history),
            "avg_quality": float(np.mean(scores)),
            "quality_trend": self.get_quality_trend(),
            "dimension_weights": self.dimension_weights
        }

    def suggest_improvements(self, decision_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        suggestions = []
        quality = self.analyze_quality(decision_data)
        biases = self.detect_biases(decision_data)
        
        for dim, score in quality.scores.items():
            if score < 0.5:
                suggestions.append({
                    "dimension": dim,
                    "current_score": score,
                    "suggestion": self._get_dimension_suggestion(dim),
                    "priority": "high" if score < 0.3 else "medium"
                })
        
        for bias in biases:
            suggestions.append({
                "dimension": "bias",
                "bias_type": bias["bias_type"],
                "severity": bias["severity"],
                "suggestion": self._get_bias_mitigation(bias["bias_type"]),
                "priority": "high" if bias["severity"] > 0.6 else "medium"
            })
        
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions

    def _get_dimension_suggestion(self, dimension: str) -> str:
        suggestions = {
            QualityDimension.ACCURACY.value: "收集更多验证数据以提高决策准确性",
            QualityDimension.COMPLETENESS.value: "考虑更多备选方案和影响因素",
            QualityDimension.CONSISTENCY.value: "检查决策与历史模式的一致性",
            QualityDimension.TIMELINESS.value: "优化决策流程以减少延迟",
            QualityDimension.RELEVANCE.value: "确保考虑的因素与决策目标相关",
            QualityDimension.ROBUSTNESS.value: "考虑边界情况和异常场景",
            QualityDimension.TRANSPARENCY.value: "记录决策过程以便审查",
            QualityDimension.FAIRNESS.value: "评估决策对不同群体的影响",
        }
        return suggestions.get(dimension, "审查决策过程")

    def _get_bias_mitigation(self, bias_type: str) -> str:
        mitigations = {
            BiasType.CONFIRMATION.value: "主动寻找反驳证据",
            BiasType.ANCHORING.value: "延迟初始判断，收集更多信息",
            BiasType.AVAILABILITY.value: "使用数据驱动方法而非直觉",
            BiasType.OVERCONFIDENCE.value: "进行置信度校准",
            BiasType.STATUS_QUO.value: "积极考虑变革选项",
            BiasType.GROUPTHINK.value: "鼓励不同意见",
            BiasType.HINDSIGHT.value: "记录决策前的预期",
            BiasType.FRAMING.value: "从多个角度审视问题",
        }
        return mitigations.get(bias_type, "实施偏见检查流程")