"""
decision/risk_assessment.py - 风险评估引擎

识别风险因子，评估概率和影响，生成风险缓解建议
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RiskLevel(Enum):
    """风险等级"""
    NEGLIGIBLE = "negligible"   # 可忽略
    LOW = "low"                 # 低风险
    MODERATE = "moderate"       # 中等风险
    HIGH = "high"               # 高风险
    CRITICAL = "critical"       # 严重风险


class RiskFactorType(Enum):
    """风险因子类型"""
    TECHNICAL = "technical"         # 技术风险
    OPERATIONAL = "operational"     # 运营风险
    RESOURCE = "resource"           # 资源风险
    EXTERNAL = "external"           # 外部风险
    TEMPORAL = "temporal"           # 时间风险
    SECURITY = "security"           # 安全风险
    FINANCIAL = "financial"         # 财务风险


@dataclass
class RiskFactor:
    """风险因子

    Attributes:
        name: 风险名称
        factor_type: 风险类型
        probability: 发生概率 (0-1)
        impact: 影响程度 (0-1)
        description: 描述
        mitigations: 缓解措施
    """

    name: str
    factor_type: RiskFactorType
    probability: float = 0.5
    impact: float = 0.5
    description: str = ""
    mitigations: List[str] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        """风险评分 = 概率 × 影响"""
        return float(self.probability * self.impact)

    @property
    def risk_level(self) -> RiskLevel:
        """风险等级"""
        score = self.risk_score
        if score < 0.05:
            return RiskLevel.NEGLIGIBLE
        elif score < 0.15:
            return RiskLevel.LOW
        elif score < 0.35:
            return RiskLevel.MODERATE
        elif score < 0.6:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "factor_type": self.factor_type.value,
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "mitigations": self.mitigations,
        }


@dataclass
class RiskProfile:
    """风险画像

    完整的风险评估结果
    """

    overall_risk: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    risk_factors: List[RiskFactor] = field(default_factory=list)
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    impact_assessment: Dict[str, float] = field(default_factory=dict)
    mitigation_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def factor_count(self) -> int:
        return len(self.risk_factors)

    @property
    def critical_factor_count(self) -> int:
        return sum(1 for f in self.risk_factors
                   if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level.value,
            "factor_count": self.factor_count,
            "critical_factor_count": self.critical_factor_count,
            "risk_factors": [f.to_dict() for f in self.risk_factors],
            "probability_distribution": self.probability_distribution,
            "impact_assessment": self.impact_assessment,
            "mitigation_suggestions": self.mitigation_suggestions,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class RiskAssessmentEngine:
    """风险评估引擎

    对决策选项进行风险评估，包括：
    1. 识别风险因子
    2. 估计发生概率
    3. 评估影响程度
    4. 综合风险评分
    5. 生成缓解建议
    """

    def __init__(self,
                 risk_threshold: float = 0.3,
                 high_risk_threshold: float = 0.5):
        self.risk_threshold = risk_threshold
        self.high_risk_threshold = high_risk_threshold

        self._risk_history: List[RiskProfile] = []
        self._risk_patterns: Dict[str, Dict[str, float]] = {}

    def assess_risk(self, option: Any,
                     context: Dict[str, Any] = None) -> RiskProfile:
        """评估决策选项的风险

        Args:
            option: 决策选项
            context: 决策上下文

        Returns:
            风险画像
        """
        context = context or {}

        risk_factors = self._identify_risk_factors(option, context)
        probabilities = self._estimate_probabilities(risk_factors, context)
        impacts = self._assess_impacts(risk_factors, context)
        overall_risk = self._calculate_overall_risk(risk_factors)
        risk_level = self._categorize_risk(overall_risk)
        mitigation = self._generate_mitigation(risk_factors, overall_risk)
        confidence = self._calculate_confidence(risk_factors, context)

        profile = RiskProfile(
            overall_risk=overall_risk,
            risk_level=risk_level,
            risk_factors=risk_factors,
            probability_distribution=probabilities,
            impact_assessment=impacts,
            mitigation_suggestions=mitigation,
            confidence=confidence,
            metadata={
                "option": getattr(option, 'option_id', str(option)),
                "option_name": getattr(option, 'name', ''),
            },
        )

        self._risk_history.append(profile)
        return profile

    def assess_multiple(self, options: List[Any],
                         context: Dict[str, Any] = None) -> Dict[str, RiskProfile]:
        """评估多个选项的风险"""
        results = {}
        for option in options:
            key = getattr(option, 'option_id', str(option))
            results[key] = self.assess_risk(option, context)
        return results

    def compare_risks(self, options: List[Any],
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """比较多个选项的风险"""
        profiles = self.assess_multiple(options, context)

        if not profiles:
            return {"lowest_risk": None, "comparison": {}}

        sorted_options = sorted(
            profiles.items(),
            key=lambda x: x[1].overall_risk
        )

        return {
            "lowest_risk": sorted_options[0][0],
            "highest_risk": sorted_options[-1][0],
            "comparison": {
                key: {
                    "overall_risk": profile.overall_risk,
                    "risk_level": profile.risk_level.value,
                    "factor_count": profile.factor_count,
                }
                for key, profile in profiles.items()
            },
            "ranking": [opt_id for opt_id, _ in sorted_options],
        }

    def _identify_risk_factors(self, option: Any,
                                context: Dict[str, Any]) -> List[RiskFactor]:
        """识别风险因子"""
        factors = []

        success_prob = getattr(option, 'success_probability', 0.5)
        failure_prob = 1.0 - success_prob

        if failure_prob > 0.3:
            factors.append(RiskFactor(
                name="执行失败",
                factor_type=RiskFactorType.OPERATIONAL,
                probability=failure_prob,
                impact=0.6,
                description=f"选项执行失败概率为 {failure_prob:.2f}",
                mitigations=["增加备选方案", "设置检查点", "准备回退路径"],
            ))

        cost = getattr(option, 'estimated_cost', 0.1)
        if cost > 0.5:
            factors.append(RiskFactor(
                name="资源消耗过大",
                factor_type=RiskFactorType.RESOURCE,
                probability=min(1.0, cost),
                impact=cost * 0.7,
                description=f"选项成本较高 ({cost:.2f})",
                mitigations=["优化资源使用", "分阶段执行", "寻找替代方案"],
            ))

        risk_level_str = getattr(option, 'risk_level', 'low')
        if risk_level_str == "high":
            factors.append(RiskFactor(
                name="高风险操作",
                factor_type=RiskFactorType.SECURITY,
                probability=0.7,
                impact=0.8,
                description="选项被标记为高风险",
                mitigations=["增加审批流程", "强化监控", "限制执行范围"],
            ))
        elif risk_level_str == "medium":
            factors.append(RiskFactor(
                name="中等风险操作",
                factor_type=RiskFactorType.SECURITY,
                probability=0.4,
                impact=0.5,
                description="选项被标记为中等风险",
                mitigations=["增加监控", "准备应急方案"],
            ))

        if context.get("time_pressure", False):
            factors.append(RiskFactor(
                name="时间压力",
                factor_type=RiskFactorType.TEMPORAL,
                probability=0.6,
                impact=0.5,
                description="存在时间约束",
                mitigations=["简化流程", "优先关键步骤"],
            ))

        if context.get("novelty", 0.0) > 0.6:
            factors.append(RiskFactor(
                name="新颖性风险",
                factor_type=RiskFactorType.TECHNICAL,
                probability=context["novelty"],
                impact=0.6,
                description="选项涉及不熟悉的情况",
                mitigations=["小规模试验", "逐步推广", "收集更多信息"],
            ))

        return factors

    def _estimate_probabilities(self, factors: List[RiskFactor],
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """估计概率分布"""
        return {f.name: f.probability for f in factors}

    def _assess_impacts(self, factors: List[RiskFactor],
                          context: Dict[str, Any]) -> Dict[str, float]:
        """评估影响"""
        return {f.name: f.impact for f in factors}

    def _calculate_overall_risk(self, factors: List[RiskFactor]) -> float:
        """计算综合风险评分

        使用加权平均，考虑风险因子间的累积效应
        """
        if not factors:
            return 0.0

        individual_risks = [f.risk_score for f in factors]

        avg_risk = sum(individual_risks) / len(individual_risks)
        max_risk = max(individual_risks)

        cumulative_factor = 1.0 + 0.1 * (len(factors) - 1)
        cumulative_factor = min(cumulative_factor, 1.5)

        overall = (0.6 * avg_risk + 0.4 * max_risk) * cumulative_factor
        return float(min(1.0, overall))

    def _categorize_risk(self, score: float) -> RiskLevel:
        """风险分类"""
        if score < 0.05:
            return RiskLevel.NEGLIGIBLE
        elif score < 0.15:
            return RiskLevel.LOW
        elif score < 0.35:
            return RiskLevel.MODERATE
        elif score < 0.6:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_mitigation(self, factors: List[RiskFactor],
                               overall_risk: float) -> List[str]:
        """生成缓解建议"""
        suggestions = []

        high_risk_factors = [
            f for f in factors
            if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]

        for factor in high_risk_factors:
            for mitigation in factor.mitigations:
                if mitigation not in suggestions:
                    suggestions.append(mitigation)

        if overall_risk > self.high_risk_threshold:
            suggestions.append("考虑放弃或推迟该选项")
            suggestions.append("寻求额外资源和权限")

        if overall_risk > self.risk_threshold and not suggestions:
            suggestions.append("增加监控和检查点")
            suggestions.append("准备应急方案")

        if not suggestions:
            suggestions.append("维持常规监控")

        return suggestions

    def _calculate_confidence(self, factors: List[RiskFactor],
                               context: Dict[str, Any]) -> float:
        """计算评估置信度"""
        if not factors:
            return 0.3

        factor_count_factor = min(1.0, len(factors) / 5.0)
        data_quality = context.get("data_quality", 0.5)

        return float(0.5 * factor_count_factor + 0.5 * data_quality)

    def get_risk_history(self, limit: int = 100) -> List[RiskProfile]:
        """获取风险评估历史"""
        return self._risk_history[-limit:]

    def get_risk_trends(self) -> Dict[str, Any]:
        """获取风险趋势"""
        if not self._risk_history:
            return {"trend": "no_data"}

        recent = self._risk_history[-10:]
        risks = [p.overall_risk for p in recent]

        if len(risks) >= 2:
            trend = "increasing" if risks[-1] > risks[0] else "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "average_risk": sum(risks) / len(risks),
            "max_risk": max(risks),
            "min_risk": min(risks),
            "assessments_count": len(self._risk_history),
        }
