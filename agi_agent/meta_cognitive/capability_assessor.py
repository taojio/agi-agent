"""
meta_cognitive/capability_assessor.py - 能力评估器

多维度能力画像：技术能力、认知能力、社交能力、适应性等。
支持能力趋势分析、能力差距识别、能力发展建议。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class CapabilityDimension(Enum):
    """能力维度"""
    TECHNICAL = "technical"          # 技术能力：算法执行、数据处理
    COGNITIVE = "cognitive"          # 认知能力：推理、决策、规划
    LEARNING = "learning"            # 学习能力：知识获取、技能习得
    ADAPTIVE = "adaptive"            # 适应能力：环境适应、变化响应
    SOCIAL = "social"                # 社交能力：协作、沟通
    CREATIVE = "creative"            # 创造能力：创新、问题重构
    EXECUTION = "execution"          # 执行能力：任务完成、目标达成
    RESILIENCE = "resilience"        # 韧性：错误恢复、压力承受


class ProficiencyLevel(Enum):
    """能力等级"""
    NOVICE = 1       # 新手 (0.0-0.2)
    BEGINNER = 2     # 初级 (0.2-0.4)
    INTERMEDIATE = 3 # 中级 (0.4-0.6)
    ADVANCED = 4     # 高级 (0.6-0.8)
    EXPERT = 5       # 专家 (0.8-1.0)

    @classmethod
    def from_score(cls, score: float) -> "ProficiencyLevel":
        if score < 0.2:
            return cls.NOVICE
        elif score < 0.4:
            return cls.BEGINNER
        elif score < 0.6:
            return cls.INTERMEDIATE
        elif score < 0.8:
            return cls.ADVANCED
        else:
            return cls.EXPERT


@dataclass
class CapabilityMetric:
    """能力指标"""
    name: str
    dimension: CapabilityDimension
    score: float                # 0.0-1.0
    confidence: float = 0.5     # 0.0-1.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)
    trend: float = 0.0          # 正值=改善，负值=退化
    history: deque = field(default_factory=lambda: deque(maxlen=50))

    def update(self, new_score: float, weight: float = 1.0) -> None:
        """增量更新能力分数"""
        self.history.append(new_score)
        # 加权移动平均
        total_weight = weight
        weighted_sum = new_score * weight
        if len(self.history) > 1:
            prev = self.history[-2]
            weighted_sum += prev * 0.3
            total_weight += 0.3
        self.score = max(0.0, min(1.0, weighted_sum / total_weight))
        self.sample_count += 1
        self.last_updated = time.time()
        self.confidence = min(1.0, self.sample_count / 20.0)
        # 计算趋势
        if len(self.history) >= 5:
            recent = list(self.history)[-5:]
            self.trend = float(np.polyfit(range(len(recent)), recent, 1)[0])

    @property
    def proficiency(self) -> ProficiencyLevel:
        return ProficiencyLevel.from_score(self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension.value,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "trend": float(self.trend),
            "proficiency": self.proficiency.name,
        }


@dataclass
class CapabilityGap:
    """能力差距"""
    dimension: CapabilityDimension
    current_score: float
    target_score: float
    gap: float
    priority: float  # 0-1，越高越优先

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "current_score": float(self.current_score),
            "target_score": float(self.target_score),
            "gap": float(self.gap),
            "priority": float(self.priority),
        }


class CapabilityAssessor:
    """能力评估器

    综合评估多维度能力，生成能力画像，识别能力差距与发展方向。

    Attributes:
        metrics: 能力指标字典 {metric_name: CapabilityMetric}
        assessment_history: 评估历史
    """

    def __init__(self):
        self.metrics: Dict[str, CapabilityMetric] = {}
        self.assessment_history: deque = deque(maxlen=100)
        self._dimension_aggregators: Dict[CapabilityDimension, List[str]] = {
            dim: [] for dim in CapabilityDimension
        }
        # 能力目标（每维度）
        self.targets: Dict[CapabilityDimension, float] = {
            CapabilityDimension.TECHNICAL: 0.8,
            CapabilityDimension.COGNITIVE: 0.75,
            CapabilityDimension.LEARNING: 0.7,
            CapabilityDimension.ADAPTIVE: 0.7,
            CapabilityDimension.SOCIAL: 0.6,
            CapabilityDimension.CREATIVE: 0.65,
            CapabilityDimension.EXECUTION: 0.8,
            CapabilityDimension.RESILIENCE: 0.75,
        }
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """初始化默认能力指标"""
        defaults = [
            ("algorithm_execution", CapabilityDimension.TECHNICAL),
            ("data_processing", CapabilityDimension.TECHNICAL),
            ("reasoning", CapabilityDimension.COGNITIVE),
            ("decision_making", CapabilityDimension.COGNITIVE),
            ("planning", CapabilityDimension.COGNITIVE),
            ("knowledge_acquisition", CapabilityDimension.LEARNING),
            ("skill_transfer", CapabilityDimension.LEARNING),
            ("environment_adaptation", CapabilityDimension.ADAPTIVE),
            ("change_response", CapabilityDimension.ADAPTIVE),
            ("collaboration", CapabilityDimension.SOCIAL),
            ("communication", CapabilityDimension.SOCIAL),
            ("innovation", CapabilityDimension.CREATIVE),
            ("problem_reformulation", CapabilityDimension.CREATIVE),
            ("task_completion", CapabilityDimension.EXECUTION),
            ("goal_achievement", CapabilityDimension.EXECUTION),
            ("error_recovery", CapabilityDimension.RESILIENCE),
            ("stress_tolerance", CapabilityDimension.RESILIENCE),
        ]
        for name, dim in defaults:
            self.register_metric(name, dim, initial_score=0.5)

    def register_metric(self, name: str, dimension: CapabilityDimension,
                         initial_score: float = 0.5) -> None:
        """注册新的能力指标"""
        if name not in self.metrics:
            self.metrics[name] = CapabilityMetric(
                name=name,
                dimension=dimension,
                score=max(0.0, min(1.0, initial_score)),
            )
            self._dimension_aggregators[dimension].append(name)

    def record_performance(self, metric_name: str, score: float,
                            weight: float = 1.0) -> None:
        """记录能力表现"""
        if metric_name not in self.metrics:
            # 自动注册为技术能力
            self.register_metric(metric_name, CapabilityDimension.TECHNICAL)
        self.metrics[metric_name].update(score, weight)

    def assess_dimension(self, dimension: CapabilityDimension) -> float:
        """评估单维度能力（取该维度所有指标的加权平均）"""
        metric_names = self._dimension_aggregators[dimension]
        if not metric_names:
            return 0.5
        scores = []
        weights = []
        for name in metric_names:
            metric = self.metrics[name]
            scores.append(metric.score * metric.confidence)
            weights.append(metric.confidence)
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        return sum(scores) / total_weight

    def assess_all(self) -> Dict[CapabilityDimension, float]:
        """评估所有维度"""
        return {dim: self.assess_dimension(dim) for dim in CapabilityDimension}

    def get_capability_profile(self) -> Dict[str, Any]:
        """生成完整能力画像"""
        dimensions = self.assess_all()
        # 总体能力分数
        overall = float(np.mean(list(dimensions.values()))) if dimensions else 0.5

        # 优势维度（最高3个）
        sorted_dims = sorted(dimensions.items(), key=lambda x: -x[1])
        strengths = [{"dimension": d.value, "score": s}
                     for d, s in sorted_dims[:3] if s >= 0.6]

        # 弱势维度（最低3个）
        weaknesses = [{"dimension": d.value, "score": s}
                      for d, s in sorted_dims[-3:] if s < 0.5]

        # 能力差距
        gaps = self.identify_gaps()

        profile = {
            "overall_score": overall,
            "overall_proficiency": ProficiencyLevel.from_score(overall).name,
            "dimensions": {d.value: s for d, s in dimensions.items()},
            "strengths": strengths,
            "weaknesses": weaknesses,
            "gaps": [g.to_dict() for g in gaps],
            "metrics_count": len(self.metrics),
            "total_samples": sum(m.sample_count for m in self.metrics.values()),
            "assessed_at": time.time(),
        }
        return profile

    def identify_gaps(self) -> List[CapabilityGap]:
        """识别能力差距"""
        gaps = []
        for dim, target in self.targets.items():
            current = self.assess_dimension(dim)
            gap = target - current
            if gap > 0.1:  # 显著差距阈值
                # 优先级：差距越大、维度重要性越高
                priority = min(1.0, gap * 1.5)
                gaps.append(CapabilityGap(
                    dimension=dim,
                    current_score=current,
                    target_score=target,
                    gap=gap,
                    priority=priority,
                ))
        gaps.sort(key=lambda g: -g.priority)
        return gaps

    def get_development_plan(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """生成能力发展计划"""
        gaps = self.identify_gaps()[:top_n]
        plan = []
        for gap in gaps:
            # 找到该维度下最弱的指标
            dim_metrics = [self.metrics[name]
                           for name in self._dimension_aggregators[gap.dimension]]
            dim_metrics.sort(key=lambda m: m.score)
            weakest = dim_metrics[0] if dim_metrics else None

            plan.append({
                "dimension": gap.dimension.value,
                "current_score": gap.current_score,
                "target_score": gap.target_score,
                "gap": gap.gap,
                "priority": gap.priority,
                "focus_metric": weakest.name if weakest else None,
                "focus_metric_score": weakest.score if weakest else None,
                "suggested_actions": self._suggest_actions(gap.dimension, gap.gap),
            })
        return plan

    def _suggest_actions(self, dimension: CapabilityDimension, gap: float) -> List[str]:
        """生成改进建议"""
        actions = {
            CapabilityDimension.TECHNICAL: [
                "增加算法练习与执行次数",
                "学习新的算法与数据结构",
                "优化现有技术实现",
            ],
            CapabilityDimension.COGNITIVE: [
                "加强推理链条训练",
                "复盘决策过程",
                "引入多元思维模型",
            ],
            CapabilityDimension.LEARNING: [
                "扩展学习资源",
                "尝试跨领域迁移",
                "建立知识图谱",
            ],
            CapabilityDimension.ADAPTIVE: [
                "主动接触新环境",
                "增加变化场景训练",
                "建立适应性反馈机制",
            ],
            CapabilityDimension.SOCIAL: [
                "增加协作任务",
                "提升沟通清晰度",
                "学习冲突调解",
            ],
            CapabilityDimension.CREATIVE: [
                "尝试非常规解法",
                "练习问题重构",
                "引入类比思维",
            ],
            CapabilityDimension.EXECUTION: [
                "强化任务分解与跟踪",
                "提升目标达成率",
                "减少执行偏差",
            ],
            CapabilityDimension.RESILIENCE: [
                "增加错误恢复演练",
                "建立压力测试机制",
                "优化故障应对策略",
            ],
        }
        suggested = actions.get(dimension, ["综合训练"])[:]
        if gap > 0.3:
            suggested.append("差距较大，建议优先投入资源")
        return suggested

    def get_capability_trend(self, dimension: CapabilityDimension,
                              window: int = 20) -> Dict[str, Any]:
        """获取能力趋势"""
        metric_names = self._dimension_aggregators[dimension]
        if not metric_names:
            return {"dimension": dimension.value, "trend": 0.0, "samples": 0}

        trends = []
        for name in metric_names:
            metric = self.metrics[name]
            if metric.trend != 0.0:
                trends.append(metric.trend)

        avg_trend = float(np.mean(trends)) if trends else 0.0
        direction = "improving" if avg_trend > 0.01 else \
                    "declining" if avg_trend < -0.01 else "stable"

        return {
            "dimension": dimension.value,
            "trend": avg_trend,
            "direction": direction,
            "samples": sum(self.metrics[n].sample_count for n in metric_names),
        }

    def compare_with(self, other: "CapabilityAssessor") -> Dict[str, Any]:
        """与其他评估器对比"""
        my_profile = self.assess_all()
        other_profile = other.assess_all()
        diffs = {}
        for dim in CapabilityDimension:
            diffs[dim.value] = my_profile[dim] - other_profile[dim]
        return {
            "my_overall": float(np.mean(list(my_profile.values()))),
            "other_overall": float(np.mean(list(other_profile.values()))),
            "dimension_diffs": diffs,
            "advantages": [d for d, v in diffs.items() if v > 0.1],
            "disadvantages": [d for d, v in diffs.items() if v < -0.1],
        }

    def save_snapshot(self) -> Dict[str, Any]:
        """保存当前能力快照"""
        snapshot = {
            "timestamp": time.time(),
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "dimensions": {d.value: self.assess_dimension(d)
                          for d in CapabilityDimension},
        }
        self.assessment_history.append(snapshot)
        return snapshot

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return list(self.assessment_history)[-limit:]

    def set_target(self, dimension: CapabilityDimension, target: float) -> None:
        """设置能力目标"""
        self.targets[dimension] = max(0.0, min(1.0, target))

    def get_stats(self) -> Dict[str, Any]:
        """获取评估器统计"""
        return {
            "total_metrics": len(self.metrics),
            "total_samples": sum(m.sample_count for m in self.metrics.values()),
            "total_assessments": len(self.assessment_history),
            "dimensions_covered": len(CapabilityDimension),
            "avg_confidence": (float(np.mean([m.confidence for m in self.metrics.values()]))
                              if self.metrics else 0.0),
        }
