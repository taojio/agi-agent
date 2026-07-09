"""
meta_cognitive/reflection_engine.py - 元认知反思引擎

负责深度自我反思、经验提炼、行为模式识别。
通过定期反思历史行为，识别成功/失败模式，提炼可复用经验。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ReflectionType(Enum):
    """反思类型"""
    ACTION_REVIEW = "action_review"            # 行动回顾
    DECISION_AUDIT = "decision_audit"          # 决策审计
    PERFORMANCE_ANALYSIS = "performance_analysis"  # 性能分析
    LEARNING_REVIEW = "learning_review"        # 学习回顾
    GOAL_REFLECTION = "goal_reflection"        # 目标反思
    STRATEGY_EVALUATION = "strategy_evaluation"  # 策略评估


class ReflectionDepth(Enum):
    """反思深度"""
    SURFACE = 1       # 表层：仅统计汇总
    PATTERN = 2       # 模式层：识别重复模式
    CAUSAL = 3        # 因果层：分析原因
    META = 4          # 元层：反思反思过程本身


@dataclass
class ReflectionResult:
    """反思结果"""
    reflection_id: str
    reflection_type: ReflectionType
    depth: ReflectionDepth
    timestamp: float
    observations: List[str] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_insight(self, insight: str) -> None:
        self.insights.append(insight)

    def add_recommendation(self, rec: str) -> None:
        self.recommendations.append(rec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "reflection_type": self.reflection_type.value,
            "depth": self.depth.value,
            "timestamp": self.timestamp,
            "observations": self.observations,
            "patterns": self.patterns,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ExperienceEntry:
    """经验条目"""
    experience_id: str
    timestamp: float
    context: Dict[str, Any]
    action: str
    outcome: str           # success / failure / partial
    reward: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    generalization: Optional[str] = None  # 提炼出的通用经验

    @property
    def is_success(self) -> bool:
        return self.outcome == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "reward": self.reward,
            "duration": self.duration,
            "metadata": self.metadata,
            "generalization": self.generalization,
        }


class ReflectionEngine:
    """元认知反思引擎

    定期回顾历史行为，识别模式，提炼经验，生成改进建议。
    支持多种反思深度和类型。

    Attributes:
        experiences: 经验库
        reflections: 反思历史
        pattern_detectors: 模式检测器
    """

    def __init__(self, max_experiences: int = 1000, max_reflections: int = 200):
        self.experiences: deque = deque(maxlen=max_experiences)
        self.reflections: deque = deque(maxlen=max_reflections)
        self._experience_counter = 0
        self._reflection_counter = 0
        self._pattern_detectors: Dict[str, Callable] = {
            "success_pattern": self._detect_success_patterns,
            "failure_pattern": self._detect_failure_patterns,
            "context_pattern": self._detect_context_patterns,
            "temporal_pattern": self._detect_temporal_patterns,
        }
        # 反思质量追踪
        self._reflection_quality_history: deque = deque(maxlen=50)
        # 已提炼的通用经验库
        self.distilled_lessons: Dict[str, Dict[str, Any]] = {}

    def record_experience(self, context: Dict[str, Any], action: str,
                          outcome: str, reward: float = 0.0,
                          duration: float = 0.0, **metadata) -> str:
        """记录一条经验"""
        self._experience_counter += 1
        exp_id = f"exp_{self._experience_counter}"
        entry = ExperienceEntry(
            experience_id=exp_id,
            timestamp=time.time(),
            context=context,
            action=action,
            outcome=outcome,
            reward=reward,
            duration=duration,
            metadata=metadata,
        )
        self.experiences.append(entry)
        return exp_id

    def reflect(self, reflection_type: ReflectionType = ReflectionType.ACTION_REVIEW,
                depth: ReflectionDepth = ReflectionDepth.PATTERN,
                window: Optional[int] = None) -> ReflectionResult:
        """执行反思

        Args:
            reflection_type: 反思类型
            depth: 反思深度
            window: 回顾窗口大小（None 表示全部）

        Returns:
            ReflectionResult: 反思结果
        """
        self._reflection_counter += 1
        result = ReflectionResult(
            reflection_id=f"refl_{self._reflection_counter}",
            reflection_type=reflection_type,
            depth=depth,
            timestamp=time.time(),
        )

        experiences = list(self.experiences)
        if window is not None:
            experiences = experiences[-window:]

        if not experiences:
            result.observations.append("No experiences to reflect upon")
            result.confidence = 0.1
            self.reflections.append(result)
            return result

        # 阶段1: 观察（统计汇总）
        self._observe(experiences, result)

        # 阶段2: 模式识别
        if depth.value >= ReflectionDepth.PATTERN.value:
            self._identify_patterns(experiences, result)

        # 阶段3: 因果分析
        if depth.value >= ReflectionDepth.CAUSAL.value:
            self._analyze_causes(experiences, result)

        # 阶段4: 元反思（反思反思过程）
        if depth.value >= ReflectionDepth.META.value:
            self._meta_reflect(result)

        # 阶段5: 提炼洞察与建议
        self._generate_insights_and_recommendations(experiences, result)

        # 评估反思质量
        quality_score = self._evaluate_reflection_quality(result, experiences)
        result.confidence = quality_score
        self._reflection_quality_history.append(quality_score)

        self.reflections.append(result)
        return result

    def _observe(self, experiences: List[ExperienceEntry], result: ReflectionResult) -> None:
        """统计观察"""
        total = len(experiences)
        successes = sum(1 for e in experiences if e.is_success)
        failures = sum(1 for e in experiences if e.outcome == "failure")
        partials = total - successes - failures

        avg_reward = float(np.mean([e.reward for e in experiences])) if experiences else 0.0
        avg_duration = float(np.mean([e.duration for e in experiences])) if experiences else 0.0

        result.observations.extend([
            f"Total experiences: {total}",
            f"Success rate: {successes / total:.2%}" if total > 0 else "No data",
            f"Failure rate: {failures / total:.2%}" if total > 0 else "No data",
            f"Average reward: {avg_reward:.3f}",
            f"Average duration: {avg_duration:.3f}",
        ])

        result.metadata["stats"] = {
            "total": total,
            "successes": successes,
            "failures": failures,
            "partials": partials,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_reward": avg_reward,
            "avg_duration": avg_duration,
        }

    def _identify_patterns(self, experiences: List[ExperienceEntry],
                            result: ReflectionResult) -> None:
        """模式识别"""
        for detector_name, detector in self._pattern_detectors.items():
            try:
                patterns = detector(experiences)
                for p in patterns:
                    result.patterns.append({
                        "detector": detector_name,
                        **p,
                    })
            except Exception as e:
                result.observations.append(
                    f"Pattern detector {detector_name} failed: {e}"
                )

    def _detect_success_patterns(self, experiences: List[ExperienceEntry]) -> List[Dict[str, Any]]:
        """识别成功模式"""
        successes = [e for e in experiences if e.is_success]
        if len(successes) < 3:
            return []

        patterns = []
        # 按动作分组
        action_groups: Dict[str, List[ExperienceEntry]] = {}
        for exp in successes:
            action_groups.setdefault(exp.action, []).append(exp)

        for action, group in action_groups.items():
            if len(group) >= 2:
                avg_reward = float(np.mean([e.reward for e in group]))
                patterns.append({
                    "type": "success_action",
                    "action": action,
                    "count": len(group),
                    "avg_reward": avg_reward,
                    "description": f"Action '{action}' succeeded {len(group)} times with avg reward {avg_reward:.3f}",
                })

        # 按上下文关键词分组
        context_keywords: Dict[str, int] = {}
        for exp in successes:
            for key, value in exp.context.items():
                if isinstance(value, (int, float)) and value > 0.7:
                    context_keywords[str(key)] = context_keywords.get(str(key), 0) + 1
        for key, count in context_keywords.items():
            if count >= 2:
                patterns.append({
                    "type": "success_context",
                    "context_key": key,
                    "count": count,
                    "description": f"High {key} correlates with success ({count} times)",
                })

        return patterns

    def _detect_failure_patterns(self, experiences: List[ExperienceEntry]) -> List[Dict[str, Any]]:
        """识别失败模式"""
        failures = [e for e in experiences if e.outcome == "failure"]
        if len(failures) < 2:
            return []

        patterns = []
        action_groups: Dict[str, List[ExperienceEntry]] = {}
        for exp in failures:
            action_groups.setdefault(exp.action, []).append(exp)

        for action, group in action_groups.items():
            if len(group) >= 2:
                patterns.append({
                    "type": "failure_action",
                    "action": action,
                    "count": len(group),
                    "description": f"Action '{action}' failed {len(group)} times - candidate for redesign",
                })

        # 识别高失败率的上下文
        context_fail_rate: Dict[str, float] = {}
        context_total: Dict[str, int] = {}
        for exp in experiences:
            for key, value in exp.context.items():
                if isinstance(value, (int, float)) and value < 0.3:
                    context_total[str(key)] = context_total.get(str(key), 0) + 1
                    if exp.outcome == "failure":
                        context_fail_rate[str(key)] = context_fail_rate.get(str(key), 0) + 1

        for key, fail_count in context_fail_rate.items():
            total = context_total.get(key, 0)
            if total >= 2 and fail_count / total >= 0.5:
                patterns.append({
                    "type": "failure_context",
                    "context_key": key,
                    "failure_rate": fail_count / total,
                    "description": f"Low {key} correlates with {fail_count / total:.0%} failure rate",
                })

        return patterns

    def _detect_context_patterns(self, experiences: List[ExperienceEntry]) -> List[Dict[str, Any]]:
        """识别上下文模式"""
        patterns = []
        # 识别频繁共现的上下文键
        key_cooccurrence: Dict[Tuple[str, str], int] = {}
        for exp in experiences:
            keys = sorted(exp.context.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pair = (keys[i], keys[j])
                    key_cooccurrence[pair] = key_cooccurrence.get(pair, 0) + 1

        frequent_pairs = [(pair, count) for pair, count in key_cooccurrence.items()
                          if count >= 3]
        frequent_pairs.sort(key=lambda x: -x[1])

        for (k1, k2), count in frequent_pairs[:5]:
            patterns.append({
                "type": "context_cooccurrence",
                "keys": [k1, k2],
                "count": count,
                "description": f"Context keys '{k1}' and '{k2}' frequently co-occur ({count} times)",
            })

        return patterns

    def _detect_temporal_patterns(self, experiences: List[ExperienceEntry]) -> List[Dict[str, Any]]:
        """识别时间模式"""
        if len(experiences) < 5:
            return []

        patterns = []
        timestamps = [e.timestamp for e in experiences]
        if len(timestamps) < 2:
            return patterns

        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        avg_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals)) if len(intervals) > 1 else 0.0

        patterns.append({
            "type": "temporal_frequency",
            "avg_interval": avg_interval,
            "std_interval": std_interval,
            "description": f"Action frequency: every {avg_interval:.2f}s "
                          f"(variability: {std_interval:.2f}s)",
        })

        # 识别成功/失败的时间聚集
        success_times = [e.timestamp for e in experiences if e.is_success]
        failure_times = [e.timestamp for e in experiences if e.outcome == "failure"]

        if len(success_times) >= 3:
            s_intervals = [success_times[i + 1] - success_times[i]
                          for i in range(len(success_times) - 1)]
            patterns.append({
                "type": "success_burst",
                "description": f"Successes cluster with avg interval {np.mean(s_intervals):.2f}s",
            })

        if len(failure_times) >= 3:
            f_intervals = [failure_times[i + 1] - failure_times[i]
                          for i in range(len(failure_times) - 1)]
            patterns.append({
                "type": "failure_burst",
                "description": f"Failures cluster with avg interval {np.mean(f_intervals):.2f}s",
            })

        return patterns

    def _analyze_causes(self, experiences: List[ExperienceEntry],
                         result: ReflectionResult) -> None:
        """因果分析（基于启发式）"""
        successes = [e for e in experiences if e.is_success]
        failures = [e for e in experiences if e.outcome == "failure"]

        if successes and failures:
            # 比较成功与失败的上下文差异
            success_context = self._aggregate_context(successes)
            failure_context = self._aggregate_context(failures)

            for key in success_context:
                if key in failure_context:
                    diff = success_context[key] - failure_context[key]
                    if abs(diff) > 0.2:
                        direction = "higher" if diff > 0 else "lower"
                        result.insights.append(
                            f"Causal hint: {key} is {direction} in successful cases "
                            f"(diff={diff:+.3f})"
                        )

        # 分析奖励与持续时间的关系
        if len(experiences) >= 5:
            rewards = [e.reward for e in experiences]
            durations = [e.duration for e in experiences]
            try:
                correlation = float(np.corrcoef(rewards, durations)[0, 1])
                if abs(correlation) > 0.4:
                    result.insights.append(
                        f"Reward-duration correlation: {correlation:+.3f} "
                        f"({'longer actions yield higher reward' if correlation > 0 else 'shorter actions yield higher reward'})"
                    )
            except (ValueError, np.linalg.LinAlgError):
                pass

    def _aggregate_context(self, experiences: List[ExperienceEntry]) -> Dict[str, float]:
        """聚合上下文的平均值"""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for exp in experiences:
            for key, value in exp.context.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
        return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}

    def _meta_reflect(self, result: ReflectionResult) -> None:
        """元反思：反思反思过程本身"""
        if len(self._reflection_quality_history) < 3:
            return

        avg_quality = float(np.mean(list(self._reflection_quality_history)))
        recent_quality = float(np.mean(list(self._reflection_quality_history)[-5:]))

        if recent_quality < avg_quality - 0.1:
            result.insights.append(
                "Meta: Recent reflection quality is declining - "
                "consider adjusting reflection depth or frequency"
            )
        elif recent_quality > avg_quality + 0.1:
            result.insights.append(
                "Meta: Recent reflection quality is improving - "
                "current reflection strategy is effective"
            )

        # 反思模式分布
        if len(self.reflections) >= 10:
            type_counts: Dict[str, int] = {}
            for r in self.reflections:
                type_counts[r.reflection_type.value] = type_counts.get(
                    r.reflection_type.value, 0) + 1
            dominant_type = max(type_counts, key=type_counts.get)
            if type_counts[dominant_type] / len(self.reflections) > 0.7:
                result.insights.append(
                    f"Meta: Reflection type '{dominant_type}' dominates "
                    f"({type_counts[dominant_type] / len(self.reflections):.0%}) - "
                    "consider diversifying reflection types"
                )

    def _generate_insights_and_recommendations(self, experiences: List[ExperienceEntry],
                                                 result: ReflectionResult) -> None:
        """基于反思结果生成洞察与建议"""
        stats = result.metadata.get("stats", {})

        # 基于统计的洞察
        success_rate = stats.get("success_rate", 0.0)
        if success_rate > 0.8:
            result.add_insight("Performance is strong - current strategy is effective")
        elif success_rate < 0.4:
            result.add_insight("Performance is weak - significant improvement needed")
            result.add_recommendation("Review failed actions and consider strategy change")

        # 基于模式的建议
        success_patterns = [p for p in result.patterns if p.get("type", "").startswith("success")]
        failure_patterns = [p for p in result.patterns if p.get("type", "").startswith("failure")]

        if success_patterns:
            result.add_recommendation(
                f"Reinforce successful patterns: {len(success_patterns)} identified"
            )
        if failure_patterns:
            result.add_recommendation(
                f"Address failure patterns: {len(failure_patterns)} identified - "
                "consider alternative actions or context adjustments"
            )

        # 基于奖励的建议
        avg_reward = stats.get("avg_reward", 0.0)
        if avg_reward < 0.3:
            result.add_recommendation("Reward signal is low - check action selection logic")
        elif avg_reward > 0.7:
            result.add_insight("Reward signal is healthy")

    def _evaluate_reflection_quality(self, result: ReflectionResult,
                                      experiences: List[ExperienceEntry]) -> float:
        """评估反思质量 (0-1)"""
        if not experiences:
            return 0.1

        score = 0.0
        # 有观察
        if result.observations:
            score += 0.2
        # 有模式
        if result.patterns:
            score += 0.3
        # 有洞察
        if result.insights:
            score += 0.2
        # 有建议
        if result.recommendations:
            score += 0.2
        # 深度更高加分
        score += (result.depth.value - 1) * 0.05

        return min(1.0, score)

    def distill_lesson(self, pattern: Dict[str, Any], lesson_id: Optional[str] = None) -> str:
        """将模式提炼为通用经验"""
        lesson_id = lesson_id or f"lesson_{len(self.distilled_lessons) + 1}"
        self.distilled_lessons[lesson_id] = {
            "lesson_id": lesson_id,
            "source_pattern": pattern,
            "created_at": time.time(),
            "applied_count": 0,
            "success_count": 0,
        }
        return lesson_id

    def get_lesson(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        return self.distilled_lessons.get(lesson_id)

    def record_lesson_application(self, lesson_id: str, success: bool) -> None:
        """记录经验应用结果"""
        if lesson_id in self.distilled_lessons:
            lesson = self.distilled_lessons[lesson_id]
            lesson["applied_count"] += 1
            if success:
                lesson["success_count"] += 1

    def get_reflection_history(self, limit: int = 20) -> List[ReflectionResult]:
        return list(self.reflections)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取反思引擎统计"""
        reflection_list = list(self.reflections)
        type_dist: Dict[str, int] = {}
        for r in reflection_list:
            type_dist[r.reflection_type.value] = type_dist.get(
                r.reflection_type.value, 0) + 1

        avg_quality = (float(np.mean(list(self._reflection_quality_history)))
                       if self._reflection_quality_history else 0.0)

        return {
            "total_experiences": len(self.experiences),
            "total_reflections": len(self.reflections),
            "distilled_lessons": len(self.distilled_lessons),
            "avg_reflection_quality": avg_quality,
            "reflection_type_distribution": type_dist,
        }
