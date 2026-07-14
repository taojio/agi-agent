"""
decision/decision_tracker.py - 决策质量追踪

记录决策过程，追踪决策结果，形成反馈闭环
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DecisionStatus(Enum):
    """决策状态"""
    PENDING = "pending"          # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    ABORTED = "aborted"          # 已中止
    TIMEOUT = "timeout"          # 超时


class BiasType(Enum):
    """偏差类型"""
    OVERCONFIDENCE = "overconfidence"      # 过度自信
    RISK_AVERSION = "risk_aversion"        # 风险规避
    RISK_SEEKING = "risk_seeking"          # 风险偏好
    ANCHORING = "anchoring"                # 锚定效应
    RECENCY = "recency"                    # 近因效应
    CONFIRMATION = "confirmation"          # 确认偏差


@dataclass
class DecisionRecord:
    """决策记录

    完整记录一次决策的上下文、选择和结果
    """

    decision_id: str
    timestamp: float = field(default_factory=time.time)
    goal: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, Any]] = field(default_factory=list)
    selected_option: Optional[Dict[str, Any]] = None
    strategy_used: str = ""
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    actual_outcome: Optional[Dict[str, Any]] = None
    status: DecisionStatus = DecisionStatus.PENDING
    execution_time: float = 0.0
    quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status in (DecisionStatus.COMPLETED,
                                DecisionStatus.FAILED,
                                DecisionStatus.ABORTED,
                                DecisionStatus.TIMEOUT)

    @property
    def is_successful(self) -> bool:
        return self.status == DecisionStatus.COMPLETED

    @property
    def prediction_accuracy(self) -> Optional[float]:
        """预测准确度"""
        if not self.actual_outcome or not self.expected_outcome:
            return None

        common_keys = set(self.expected_outcome.keys()) & set(self.actual_outcome.keys())
        if not common_keys:
            return None

        errors = []
        for key in common_keys:
            expected = self.expected_outcome[key]
            actual = self.actual_outcome[key]
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if abs(expected) > 1e-10:
                    error = abs(actual - expected) / abs(expected)
                    errors.append(error)

        if not errors:
            return None

        return float(1.0 - np.mean(errors))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "goal": self.goal,
            "context": self.context,
            "options": self.options,
            "selected_option": self.selected_option,
            "strategy_used": self.strategy_used,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "quality_score": self.quality_score,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class BiasPattern:
    """偏差模式"""

    bias_type: BiasType
    description: str
    frequency: int = 0
    severity: float = 0.0
    examples: List[str] = field(default_factory=list)
    correction_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias_type": self.bias_type.value,
            "description": self.description,
            "frequency": self.frequency,
            "severity": self.severity,
            "examples": self.examples,
            "correction_suggestions": self.correction_suggestions,
        }


class DecisionQualityTracker:
    """决策质量追踪器

    记录决策、追踪结果、分析质量、识别偏差
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._decisions: deque = deque(maxlen=max_history)
        self._pending: Dict[str, DecisionRecord] = {}

        self._stats = {
            "total_decisions": 0,
            "successful": 0,
            "failed": 0,
            "aborted": 0,
            "average_quality": 0.0,
            "average_prediction_accuracy": 0.0,
        }

    def record_decision(self, decision_id: str,
                         goal: str,
                         context: Dict[str, Any],
                         options: List[Any],
                         selected_option: Any,
                         strategy_used: str = "",
                         expected_outcome: Dict[str, Any] = None) -> DecisionRecord:
        """记录决策

        Args:
            decision_id: 决策ID
            goal: 决策目标
            context: 决策上下文
            options: 可用选项列表
            selected_option: 选中的选项
            strategy_used: 使用的策略
            expected_outcome: 预期结果

        Returns:
            决策记录
        """
        record = DecisionRecord(
            decision_id=decision_id,
            goal=goal,
            context=self._serialize_dict(context),
            options=[self._serialize_option(o) for o in options],
            selected_option=self._serialize_option(selected_option),
            strategy_used=strategy_used,
            expected_outcome=expected_outcome or {},
        )

        self._pending[decision_id] = record
        self._stats["total_decisions"] += 1

        return record

    def record_outcome(self, decision_id: str,
                        actual_outcome: Dict[str, Any],
                        status: DecisionStatus = DecisionStatus.COMPLETED,
                        execution_time: float = 0.0,
                        errors: List[str] = None) -> Optional[DecisionRecord]:
        """记录决策结果

        Args:
            decision_id: 决策ID
            actual_outcome: 实际结果
            status: 决策状态
            execution_time: 执行时间
            errors: 错误列表

        Returns:
            更新后的决策记录
        """
        record = self._pending.pop(decision_id, None)
        if record is None:
            return None

        record.actual_outcome = self._serialize_dict(actual_outcome)
        record.status = status
        record.execution_time = execution_time
        record.errors = errors or []

        record.quality_score = self._calculate_quality(record)

        self._decisions.append(record)
        self._update_stats(record)

        return record

    def calculate_decision_quality(self, decision_id: str) -> float:
        """计算决策质量评分"""
        record = self._find_decision(decision_id)
        if record is None:
            return 0.0
        return record.quality_score

    def get_decision_history(self, limit: int = 100) -> List[DecisionRecord]:
        """获取决策历史"""
        return list(self._decisions)[-limit:]

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """获取指定决策"""
        return self._find_decision(decision_id)

    def get_pending_decisions(self) -> List[DecisionRecord]:
        """获取待完成的决策"""
        return list(self._pending.values())

    def identify_bias_patterns(self) -> List[BiasPattern]:
        """识别决策偏差模式"""
        patterns = []

        completed = [d for d in self._decisions if d.is_completed]
        if len(completed) < 5:
            return patterns

        overconfidence = self._detect_overconfidence(completed)
        if overconfidence:
            patterns.append(overconfidence)

        risk_aversion = self._detect_risk_aversion(completed)
        if risk_aversion:
            patterns.append(risk_aversion)

        recency = self._detect_recency_bias(completed)
        if recency:
            patterns.append(recency)

        return patterns

    def get_quality_metrics(self) -> Dict[str, Any]:
        """获取质量指标"""
        if not self._decisions:
            return self._stats.copy()

        completed = [d for d in self._decisions if d.is_completed]
        successful = [d for d in completed if d.is_successful]

        accuracies = [d.prediction_accuracy for d in completed
                      if d.prediction_accuracy is not None]

        return {
            "total_decisions": self._stats["total_decisions"],
            "completed": len(completed),
            "successful": len(successful),
            "success_rate": len(successful) / max(1, len(completed)),
            "average_quality": float(np.mean([d.quality_score for d in completed]))
                               if completed else 0.0,
            "average_prediction_accuracy": float(np.mean(accuracies))
                                            if accuracies else 0.0,
            "average_execution_time": float(np.mean([d.execution_time for d in completed]))
                                       if completed else 0.0,
            "pending_count": len(self._pending),
        }

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """获取各策略的性能表现"""
        strategy_stats: Dict[str, List[float]] = {}

        for d in self._decisions:
            if not d.is_completed:
                continue
            strategy = d.strategy_used or "unknown"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(d.quality_score)

        return {
            strategy: {
                "count": len(scores),
                "average_quality": float(np.mean(scores)),
                "min_quality": float(np.min(scores)),
                "max_quality": float(np.max(scores)),
            }
            for strategy, scores in strategy_stats.items()
        }

    def generate_report(self) -> Dict[str, Any]:
        """生成完整报告"""
        return {
            "quality_metrics": self.get_quality_metrics(),
            "strategy_performance": self.get_strategy_performance(),
            "bias_patterns": [p.to_dict() for p in self.identify_bias_patterns()],
            "recent_decisions": [d.to_dict() for d in self.get_decision_history(10)],
            "pending_count": len(self._pending),
        }

    # ====== 内部方法 ======

    def _calculate_quality(self, record: DecisionRecord) -> float:
        """计算决策质量评分"""
        score = 0.0

        if record.is_successful:
            score += 0.4
        elif record.status == DecisionStatus.FAILED:
            score -= 0.2

        accuracy = record.prediction_accuracy
        if accuracy is not None:
            score += 0.3 * accuracy

        if not record.errors:
            score += 0.1

        if record.execution_time > 0:
            time_factor = max(0, 1.0 - record.execution_time / 60.0)
            score += 0.2 * time_factor

        return float(max(0.0, min(1.0, score)))

    def _update_stats(self, record: DecisionRecord) -> None:
        """更新统计"""
        if record.is_successful:
            self._stats["successful"] += 1
        elif record.status == DecisionStatus.FAILED:
            self._stats["failed"] += 1
        elif record.status == DecisionStatus.ABORTED:
            self._stats["aborted"] += 1

        completed = [d for d in self._decisions if d.is_completed]
        if completed:
            self._stats["average_quality"] = float(
                np.mean([d.quality_score for d in completed])
            )

            accuracies = [d.prediction_accuracy for d in completed
                          if d.prediction_accuracy is not None]
            if accuracies:
                self._stats["average_prediction_accuracy"] = float(np.mean(accuracies))

    def _find_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """查找决策"""
        if decision_id in self._pending:
            return self._pending[decision_id]
        for d in self._decisions:
            if d.decision_id == decision_id:
                return d
        return None

    def _serialize_option(self, option: Any) -> Dict[str, Any]:
        """序列化选项"""
        if hasattr(option, 'to_dict'):
            return option.to_dict()
        if isinstance(option, dict):
            return option
        return {"value": str(option)}

    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """序列化字典"""
        result = {}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                result[k] = v
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [str(x) if not isinstance(x, (int, float, str, bool))
                            else x for x in v]
            else:
                result[k] = str(v)
        return result

    def _detect_overconfidence(self, decisions: List[DecisionRecord]) -> Optional[BiasPattern]:
        """检测过度自信"""
        overconfident = []
        for d in decisions:
            if d.prediction_accuracy is not None and d.prediction_accuracy < 0.5:
                expected_reward = d.expected_outcome.get("reward", 0.5)
                if expected_reward > 0.7:
                    overconfident.append(d)

        if len(overconfident) >= len(decisions) * 0.3:
            return BiasPattern(
                bias_type=BiasType.OVERCONFIDENCE,
                description="决策时预期收益过高，实际表现不及预期",
                frequency=len(overconfident),
                severity=len(overconfident) / max(1, len(decisions)),
                examples=[d.decision_id for d in overconfident[:3]],
                correction_suggestions=[
                    "降低预期收益估计",
                    "增加不确定性考量",
                    "参考历史成功率",
                ],
            )
        return None

    def _detect_risk_aversion(self, decisions: List[DecisionRecord]) -> Optional[BiasPattern]:
        """检测风险规避"""
        low_risk_choices = 0
        total = 0

        for d in decisions:
            if d.selected_option and "risk_level" in d.selected_option:
                total += 1
                if d.selected_option["risk_level"] == "low":
                    low_risk_choices += 1

        if total > 0 and low_risk_choices / total > 0.8:
            return BiasPattern(
                bias_type=BiasType.RISK_AVERSION,
                description="过度偏好低风险选项",
                frequency=low_risk_choices,
                severity=low_risk_choices / total,
                correction_suggestions=[
                    "适度接受中等风险选项",
                    "评估高风险选项的潜在收益",
                ],
            )
        return None

    def _detect_recency_bias(self, decisions: List[DecisionRecord]) -> Optional[BiasPattern]:
        """检测近因效应"""
        if len(decisions) < 10:
            return None

        recent = decisions[-5:]
        older = decisions[-10:-5]

        recent_strategies = [d.strategy_used for d in recent]
        older_strategies = [d.strategy_used for d in older]

        from collections import Counter
        recent_counter = Counter(recent_strategies)
        older_counter = Counter(older_strategies)

        for strategy, count in recent_counter.items():
            if count > 3 and older_counter.get(strategy, 0) <= 1:
                return BiasPattern(
                    bias_type=BiasType.RECENCY,
                    description=f"近期过度使用策略 '{strategy}'",
                    frequency=count,
                    severity=count / len(recent),
                    correction_suggestions=[
                        "考虑策略多样性",
                        "根据场景选择合适策略",
                    ],
                )
        return None
