"""
learning/experience_distiller.py - 经验沉淀器

从历史经验中提取可复用模式，进行策略演化。
将原始经验蒸馏为通用规则、技能模板、策略原型。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PatternType(Enum):
    """模式类型"""
    ACTION_SEQUENCE = "action_sequence"    # 动作序列模式
    CONTEXT_ACTION = "context_action"      # 上下文-动作关联
    SUCCESS_FACTOR = "success_factor"      # 成功因素
    FAILURE_FACTOR = "failure_factor"      # 失败因素
    TEMPORAL = "temporal"                  # 时间模式
    CAUSAL = "causal"                      # 因果模式


class PatternStatus(Enum):
    """模式状态"""
    CANDIDATE = "candidate"      # 候选
    VALIDATED = "validated"       # 已验证
    DEPRECATED = "deprecated"     # 已废弃
    EVOLVED = "evolved"           # 已演化


@dataclass
class Pattern:
    """经验模式"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    conditions: Dict[str, Any]       # 触发条件
    actions: List[str]               # 关联动作
    expected_outcome: str
    confidence: float = 0.5
    support: int = 0                  # 支持样本数
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_applied: Optional[float] = None
    status: PatternStatus = PatternStatus.CANDIDATE
    variants: List[str] = field(default_factory=list)  # 演化变体 ID

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def record_application(self, success: bool) -> None:
        """记录一次应用结果"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_applied = time.time()
        self.support += 1
        # 更新置信度
        total = self.success_count + self.failure_count
        if total >= 5:
            self.confidence = min(1.0, self.success_rate * 0.7 + 0.3 * min(1.0, total / 20))
        # 状态升级
        if total >= 10 and self.success_rate >= 0.7:
            self.status = PatternStatus.VALIDATED
        elif total >= 10 and self.success_rate < 0.3:
            self.status = PatternStatus.DEPRECATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "conditions": self.conditions,
            "actions": self.actions,
            "expected_outcome": self.expected_outcome,
            "confidence": float(self.confidence),
            "support": self.support,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": float(self.success_rate),
            "created_at": self.created_at,
            "last_applied": self.last_applied,
            "status": self.status.value,
            "variants": self.variants,
        }


@dataclass
class StrategyPrototype:
    """策略原型"""
    prototype_id: str
    name: str
    description: str
    patterns: List[str]  # 关联模式 ID
    priority: float
    created_at: float = field(default_factory=time.time)
    effectiveness: float = 0.5
    application_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prototype_id": self.prototype_id,
            "name": self.name,
            "description": self.description,
            "patterns": self.patterns,
            "priority": float(self.priority),
            "created_at": self.created_at,
            "effectiveness": float(self.effectiveness),
            "application_count": self.application_count,
        }


class ExperienceDistiller:
    """经验沉淀器

    从原始经验流中提取、验证、演化可复用模式。
    工作流程：候选模式生成 → 验证 → 演化 → 沉淀为策略原型

    Attributes:
        patterns: 已发现的模式库
        prototypes: 策略原型库
        raw_experiences: 原始经验流
    """

    def __init__(self, min_support: int = 3, min_confidence: float = 0.6,
                 max_patterns: int = 500):
        self.patterns: Dict[str, Pattern] = {}
        self.prototypes: Dict[str, StrategyPrototype] = {}
        self.raw_experiences: deque = deque(maxlen=2000)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_patterns = max_patterns

        self._pattern_counter = 0
        self._prototype_counter = 0
        self._distillation_history: deque = deque(maxlen=50)

    def ingest_experience(self, context: Dict[str, Any], action: str,
                          outcome: str, reward: float = 0.0, **metadata) -> None:
        """摄入一条原始经验"""
        self.raw_experiences.append({
            "timestamp": time.time(),
            "context": context,
            "action": action,
            "outcome": outcome,
            "reward": reward,
            "metadata": metadata,
        })

    def distill(self) -> Dict[str, Any]:
        """执行一轮经验蒸馏

        Returns:
            蒸馏报告
        """
        start_time = time.time()
        experiences = list(self.raw_experiences)
        if len(experiences) < self.min_support:
            return {
                "status": "insufficient_data",
                "experiences": len(experiences),
                "required": self.min_support,
            }

        # 1. 候选模式生成
        new_candidates = self._generate_candidates(experiences)

        # 2. 模式合并（与已有模式去重/合并）
        merged_count = self._merge_patterns(new_candidates)

        # 3. 模式验证
        validated = self._validate_patterns(experiences)

        # 4. 模式演化（生成变体）
        evolved = self._evolve_patterns()

        # 5. 策略原型生成
        new_prototypes = self._generate_prototypes()

        report = {
            "status": "completed",
            "duration": time.time() - start_time,
            "experiences_processed": len(experiences),
            "new_candidates": len(new_candidates),
            "merged": merged_count,
            "validated": len(validated),
            "evolved": len(evolved),
            "new_prototypes": len(new_prototypes),
            "total_patterns": len(self.patterns),
            "total_prototypes": len(self.prototypes),
        }
        self._distillation_history.append(report)
        return report

    def _generate_candidates(self, experiences: List[Dict[str, Any]]) -> List[Pattern]:
        """生成候选模式"""
        candidates = []

        # 动作序列模式
        action_sequences = self._find_action_sequences(experiences)
        for seq, count, success_rate in action_sequences:
            if count >= self.min_support:
                self._pattern_counter += 1
                candidates.append(Pattern(
                    pattern_id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.ACTION_SEQUENCE,
                    description=f"Action sequence: {' -> '.join(seq)}",
                    conditions={"sequence_length": len(seq)},
                    actions=seq,
                    expected_outcome="success" if success_rate > 0.5 else "failure",
                    confidence=success_rate,
                    support=count,
                ))

        # 上下文-动作关联
        context_action = self._find_context_action_pairs(experiences)
        for ctx_key, ctx_value, action, count, success_rate in context_action:
            if count >= self.min_support:
                self._pattern_counter += 1
                candidates.append(Pattern(
                    pattern_id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.CONTEXT_ACTION,
                    description=f"When {ctx_key}={ctx_value}, take action '{action}'",
                    conditions={ctx_key: ctx_value},
                    actions=[action],
                    expected_outcome="success" if success_rate > 0.5 else "failure",
                    confidence=success_rate,
                    support=count,
                ))

        # 成功/失败因素
        success_factors = self._find_key_factors(experiences, "success")
        for factor, value, count, success_rate in success_factors:
            if count >= self.min_support:
                self._pattern_counter += 1
                candidates.append(Pattern(
                    pattern_id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.SUCCESS_FACTOR,
                    description=f"High {factor} contributes to success",
                    conditions={factor: value},
                    actions=[],
                    expected_outcome="success",
                    confidence=success_rate,
                    support=count,
                ))

        failure_factors = self._find_key_factors(experiences, "failure")
        for factor, value, count, fail_rate in failure_factors:
            if count >= self.min_support:
                self._pattern_counter += 1
                candidates.append(Pattern(
                    pattern_id=f"pat_{self._pattern_counter}",
                    pattern_type=PatternType.FAILURE_FACTOR,
                    description=f"Low {factor} contributes to failure",
                    conditions={factor: value},
                    actions=[],
                    expected_outcome="failure",
                    confidence=fail_rate,
                    support=count,
                ))

        return candidates

    def _find_action_sequences(self, experiences: List[Dict[str, Any]]
                                ) -> List[Tuple[List[str], int, float]]:
        """发现频繁动作序列"""
        sequences = []
        # 长度2和3的序列
        for seq_len in [2, 3]:
            seq_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
            for i in range(len(experiences) - seq_len + 1):
                seq = tuple(exp["action"] for exp in experiences[i:i + seq_len])
                if seq not in seq_counts:
                    seq_counts[seq] = {"total": 0, "success": 0}
                seq_counts[seq]["total"] += 1
                # 序列最后动作的结果作为序列结果
                if experiences[i + seq_len - 1]["outcome"] == "success":
                    seq_counts[seq]["success"] += 1

            for seq, counts in seq_counts.items():
                if counts["total"] >= self.min_support:
                    success_rate = counts["success"] / counts["total"]
                    sequences.append((list(seq), counts["total"], success_rate))

        return sequences

    def _find_context_action_pairs(self, experiences: List[Dict[str, Any]]
                                    ) -> List[Tuple[str, Any, str, int, float]]:
        """发现上下文-动作关联"""
        pairs = []
        pair_counts: Dict[Tuple[str, Any, str], Dict[str, int]] = {}

        for exp in experiences:
            for ctx_key, ctx_value in exp["context"].items():
                if not isinstance(ctx_value, (int, float)):
                    continue
                # 离散化为高/中/低
                if ctx_value > 0.7:
                    bucket = "high"
                elif ctx_value < 0.3:
                    bucket = "low"
                else:
                    bucket = "medium"

                key = (ctx_key, bucket, exp["action"])
                if key not in pair_counts:
                    pair_counts[key] = {"total": 0, "success": 0}
                pair_counts[key]["total"] += 1
                if exp["outcome"] == "success":
                    pair_counts[key]["success"] += 1

        for (ctx_key, bucket, action), counts in pair_counts.items():
            if counts["total"] >= self.min_support:
                success_rate = counts["success"] / counts["total"]
                pairs.append((ctx_key, bucket, action, counts["total"], success_rate))

        return pairs

    def _find_key_factors(self, experiences: List[Dict[str, Any]],
                           outcome: str) -> List[Tuple[str, Any, int, float]]:
        """发现成功/失败关键因素"""
        factors = []
        target_experiences = [e for e in experiences if e["outcome"] == outcome]
        if len(target_experiences) < self.min_support:
            return factors

        # 找出在目标结果中出现频率高的特征值
        factor_counts: Dict[Tuple[str, Any], int] = {}
        for exp in target_experiences:
            for key, value in exp["context"].items():
                if isinstance(value, (int, float)):
                    if value > 0.7:
                        bucket = "high"
                    elif value < 0.3:
                        bucket = "low"
                    else:
                        continue
                    factor_counts[(key, bucket)] = factor_counts.get((key, bucket), 0) + 1

        # 计算相对于整体的比例
        for (key, bucket), count in factor_counts.items():
            if count >= self.min_support:
                rate = count / len(target_experiences)
                factors.append((key, bucket, count, rate))

        return factors

    def _merge_patterns(self, candidates: List[Pattern]) -> int:
        """合并相似模式"""
        merged = 0
        for candidate in candidates:
            # 寻找相似已有模式
            similar = self._find_similar_pattern(candidate)
            if similar:
                # 合并：增加支持度
                similar.support += candidate.support
                similar.confidence = (similar.confidence * similar.support +
                                       candidate.confidence * candidate.support) / \
                                      (similar.support + candidate.support)
                merged += 1
            else:
                if len(self.patterns) < self.max_patterns:
                    self.patterns[candidate.pattern_id] = candidate
        return merged

    def _find_similar_pattern(self, candidate: Pattern) -> Optional[Pattern]:
        """寻找相似模式"""
        for existing in self.patterns.values():
            if existing.pattern_type != candidate.pattern_type:
                continue
            if existing.actions == candidate.actions and \
               self._conditions_similar(existing.conditions, candidate.conditions):
                return existing
        return None

    def _conditions_similar(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        """判断条件是否相似"""
        if set(c1.keys()) != set(c2.keys()):
            return False
        for k in c1:
            if c1[k] != c2[k]:
                return False
        return True

    def _validate_patterns(self, experiences: List[Dict[str, Any]]) -> List[Pattern]:
        """验证模式：在历史经验上回测"""
        validated = []
        for pattern in self.patterns.values():
            if pattern.status != PatternStatus.CANDIDATE:
                continue
            # 在经验上回测
            matches = 0
            successes = 0
            for exp in experiences:
                if self._pattern_matches(pattern, exp):
                    matches += 1
                    if (exp["outcome"] == "success") == (pattern.expected_outcome == "success"):
                        successes += 1

            if matches >= self.min_support:
                pattern.success_count = successes
                pattern.failure_count = matches - successes
                pattern.support = matches
                pattern.confidence = successes / matches if matches > 0 else 0.0
                if pattern.confidence >= self.min_confidence:
                    pattern.status = PatternStatus.VALIDATED
                    validated.append(pattern)
        return validated

    def _pattern_matches(self, pattern: Pattern, experience: Dict[str, Any]) -> bool:
        """检查模式是否匹配经验"""
        # 检查动作匹配
        if pattern.actions:
            if experience["action"] not in pattern.actions:
                return False
        # 检查条件匹配
        for key, value in pattern.conditions.items():
            if key in experience["context"]:
                exp_value = experience["context"][key]
                if isinstance(exp_value, (int, float)) and isinstance(value, str):
                    if value == "high" and exp_value <= 0.7:
                        return False
                    if value == "low" and exp_value >= 0.3:
                        return False
                elif exp_value != value:
                    return False
            else:
                return False
        return True

    def _evolve_patterns(self) -> List[Pattern]:
        """模式演化：生成变体"""
        evolved = []
        for pattern in list(self.patterns.values()):
            if pattern.status != PatternStatus.VALIDATED:
                continue
            if pattern.success_rate < 0.7:
                # 生成更严格的变体
                self._pattern_counter += 1
                variant = Pattern(
                    pattern_id=f"pat_{self._pattern_counter}",
                    pattern_type=pattern.pattern_type,
                    description=f"Variant of {pattern.pattern_id}: stricter conditions",
                    conditions=dict(pattern.conditions),
                    actions=list(pattern.actions),
                    expected_outcome=pattern.expected_outcome,
                    confidence=pattern.confidence * 0.9,
                    support=0,
                )
                self.patterns[variant.pattern_id] = variant
                pattern.variants.append(variant.pattern_id)
                pattern.status = PatternStatus.EVOLVED
                evolved.append(variant)
        return evolved

    def _generate_prototypes(self) -> List[StrategyPrototype]:
        """从已验证模式生成策略原型"""
        new_prototypes = []
        validated = [p for p in self.patterns.values()
                     if p.status == PatternStatus.VALIDATED]
        # 按预期结果分组
        groups: Dict[str, List[Pattern]] = {}
        for p in validated:
            groups.setdefault(p.expected_outcome, []).append(p)

        for outcome, group in groups.items():
            if len(group) < 2:
                continue
            self._prototype_counter += 1
            proto = StrategyPrototype(
                prototype_id=f"proto_{self._prototype_counter}",
                name=f"Strategy for {outcome}",
                description=f"Auto-generated strategy targeting {outcome} "
                           f"with {len(group)} patterns",
                patterns=[p.pattern_id for p in group],
                priority=float(np.mean([p.confidence for p in group])),
                effectiveness=float(np.mean([p.success_rate for p in group])),
            )
            self.prototypes[proto.prototype_id] = proto
            new_prototypes.append(proto)

        return new_prototypes

    def retrieve_patterns(self, context: Dict[str, Any],
                           action: Optional[str] = None,
                           top_k: int = 5) -> List[Pattern]:
        """根据上下文检索匹配模式"""
        matches = []
        for pattern in self.patterns.values():
            if pattern.status == PatternStatus.DEPRECATED:
                continue
            if action and action not in pattern.actions:
                continue
            if self._conditions_match(pattern.conditions, context):
                matches.append(pattern)

        matches.sort(key=lambda p: -p.confidence)
        return matches[:top_k]

    def _conditions_match(self, conditions: Dict[str, Any],
                           context: Dict[str, Any]) -> bool:
        """条件匹配检查（宽松）"""
        for key, value in conditions.items():
            if key not in context:
                return False
            ctx_value = context[key]
            if isinstance(value, str) and isinstance(ctx_value, (int, float)):
                if value == "high" and ctx_value <= 0.7:
                    return False
                if value == "low" and ctx_value >= 0.3:
                    return False
            elif ctx_value != value:
                return False
        return True

    def record_pattern_application(self, pattern_id: str, success: bool) -> None:
        """记录模式应用结果"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].record_application(success)

    def get_distillation_history(self) -> List[Dict[str, Any]]:
        return list(self._distillation_history)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        type_dist: Dict[str, int] = {}
        status_dist: Dict[str, int] = {}
        for p in self.patterns.values():
            type_dist[p.pattern_type.value] = type_dist.get(p.pattern_type.value, 0) + 1
            status_dist[p.status.value] = status_dist.get(p.status.value, 0) + 1

        return {
            "total_experiences": len(self.raw_experiences),
            "total_patterns": len(self.patterns),
            "total_prototypes": len(self.prototypes),
            "pattern_types": type_dist,
            "pattern_statuses": status_dist,
            "distillation_count": len(self._distillation_history),
            "avg_confidence": (float(np.mean([p.confidence for p in self.patterns.values()]))
                               if self.patterns else 0.0),
        }
