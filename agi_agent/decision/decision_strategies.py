"""
decision/decision_strategies.py - 智能决策策略

实现 6 种决策策略，支持不同场景的决策需求
"""
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class StrategyType(Enum):
    """决策策略类型"""
    UTILITY_MAXIMIZATION = "utility_maximization"   # 效用最大化
    BAYESIAN = "bayesian"                           # 贝叶斯决策
    MARKOV = "markov"                               # 马尔可夫决策
    MULTI_OBJECTIVE = "multi_objective"             # 多目标优化
    FUZZY = "fuzzy"                                 # 模糊决策
    CASE_BASED = "case_based"                       # 案例推理


@dataclass
class DecisionContext:
    """决策上下文

    包含决策所需的所有信息
    """

    goal: str = ""
    options: List[Any] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0
    time_horizon: int = 1
    risk_tolerance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """策略决策结果"""

    strategy: StrategyType
    selected_option: Optional[Any] = None
    option_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    reasoning: str = ""
    alternatives: List[Tuple[Any, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "selected_option": str(self.selected_option),
            "option_scores": self.option_scores,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [(str(o), s) for o, s in self.alternatives],
            "timestamp": self.timestamp,
        }


class DecisionStrategy(ABC):
    """决策策略基类"""

    strategy_type: StrategyType = StrategyType.UTILITY_MAXIMIZATION

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def decide(self, context: DecisionContext) -> StrategyResult:
        """执行决策"""
        ...

    def get_strategy_type(self) -> StrategyType:
        return self.strategy_type


class UtilityMaximizationStrategy(DecisionStrategy):
    """效用最大化策略

    计算各选项的期望效用，选择效用最大的选项
    utility = reward * success_probability - cost
    """

    strategy_type = StrategyType.UTILITY_MAXIMIZATION

    def __init__(self, risk_aversion: float = 0.5):
        super().__init__()
        self.risk_aversion = risk_aversion

    def decide(self, context: DecisionContext) -> StrategyResult:
        result = StrategyResult(strategy=self.strategy_type)

        if not context.options:
            result.reasoning = "无可用选项"
            return result

        scores = {}
        for option in context.options:
            utility = self._calculate_utility(option, context)
            scores[self._option_key(option)] = utility

        result.option_scores = scores

        best_option = max(context.options,
                          key=lambda o: scores[self._option_key(o)])
        result.selected_option = best_option

        sorted_options = sorted(
            context.options,
            key=lambda o: scores[self._option_key(o)],
            reverse=True
        )
        result.alternatives = [
            (opt, scores[self._option_key(opt)])
            for opt in sorted_options[1:4]
        ]

        best_score = scores[self._option_key(best_option)]
        result.confidence = float(max(0.0, min(1.0, best_score)))

        result.reasoning = (
            f"选择效用最高的选项（效用={best_score:.4f}），"
            f"基于收益×成功率-成本-风险调整"
        )

        return result

    def _calculate_utility(self, option: Any,
                            context: DecisionContext) -> float:
        """计算选项效用"""
        reward = getattr(option, 'estimated_reward', 0.5)
        cost = getattr(option, 'estimated_cost', 0.1)
        success_prob = getattr(option, 'success_probability', 0.5)
        risk_level = getattr(option, 'risk_level', 'low')

        risk_factor = {"low": 0.1, "medium": 0.3, "high": 0.5}.get(risk_level, 0.2)

        expected_reward = reward * success_prob
        risk_adjustment = self.risk_aversion * risk_factor * expected_reward

        return float(expected_reward - cost - risk_adjustment)

    def _option_key(self, option: Any) -> str:
        return getattr(option, 'option_id', None) or str(option)


class BayesianStrategy(DecisionStrategy):
    """贝叶斯决策策略

    结合先验概率与新证据更新信念，基于期望损失决策
    """

    strategy_type = StrategyType.BAYESIAN

    def __init__(self, prior_strength: float = 1.0):
        super().__init__()
        self.prior_strength = prior_strength

    def decide(self, context: DecisionContext) -> StrategyResult:
        result = StrategyResult(strategy=self.strategy_type)

        if not context.options:
            result.reasoning = "无可用选项"
            return result

        scores = {}
        posterior_probs = {}

        for option in context.options:
            prior = self._get_prior(option, context)
            likelihood = self._compute_likelihood(option, context)
            posterior = prior * likelihood

            posterior_probs[self._option_key(option)] = posterior

            expected_value = self._compute_expected_value(option, posterior)
            scores[self._option_key(option)] = expected_value

        total_posterior = sum(posterior_probs.values())
        if total_posterior > 0:
            for k in posterior_probs:
                posterior_probs[k] /= total_posterior

        result.option_scores = scores
        result.metadata["posterior_probabilities"] = posterior_probs

        best_option = max(context.options,
                          key=lambda o: scores[self._option_key(o)])
        result.selected_option = best_option

        best_score = scores[self._option_key(best_option)]
        best_posterior = posterior_probs[self._option_key(best_option)]
        result.confidence = float(best_posterior)

        result.reasoning = (
            f"基于贝叶斯推理选择后验概率最高的选项"
            f"（后验={best_posterior:.4f}，期望值={best_score:.4f}）"
        )

        return result

    def _get_prior(self, option: Any, context: DecisionContext) -> float:
        """获取先验概率"""
        success_prob = getattr(option, 'success_probability', 0.5)
        return float(success_prob)

    def _compute_likelihood(self, option: Any,
                             context: DecisionContext) -> float:
        """计算似然"""
        reward = getattr(option, 'estimated_reward', 0.5)
        cost = getattr(option, 'estimated_cost', 0.1)
        return float(max(0.01, reward - cost))

    def _compute_expected_value(self, option: Any,
                                 posterior: float) -> float:
        """计算期望值"""
        reward = getattr(option, 'estimated_reward', 0.5)
        cost = getattr(option, 'estimated_cost', 0.1)
        return float(posterior * (reward - cost))

    def _option_key(self, option: Any) -> str:
        return getattr(option, 'option_id', None) or str(option)


class MultiObjectiveStrategy(DecisionStrategy):
    """多目标优化策略

    处理多目标冲突，使用加权和法求 Pareto 最优解
    """

    strategy_type = StrategyType.MULTI_OBJECTIVE

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            "reward": 0.4,
            "cost": 0.3,
            "risk": 0.2,
            "success": 0.1,
        }

    def decide(self, context: DecisionContext) -> StrategyResult:
        result = StrategyResult(strategy=self.strategy_type)

        if not context.options:
            result.reasoning = "无可用选项"
            return result

        if context.preferences:
            weights = {**self.weights, **context.preferences}
        else:
            weights = self.weights

        scores = {}
        pareto_optimal = self._find_pareto_optimal(context.options)

        for option in context.options:
            score = self._compute_weighted_score(option, weights)
            scores[self._option_key(option)] = score

        result.option_scores = scores
        result.metadata["pareto_optimal"] = [
            self._option_key(o) for o in pareto_optimal
        ]
        result.metadata["weights"] = weights

        best_option = max(context.options,
                          key=lambda o: scores[self._option_key(o)])
        result.selected_option = best_option

        best_score = scores[self._option_key(best_option)]
        is_pareto = best_option in pareto_optimal
        result.confidence = float(best_score * (1.2 if is_pareto else 1.0))
        result.confidence = min(1.0, result.confidence)

        result.reasoning = (
            f"基于多目标加权求和选择最优解"
            f"（得分={best_score:.4f}，Pareto最优={'是' if is_pareto else '否'}）"
        )

        return result

    def _compute_weighted_score(self, option: Any,
                                 weights: Dict[str, float]) -> float:
        """计算加权得分"""
        reward = getattr(option, 'estimated_reward', 0.5)
        cost = getattr(option, 'estimated_cost', 0.1)
        success_prob = getattr(option, 'success_probability', 0.5)
        risk_level = getattr(option, 'risk_level', 'low')

        risk_score = {"low": 0.1, "medium": 0.3, "high": 0.5}.get(risk_level, 0.2)

        w_reward = weights.get("reward", 0.4)
        w_cost = weights.get("cost", 0.3)
        w_risk = weights.get("risk", 0.2)
        w_success = weights.get("success", 0.1)

        normalized_cost = min(1.0, cost)

        score = (w_reward * reward +
                 w_success * success_prob -
                 w_cost * normalized_cost -
                 w_risk * risk_score)

        return float(max(0.0, min(1.0, score)))

    def _find_pareto_optimal(self, options: List[Any]) -> List[Any]:
        """寻找 Pareto 最优解"""
        pareto = []

        for opt in options:
            dominated = False
            for other in options:
                if other is opt:
                    continue
                if self._dominates(other, opt):
                    dominated = True
                    break

            if not dominated:
                pareto.append(opt)

        return pareto

    def _dominates(self, a: Any, b: Any) -> bool:
        """判断 a 是否支配 b"""
        a_reward = getattr(a, 'estimated_reward', 0.5)
        b_reward = getattr(b, 'estimated_reward', 0.5)
        a_cost = getattr(a, 'estimated_cost', 0.1)
        b_cost = getattr(b, 'estimated_cost', 0.1)
        a_success = getattr(a, 'success_probability', 0.5)
        b_success = getattr(b, 'success_probability', 0.5)

        return (a_reward >= b_reward and
                a_cost <= b_cost and
                a_success >= b_success and
                (a_reward > b_reward or a_cost < b_cost or a_success > b_success))

    def _option_key(self, option: Any) -> str:
        return getattr(option, 'option_id', None) or str(option)


class FuzzyStrategy(DecisionStrategy):
    """模糊决策策略

    处理模糊信息和主观判断，使用模糊集和模糊推理
    """

    strategy_type = StrategyType.FUZZY

    def __init__(self, thresholds: Dict[str, Tuple[float, float]] = None):
        super().__init__()
        self.thresholds = thresholds or {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0),
        }

    def decide(self, context: DecisionContext) -> StrategyResult:
        result = StrategyResult(strategy=self.strategy_type)

        if not context.options:
            result.reasoning = "无可用选项"
            return result

        scores = {}
        fuzzy_evaluations = {}

        for option in context.options:
            fuzzy_eval = self._fuzzy_evaluate(option, context)
            fuzzy_evaluations[self._option_key(option)] = fuzzy_eval
            scores[self._option_key(option)] = fuzzy_eval["defuzzified"]

        result.option_scores = scores
        result.metadata["fuzzy_evaluations"] = fuzzy_evaluations

        best_option = max(context.options,
                          key=lambda o: scores[self._option_key(o)])
        result.selected_option = best_option

        best_score = scores[self._option_key(best_option)]
        result.confidence = float(best_score)

        best_fuzzy = fuzzy_evaluations[self._option_key(best_option)]
        result.reasoning = (
            f"基于模糊推理选择最优选项"
            f"（去模糊化值={best_score:.4f}，"
            f"主要隶属度={best_fuzzy['dominant_set']}）"
        )

        return result

    def _fuzzy_evaluate(self, option: Any,
                         context: DecisionContext) -> Dict[str, Any]:
        """模糊评估"""
        reward = getattr(option, 'estimated_reward', 0.5)
        cost = getattr(option, 'estimated_cost', 0.1)
        success_prob = getattr(option, 'success_probability', 0.5)

        reward_membership = self._compute_membership(reward)
        cost_membership = self._compute_membership(1.0 - min(1.0, cost))
        success_membership = self._compute_membership(success_prob)

        combined = {
            level: (reward_membership[level] * 0.4 +
                    cost_membership[level] * 0.3 +
                    success_membership[level] * 0.3)
            for level in self.thresholds
        }

        dominant_set = max(combined, key=combined.get)
        defuzzified = self._defuzzify(combined)

        return {
            "memberships": combined,
            "dominant_set": dominant_set,
            "defuzzified": defuzzified,
        }

    def _compute_membership(self, value: float) -> Dict[str, float]:
        """计算隶属度"""
        memberships = {}
        for level, (low, high) in self.thresholds.items():
            if level == "low":
                if value <= low:
                    memberships[level] = 1.0
                elif value < high:
                    memberships[level] = (high - value) / (high - low)
                else:
                    memberships[level] = 0.0
            elif level == "high":
                if value >= high:
                    memberships[level] = 1.0
                elif value > low:
                    memberships[level] = (value - low) / (high - low)
                else:
                    memberships[level] = 0.0
            else:
                center = (low + high) / 2
                if value <= low or value >= high:
                    memberships[level] = 0.0
                else:
                    memberships[level] = 1.0 - abs(value - center) / (high - low) * 2
                    memberships[level] = max(0.0, memberships[level])
        return memberships

    def _defuzzify(self, memberships: Dict[str, float]) -> float:
        """去模糊化（加权平均法）"""
        centers = {
            level: (low + high) / 2
            for level, (low, high) in self.thresholds.items()
        }

        total_weight = sum(memberships.values())
        if total_weight < 1e-10:
            return 0.5

        return sum(centers[level] * memberships[level]
                   for level in memberships) / total_weight

    def _option_key(self, option: Any) -> str:
        return getattr(option, 'option_id', None) or str(option)


class CaseBasedStrategy(DecisionStrategy):
    """案例推理策略

    基于历史相似案例进行决策
    """

    strategy_type = StrategyType.CASE_BASED

    def __init__(self, similarity_threshold: float = 0.3):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self._case_base: List[Dict[str, Any]] = []

    def add_case(self, case: Dict[str, Any]) -> None:
        """添加案例到案例库"""
        self._case_base.append(case)

    def add_cases(self, cases: List[Dict[str, Any]]) -> None:
        """批量添加案例"""
        self._case_base.extend(cases)

    def decide(self, context: DecisionContext) -> StrategyResult:
        result = StrategyResult(strategy=self.strategy_type)

        if not context.options:
            result.reasoning = "无可用选项"
            return result

        if not self._case_base:
            result.reasoning = "案例库为空，无法执行案例推理"
            result.confidence = 0.2
            return result

        query_case = self._context_to_case(context)

        similar_cases = self._retrieve_similar(query_case, top_k=5)

        if not similar_cases:
            result.reasoning = "未找到相似案例"
            result.confidence = 0.3
            return result

        scores = {}
        for option in context.options:
            score = self._compute_case_based_score(option, similar_cases)
            scores[self._option_key(option)] = score

        result.option_scores = scores
        result.metadata["similar_cases"] = len(similar_cases)
        result.metadata["best_similarity"] = similar_cases[0][1] if similar_cases else 0

        best_option = max(context.options,
                          key=lambda o: scores[self._option_key(o)])
        result.selected_option = best_option

        best_score = scores[self._option_key(best_option)]
        best_sim = similar_cases[0][1] if similar_cases else 0
        result.confidence = float(best_sim * 0.7 + best_score * 0.3)

        result.reasoning = (
            f"基于 {len(similar_cases)} 个相似案例推理"
            f"（最高相似度={best_sim:.4f}）"
        )

        return result

    def _context_to_case(self, context: DecisionContext) -> Dict[str, Any]:
        """将决策上下文转换为案例表示"""
        return {
            "goal": context.goal,
            "state": context.current_state,
            "constraints": context.constraints,
        }

    def _retrieve_similar(self, query: Dict[str, Any],
                           top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """检索相似案例"""
        scored = []
        for case in self._case_base:
            similarity = self._compute_similarity(query, case)
            if similarity >= self.similarity_threshold:
                scored.append((case, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _compute_similarity(self, a: Dict[str, Any],
                             b: Dict[str, Any]) -> float:
        """计算案例相似度"""
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        common_keys = a_keys & b_keys

        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            a_val = a[key]
            b_val = b[key]

            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                diff = abs(a_val - b_val)
                sim = max(0.0, 1.0 - diff)
            elif isinstance(a_val, str) and isinstance(b_val, str):
                sim = 1.0 if a_val == b_val else 0.5
            elif isinstance(a_val, dict) and isinstance(b_val, dict):
                sim = self._compute_similarity(a_val, b_val)
            else:
                sim = 0.5

            similarities.append(sim)

        return sum(similarities) / len(similarities)

    def _compute_case_based_score(self, option: Any,
                                    similar_cases: List[Tuple[Dict, float]]) -> float:
        """基于相似案例计算选项得分"""
        option_key = self._option_key(option)
        total_score = 0.0
        total_weight = 0.0

        for case, similarity in similar_cases:
            case_outcomes = case.get("outcomes", {})
            if option_key in case_outcomes:
                outcome_score = case_outcomes[option_key]
            else:
                success_prob = getattr(option, 'success_probability', 0.5)
                outcome_score = success_prob

            total_score += outcome_score * similarity
            total_weight += similarity

        if total_weight < 1e-10:
            return 0.5

        return float(total_score / total_weight)

    def _option_key(self, option: Any) -> str:
        return getattr(option, 'option_id', None) or str(option)


class StrategyRegistry:
    """策略注册中心"""

    def __init__(self):
        self._strategies: Dict[StrategyType, DecisionStrategy] = {}

    def register(self, strategy: DecisionStrategy) -> None:
        """注册策略"""
        self._strategies[strategy.get_strategy_type()] = strategy

    def get(self, strategy_type: StrategyType) -> Optional[DecisionStrategy]:
        """获取策略"""
        return self._strategies.get(strategy_type)

    def list_strategies(self) -> List[StrategyType]:
        """列出所有已注册策略"""
        return list(self._strategies.keys())

    def create_default_registry() -> "StrategyRegistry":
        """创建默认策略注册中心"""
        registry = StrategyRegistry()
        registry.register(UtilityMaximizationStrategy())
        registry.register(BayesianStrategy())
        registry.register(MultiObjectiveStrategy())
        registry.register(FuzzyStrategy())
        registry.register(CaseBasedStrategy())
        return registry
