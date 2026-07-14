"""
ai_algorithms/pattern_recognition.py - 模式识别组件

支持：
- 频繁项集挖掘（Apriori 算法）
- 序列模式挖掘
- 关联规则提取
"""
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


@dataclass
class FrequentItemset:
    """频繁项集"""
    items: FrozenSet[str]
    support: float          # 支持度
    count: int              # 出现次数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": list(self.items),
            "support": float(self.support),
            "count": self.count,
        }


@dataclass
class AssociationRule:
    """关联规则"""
    antecedent: FrozenSet[str]    # 前件
    consequent: FrozenSet[str]    # 后件
    support: float                # 支持度
    confidence: float             # 置信度
    lift: float                   # 提升度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antecedent": list(self.antecedent),
            "consequent": list(self.consequent),
            "support": float(self.support),
            "confidence": float(self.confidence),
            "lift": float(self.lift),
        }


@dataclass
class SequencePattern:
    """序列模式"""
    sequence: Tuple[str, ...]
    support: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "support": float(self.support),
            "count": self.count,
        }


@dataclass
class PatternResult:
    """模式识别结果"""
    frequent_itemsets: List[Dict[str, Any]] = field(default_factory=list)
    association_rules: List[Dict[str, Any]] = field(default_factory=list)
    sequence_patterns: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequent_itemsets": self.frequent_itemsets,
            "association_rules": self.association_rules,
            "sequence_patterns": self.sequence_patterns,
            "method": self.method,
        }


class AprioriMiner(AIAlgorithmComponent):
    """Apriori 频繁项集挖掘器"""

    def __init__(self, name: str = "apriori",
                 min_support: float = 0.1, min_confidence: float = 0.5):
        super().__init__(name, min_support=min_support,
                         min_confidence=min_confidence)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._transactions: List[Set[str]] = []
        self._frequent_itemsets: Dict[FrozenSet[str], int] = {}

    @property
    def component_type(self) -> str:
        return "pattern_recognition"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AprioriMiner":
        start = self._start_training()
        try:
            # X 可以是交易列表或 numpy 数组
            if isinstance(X, list):
                self._transactions = [set(t) for t in X]
            else:
                data = np.asarray(X)
                if data.ndim == 2:
                    # 假设每行是一个交易，列为 one-hot
                    self._transactions = [set(np.where(row)[0].astype(str))
                                          for row in data]
                else:
                    raise ValueError("X should be list of transactions or 2D array")

            self._frequent_itemsets = self._find_frequent_itemsets()
            self.metrics.sample_count = len(self._transactions)
            self.metrics.custom["n_frequent_itemsets"] = len(self._frequent_itemsets)
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _find_frequent_itemsets(self) -> Dict[FrozenSet[str], int]:
        """Apriori 算法"""
        n_transactions = len(self._transactions)
        min_count = int(np.ceil(self.min_support * n_transactions))

        # 1-项集
        item_counts: Counter = Counter()
        for t in self._transactions:
            for item in t:
                item_counts[frozenset([item])] += 1

        frequent = {k: v for k, v in item_counts.items() if v >= min_count}
        all_frequent = dict(frequent)

        # k-项集
        k = 2
        while frequent:
            # 生成候选
            candidates = self._generate_candidates(frequent, k)
            if not candidates:
                break
            # 计数
            candidate_counts = {c: 0 for c in candidates}
            for t in self._transactions:
                for c in candidates:
                    if c.issubset(t):
                        candidate_counts[c] += 1
            # 过滤
            frequent = {k: v for k, v in candidate_counts.items() if v >= min_count}
            all_frequent.update(frequent)
            k += 1

        return all_frequent

    @staticmethod
    def _generate_candidates(prev_frequent: Dict[FrozenSet[str], int],
                              k: int) -> Set[FrozenSet[str]]:
        """生成 k 项候选集"""
        candidates = set()
        prev_items = list(prev_frequent.keys())
        for i in range(len(prev_items)):
            for j in range(i + 1, len(prev_items)):
                union = prev_items[i] | prev_items[j]
                if len(union) == k:
                    candidates.add(union)
        return candidates

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回频繁项集数量"""
        return np.array([len(self._frequent_itemsets)])

    def get_rules(self) -> List[AssociationRule]:
        """生成关联规则"""
        rules = []
        n_transactions = len(self._transactions)
        for itemset, count in self._frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            support = count / n_transactions
            # 生成所有可能的前件→后件
            items = list(itemset)
            for i in range(1, len(items)):
                for antecedent in self._combinations(items, i):
                    antecedent_set = frozenset(antecedent)
                    consequent_set = itemset - antecedent_set
                    if not consequent_set:
                        continue
                    antecedent_count = self._frequent_itemsets.get(antecedent_set, 0)
                    if antecedent_count == 0:
                        continue
                    confidence = count / antecedent_count
                    if confidence < self.min_confidence:
                        continue
                    consequent_count = self._frequent_itemsets.get(consequent_set, 0)
                    if consequent_count == 0:
                        continue
                    consequent_support = consequent_count / n_transactions
                    lift = confidence / consequent_support if consequent_support > 0 else 0
                    rules.append(AssociationRule(
                        antecedent=antecedent_set,
                        consequent=consequent_set,
                        support=support,
                        confidence=confidence,
                        lift=lift,
                    ))
        return rules

    @staticmethod
    def _combinations(items: List[str], k: int) -> List[List[str]]:
        """生成组合"""
        from itertools import combinations
        return [list(c) for c in combinations(items, k)]

    def get_result(self) -> PatternResult:
        n_transactions = len(self._transactions)
        itemsets = [
            FrequentItemset(items=k, support=v / n_transactions, count=v).to_dict()
            for k, v in self._frequent_itemsets.items()
        ]
        rules = [r.to_dict() for r in self.get_rules()]
        return PatternResult(
            frequent_itemsets=itemsets,
            association_rules=rules,
            method="apriori",
        )


class SequencePatternMiner(AIAlgorithmComponent):
    """序列模式挖掘器"""

    def __init__(self, name: str = "sequence_miner",
                 min_support: float = 0.2, max_length: int = 5):
        super().__init__(name, min_support=min_support, max_length=max_length)
        self.min_support = min_support
        self.max_length = max_length
        self._sequences: List[List[str]] = []
        self._patterns: Dict[Tuple[str, ...], int] = {}

    @property
    def component_type(self) -> str:
        return "pattern_recognition"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SequencePatternMiner":
        start = self._start_training()
        try:
            if isinstance(X, list):
                self._sequences = [list(s) for s in X]
            else:
                data = np.asarray(X)
                if data.ndim == 2:
                    self._sequences = [list(row.astype(str)) for row in data]
                else:
                    self._sequences = [[str(x)] for x in data]

            self._patterns = self._mine_patterns()
            self.metrics.sample_count = len(self._sequences)
            self.metrics.custom["n_patterns"] = len(self._patterns)
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _mine_patterns(self) -> Dict[Tuple[str, ...], int]:
        """挖掘序列模式"""
        n_sequences = len(self._sequences)
        min_count = int(np.ceil(self.min_support * n_sequences))

        # 1-gram
        pattern_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for seq in self._sequences:
            for item in set(seq):
                pattern_counts[(item,)] += 1

        frequent = {k: v for k, v in pattern_counts.items() if v >= min_count}
        all_patterns = dict(frequent)

        # k-gram
        for k in range(2, self.max_length + 1):
            new_patterns: Dict[Tuple[str, ...], int] = defaultdict(int)
            for seq in self._sequences:
                if len(seq) < k:
                    continue
                # 提取所有 k-gram（允许间隔）
                for combo in self._subsequences(seq, k):
                    new_patterns[combo] += 1
            frequent = {k: v for k, v in new_patterns.items() if v >= min_count}
            if not frequent:
                break
            all_patterns.update(frequent)

        return all_patterns

    @staticmethod
    def _subsequences(seq: List[str], k: int) -> Set[Tuple[str, ...]]:
        """生成所有 k 长度子序列（保持顺序）"""
        from itertools import combinations
        result = set()
        for indices in combinations(range(len(seq)), k):
            result.add(tuple(seq[i] for i in indices))
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([len(self._patterns)])

    def get_result(self) -> PatternResult:
        n_sequences = len(self._sequences)
        patterns = [
            SequencePattern(
                sequence=k, support=v / n_sequences, count=v
            ).to_dict()
            for k, v in sorted(self._patterns.items(), key=lambda x: -x[1])[:20]
        ]
        return PatternResult(
            sequence_patterns=patterns,
            method="sequence_mining",
        )


class PatternRecognizer:
    """模式识别统一接口"""

    def __init__(self):
        self._components: Dict[str, AIAlgorithmComponent] = {}

    def mine_frequent_itemsets(self, transactions: List[Set[str]],
                                min_support: float = 0.1,
                                min_confidence: float = 0.5) -> PatternResult:
        """频繁项集挖掘"""
        miner = AprioriMiner(min_support=min_support, min_confidence=min_confidence)
        miner.fit(transactions)
        self._components["apriori"] = miner
        return miner.get_result()

    def mine_sequence_patterns(self, sequences: List[List[str]],
                                min_support: float = 0.2) -> PatternResult:
        """序列模式挖掘"""
        miner = SequencePatternMiner(min_support=min_support)
        miner.fit(sequences)
        self._components["sequence"] = miner
        return miner.get_result()

    def find_rules(self, transactions: List[Set[str]],
                    min_support: float = 0.1,
                    min_confidence: float = 0.5) -> List[AssociationRule]:
        """提取关联规则"""
        miner = AprioriMiner(min_support=min_support, min_confidence=min_confidence)
        miner.fit(transactions)
        return miner.get_rules()
