"""
reasoning_strategy_selector.py - 推理策略选择器

构建智能策略选择机制，能够根据问题类型、复杂度及上下文自动选择最优推理策略。

核心功能：
1. 问题特征分析与分类
2. 推理策略评估指标体系
3. 策略选择决策算法
4. 动态调整机制与自学习能力
"""
import time
import uuid
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple

from ..meta_orchestration.data_contract import (
    ReasoningStrategy, ProblemFeature, ProblemDomain, ProblemComplexity,
    ReasoningTrace, ReasoningStep
)
from .chain_of_thought_engine import ChainOfThoughtEngine, ReasoningMode
from .enhanced_symbolic_reasoner import EnhancedSymbolicReasoner
from .logical_deductor import LogicalDeductor
from .neuro_symbolic_reasoner import NeuroSymbolicReasoner


class StrategyPerformanceMetric(Enum):
    """策略性能指标"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"
    EXPLAINABILITY = "explainability"
    COVERAGE = "coverage"


class StrategySelectionMode(Enum):
    """策略选择模式"""
    RULE_BASED = "rule_based"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"


class StrategySelector:
    """推理策略选择器

    根据问题特征自动选择最优推理策略：
    - 问题分析：提取问题领域、复杂度、关键词等特征
    - 策略匹配：基于规则和机器学习模型匹配合适策略
    - 决策执行：调用对应推理引擎执行推理
    - 反馈学习：根据执行结果更新策略选择模型
    """

    def __init__(self, event_bus=None):
        self._event_bus = event_bus
        self._selection_mode = StrategySelectionMode.HYBRID
        self._strategy_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._selection_history = deque(maxlen=100)
        self._strategy_priority: Dict[ReasoningStrategy, int] = {}

        self._chain_of_thought_engine = ChainOfThoughtEngine()
        self._enhanced_symbolic_reasoner = EnhancedSymbolicReasoner()
        self._logical_deductor = LogicalDeductor()
        self._neuro_symbolic_reasoner = NeuroSymbolicReasoner()

        self._init_strategy_priorities()
        self._init_strategy_rules()

    def _init_strategy_priorities(self):
        """初始化策略优先级"""
        self._strategy_priority = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 5,
            ReasoningStrategy.DEDUCTIVE_REASONING: 5,
            ReasoningStrategy.INDUCTIVE_REASONING: 4,
            ReasoningStrategy.ANALOGICAL_REASONING: 4,
            ReasoningStrategy.FORWARD_CHAINING: 3,
            ReasoningStrategy.BACKWARD_CHAINING: 3,
            ReasoningStrategy.NEURAL_SYMBOLIC: 2,
            ReasoningStrategy.DEFAULT: 1,
        }

    def _init_strategy_rules(self):
        """初始化策略选择规则"""
        self._strategy_rules = [
            {
                "name": "mathematics_multistep",
                "domain": ProblemDomain.MATHEMATICS,
                "requires_multistep": True,
                "strategy": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "confidence": 0.95
            },
            {
                "name": "mathematics_any",
                "domain": ProblemDomain.MATHEMATICS,
                "strategy": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "confidence": 0.9
            },
            {
                "name": "logic_deductive",
                "domain": ProblemDomain.LOGIC,
                "strategy": ReasoningStrategy.DEDUCTIVE_REASONING,
                "confidence": 0.9
            },
            {
                "name": "common_sense_logic",
                "domain": ProblemDomain.COMMON_SENSE,
                "requires_symbolic_reasoning": True,
                "strategy": ReasoningStrategy.DEDUCTIVE_REASONING,
                "confidence": 0.85
            },
            {
                "name": "science_inductive",
                "domain": ProblemDomain.SCIENCE,
                "requires_inductive_reasoning": True,
                "strategy": ReasoningStrategy.INDUCTIVE_REASONING,
                "confidence": 0.9
            },
            {
                "name": "science_general",
                "domain": ProblemDomain.SCIENCE,
                "strategy": ReasoningStrategy.INDUCTIVE_REASONING,
                "confidence": 0.85
            },
            {
                "name": "common_sense_analogical",
                "domain": ProblemDomain.COMMON_SENSE,
                "requires_analogical_reasoning": True,
                "strategy": ReasoningStrategy.ANALOGICAL_REASONING,
                "confidence": 0.9
            },
            {
                "name": "requires_analogical",
                "requires_analogical_reasoning": True,
                "strategy": ReasoningStrategy.ANALOGICAL_REASONING,
                "confidence": 0.9
            },
            {
                "name": "requires_inductive",
                "requires_inductive_reasoning": True,
                "strategy": ReasoningStrategy.INDUCTIVE_REASONING,
                "confidence": 0.9
            },
            {
                "name": "multistep_reasoning",
                "requires_multistep": True,
                "strategy": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "confidence": 0.95
            },
            {
                "name": "simple_trivial",
                "complexity": [ProblemComplexity.TRIVIAL],
                "strategy": ReasoningStrategy.FORWARD_CHAINING,
                "confidence": 0.9
            },
            {
                "name": "simple_problems",
                "complexity": [ProblemComplexity.SIMPLE],
                "strategy": ReasoningStrategy.FORWARD_CHAINING,
                "confidence": 0.85
            },
            {
                "name": "requires_symbolic",
                "requires_symbolic_reasoning": True,
                "strategy": ReasoningStrategy.NEURAL_SYMBOLIC,
                "confidence": 0.8
            },
        ]

    def analyze_problem(self, problem_text: str, facts: Optional[Dict[str, Any]] = None) -> ProblemFeature:
        """分析问题，提取特征

        Args:
            problem_text: 问题文本
            facts: 已知事实

        Returns:
            ProblemFeature: 问题特征描述
        """
        feature = ProblemFeature(
            problem_text=problem_text,
            context_length=len(problem_text),
            fact_count=len(facts) if facts else 0
        )

        keywords = self._extract_keywords(problem_text)
        feature.keywords = keywords

        domain = self._classify_domain(problem_text, keywords)
        feature.domain = domain

        complexity = self._estimate_complexity(problem_text, facts)
        feature.complexity = complexity

        feature.requires_multistep = complexity in (ProblemComplexity.COMPLEX, ProblemComplexity.HIGHLY_COMPLEX)
        feature.requires_symbolic_reasoning = any(k in keywords for k in ["if", "then", "because", "therefore", "implies"])
        feature.requires_inductive_reasoning = any(k in keywords for k in ["all", "every", "always", "never", "observed"])
        feature.requires_analogical_reasoning = any(k in keywords for k in ["like", "similar", "analogous", "compare", "as", "just as", "resemble"])

        feature.estimated_difficulty = self._estimate_difficulty(feature)
        feature.novelty_score = self._calculate_novelty(problem_text)

        return feature

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        text_lower = text.lower()

        logic_keywords = ["if", "then", "else", "because", "therefore", "however", "but", "and", "or", "not", "implies", "prove", "show", "demonstrate", "conclude", "infer", "derive"]
        quantifier_keywords = ["all", "every", "some", "none", "always", "never", "sometimes", "observed", "each"]
        comparison_keywords = ["like", "similar", "analogous", "compare", "different", "as", "just as", "resemble"]
        math_keywords = ["calculate", "sum", "product", "add", "subtract", "multiply", "divide", "what is", "how much", "total", "percent"]

        for kw in logic_keywords + quantifier_keywords + comparison_keywords + math_keywords:
            if kw in text_lower:
                keywords.append(kw)

        return keywords

    def _classify_domain(self, text: str, keywords: List[str]) -> ProblemDomain:
        """分类问题领域"""
        text_lower = text.lower()

        math_terms = ["math", "equation", "solve", "calculate", "sum", "product", "factor", "number", "add", "subtract",
                      "multiply", "divide", "total", "percent", "km/h", "distance", "speed", "time"]
        if any(term in text_lower for term in math_terms):
            return ProblemDomain.MATHEMATICS

        logic_terms = ["logic", "prove", "theorem", "argument", "valid", "invalid", "inference",
                       "if", "then", "implies", "all humans", "socrates", "mortal"]
        if any(term in text_lower for term in logic_terms):
            return ProblemDomain.LOGIC

        science_terms = ["physics", "chemistry", "biology", "science", "experiment", "observed", "swan", "white"]
        if any(term in text_lower for term in science_terms):
            return ProblemDomain.SCIENCE

        medical_terms = ["medical", "health", "disease", "treatment"]
        if any(term in text_lower for term in medical_terms):
            return ProblemDomain.MEDICAL

        financial_terms = ["finance", "money", "investment", "stock", "price"]
        if any(term in text_lower for term in financial_terms):
            return ProblemDomain.FINANCIAL

        engineering_terms = ["engineer", "design", "build", "structure"]
        if any(term in text_lower for term in engineering_terms):
            return ProblemDomain.ENGINEERING

        philosophy_terms = ["philosophy", "ethics", "moral", "exist"]
        if any(term in text_lower for term in philosophy_terms):
            return ProblemDomain.PHILOSOPHY

        return ProblemDomain.COMMON_SENSE

    def _estimate_complexity(self, text: str, facts: Optional[Dict[str, Any]]) -> ProblemComplexity:
        """估算问题复杂度"""
        length = len(text)
        fact_count = len(facts) if facts else 0

        if length < 50 and fact_count <= 2:
            return ProblemComplexity.TRIVIAL
        elif length < 100 and fact_count <= 4:
            return ProblemComplexity.SIMPLE
        elif length < 200 and fact_count <= 8:
            return ProblemComplexity.MODERATE
        elif length < 400 and fact_count <= 15:
            return ProblemComplexity.COMPLEX
        else:
            return ProblemComplexity.HIGHLY_COMPLEX

    def _estimate_difficulty(self, feature: ProblemFeature) -> float:
        """估算问题难度"""
        difficulty = 0.5

        complexity_weights = {
            ProblemComplexity.TRIVIAL: 0.1,
            ProblemComplexity.SIMPLE: 0.25,
            ProblemComplexity.MODERATE: 0.5,
            ProblemComplexity.COMPLEX: 0.75,
            ProblemComplexity.HIGHLY_COMPLEX: 0.9
        }
        difficulty += complexity_weights[feature.complexity] * 0.3

        if feature.requires_multistep:
            difficulty += 0.15
        if feature.requires_symbolic_reasoning:
            difficulty += 0.1
        if feature.requires_inductive_reasoning:
            difficulty += 0.1
        if feature.requires_analogical_reasoning:
            difficulty += 0.1

        return min(1.0, difficulty)

    def _calculate_novelty(self, text: str) -> float:
        """计算新颖度"""
        novel_patterns = ["never seen before", "unprecedented", "new type of", "first time", "unknown"]
        return sum(1 for pattern in novel_patterns if pattern in text.lower()) / len(novel_patterns)

    def select_strategy(self, feature: ProblemFeature) -> Tuple[ReasoningStrategy, float]:
        """选择推理策略

        Args:
            feature: 问题特征

        Returns:
            Tuple[ReasoningStrategy, float]: (策略, 置信度)
        """
        if self._selection_mode == StrategySelectionMode.RULE_BASED:
            return self._rule_based_selection(feature)
        elif self._selection_mode == StrategySelectionMode.MODEL_BASED:
            return self._model_based_selection(feature)
        else:
            return self._hybrid_selection(feature)

    def _rule_based_selection(self, feature: ProblemFeature) -> Tuple[ReasoningStrategy, float]:
        """基于规则的策略选择"""
        matched_rules = []

        for rule in self._strategy_rules:
            matches = True
            condition_count = 0

            if "domain" in rule:
                condition_count += 1
                if rule["domain"] != feature.domain:
                    matches = False
            if "complexity" in rule:
                condition_count += 1
                if feature.complexity not in rule["complexity"]:
                    matches = False
            if "requires_symbolic_reasoning" in rule:
                condition_count += 1
                if rule["requires_symbolic_reasoning"] != feature.requires_symbolic_reasoning:
                    matches = False
            if "requires_inductive_reasoning" in rule:
                condition_count += 1
                if rule["requires_inductive_reasoning"] != feature.requires_inductive_reasoning:
                    matches = False
            if "requires_analogical_reasoning" in rule:
                condition_count += 1
                if rule["requires_analogical_reasoning"] != feature.requires_analogical_reasoning:
                    matches = False
            if "requires_multistep" in rule:
                condition_count += 1
                if rule["requires_multistep"] != feature.requires_multistep:
                    matches = False

            if matches:
                matched_rules.append((rule["strategy"], rule["confidence"], rule["name"], condition_count))

        if matched_rules:
            matched_rules.sort(key=lambda x: (x[3], self._strategy_priority[x[0]], -x[1]), reverse=True)
            best_strategy, confidence, _, _ = matched_rules[0]
            return best_strategy, confidence

        return ReasoningStrategy.DEFAULT, 0.5

    def _model_based_selection(self, feature: ProblemFeature) -> Tuple[ReasoningStrategy, float]:
        """基于模型的策略选择

        使用历史性能数据进行策略选择
        """
        strategy_scores = {}

        for strategy in ReasoningStrategy:
            if strategy == ReasoningStrategy.DEFAULT:
                continue

            perf_data = self._strategy_performance.get(strategy.value, {})
            if not perf_data:
                continue

            avg_accuracy = np.mean(perf_data.get("accuracy", [0.5])) if perf_data.get("accuracy") else 0.5
            avg_efficiency = np.mean(perf_data.get("efficiency", [1.0])) if perf_data.get("efficiency") else 1.0
            avg_consistency = np.mean(perf_data.get("consistency", [0.5])) if perf_data.get("consistency") else 0.5

            score = avg_accuracy * 0.4 + (1.0 / avg_efficiency) * 0.3 + avg_consistency * 0.3
            strategy_scores[strategy] = score

        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            confidence = strategy_scores[best_strategy]
            return best_strategy, min(1.0, confidence)

        return ReasoningStrategy.DEFAULT, 0.5

    def _hybrid_selection(self, feature: ProblemFeature) -> Tuple[ReasoningStrategy, float]:
        """混合策略选择"""
        rule_strategy, rule_confidence = self._rule_based_selection(feature)
        model_strategy, model_confidence = self._model_based_selection(feature)

        if rule_confidence >= 0.7:
            return rule_strategy, rule_confidence
        elif model_confidence >= rule_confidence and model_confidence >= 0.6:
            return model_strategy, model_confidence
        else:
            return rule_strategy, rule_confidence

    def execute_reasoning(self, problem_text: str, facts: Optional[Dict[str, Any]] = None,
                          goal: Optional[str] = None) -> ReasoningTrace:
        """执行推理

        根据问题特征选择策略并执行推理

        Args:
            problem_text: 问题文本
            facts: 已知事实
            goal: 推理目标

        Returns:
            ReasoningTrace: 推理轨迹
        """
        start_time = time.time()

        feature = self.analyze_problem(problem_text, facts)
        strategy, strategy_confidence = self.select_strategy(feature)

        trace = ReasoningTrace(
            trace_id=f"strategy_trace_{uuid.uuid4().hex[:12]}",
            problem_text=problem_text,
            problem_feature=feature,
            strategy=strategy,
            strategy_confidence=strategy_confidence
        )

        result = self._execute_strategy(strategy, problem_text, facts, goal)

        trace.steps = result.get("steps", [])
        trace.final_conclusion = result.get("final_conclusion", "")
        trace.final_confidence = result.get("final_confidence", 0.0)
        trace.is_complete = result.get("is_complete", True)
        trace.is_success = result.get("is_success", False)
        trace.total_steps = len(result.get("steps", []))
        trace.execution_time = time.time() - start_time
        trace.explanation = result.get("explanation", "")

        self._record_selection(feature, strategy, strategy_confidence, trace)
        self._update_performance(strategy, trace)

        return trace

    def _execute_strategy(self, strategy: ReasoningStrategy, problem_text: str,
                          facts: Optional[Dict[str, Any]], goal: Optional[str]) -> Dict[str, Any]:
        """执行选定的策略"""
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            trace = self._chain_of_thought_engine.reason(problem_text, facts, goal)
            return {
                "steps": trace.steps,
                "final_conclusion": trace.final_conclusion,
                "final_confidence": trace.final_confidence,
                "is_complete": trace.is_complete,
                "is_success": trace.is_success,
                "explanation": trace.explanation
            }

        elif strategy == ReasoningStrategy.DEDUCTIVE_REASONING:
            trace = self._enhanced_symbolic_reasoner.reason(problem_text, facts, mode="deductive", goal=goal)
            return {
                "steps": trace.steps,
                "final_conclusion": trace.final_confidence >= 0.7 and "演绎推理成功" or "演绎推理不确定",
                "final_confidence": trace.final_confidence,
                "is_complete": trace.is_complete,
                "is_success": trace.is_success,
                "explanation": trace.explanation
            }

        elif strategy == ReasoningStrategy.INDUCTIVE_REASONING:
            observations = facts.get("observations", []) if isinstance(facts, dict) else []
            trace = self._enhanced_symbolic_reasoner.reason(problem_text, {"observations": observations}, mode="inductive")
            return {
                "steps": trace.steps,
                "final_conclusion": trace.final_confidence >= 0.7 and "归纳推理成功" or "归纳推理不确定",
                "final_confidence": trace.final_confidence,
                "is_complete": trace.is_complete,
                "is_success": trace.is_success,
                "explanation": trace.explanation
            }

        elif strategy == ReasoningStrategy.ANALOGICAL_REASONING:
            source = facts.get("source", {}) if isinstance(facts, dict) else {}
            target = facts.get("target", {}) if isinstance(facts, dict) else {}
            trace = self._enhanced_symbolic_reasoner.reason(problem_text, {"source": source, "target": target}, mode="analogical")
            return {
                "steps": trace.steps,
                "final_conclusion": trace.final_confidence >= 0.7 and "类比推理成功" or "类比推理不确定",
                "final_confidence": trace.final_confidence,
                "is_complete": trace.is_complete,
                "is_success": trace.is_success,
                "explanation": trace.explanation
            }

        elif strategy == ReasoningStrategy.FORWARD_CHAINING:
            if facts:
                result = self._logical_deductor.forward_chain_reasoning(facts, goal)
                steps = []
                for i, step_data in enumerate(result.get("steps", []), 1):
                    step = ReasoningStep(
                        step_id=f"fc_step_{i:04d}",
                        step_number=i,
                        type="forward_chain",
                        description=step_data.get("rule_name", "前向推理"),
                        natural_language=f"应用规则 '{step_data.get('rule_name', '未知')}'",
                        premise=str(step_data.get("inputs", {})),
                        result=str(step_data.get("output", {})),
                        confidence=step_data.get("confidence", 0.5)
                    )
                    steps.append(step)

                return {
                    "steps": steps,
                    "final_conclusion": result.get("goal_achieved") and "目标达成" or "推理完成",
                    "final_confidence": result.get("final_confidence", 0.5),
                    "is_complete": True,
                    "is_success": result.get("goal_achieved", False),
                    "explanation": f"前向链推理完成，共{len(steps)}步"
                }
            else:
                return {
                    "steps": [],
                    "final_conclusion": "没有足够事实进行前向链推理",
                    "final_confidence": 0.3,
                    "is_complete": True,
                    "is_success": False,
                    "explanation": "缺少事实数据"
                }

        elif strategy == ReasoningStrategy.NEURAL_SYMBOLIC:
            self._neuro_symbolic_reasoner.symbol_memory.clear()
            if facts:
                for key, value in facts.items():
                    self._neuro_symbolic_reasoner.add_symbol(str(key), "concept")

                query_expr = self._create_query_expression(facts)
                result = self._neuro_symbolic_reasoner.reason(query_expr)

                step = ReasoningStep(
                    step_id="ns_step_0001",
                    step_number=1,
                    type="neural_symbolic",
                    description="神经符号推理",
                    natural_language=result.get("explanation", ""),
                    premise=str(facts),
                    result=f"推理真值: {result.get('truth_value', 0.5):.3f}",
                    confidence=result.get("confidence", 0.5)
                )

                return {
                    "steps": [step],
                    "final_conclusion": f"神经符号推理结果: {result.get('truth_value', 0.5):.3f}",
                    "final_confidence": result.get("confidence", 0.5),
                    "is_complete": True,
                    "is_success": result.get("confidence", 0.5) >= 0.5,
                    "explanation": result.get("explanation", "")
                }
            else:
                return {
                    "steps": [],
                    "final_conclusion": "没有足够数据进行神经符号推理",
                    "final_confidence": 0.3,
                    "is_complete": True,
                    "is_success": False,
                    "explanation": "缺少输入数据"
                }

        else:
            trace = self._chain_of_thought_engine.reason(problem_text, facts, goal)
            return {
                "steps": trace.steps,
                "final_conclusion": trace.final_conclusion,
                "final_confidence": trace.final_confidence,
                "is_complete": trace.is_complete,
                "is_success": trace.is_success,
                "explanation": trace.explanation
            }

    def _create_query_expression(self, facts: Dict[str, Any]):
        """创建查询表达式"""
        from .neuro_symbolic_reasoner import SymbolicExpression, NeuralSymbol

        symbols = []
        for key in facts:
            if key in self._neuro_symbolic_reasoner.symbol_memory:
                symbols.append(self._neuro_symbolic_reasoner.symbol_memory[key])

        if len(symbols) >= 2:
            return SymbolicExpression("AND", symbols[:2])
        elif symbols:
            return SymbolicExpression("AND", symbols)
        else:
            return SymbolicExpression("AND", [])

    def _record_selection(self, feature: ProblemFeature, strategy: ReasoningStrategy,
                          confidence: float, trace: ReasoningTrace) -> None:
        """记录策略选择"""
        self._selection_history.append({
            "timestamp": time.time(),
            "domain": feature.domain.value,
            "complexity": feature.complexity.value,
            "strategy": strategy.value,
            "selection_confidence": confidence,
            "execution_confidence": trace.final_confidence,
            "success": trace.is_success,
            "execution_time": trace.execution_time
        })

    def _update_performance(self, strategy: ReasoningStrategy, trace: ReasoningTrace) -> None:
        """更新策略性能数据"""
        perf = self._strategy_performance[strategy.value]

        perf["accuracy"].append(trace.final_confidence)
        perf["efficiency"].append(trace.execution_time)
        perf["consistency"].append(1.0 if trace.is_success else 0.0)
        perf["explainability"].append(len(trace.steps) > 0)
        perf["coverage"].append(1.0)

        for key in perf:
            if len(perf[key]) > 100:
                perf[key] = perf[key][-100:]

    def get_strategy_performance(self, strategy: Optional[ReasoningStrategy] = None) -> Dict[str, Any]:
        """获取策略性能统计"""
        if strategy:
            perf = self._strategy_performance.get(strategy.value, {})
            return self._compute_performance_summary(perf)

        results = {}
        for strategy_enum in ReasoningStrategy:
            perf = self._strategy_performance.get(strategy_enum.value, {})
            results[strategy_enum.value] = self._compute_performance_summary(perf)

        return results

    def _compute_performance_summary(self, perf: Dict[str, List[float]]) -> Dict[str, float]:
        """计算性能摘要"""
        if not perf or not any(perf.values()):
            return {
                "avg_accuracy": 0.5,
                "avg_efficiency": 1.0,
                "avg_consistency": 0.5,
                "sample_count": 0
            }

        return {
            "avg_accuracy": float(np.mean(perf.get("accuracy", [0.5]))),
            "avg_efficiency": float(np.mean(perf.get("efficiency", [1.0]))),
            "avg_consistency": float(np.mean(perf.get("consistency", [0.5]))),
            "sample_count": sum(len(v) for v in perf.values()) // len(perf)
        }

    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        if not self._selection_history:
            return {
                "total_selections": 0,
                "strategy_distribution": {},
                "avg_selection_confidence": 0.0,
                "avg_execution_confidence": 0.0,
                "success_rate": 0.0
            }

        strategy_dist = defaultdict(int)
        total_selection_conf = 0.0
        total_execution_conf = 0.0
        success_count = 0

        for record in self._selection_history:
            strategy_dist[record["strategy"]] += 1
            total_selection_conf += record["selection_confidence"]
            total_execution_conf += record["execution_confidence"]
            if record["success"]:
                success_count += 1

        return {
            "total_selections": len(self._selection_history),
            "strategy_distribution": dict(strategy_dist),
            "avg_selection_confidence": total_selection_conf / len(self._selection_history),
            "avg_execution_confidence": total_execution_conf / len(self._selection_history),
            "success_rate": success_count / len(self._selection_history)
        }

    def set_selection_mode(self, mode: StrategySelectionMode) -> None:
        """设置选择模式"""
        self._selection_mode = mode

    def update_strategy_priority(self, strategy: ReasoningStrategy, priority: int) -> None:
        """更新策略优先级"""
        self._strategy_priority[strategy] = priority

    def add_strategy_rule(self, name: str, strategy: ReasoningStrategy, confidence: float,
                          **conditions) -> None:
        """添加策略选择规则"""
        rule = {
            "name": name,
            "strategy": strategy,
            "confidence": confidence,
            **conditions
        }
        self._strategy_rules.append(rule)

    def get_recent_traces(self, count: int = 10) -> List[ReasoningTrace]:
        """获取最近的推理轨迹"""
        return self._chain_of_thought_engine.get_recent_traces(count)


import numpy as np