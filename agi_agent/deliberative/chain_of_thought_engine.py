"""
chain_of_thought_engine.py - 思维链推理引擎

实现基于自然语言的多步骤逻辑推理能力，能够将复杂问题分解为有序的推理步骤
并生成可解释的中间过程。

设计思路：
1. 包装现有LogicalDeductor和NeuroSymbolicReasoner作为推理核心
2. 添加步骤分解机制，将复杂问题分解为可管理的子问题
3. 实现自然语言解释生成，使推理过程可解释
4. 支持推理轨迹的完整记录与回溯
"""
import time
import uuid
from enum import Enum
from collections import deque
from typing import Dict, List, Optional, Any

from ..meta_orchestration.data_contract import (
    ReasoningTrace, ReasoningStep, ProblemFeature,
    ReasoningStrategy, ProblemDomain, ProblemComplexity
)
from .logical_deductor import LogicalDeductor, DeductionStep
from .neuro_symbolic_reasoner import NeuroSymbolicReasoner, SymbolicExpression, NeuralSymbol, SymbolType
from ..knowledge_rulebase.rule_registry import DisciplinaryRuleRegistry as RuleRegistry


class ReasoningMode(Enum):
    """推理模式"""
    STRICT = "strict"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class ChainOfThoughtEngine:
    """思维链推理引擎

    核心能力：
    - 问题分解：将复杂问题分解为有序的推理步骤
    - 多步骤推理：支持链式推理，每一步的输出作为下一步的输入
    - 可解释性输出：生成自然语言描述的推理过程
    - 推理轨迹管理：完整记录推理过程，支持回溯和分析
    """

    def __init__(self, rule_registry: Optional[RuleRegistry] = None, reasoning_mode: ReasoningMode = ReasoningMode.HYBRID):
        self._rule_registry = rule_registry
        self._reasoning_mode = reasoning_mode
        self._logical_deductor = LogicalDeductor(rule_registry=rule_registry)
        self._neuro_symbolic_reasoner = NeuroSymbolicReasoner()
        self._reasoning_history = deque(maxlen=100)
        self._step_id_counter = 0

    def reason(self, problem_text: str, facts: Optional[Dict[str, Any]] = None,
               goal: Optional[str] = None, max_steps: int = 20) -> ReasoningTrace:
        """执行思维链推理

        Args:
            problem_text: 问题描述文本
            facts: 初始事实集合
            goal: 推理目标
            max_steps: 最大推理步骤数

        Returns:
            ReasoningTrace: 完整的推理轨迹
        """
        start_time = time.time()
        trace = ReasoningTrace(
            trace_id=f"cot_trace_{uuid.uuid4().hex[:12]}",
            problem_text=problem_text,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            strategy_confidence=0.8
        )

        facts = facts or {}
        current_facts = facts.copy()

        problem_feature = self._extract_problem_features(problem_text, facts)
        trace.problem_feature = problem_feature

        steps = self._decompose_problem(problem_text, current_facts, goal, max_steps)

        for step_data in steps:
            reasoning_step = self._execute_reasoning_step(step_data, current_facts)
            trace.add_step(reasoning_step)

            if reasoning_step.derived_facts:
                current_facts.update(reasoning_step.derived_facts)

            if reasoning_step.is_conclusion:
                break

        trace.final_conclusion = self._generate_final_conclusion(trace, current_facts, goal)
        trace.final_confidence = self._calculate_final_confidence(trace)
        trace.is_complete = True
        trace.is_success = trace.final_confidence >= 0.5
        trace.execution_time = time.time() - start_time
        trace.explanation = self._generate_explanation(trace)

        self._reasoning_history.append(trace)

        return trace

    def _extract_problem_features(self, problem_text: str, facts: Dict[str, Any]) -> ProblemFeature:
        """提取问题特征

        Args:
            problem_text: 问题文本
            facts: 事实集合

        Returns:
            ProblemFeature: 问题特征描述
        """
        feature = ProblemFeature(
            problem_text=problem_text,
            context_length=len(problem_text),
            fact_count=len(facts)
        )

        keywords = self._extract_keywords(problem_text)
        feature.keywords = keywords

        domain = self._classify_domain(problem_text, keywords)
        feature.domain = domain

        complexity = self._estimate_complexity(problem_text, facts)
        feature.complexity = complexity

        feature.requires_multistep = complexity in (ProblemComplexity.COMPLEX, ProblemComplexity.HIGHLY_COMPLEX)
        feature.requires_symbolic_reasoning = any(k in keywords for k in ["if", "then", "because", "therefore", "implies"])
        feature.requires_inductive_reasoning = any(k in keywords for k in ["all", "every", "always", "never"])
        feature.requires_analogical_reasoning = any(k in keywords for k in ["like", "similar", "analogous", "compare"])

        feature.target_variables = self._extract_target_variables(problem_text)

        return feature

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        text_lower = text.lower()

        logic_keywords = ["if", "then", "else", "because", "therefore", "however", "but", "and", "or", "not", "implies"]
        quantifier_keywords = ["all", "every", "some", "none", "always", "never", "sometimes"]
        reasoning_keywords = ["prove", "show", "demonstrate", "conclude", "infer", "derive"]
        comparison_keywords = ["like", "similar", "analogous", "compare", "different"]

        for kw in logic_keywords + quantifier_keywords + reasoning_keywords + comparison_keywords:
            if kw in text_lower:
                keywords.append(kw)

        return keywords

    def _classify_domain(self, text: str, keywords: List[str]) -> ProblemDomain:
        """分类问题领域"""
        text_lower = text.lower()

        if any(term in text_lower for term in ["math", "equation", "solve", "calculate", "sum", "product", "factor"]):
            return ProblemDomain.MATHEMATICS
        if any(term in text_lower for term in ["logic", "prove", "theorem", "argument", "valid", "invalid"]):
            return ProblemDomain.LOGIC
        if any(term in text_lower for term in ["physics", "chemistry", "biology", "science", "experiment"]):
            return ProblemDomain.SCIENCE
        if any(term in text_lower for term in ["medical", "health", "disease", "treatment"]):
            return ProblemDomain.MEDICAL
        if any(term in text_lower for term in ["finance", "money", "investment", "stock", "price"]):
            return ProblemDomain.FINANCIAL
        if any(term in text_lower for term in ["engineer", "design", "build", "structure"]):
            return ProblemDomain.ENGINEERING
        if any(term in text_lower for term in ["philosophy", "ethics", "moral", "exist"]):
            return ProblemDomain.PHILOSOPHY

        return ProblemDomain.COMMON_SENSE

    def _estimate_complexity(self, text: str, facts: Dict[str, Any]) -> ProblemComplexity:
        """估算问题复杂度"""
        length = len(text)
        fact_count = len(facts)

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

    def _extract_target_variables(self, text: str) -> List[str]:
        """提取目标变量"""
        targets = []
        question_words = ["what", "who", "when", "where", "why", "how", "which"]

        for word in question_words:
            idx = text.lower().find(word)
            if idx != -1:
                remaining = text[idx + len(word):].strip()
                if remaining.startswith("is") or remaining.startswith("are") or remaining.startswith("was"):
                    targets.append(remaining.split()[0] if remaining.split() else "")

        return [t for t in targets if t]

    def _decompose_problem(self, problem_text: str, facts: Dict[str, Any],
                           goal: Optional[str], max_steps: int) -> List[Dict[str, Any]]:
        """分解问题为推理步骤

        Args:
            problem_text: 问题文本
            facts: 事实集合
            goal: 推理目标
            max_steps: 最大步骤数

        Returns:
            List[Dict]: 步骤描述列表
        """
        steps = []

        steps.append({
            "type": "analysis",
            "description": "分析问题结构",
            "premise": f"问题：{problem_text}",
            "operation": "问题分析"
        })

        if facts:
            steps.append({
                "type": "fact_extraction",
                "description": "提取已知事实",
                "premise": f"已知事实：{list(facts.keys())}",
                "operation": "事实提取"
            })

        steps.append({
            "type": "inference",
            "description": "执行推理",
            "premise": "应用推理规则",
            "operation": "链式推理"
        })

        if goal:
            steps.append({
                "type": "goal_check",
                "description": "验证目标达成",
                "premise": f"目标：{goal}",
                "operation": "目标验证"
            })

        steps.append({
            "type": "conclusion",
            "description": "生成结论",
            "premise": "综合推理结果",
            "operation": "结论生成",
            "is_conclusion": True
        })

        return steps[:max_steps]

    def _execute_reasoning_step(self, step_data: Dict[str, Any], facts: Dict[str, Any]) -> ReasoningStep:
        """执行单个推理步骤

        Args:
            step_data: 步骤描述
            facts: 当前事实集合

        Returns:
            ReasoningStep: 推理步骤结果
        """
        step = ReasoningStep(
            step_id=f"cot_step_{self._step_id_counter:04d}",
            type=step_data.get("type", "inference"),
            description=step_data.get("description", ""),
            premise=step_data.get("premise", ""),
            operation=step_data.get("operation", "")
        )

        self._step_id_counter += 1

        if step.type == "analysis":
            step.natural_language = f"我需要分析这个问题：{step.premise}"
            step.confidence = 0.9
            step.truth_value = 1.0

        elif step.type == "fact_extraction":
            step.natural_language = f"已知事实包括：{', '.join(facts.keys())}" if facts else "没有明确的已知事实"
            step.confidence = 1.0
            step.truth_value = 1.0
            step.derived_facts = dict(facts)

        elif step.type == "inference":
            inference_result = self._perform_inference(facts)
            step.natural_language = inference_result.get("explanation", "")
            step.result = inference_result.get("result", "")
            step.confidence = inference_result.get("confidence", 0.5)
            step.truth_value = inference_result.get("truth_value", 0.5)
            step.derived_facts = inference_result.get("derived_facts", {})
            step.used_rules = inference_result.get("used_rules", [])
            step.evidence_support = inference_result.get("evidence_support", 0.0)

        elif step.type == "goal_check":
            goal = step.premise.replace("目标：", "") if "目标：" in step.premise else step.premise
            achieved = self._check_goal_achieved(goal, facts)
            step.natural_language = f"目标 '{goal}' 是否已达成：{'是' if achieved else '否'}"
            step.result = str(achieved)
            step.confidence = 0.9 if achieved else 0.3
            step.truth_value = 1.0 if achieved else 0.0

        elif step.type == "conclusion":
            step.is_conclusion = True
            step.is_intermediate = False
            step.natural_language = "综合所有推理步骤，得出最终结论"
            step.confidence = 0.8
            step.truth_value = 0.8

        return step

    def _perform_inference(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理

        根据推理模式选择合适的推理方式
        """
        if self._reasoning_mode == ReasoningMode.STRICT:
            return self._strict_inference(facts)
        elif self._reasoning_mode == ReasoningMode.HEURISTIC:
            return self._heuristic_inference(facts)
        else:
            return self._hybrid_inference(facts)

    def _strict_inference(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """严格推理模式

        使用逻辑演绎器进行形式化推理
        """
        if self._rule_registry:
            result = self._logical_deductor.forward_chain_reasoning(facts)

            explanation_parts = []
            for i, step in enumerate(result.get("steps", []), 1):
                rule_name = step.get("rule_name", "未知规则")
                inputs = step.get("inputs", {})
                output = step.get("output", {})
                explanation_parts.append(f"步骤{i}：应用{rule_name}规则，输入{inputs}，得出{output}")

            return {
                "result": str(result.get("derived_facts", {})),
                "explanation": "\n".join(explanation_parts) if explanation_parts else "应用逻辑规则进行推理",
                "confidence": result.get("final_confidence", 0.5),
                "truth_value": result.get("final_confidence", 0.5),
                "derived_facts": result.get("derived_facts", {}),
                "used_rules": [s.get("rule_name") for s in result.get("steps", [])],
                "evidence_support": result.get("final_confidence", 0.5)
            }

        return {
            "result": "",
            "explanation": "无法进行严格推理：未配置规则注册表",
            "confidence": 0.3,
            "truth_value": 0.3,
            "derived_facts": {},
            "used_rules": [],
            "evidence_support": 0.0
        }

    def _heuristic_inference(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """启发式推理模式

        使用神经符号推理器进行灵活推理
        """
        self._neuro_symbolic_reasoner.symbol_memory.clear()

        for key, value in facts.items():
            symbol_type = SymbolType.CONCEPT if isinstance(value, (int, float)) else SymbolType.PREDICATE
            self._neuro_symbolic_reasoner.add_symbol(str(key), symbol_type)

        if facts:
            keys = list(facts.keys())
            for i in range(len(keys) - 1):
                self._neuro_symbolic_reasoner.add_relation(keys[i], keys[i + 1], "related")

            query_expr = SymbolicExpression("AND", [
                self._neuro_symbolic_reasoner.symbol_memory.get(k) for k in keys
                if k in self._neuro_symbolic_reasoner.symbol_memory
            ])

            result = self._neuro_symbolic_reasoner.reason(query_expr)

            steps = []
            for step_data in result.get("steps", []):
                desc = step_data.get("description", "")
                value = step_data.get("value", "")
                if desc:
                    steps.append(f"- {desc}" + (f" (值: {value:.3f})" if value else ""))

            return {
                "result": f"推理真值: {result.get('truth_value', 0.5):.3f}",
                "explanation": "\n".join(steps) if steps else "使用神经符号推理进行分析",
                "confidence": result.get("confidence", 0.5),
                "truth_value": result.get("truth_value", 0.5),
                "derived_facts": {},
                "used_rules": result.get("deduced_symbols", []),
                "evidence_support": result.get("confidence", 0.5)
            }

        return {
            "result": "",
            "explanation": "无法进行启发式推理：没有足够的事实",
            "confidence": 0.3,
            "truth_value": 0.3,
            "derived_facts": {},
            "used_rules": [],
            "evidence_support": 0.0
        }

    def _hybrid_inference(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """混合推理模式

        结合严格推理和启发式推理的优点
        """
        strict_result = self._strict_inference(facts)
        heuristic_result = self._heuristic_inference(facts)

        combined_confidence = (strict_result["confidence"] + heuristic_result["confidence"]) / 2

        explanation_parts = [
            f"逻辑推理结果：{strict_result['explanation']}",
            f"神经符号推理结果：{heuristic_result['explanation']}",
            f"综合置信度：{combined_confidence:.3f}"
        ]

        derived_facts = {**strict_result["derived_facts"]}

        return {
            "result": f"综合推理完成，置信度{combined_confidence:.3f}",
            "explanation": "\n".join(explanation_parts),
            "confidence": combined_confidence,
            "truth_value": combined_confidence,
            "derived_facts": derived_facts,
            "used_rules": list(set(strict_result["used_rules"] + heuristic_result["used_rules"])),
            "evidence_support": combined_confidence
        }

    def _check_goal_achieved(self, goal: str, facts: Dict[str, Any]) -> bool:
        """检查目标是否达成"""
        goal_lower = goal.lower().strip()

        if not goal_lower:
            return True

        for key, value in facts.items():
            key_lower = str(key).lower()
            value_lower = str(value).lower()

            if key_lower in goal_lower or value_lower in goal_lower:
                return True

            if "=" in goal_lower and "=" not in key_lower:
                parts = goal_lower.split("=")
                if len(parts) == 2:
                    target_var = parts[0].strip()
                    if target_var in key_lower:
                        return True

        return False

    def _generate_final_conclusion(self, trace: ReasoningTrace, facts: Dict[str, Any], goal: Optional[str]) -> str:
        """生成最终结论"""
        if trace.final_confidence >= 0.7:
            if facts:
                return f"根据推理，{', '.join([f'{k} = {v}' for k, v in facts.items()])}"
            return "推理完成，得出合理结论"
        elif trace.final_confidence >= 0.5:
            return f"推理结果不确定，置信度{trace.final_confidence:.2f}。建议进一步收集信息。"
        else:
            return f"无法得出有效结论，置信度{trace.final_confidence:.2f}。"

    def _calculate_final_confidence(self, trace: ReasoningTrace) -> float:
        """计算最终置信度"""
        if not trace.steps:
            return 0.0

        step_confidences = [step.confidence for step in trace.steps]
        avg_confidence = sum(step_confidences) / len(step_confidences)

        verified_steps = sum(1 for step in trace.steps if step.is_verified)
        verification_bonus = verified_steps / len(trace.steps) * 0.2

        conclusion_bonus = 0.1 if any(step.is_conclusion for step in trace.steps) else 0.0

        return min(1.0, avg_confidence + verification_bonus + conclusion_bonus)

    def _generate_explanation(self, trace: ReasoningTrace) -> str:
        """生成自然语言解释"""
        lines = []
        lines.append(f"问题：{trace.problem_text}")
        lines.append(f"推理策略：{trace.strategy.value}")
        lines.append(f"策略置信度：{trace.strategy_confidence:.2f}")
        lines.append("")
        lines.append("推理步骤：")

        for step in trace.steps:
            prefix = f"[{step.step_number}]"
            step_type = f"({step.type})"
            description = step.description
            natural_lang = step.natural_language

            line = f"{prefix} {step_type} {description}"
            if natural_lang:
                line += f"\n        {natural_lang}"
            if step.result:
                line += f"\n        结果：{step.result}"
            if step.confidence < 0.7:
                line += f" (置信度: {step.confidence:.2f})"

            lines.append(line)

        lines.append("")
        lines.append(f"最终结论：{trace.final_conclusion}")
        lines.append(f"最终置信度：{trace.final_confidence:.2f}")
        lines.append(f"推理耗时：{trace.execution_time:.2f}秒")

        return "\n".join(lines)

    def get_recent_traces(self, count: int = 10) -> List[ReasoningTrace]:
        """获取最近的推理轨迹"""
        return list(self._reasoning_history)[-count:]

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        recent = list(self._reasoning_history)
        if recent:
            avg_confidence = sum(t.final_confidence for t in recent) / len(recent)
            avg_steps = sum(t.total_steps for t in recent) / len(recent)
            avg_time = sum(t.execution_time for t in recent) / len(recent)
            success_rate = sum(1 for t in recent if t.is_success) / len(recent)
        else:
            avg_confidence = 0.0
            avg_steps = 0.0
            avg_time = 0.0
            success_rate = 0.0

        return {
            "total_reasonings": len(self._reasoning_history),
            "avg_confidence": avg_confidence,
            "avg_steps": avg_steps,
            "avg_execution_time": avg_time,
            "success_rate": success_rate,
            "reasoning_mode": self._reasoning_mode.value
        }

    def set_reasoning_mode(self, mode: ReasoningMode) -> None:
        """设置推理模式"""
        self._reasoning_mode = mode

    def set_rule_registry(self, rule_registry: RuleRegistry) -> None:
        """设置规则注册表"""
        self._rule_registry = rule_registry
        self._logical_deductor.set_rule_registry(rule_registry)