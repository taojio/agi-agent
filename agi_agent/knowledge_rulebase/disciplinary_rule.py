"""
跨学科知识强化规则体系 - 规则数据模型

定义结构化的学科规则数据模型，支持：
- 多学科覆盖（物理、数学、化学、生物等）
- 规则间关联分析（派生关系、依赖关系）
- 知识推理扩展（前向链式推理）
- 复杂问题解决（规则组合应用）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable


class Discipline(Enum):
    """学科领域枚举"""
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    CHINESE = "chinese"
    COMPUTER_SCIENCE = "computer_science"
    ENGINEERING = "engineering"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    ECONOMICS = "economics"
    MEDICINE = "medicine"


class RuleDifficulty(Enum):
    """规则难度等级"""
    INTRODUCTORY = 1    # 入门级
    INTERMEDIATE = 2    # 中级
    ADVANCED = 3        # 高级
    SPECIALIZED = 4     # 专业级
    RESEARCH = 5        # 研究级


class RuleType(Enum):
    """规则类型"""
    LAW = "law"                 # 定律
    THEOREM = "theorem"         # 定理
    FORMULA = "formula"         # 公式
    PRINCIPLE = "principle"     # 原理
    DEFINITION = "definition"   # 定义
    COROLLARY = "corollary"     # 推论
    POSTULATE = "postulate"     # 假设


@dataclass
class RuleVariable:
    """规则变量定义"""
    name: str
    symbol: str
    unit: str = ""
    description: str = ""
    constraints: str = ""


@dataclass
class DerivationStep:
    """规则推导步骤"""
    step_number: int
    description: str
    prerequisite_rules: List[str] = field(default_factory=list)
    formula_transformation: str = ""


@dataclass
class DisciplinaryRule:
    """学科规则数据模型"""

    rule_id: str
    discipline: Discipline
    rule_type: RuleType
    name: str
    formula: str
    description: str
    variables: List[RuleVariable] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)
    derivation_chain: List[DerivationStep] = field(default_factory=list)
    prerequisite_rules: List[str] = field(default_factory=list)
    dependent_rules: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    difficulty: RuleDifficulty = RuleDifficulty.INTERMEDIATE
    confidence: float = 0.9
    evidence_count: int = 0
    real_world_examples: List[str] = field(default_factory=list)
    problem_solving_patterns: List[str] = field(default_factory=list)
    application_scenarios: List[str] = field(default_factory=list)
    validation_method: Optional[str] = None
    exceptions: List[str] = field(default_factory=list)
    historical_context: str = ""
    alternative_forms: List[str] = field(default_factory=list)

    def normalize_formula(self) -> str:
        """标准化公式（用于语义搜索）"""
        from agi_agent.memory.semantic_search import TextNormalizer
        normalizer = TextNormalizer()
        return normalizer.normalize(self.formula)

    def get_variable_map(self) -> Dict[str, RuleVariable]:
        """获取变量符号到变量对象的映射"""
        return {v.symbol: v for v in self.variables}

    def get_prerequisite_count(self) -> int:
        """获取前置规则数量"""
        return len(self.prerequisite_rules)

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """检查规则是否适用于当前上下文"""
        for condition in self.conditions:
            if not self._check_condition(condition, context):
                return False
        return True

    def _check_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """检查单个条件"""
        if "temperature" in condition.lower():
            temp = context.get("temperature", 0)
            if "room" in condition.lower() and not (15 <= temp <= 35):
                return False
            if "high" in condition.lower() and temp < 100:
                return False
            if "low" in condition.lower() and temp > 0:
                return False
        if "velocity" in condition.lower():
            v = context.get("velocity", 0)
            if "low" in condition.lower() and v > 100:
                return False
            if "relativistic" in condition.lower() and v < 1e8:
                return False
        return True

    def apply(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """应用规则进行计算"""
        result = {"rule_id": self.rule_id, "rule_name": self.name, "inputs": inputs}
        try:
            result["result"] = self._evaluate_formula(inputs)
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        return result

    def _evaluate_formula(self, inputs: Dict[str, float]) -> float:
        """评估公式"""
        import math
        
        computation_expr = self._get_computation_expression(inputs)
        if not computation_expr:
            raise ValueError(f"无法解析公式: {self.formula}")
        
        local_vars = {**inputs}
        local_vars.update(math.__dict__)
        for var in self.variables:
            if var.symbol in inputs:
                local_vars[var.name] = inputs[var.symbol]
        
        return eval(computation_expr, {"__builtins__": {}}, local_vars)

    def _get_computation_expression(self, inputs: Dict[str, float]) -> str:
        """根据输入变量获取可计算的表达式"""
        formula = self.formula
        
        if formula.startswith('F=ma'):
            return 'm * a'
        elif formula.startswith('F=G'):
            return '6.674e-11 * m1 * m2 / (r ** 2)'
        elif formula.startswith('W=F'):
            return 'F * d * cos(theta)' if 'theta' in inputs else 'F * d'
        elif formula.startswith('Ek='):
            return '0.5 * m * v ** 2'
        elif formula.startswith('Ep=m'):
            g = inputs.get('g', 9.8)
            return f'{g} * m * h'
        elif formula.startswith('E_total'):
            return 'Ek + Ep'
        elif formula.startswith('Q ='):
            return 'W + delta_U'
        elif formula.startswith('PV='):
            return 'n * 8.314 * T'
        elif formula.startswith('F=k'):
            return '8.988e9 * q1 * q2 / (r ** 2)'
        elif formula.startswith('V=I'):
            return 'I * R'
        elif formula.startswith('v=f'):
            return 'f * lambda'
        elif formula.startswith('p=m'):
            return 'm * v'
        elif '→' in formula:
            return 'v'
        
        if formula.startswith('x = (-b'):
            return '(-b + (b**2 - 4*a*c)**0.5) / (2*a)'
        elif formula.startswith('an = a1 +'):
            return 'a1 + (n - 1) * d'
        elif formula.startswith('an = a1 *'):
            return 'a1 * (r ** (n - 1))'
        elif formula.startswith('log_b'):
            return 'log(a) / log(b)'
        elif formula.startswith('a² + b²'):
            return '(a**2 + b**2)**0.5'
        elif formula.startswith('S = πr'):
            return 'pi * r ** 2'
        elif formula.startswith('C = 2π'):
            return '2 * pi * r'
        elif formula.startswith('S = 0.5 * base'):
            return '0.5 * base * height'
        elif formula.startswith('c² = a²'):
            return '(a**2 + b**2 - 2*a*b*cos(C))**0.5'
        elif formula.startswith('P(n,k)'):
            return 'factorial(n) / factorial(n - k)'
        elif formula.startswith('C(n,k)'):
            return 'factorial(n) / (factorial(k) * factorial(n - k))'
        elif formula.startswith('σ = sqrt'):
            return 'sqrt(sum((x - mu)**2) / len(x))'
        elif formula.startswith('M = m'):
            return 'm / n'
        elif formula.startswith('c = n'):
            return 'n / V'
        elif formula.startswith('pH = -log'):
            return '-log(H)'
        
        if formula.startswith('T=2π*sqrt(m/k)'):
            return '2 * pi * sqrt(m / k)'
        elif formula.startswith('T=2π*sqrt(L'):
            return '2 * pi * sqrt(L / g)'
        elif formula.startswith('E=mc²'):
            return 'm * (3e8) ** 2'
        elif formula.startswith('P=V*I'):
            return 'V * I'
        elif formula.startswith('C=Q/V'):
            return 'Q / V'
        elif formula.startswith('d = sqrt'):
            return 'sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)'
        
        normalized = self.normalize_formula()
        expr = normalized.replace('^', '**').replace('sqrt', 'math.sqrt')
        expr = expr.replace('→', '-&gt;').replace('|', ' or ').replace('&amp;', ' and ')
        
        if '=' in expr:
            parts = expr.split('=')
            if len(parts) >= 2:
                expr = parts[-1].strip()
        
        return expr

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "discipline": self.discipline.value,
            "rule_type": self.rule_type.value,
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "variables": [{"name": v.name, "symbol": v.symbol, "unit": v.unit, "description": v.description}
                         for v in self.variables],
            "conditions": self.conditions,
            "units": self.units,
            "prerequisite_rules": self.prerequisite_rules,
            "dependent_rules": self.dependent_rules,
            "related_concepts": self.related_concepts,
            "difficulty": self.difficulty.value,
            "confidence": self.confidence,
            "real_world_examples": self.real_world_examples,
            "application_scenarios": self.application_scenarios,
        }
