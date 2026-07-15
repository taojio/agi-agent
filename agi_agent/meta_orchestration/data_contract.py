"""
data_contract.py - 模块间统一数据格式规范

定义模块间数据传递的统一规范，包括：
- 标准数据结构定义
- 字段命名标准
- 序列化/反序列化协议
- 版本控制机制
- 数据验证规则
"""
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from ..data_standards.models import BaseDataModel, IdentifiableMixin, TimestampedMixin


class DataContractVersion(Enum):
    """数据契约版本"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class EventCategory(Enum):
    """事件分类"""
    COGNITIVE = "cognitive"
    LEARNING = "learning"
    DECISION = "decision"
    PARSING = "parsing"
    PROGRAMMING = "programming"
    SYSTEM = "system"
    MONITORING = "monitoring"


class EventAction(Enum):
    """事件动作"""
    UPDATE = "update"
    COMPLETE = "complete"
    ERROR = "error"
    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    OPTIMIZE = "optimize"
    ADAPT = "adapt"


@dataclass
class CognitiveEvent(BaseDataModel):
    """认知事件数据契约

    用于认知流程与元模块之间的实时数据传递
    """

    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    category: EventCategory = EventCategory.COGNITIVE
    action: EventAction = EventAction.UPDATE
    timestamp: float = field(default_factory=time.time)
    source_module: str = ""
    target_module: str = ""
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")

    confidence: float = 0.5
    free_energy: float = 0.0
    entropy: float = 0.0
    causal_effect: float = 0.0
    system_used: str = "system2"
    is_impasse: bool = False

    feature_dim: int = 0
    feature_vector: List[float] = field(default_factory=list)
    prediction: Any = None
    action: Any = None

    needs_status: Dict[str, Dict[str, float]] = field(default_factory=dict)
    internal_state: Dict[str, float] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result['category'] = self.category.value
        result['action'] = self.action.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveEvent":
        data = data.copy()
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = EventCategory(data['category'])
        if 'action' in data and isinstance(data['action'], str):
            data['action'] = EventAction(data['action'])
        return super().from_dict(data)


@dataclass
class LearningFeedback(BaseDataModel):
    """学习反馈数据契约

    用于元学习模块接收认知流程反馈，实现实时参数优化
    """

    feedback_id: str = field(default_factory=lambda: f"fb_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    task_id: str = ""
    task_type: str = ""

    performance_metrics: Dict[str, float] = field(default_factory=dict)
    learning_outcome: str = ""
    improvement: float = 0.0
    confidence_change: float = 0.0

    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)

    cognitive_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionFeedback(BaseDataModel):
    """决策反馈数据契约

    用于元决策模块收集决策效果反馈，实现决策策略迭代
    """

    feedback_id: str = field(default_factory=lambda: f"df_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    decision_id: str = ""
    goal: str = ""

    outcome: str = ""
    outcome_score: float = 0.0
    confidence: float = 0.5

    factors_considered: List[str] = field(default_factory=list)
    options_considered: List[str] = field(default_factory=list)
    chosen_option: str = ""

    quality_metrics: Dict[str, float] = field(default_factory=dict)
    detected_biases: List[Dict[str, Any]] = field(default_factory=list)

    suggested_improvements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingResult(BaseDataModel):
    """解析结果数据契约

    用于元解析模块传递语义理解结果
    """

    result_id: str = field(default_factory=lambda: f"pr_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    source_data: str = ""
    data_type: str = ""

    parsed_data: Dict[str, Any] = field(default_factory=dict)
    semantic_understanding: Dict[str, Any] = field(default_factory=dict)
    context_relations: List[Dict[str, Any]] = field(default_factory=list)

    data_quality: float = 0.0
    understanding_level: str = "basic"
    complexity_level: str = "simple"

    confidence: float = 0.0
    reliability: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgrammingTask(BaseDataModel):
    """编程任务数据契约

    用于元编程模块定义代码改进任务
    """

    task_id: str = field(default_factory=lambda: f"pt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    task_type: str = "analyze"

    target_code: str = ""
    target_file: str = ""
    language: str = "python"

    quality_metrics: Dict[str, float] = field(default_factory=dict)
    detected_issues: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    refactored_code: str = ""
    test_results: Dict[str, Any] = field(default_factory=dict)

    safety_verified: bool = False
    verification_report: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleStatusUpdate(BaseDataModel):
    """模块状态更新数据契约

    用于监控系统与各模块之间的状态同步
    """

    update_id: str = field(default_factory=lambda: f"su_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    module_id: str = ""
    module_name: str = ""

    status: str = "active"
    health_score: float = 1.0
    load: float = 0.0

    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRequest(BaseDataModel):
    """优化请求数据契约

    用于请求元模块执行优化操作
    """

    request_id: str = field(default_factory=lambda: f"opt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    target_module: str = ""
    optimization_type: str = ""

    current_params: Dict[str, Any] = field(default_factory=dict)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    priority: int = 1
    deadline: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult(BaseDataModel):
    """优化结果数据契约

    用于返回优化操作的结果
    """

    result_id: str = field(default_factory=lambda: f"ores_{uuid.uuid4().hex[:12]}")
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    success: bool = False

    optimized_params: Dict[str, Any] = field(default_factory=dict)
    performance_improvement: float = 0.0
    expected_effect: float = 0.0

    validation_results: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(Enum):
    """推理策略类型"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    NEURAL_SYMBOLIC = "neural_symbolic"
    INDUCTIVE_REASONING = "inductive_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    DEDUCTIVE_REASONING = "deductive_reasoning"
    DEFAULT = "default"


class ProblemDomain(Enum):
    """问题领域"""
    MATHEMATICS = "mathematics"
    LOGIC = "logic"
    COMMON_SENSE = "common_sense"
    SCIENCE = "science"
    PHILOSOPHY = "philosophy"
    ENGINEERING = "engineering"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    OTHER = "other"


class ProblemComplexity(Enum):
    """问题复杂度"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class ProblemFeature(BaseDataModel):
    """问题特征数据契约

    用于描述推理问题的特征，供策略选择器使用
    """

    feature_id: str = field(default_factory=lambda: f"pf_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    problem_text: str = ""
    problem_type: str = ""
    domain: ProblemDomain = ProblemDomain.OTHER
    complexity: ProblemComplexity = ProblemComplexity.MODERATE

    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)

    requires_multistep: bool = False
    requires_symbolic_reasoning: bool = False
    requires_inductive_reasoning: bool = False
    requires_analogical_reasoning: bool = False

    estimated_difficulty: float = 0.5
    novelty_score: float = 0.0
    ambiguity_score: float = 0.0

    context_length: int = 0
    fact_count: int = 0
    target_variables: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result['domain'] = self.domain.value
        result['complexity'] = self.complexity.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemFeature":
        data = data.copy()
        if 'domain' in data and isinstance(data['domain'], str):
            data['domain'] = ProblemDomain(data['domain'])
        if 'complexity' in data and isinstance(data['complexity'], str):
            data['complexity'] = ProblemComplexity(data['complexity'])
        return super().from_dict(data)


@dataclass
class ReasoningStep(BaseDataModel):
    """推理步骤数据契约

    描述单个推理步骤的详细信息
    """

    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    step_number: int = 0

    type: str = "inference"
    description: str = ""
    natural_language: str = ""

    premise: str = ""
    operation: str = ""
    result: str = ""

    confidence: float = 0.5
    truth_value: float = 0.5
    evidence_support: float = 0.0

    used_rules: List[str] = field(default_factory=list)
    used_symbols: List[str] = field(default_factory=list)
    derived_facts: Dict[str, Any] = field(default_factory=dict)

    is_intermediate: bool = True
    is_conclusion: bool = False
    is_verified: bool = False

    parent_steps: List[str] = field(default_factory=list)
    child_steps: List[str] = field(default_factory=list)

    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace(BaseDataModel):
    """推理轨迹数据契约

    标准化推理步骤输出，支持可解释性分析
    """

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")
    timestamp: float = field(default_factory=time.time)

    problem_id: str = ""
    problem_text: str = ""
    problem_feature: Optional[ProblemFeature] = None

    strategy: ReasoningStrategy = ReasoningStrategy.DEFAULT
    strategy_confidence: float = 0.5

    steps: List[ReasoningStep] = field(default_factory=list)
    final_conclusion: str = ""
    final_confidence: float = 0.0

    is_complete: bool = False
    is_success: bool = False

    total_steps: int = 0
    execution_time: float = 0.0

    explanation: str = ""
    supporting_evidence: List[str] = field(default_factory=list)

    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> None:
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result['strategy'] = self.strategy.value
        result['steps'] = [step.to_dict() for step in self.steps]
        if self.problem_feature:
            result['problem_feature'] = self.problem_feature.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTrace":
        data = data.copy()
        if 'strategy' in data and isinstance(data['strategy'], str):
            data['strategy'] = ReasoningStrategy(data['strategy'])
        if 'steps' in data:
            data['steps'] = [ReasoningStep.from_dict(s) for s in data['steps']]
        if 'problem_feature' in data and isinstance(data['problem_feature'], dict):
            data['problem_feature'] = ProblemFeature.from_dict(data['problem_feature'])
        return super().from_dict(data)


class DataContractSerializer:
    """数据契约序列化器

    提供统一的序列化/反序列化能力，确保模块间数据传递的兼容性
    """

    _contract_types = {
        'CognitiveEvent': CognitiveEvent,
        'LearningFeedback': LearningFeedback,
        'DecisionFeedback': DecisionFeedback,
        'ParsingResult': ParsingResult,
        'ProgrammingTask': ProgrammingTask,
        'ModuleStatusUpdate': ModuleStatusUpdate,
        'OptimizationRequest': OptimizationRequest,
        'OptimizationResult': OptimizationResult,
        'ProblemFeature': ProblemFeature,
        'ReasoningStep': ReasoningStep,
        'ReasoningTrace': ReasoningTrace,
    }

    @classmethod
    def serialize(cls, data: BaseDataModel) -> Dict[str, Any]:
        """序列化数据契约对象"""
        result = data.to_dict()
        result['_contract_type'] = type(data).__name__
        result['_contract_version'] = DataContractVersion.V1_0.value
        return result

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Optional[BaseDataModel]:
        """反序列化数据契约对象"""
        contract_type = data.get('_contract_type')
        if not contract_type:
            return None

        contract_cls = cls._contract_types.get(contract_type)
        if not contract_cls:
            return None

        return contract_cls.from_dict(data)

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证数据契约格式"""
        errors = []

        required_fields = ['_contract_type']
        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        contract_type = data.get('_contract_type')
        if contract_type and contract_type not in cls._contract_types:
            errors.append(f"Unknown contract type: {contract_type}")

        version = data.get('_contract_version', '1.0')
        try:
            DataContractVersion(version)
        except ValueError:
            errors.append(f"Invalid contract version: {version}")

        if contract_type in cls._contract_types:
            try:
                obj = cls.deserialize(data)
                if obj:
                    validation_errors = obj.validate()
                    errors.extend(validation_errors)
            except Exception as e:
                errors.append(f"Deserialization error: {e}")

        return (len(errors) == 0, errors)

    @classmethod
    def get_contract_type(cls, name: str) -> Optional[type]:
        """获取数据契约类型"""
        return cls._contract_types.get(name)

    @classmethod
    def list_contract_types(cls) -> List[str]:
        """列出所有数据契约类型"""
        return list(cls._contract_types.keys())


class DataContractFactory:
    """数据契约工厂

    提供便捷的契约对象创建方法
    """

    @staticmethod
    def create_cognitive_event(
        source_module: str,
        target_module: str,
        **kwargs
    ) -> CognitiveEvent:
        """创建认知事件"""
        return CognitiveEvent(
            source_module=source_module,
            target_module=target_module,
            **kwargs
        )

    @staticmethod
    def create_learning_feedback(
        task_id: str,
        performance_metrics: Dict[str, float],
        **kwargs
    ) -> LearningFeedback:
        """创建学习反馈"""
        return LearningFeedback(
            task_id=task_id,
            performance_metrics=performance_metrics,
            **kwargs
        )

    @staticmethod
    def create_decision_feedback(
        decision_id: str,
        outcome: str,
        **kwargs
    ) -> DecisionFeedback:
        """创建决策反馈"""
        return DecisionFeedback(
            decision_id=decision_id,
            outcome=outcome,
            **kwargs
        )

    @staticmethod
    def create_parsing_result(
        source_data: str,
        parsed_data: Dict[str, Any],
        **kwargs
    ) -> ParsingResult:
        """创建解析结果"""
        return ParsingResult(
            source_data=source_data,
            parsed_data=parsed_data,
            **kwargs
        )

    @staticmethod
    def create_programming_task(
        task_type: str,
        **kwargs
    ) -> ProgrammingTask:
        """创建编程任务"""
        return ProgrammingTask(
            task_type=task_type,
            **kwargs
        )

    @staticmethod
    def create_optimization_request(
        target_module: str,
        optimization_type: str,
        **kwargs
    ) -> OptimizationRequest:
        """创建优化请求"""
        return OptimizationRequest(
            target_module=target_module,
            optimization_type=optimization_type,
            **kwargs
        )

    @staticmethod
    def create_optimization_result(
        request_id: str,
        success: bool,
        **kwargs
    ) -> OptimizationResult:
        """创建优化结果"""
        return OptimizationResult(
            request_id=request_id,
            success=success,
            **kwargs
        )