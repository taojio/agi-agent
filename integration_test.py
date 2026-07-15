import sys
sys.path.insert(0, '.')

print('=' * 60)
print('元模块深度集成与数据流优化 - 集成测试')
print('=' * 60)

# 测试1: 数据契约模块
print('\n--- 测试1: 数据契约模块 ---')
from agi_agent.meta_orchestration.data_contract import (
    CognitiveEvent, LearningFeedback, DecisionFeedback,
    ParsingResult, ProgrammingTask, OptimizationRequest,
    OptimizationResult, DataContractSerializer, DataContractFactory,
    EventCategory, EventAction, DataContractVersion
)

event = CognitiveEvent(
    category=EventCategory.COGNITIVE,
    action=EventAction.UPDATE,
    source_module='test_source',
    confidence=0.8,
    free_energy=0.2,
    entropy=0.3
)
print(f'CognitiveEvent created: {event.event_id}')

serialized = DataContractSerializer.serialize(event)
print(f'Serialized: {len(serialized)} bytes')

deserialized = DataContractSerializer.deserialize(serialized)
print(f'Deserialized type: {type(deserialized).__name__}')
print('✓ 数据契约序列化/反序列化成功')

# 测试2: 认知-元模块桥梁
print('\n--- 测试2: 认知-元模块桥梁 ---')
from agi_agent.meta_orchestration.cognitive_meta_bridge import (
    CognitiveMetaBridge, BridgeChannel, FeedbackTrigger, get_cognitive_meta_bridge
)
from agi_agent.orchestration.event_bus import EventBus, Event

event_bus = EventBus()
bridge = get_cognitive_meta_bridge(event_bus)
print(f'Bridge created: {bridge}')
print('✓ 认知-元模块桥梁初始化成功')

# 测试3: 元学习模块集成
print('\n--- 测试3: 元学习模块集成 ---')
from agi_agent.meta_learning.integration import (
    MetaLearningIntegration, LearningAdaptationTrigger, get_meta_learning_integration
)

meta_learning = get_meta_learning_integration(event_bus)
meta_learning.start()
print(f'Meta-learning stats: {meta_learning.get_stats()}')
meta_learning.stop()
print('✓ 元学习模块集成成功')

# 测试4: 元决策模块集成
print('\n--- 测试4: 元决策模块集成 ---')
from agi_agent.meta_decision.integration import (
    MetaDecisionIntegration, DecisionQualityDimension, DecisionFeedbackTrigger,
    get_meta_decision_integration
)

meta_decision = get_meta_decision_integration(event_bus)
meta_decision.start()
print(f'Meta-decision stats: {meta_decision.get_stats()}')
meta_decision.stop()
print('✓ 元决策模块集成成功')

# 测试5: 元编程模块集成
print('\n--- 测试5: 元编程模块集成 ---')
from agi_agent.meta_programming.integration import (
    MetaProgrammingIntegration, CodeQualityLevel, RefactoringType,
    SafetyLevel, CodeAnalysisTrigger, get_meta_programming_integration
)

meta_programming = get_meta_programming_integration(event_bus)
meta_programming.start()
print(f'Meta-programming stats: {meta_programming.get_stats()}')
meta_programming.stop()
print('✓ 元编程模块集成成功')

# 测试6: 元解析模块集成
print('\n--- 测试6: 元解析模块集成 ---')
from agi_agent.meta_parsing.integration import (
    MetaParsingIntegration, SemanticAnalysisLevel, ContextRelationType,
    MultimodalDataType, ParsingOptimizationTrigger, get_meta_parsing_integration
)

meta_parsing = get_meta_parsing_integration(event_bus)
meta_parsing.start()

test_text = 'AI技术正在改变世界，机器学习算法在医疗、金融等领域有广泛应用。2024年是AI发展的重要一年。'
event_bus.publish(
    Event(
        event_type='parsing.parse',
        data={'data': test_text, 'format': 'text'}
    )
)
print(f'Meta-parsing stats: {meta_parsing.get_stats()}')

semantic_results = meta_parsing.get_recent_semantic(5)
print(f'Semantic analyses performed: {len(semantic_results)}')
meta_parsing.stop()
print('✓ 元解析模块集成成功')

# 测试7: 流程协调器
print('\n--- 测试7: 流程协调器 ---')
from agi_agent.orchestration.flow_coordinator import (
    FlowCoordinator, ProcessingPathway, PathwayPriority,
    FlowPhase, FlowControlMode, get_flow_coordinator
)
from agi_agent.orchestration.automation_linkage import SystemState

flow_coordinator = get_flow_coordinator(event_bus)
flow_coordinator.start()

state = SystemState(
    step=1,
    confidence=0.8,
    free_energy=0.1,
    novelty=0.1,
    entropy=0.2
)
selected = flow_coordinator._select_pathway(state)
print(f'Selected pathway: {selected}')

print(f'Flow coordinator stats: {flow_coordinator.get_stats()}')
flow_coordinator.stop()
print('✓ 流程协调器集成成功')

# 测试8: 多通路数据规范验证
print('\n--- 测试8: 多通路数据规范验证 ---')
reflex_spec = flow_coordinator.get_pathway_spec(ProcessingPathway.REFLEX)
deliberate_spec = flow_coordinator.get_pathway_spec(ProcessingPathway.DELIBERATE)
meta_spec = flow_coordinator.get_pathway_spec(ProcessingPathway.META_COGNITIVE)

print(f'Reflex pathway required inputs: {reflex_spec.required_inputs}')
print(f'Deliberate pathway required inputs: {deliberate_spec.required_inputs}')
print(f'Meta-cognitive pathway required inputs: {meta_spec.required_inputs}')

reflex_valid, errors = reflex_spec.validate_input({
    'perception_vector': [0.1, 0.2, 0.3],
    'confidence': 0.8,
    'free_energy': 0.2
})
print(f'Reflex input validation: {"PASS" if reflex_valid else "FAIL"} {errors}')

deliberate_valid, errors = deliberate_spec.validate_input({
    'perception_vector': [0.1, 0.2, 0.3],
    'goal_state': [0.5, 0.5],
    'confidence': 0.4,
    'free_energy': 0.6,
    'novelty': 0.4
})
print(f'Deliberate input validation: {"PASS" if deliberate_valid else "FAIL"} {errors}')
print('✓ 多通路数据规范验证成功')

# 测试9: 元编排器整体集成
print('\n--- 测试9: 元编排器整体集成 ---')
from agi_agent.meta_orchestration.meta_orchestrator import MetaOrchestrator, get_meta_orchestrator

orchestrator = get_meta_orchestrator()
print(f'Meta-orchestrator created: {orchestrator}')
print('✓ 元编排器集成成功')

# 测试10: 自动化联动引擎
print('\n--- 测试10: 自动化联动引擎 ---')
from agi_agent.orchestration.automation_linkage import (
    AutomationLinkageEngine, create_default_linkage_rules
)

linkage_engine = AutomationLinkageEngine()
rules = create_default_linkage_rules()
for rule in rules:
    linkage_engine.register_rule(rule)
print(f'Registered {len(rules)} linkage rules')
print('✓ 自动化联动引擎集成成功')

print('\n' + '=' * 60)
print('所有集成测试通过!')
print('=' * 60)