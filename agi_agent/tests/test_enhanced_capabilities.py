import sys
import os
import numpy as np
import torch
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.deliberative.advanced_reasoner import AdvancedReasoner, LogicalStatement, Predicate
from agi_agent.deliberative.abstract_thinking import AbstractionEngine, AbstractionLevel, ConceptNode
from agi_agent.cognitive.context_awareness import ContextAwarenessEngine, ContextFrame, SceneType, ContextType
from agi_agent.learning.enhanced_knowledge_graph import EnhancedKnowledgeGraph, RelationType, EntityType
from agi_agent.learning.learning_planner import LearningPlanner, LearningGoalType, LearningPriority
from agi_agent.learning.knowledge_integrator import KnowledgeIntegrator, IntegrationStrategy, DomainType
from agi_agent.perception.multimodal_fusion import MultimodalFusion


class TestAdvancedReasoner(unittest.TestCase):
    def setUp(self):
        self.reasoner = AdvancedReasoner(feature_dim=16)

    def test_reasoning_basic(self):
        predicate = Predicate("test_predicate", ["arg1"], truth_value=0.7)
        query = LogicalStatement([predicate], confidence=0.8)
        
        self.reasoner.add_predicate("test_predicate", ["arg1"], truth_value=0.7)
        
        result = self.reasoner.reason(query)
        
        self.assertIn('confidence', result)
        self.assertIn('steps', result)
        self.assertIn('heuristics_applied', result)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_add_predicate(self):
        self.reasoner.add_predicate("is_human", ["John"], truth_value=1.0)
        self.reasoner.add_predicate("is_mortal", ["John"], truth_value=1.0)
        
        pred1 = Predicate("is_human", ["John"])
        query = LogicalStatement([pred1])
        
        result = self.reasoner.reason(query)
        self.assertGreater(result['confidence'], 0.5)

    def test_add_inference_rule(self):
        self.reasoner.add_predicate("is_human", ["Socrates"], truth_value=1.0)
        
        self.reasoner.add_inference_rule(
            condition_predicates=[{"name": "is_human", "arguments": ["x"], "truth_value": 1.0}],
            condition_operator="AND",
            conclusion_predicates=[{"name": "is_mortal", "arguments": ["x"]}],
            conclusion_operator="AND",
            confidence=0.95
        )
        
        pred = Predicate("is_mortal", ["Socrates"])
        query = LogicalStatement([pred], confidence=0.5)
        
        result = self.reasoner.reason(query)
        self.assertGreaterEqual(result['confidence'], 0.5)

    def test_add_relation(self):
        self.reasoner.add_relation("likes", "Alice", "Bob", strength=0.8)
        relations = self.reasoner.knowledge_base.query_relation("likes", source="Alice")
        
        self.assertIsInstance(relations, list)
        self.assertEqual(len(relations), 1)

    def test_reasoning_with_rules(self):
        self.reasoner.add_predicate("has_wings", ["bird1"], truth_value=1.0)
        
        self.reasoner.add_inference_rule(
            condition_predicates=[{"name": "has_wings", "arguments": ["x"], "truth_value": 1.0}],
            condition_operator="AND",
            conclusion_predicates=[{"name": "can_fly", "arguments": ["x"]}],
            confidence=0.9
        )
        
        pred = Predicate("can_fly", ["bird1"])
        query = LogicalStatement([pred])
        
        result = self.reasoner.reason(query)
        self.assertIn('deduced_facts', result)


class TestAbstractionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AbstractionEngine(feature_dim=16)

    def test_add_concept(self):
        features = np.random.rand(16)
        concept_id = self.engine.add_concept("test_concept", features)
        
        self.assertIsNotNone(concept_id)
        self.assertIn(concept_id, self.engine.concepts)

    def test_abstract_from_instances(self):
        instance_ids = []
        for i in range(3):
            features = np.random.rand(16) + i * 0.1
            inst_id = self.engine.add_concept(f"instance_{i}", features, AbstractionLevel.INSTANCE)
            instance_ids.append(inst_id)
        
        result = self.engine.abstract_from_instances(instance_ids, "abstract_concept", "Test abstraction")
        
        self.assertIsNotNone(result)
        self.assertIn(result, self.engine.concepts)

    def test_generalize(self):
        parent_id = self.engine.add_concept("parent", np.random.rand(16), AbstractionLevel.CONCEPT)
        child_ids = []
        for i in range(3):
            features = np.random.rand(16) * 0.5 + self.engine.concepts[parent_id].features * 0.5
            child_id = self.engine.add_concept(f"child_{i}", features, AbstractionLevel.INSTANCE)
            child_ids.append(child_id)
            self.engine.concepts[parent_id].add_child(child_id)
            self.engine.concepts[child_id].add_parent(parent_id)
        
        result = self.engine.generalize(parent_id, "generalized_concept")
        
        self.assertIsNotNone(result)

    def test_find_analogies(self):
        for i in range(5):
            features = np.array([0.5 + i * 0.1] * 16)
            self.engine.add_concept(f"concept_{i}", features)
        
        concept_ids = list(self.engine.concepts.keys())
        if concept_ids:
            result = self.engine.find_analogies(concept_ids[0], top_k=2)
            
            self.assertIsInstance(result, list)

    def test_concept_hierarchy(self):
        root_id = self.engine.add_concept("root", np.random.rand(16), AbstractionLevel.CATEGORY)
        child_id = self.engine.add_concept("child", np.random.rand(16), AbstractionLevel.CONCEPT)
        
        self.engine.concepts[root_id].add_child(child_id)
        self.engine.concepts[child_id].add_parent(root_id)
        
        self.assertIn(child_id, self.engine.concepts[root_id].child_concepts)
        self.assertIn(root_id, self.engine.concepts[child_id].parent_concepts)


class TestContextAwarenessEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ContextAwarenessEngine()

    def test_add_context_frame(self):
        context_id = self.engine.add_context_frame(
            context_type=ContextType.PHYSICAL,
            features=np.random.rand(16)
        )
        
        self.assertIsNotNone(context_id)
        self.assertEqual(len(self.engine.context_history), 1)

    def test_detect_scene(self):
        for i in range(10):
            self.engine.add_context_frame(
                context_type=ContextType.TASK,
                features=np.random.rand(16) * 0.1
            )
        
        scene_type = self.engine.detect_scene()
        
        self.assertIsInstance(scene_type, SceneType)

    def test_predict_next_context(self):
        for i in range(10):
            self.engine.add_context_frame(
                context_type=ContextType.COGNITIVE,
                features=np.array([i * 0.1] * 16)
            )
        
        prediction = self.engine.predict_next_context()
        
        self.assertIsNotNone(prediction)


class TestEnhancedKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.kg = EnhancedKnowledgeGraph()

    def test_add_node(self):
        features = torch.randn(16)
        node_id = self.kg.add_node("test_node", EntityType.CONCEPT, features, "test_label")
        
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, self.kg.nodes)

    def test_add_edge(self):
        node1_id = self.kg.add_node("node1", EntityType.CONCEPT, torch.tensor([0.5] * 16))
        node2_id = self.kg.add_node("node2", EntityType.CONCEPT, torch.tensor([0.6] * 16))
        
        success = self.kg.add_edge(node1_id, node2_id, RelationType.RELATED_TO, weight=0.8)
        
        self.assertTrue(success)

    def test_find_path(self):
        node_a = self.kg.add_node("a", EntityType.CONCEPT, torch.tensor([0.1] * 16))
        node_b = self.kg.add_node("b", EntityType.CONCEPT, torch.tensor([0.2] * 16))
        node_c = self.kg.add_node("c", EntityType.CONCEPT, torch.tensor([0.3] * 16))
        
        self.kg.add_edge(node_a, node_b, RelationType.RELATED_TO)
        self.kg.add_edge(node_b, node_c, RelationType.RELATED_TO)
        
        path = self.kg.find_path(node_a, node_c, max_depth=3)
        
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)

    def test_get_summary(self):
        for i in range(5):
            features = torch.randn(16) * 0.5 + i * 0.5
            self.kg.add_node(f"node_{i}", EntityType.CONCEPT, features)
        
        node_ids = list(self.kg.nodes.keys())
        if len(node_ids) >= 2:
            self.kg.add_edge(node_ids[0], node_ids[1], RelationType.RELATED_TO)
        
        summary = self.kg.get_summary()
        
        self.assertIn('nodes', summary)
        self.assertIn('edges', summary)
        self.assertGreaterEqual(summary['nodes'], 1)


class TestLearningPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = LearningPlanner()

    def test_create_learning_goal(self):
        goal_id = self.planner.create_learning_goal(
            goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
            description="Test learning goal",
            priority=LearningPriority.HIGH,
            target_confidence=0.9
        )
        
        self.assertIsNotNone(goal_id)
        self.assertIn(goal_id, self.planner.goals)

    def test_create_learning_plan(self):
        goal_id1 = self.planner.create_learning_goal(
            goal_type=LearningGoalType.SKILL_DEVELOPMENT,
            description="Goal 1",
            priority=LearningPriority.HIGH
        )
        goal_id2 = self.planner.create_learning_goal(
            goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
            description="Goal 2",
            priority=LearningPriority.MEDIUM
        )
        
        plan_id = self.planner.create_learning_plan([goal_id1, goal_id2])
        
        self.assertIsNotNone(plan_id)
        self.assertIn(plan_id, self.planner.plans)

    def test_plan_status(self):
        goal_id = self.planner.create_learning_goal(
            goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
            description="Test plan",
            priority=LearningPriority.HIGH
        )
        
        plan_id = self.planner.create_learning_plan([goal_id])
        plan = self.planner.plans[plan_id]
        
        self.assertEqual(plan.status, "active")

    def test_goal_decomposition(self):
        goal_id = self.planner.create_learning_goal(
            goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
            description="Complex goal with decomposition",
            priority=LearningPriority.HIGH,
            target_confidence=0.95
        )
        
        goal = self.planner.goals[goal_id]
        self.assertIsInstance(goal.sub_goals, list)


class TestKnowledgeIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = KnowledgeIntegrator()

    def test_add_fragment(self):
        fragment_id = self.integrator.add_fragment(
            content="Test knowledge",
            domain=DomainType.GENERAL,
            features=np.array([0.5] * 16),
            confidence=0.8,
            source="test"
        )
        
        self.assertIsNotNone(fragment_id)
        self.assertIn(fragment_id, self.integrator.fragments)

    def test_integrate_merge(self):
        frag1_id = self.integrator.add_fragment(
            content="Fragment 1",
            domain=DomainType.GENERAL,
            features=np.array([0.5] * 16),
            confidence=0.8
        )
        frag2_id = self.integrator.add_fragment(
            content="Fragment 2",
            domain=DomainType.GENERAL,
            features=np.array([0.55] * 16),
            confidence=0.85
        )
        
        result = self.integrator.integrate(
            strategy=IntegrationStrategy.MERGE,
            source_ids=[frag1_id, frag2_id],
            target_id=frag1_id
        )
        
        self.assertTrue(result['success'])
        self.assertIn('record_id', result)

    def test_integrate_link(self):
        frag1_id = self.integrator.add_fragment(
            content="Science knowledge",
            domain=DomainType.SCIENCE,
            features=np.array([0.6] * 16),
            confidence=0.7
        )
        frag2_id = self.integrator.add_fragment(
            content="Math knowledge",
            domain=DomainType.MATHEMATICS,
            features=np.array([0.7] * 16),
            confidence=0.8
        )
        
        result = self.integrator.integrate(
            strategy=IntegrationStrategy.LINK,
            source_ids=[frag1_id],
            target_id=frag2_id
        )
        
        self.assertTrue(result['success'])

    def test_distill_experience(self):
        frag_ids = []
        for i in range(5):
            frag_id = self.integrator.add_fragment(
                content=f"Experience {i}",
                domain=DomainType.GENERAL,
                features=np.array([0.5 + i * 0.05] * 16),
                confidence=0.7 + i * 0.05,
                source="experience"
            )
            frag_ids.append(frag_id)
        
        result = self.integrator.distill_experience(frag_ids, DomainType.GENERAL)
        
        self.assertTrue(result['success'])
        self.assertIn('distilled_fragment_id', result)


class TestMultimodalFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = MultimodalFusion({"visual": 16, "audio": 8, "text": 12}, output_dim=16)

    def test_forward(self):
        inputs = {
            "visual": np.random.rand(16),
            "audio": np.random.rand(8),
            "text": np.random.rand(12)
        }
        
        output = self.fusion(inputs)
        
        self.assertEqual(output.shape, (1, 16))

    def test_get_modality_importance(self):
        importance = self.fusion.get_modality_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertIn('visual', importance)
        self.assertIn('audio', importance)
        self.assertIn('text', importance)

    def test_add_modality(self):
        self.fusion.add_modality("sensor", 6)
        
        self.assertIn('sensor', self.fusion.modalities)

    def test_remove_modality(self):
        self.fusion.remove_modality("audio")
        
        self.assertNotIn('audio', self.fusion.modalities)

    def test_get_fusion_summary(self):
        summary = self.fusion.get_fusion_summary()
        
        self.assertIn('modalities', summary)
        self.assertIn('modality_importance', summary)
        self.assertIn('total_parameters', summary)


def run_all_tests():
    print("=" * 60)
    print("Running Enhanced Capabilities Test Suite")
    print("=" * 60)
    
    unittest.main(module=__name__, exit=False, verbosity=2)


def calculate_metrics():
    print("\n" + "=" * 60)
    print("Capability Enhancement Metrics")
    print("=" * 60)

    metrics = {}

    reasoner = AdvancedReasoner(feature_dim=16)
    predicate = Predicate("test", ["arg"], truth_value=0.7)
    query = LogicalStatement([predicate], confidence=0.8)
    reasoner.add_predicate("test", ["arg"], truth_value=0.7)
    reasoning_result = reasoner.reason(query)
    metrics['reasoning_confidence'] = reasoning_result['confidence']
    metrics['reasoning_steps'] = len(reasoning_result['steps'])
    
    abstraction_engine = AbstractionEngine(feature_dim=16)
    for i in range(10):
        abstraction_engine.add_concept(f"concept_{i}", np.random.rand(16))
    concept_ids = list(abstraction_engine.concepts.keys())
    analogy_result = abstraction_engine.find_analogies(concept_ids[0], top_k=3) if concept_ids else []
    metrics['analogy_quality'] = analogy_result[0]['similarity'] if analogy_result else 0.0
    metrics['concept_count'] = len(abstraction_engine.concepts)

    context_engine = ContextAwarenessEngine()
    for i in range(10):
        context_engine.add_context_frame(
            context_type=ContextType.COGNITIVE,
            features=np.random.rand(16)
        )
    scene_type = context_engine.detect_scene()
    prediction = context_engine.predict_next_context()
    metrics['scene_detection'] = scene_type.value
    metrics['context_prediction_confidence'] = prediction.get('confidence', 0.0) if prediction else 0.0

    kg = EnhancedKnowledgeGraph()
    node_ids = []
    for i in range(20):
        node_id = kg.add_node(f"node_{i}", EntityType.CONCEPT, torch.randn(16))
        node_ids.append(node_id)
    for i in range(15):
        kg.add_edge(node_ids[i], node_ids[i+1], RelationType.RELATED_TO)
    metrics['knowledge_graph_nodes'] = kg.get_summary()['nodes']
    metrics['knowledge_graph_edges'] = kg.get_summary()['edges']

    learning_planner = LearningPlanner()
    for i in range(3):
        learning_planner.create_learning_goal(
            goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
            description=f"Goal {i}",
            priority=LearningPriority.MEDIUM
        )
    plan_id = learning_planner.create_learning_plan(list(learning_planner.goals.keys()))
    plan = learning_planner.plans[plan_id]
    metrics['learning_goals'] = len(learning_planner.goals)
    metrics['learning_plan_status'] = plan.status

    integrator = KnowledgeIntegrator()
    for i in range(5):
        integrator.add_fragment(
            domain=DomainType.GENERAL,
            content=f"Fragment {i}",
            features=np.random.rand(16),
            confidence=0.7 + i * 0.05
        )
    metrics['knowledge_fragments'] = len(integrator.fragments)

    fusion = MultimodalFusion({"visual": 16, "audio": 8, "text": 12}, output_dim=16)
    importance = fusion.get_modality_importance()
    metrics['multimodal_importance'] = importance
    metrics['multimodal_params'] = fusion.get_fusion_summary()['total_parameters']

    print("\n--- Intelligence Level Metrics ---")
    print(f"  Reasoning Confidence: {metrics['reasoning_confidence']:.4f}")
    print(f"  Reasoning Steps: {metrics['reasoning_steps']}")
    print(f"  Analogy Quality: {metrics['analogy_quality']:.4f}")
    print(f"  Concept Count: {metrics['concept_count']}")

    print("\n--- Cognitive Ability Metrics ---")
    print(f"  Scene Detection: {metrics['scene_detection']}")
    print(f"  Context Prediction Confidence: {metrics['context_prediction_confidence']:.4f}")
    print(f"  Knowledge Graph Nodes: {metrics['knowledge_graph_nodes']}")
    print(f"  Knowledge Graph Edges: {metrics['knowledge_graph_edges']}")

    print("\n--- Active Learning Ability Metrics ---")
    print(f"  Learning Goals: {metrics['learning_goals']}")
    print(f"  Learning Plan Status: {metrics['learning_plan_status']}")
    print(f"  Knowledge Fragments: {metrics['knowledge_fragments']}")

    print("\n--- Multimodal Processing Metrics ---")
    print(f"  Modality Importance: {metrics['multimodal_importance']}")
    print(f"  Total Parameters: {metrics['multimodal_params']}")

    print("\n" + "=" * 60)
    print("All capability metrics calculated successfully!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_all_tests()
    calculate_metrics()