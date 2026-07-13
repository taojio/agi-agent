import unittest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.deliberative.neuro_symbolic_reasoner import NeuroSymbolicReasoner, SymbolType, NeuralSymbol, SymbolicExpression, InferenceRule
from agi_agent.cognitive.world_model import WorldModelEngine, EntityCategory, WorldEntity, CausalRelation, SocialRule, SimulationResult
from agi_agent.deliberative.neuro_symbolic_world_coordinator import NeuroSymbolicWorldCoordinator, InteractionProtocol


class TestNeuroSymbolicReasoner(unittest.TestCase):
    def setUp(self):
        self.reasoner = NeuroSymbolicReasoner(symbol_dim=32, hidden_dim=64)

    def test_add_symbol(self):
        symbol = self.reasoner.add_symbol("test_concept", SymbolType.CONCEPT, np.random.rand(32))
        
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.symbol_id, "test_concept")
        self.assertEqual(symbol.symbol_type, SymbolType.CONCEPT)
        self.assertIn("test_concept", self.reasoner.symbol_memory)

    def test_add_relation(self):
        self.reasoner.add_symbol("source", SymbolType.CONCEPT, np.random.rand(32))
        self.reasoner.add_symbol("target", SymbolType.CONCEPT, np.random.rand(32))
        
        result = self.reasoner.add_relation("source", "target", "causes", weight=0.8, confidence=0.9)
        
        self.assertTrue(result)
        self.assertIn(("source", "target"), self.reasoner.relation_memory)

    def test_reasoning_with_expression(self):
        symbol_a = self.reasoner.add_symbol("A", SymbolType.PREDICATE, np.array([0.9] * 32))
        symbol_a.confidence = 0.9
        
        query = SymbolicExpression("NOT", [symbol_a])
        result = self.reasoner.reason(query)
        
        self.assertIn("truth_value", result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0.0)

    def test_add_inference_rule(self):
        symbol_p = self.reasoner.add_symbol("P", SymbolType.PREDICATE, np.array([0.9] * 32))
        symbol_q = self.reasoner.add_symbol("Q", SymbolType.PREDICATE, np.array([0.5] * 32))
        
        condition = SymbolicExpression("AND", [symbol_p])
        conclusion = SymbolicExpression("IMPLIES", [symbol_p, symbol_q])
        
        self.reasoner.add_inference_rule("modus_ponens", condition, conclusion, confidence=0.95, priority=1)
        
        self.assertEqual(len(self.reasoner.inference_rules), 1)
        self.assertEqual(self.reasoner.inference_rules[0].rule_id, "modus_ponens")

    def test_explain_symbol(self):
        self.reasoner.add_symbol("explain_test", SymbolType.CONCEPT, np.random.rand(32))
        
        explanation = self.reasoner.explain_symbol("explain_test")
        
        self.assertIn("symbol_id", explanation)
        self.assertIn("symbol_type", explanation)
        self.assertIn("confidence", explanation)

    def test_get_summary(self):
        self.reasoner.add_symbol("summary_test", SymbolType.CONCEPT, np.random.rand(32))
        
        summary = self.reasoner.get_summary()
        
        self.assertIn("symbol_count", summary)
        self.assertIn("relation_count", summary)
        self.assertIn("rule_count", summary)
        self.assertEqual(summary["symbol_count"], 1)


class TestWorldModelEngine(unittest.TestCase):
    def setUp(self):
        self.engine = WorldModelEngine(feature_dim=32, history_length=50)

    def test_add_entity(self):
        entity = self.engine.add_entity("test_agent", EntityCategory.AGENT, np.random.rand(32))
        
        self.assertIsNotNone(entity)
        self.assertEqual(entity.entity_id, "test_agent")
        self.assertEqual(entity.category, EntityCategory.AGENT)
        self.assertIn("test_agent", self.engine.entities)

    def test_add_causal_relation(self):
        self.engine.add_entity("cause_entity", EntityCategory.ACTION, np.random.rand(32))
        self.engine.add_entity("effect_entity", EntityCategory.EVENT, np.random.rand(32))
        
        result = self.engine.add_causal_relation("cause_entity", "effect_entity", strength=0.8, delay=0.1)
        
        self.assertTrue(result)
        self.assertEqual(len(self.engine.causal_relations), 1)

    def test_add_social_rule(self):
        self.engine.add_social_rule(
            rule_id="test_rule",
            description="Test social rule",
            conditions=["condition1"],
            consequences=["consequence1"],
            enforcement_strength=0.9,
            domain="test"
        )
        
        self.assertEqual(len(self.engine.social_rules), 4)

    def test_simulate_scenario(self):
        self.engine.add_entity("player", EntityCategory.AGENT, np.random.rand(32))
        
        initial_state = {"player": {"activated": True, "confidence": 0.8}}
        actions = [{"entity_id": "player"}]
        
        simulation = self.engine.simulate_scenario("test_scenario", initial_state, actions, max_steps=5)
        
        self.assertIsInstance(simulation, SimulationResult)
        self.assertEqual(simulation.simulation_id, "test_scenario")
        self.assertGreater(len(simulation.steps), 0)

    def test_simulate_counterfactual(self):
        self.engine.add_entity("base_entity", EntityCategory.AGENT, np.random.rand(32))
        
        initial_state = {"base_entity": {"activated": False, "confidence": 0.5}}
        actions = [{"entity_id": "base_entity"}]
        
        base_simulation = self.engine.simulate_scenario("base_scenario", initial_state, actions, max_steps=3)
        
        modified_actions = [{"entity_id": "base_entity", "state": {"activated": True}}]
        counterfactual = self.engine.simulate_counterfactual(base_simulation, modified_actions)
        
        self.assertIsInstance(counterfactual, SimulationResult)
        self.assertTrue(counterfactual.simulation_id.startswith("counterfactual_"))

    def test_plan_long_term(self):
        self.engine.add_entity("goal_entity", EntityCategory.OBJECT, np.random.rand(32))
        self.engine.add_causal_relation("goal_entity", "goal_entity", strength=0.5)
        
        goal_state = {"goal_entity": {"activated": True}}
        current_state = {"goal_entity": {"activated": False}}
        
        plan = self.engine.plan_long_term(goal_state, current_state, max_planning_steps=5)
        
        self.assertIsInstance(plan, list)

    def test_get_causal_graph(self):
        self.engine.add_entity("cause", EntityCategory.ACTION, np.random.rand(32))
        self.engine.add_entity("effect", EntityCategory.EVENT, np.random.rand(32))
        self.engine.add_causal_relation("cause", "effect", strength=0.7)
        
        graph = self.engine.get_causal_graph()
        
        self.assertIn("cause", graph)
        self.assertEqual(len(graph["cause"]), 1)

    def test_get_entity_state(self):
        self.engine.add_entity("state_entity", EntityCategory.OBJECT, np.random.rand(32))
        
        state = self.engine.get_entity_state("state_entity")
        
        self.assertIn("entity_id", state)
        self.assertIn("category", state)
        self.assertIn("confidence", state)

    def test_get_summary(self):
        self.engine.add_entity("summary_entity", EntityCategory.CONCEPT, np.random.rand(32))
        
        summary = self.engine.get_summary()
        
        self.assertIn("entity_count", summary)
        self.assertIn("causal_relation_count", summary)
        self.assertIn("social_rule_count", summary)
        self.assertEqual(summary["entity_count"], 1)


class TestNeuroSymbolicWorldCoordinator(unittest.TestCase):
    def setUp(self):
        self.reasoner = NeuroSymbolicReasoner(symbol_dim=32, hidden_dim=64)
        self.world_model = WorldModelEngine(feature_dim=32, history_length=50)
        self.coordinator = NeuroSymbolicWorldCoordinator(
            neuro_symbolic_reasoner=self.reasoner,
            world_model_engine=self.world_model
        )

    def test_add_message(self):
        self.coordinator.add_message(
            protocol=InteractionProtocol.SYMBOL_TO_ENTITY,
            sender="test",
            receiver="test",
            payload={"test": "data"},
            priority=2
        )
        
        self.assertEqual(len(self.coordinator.message_queue), 1)

    def test_process_messages(self):
        self.coordinator.add_message(
            protocol=InteractionProtocol.SYMBOL_TO_ENTITY,
            sender="test",
            receiver="test",
            payload={"test": "data"},
            priority=2
        )
        
        processed_count = self.coordinator.process_messages()
        
        self.assertEqual(processed_count, 1)
        self.assertEqual(len(self.coordinator.message_queue), 0)

    def test_synchronize_knowledge(self):
        self.reasoner.add_symbol("sync_symbol", SymbolType.CONCEPT, np.random.rand(32))
        
        self.coordinator.synchronize_knowledge()
        
        summary = self.coordinator.get_coordination_summary()
        self.assertGreater(summary["symbol_entity_mappings"], 0)

    def test_coordinated_reasoning(self):
        symbol = self.reasoner.add_symbol("query_symbol", SymbolType.PREDICATE, np.array([0.8] * 32))
        symbol.confidence = 0.8
        
        query = SymbolicExpression("AND", [symbol])
        
        result = self.coordinator.coordinated_reasoning(query)
        
        self.assertIn("reasoning_result", result)
        self.assertIn("confidence", result)

    def test_plan_with_world_model(self):
        self.world_model.add_entity("plan_entity", EntityCategory.AGENT, np.random.rand(32))
        
        goal_state = {"plan_entity": {"activated": True}}
        current_state = {"plan_entity": {"activated": False}}
        
        result = self.coordinator.plan_with_world_model(goal_state, current_state)
        
        self.assertIn("plan", result)
        self.assertIn("confidence", result)

    def test_get_coordination_summary(self):
        summary = self.coordinator.get_coordination_summary()
        
        self.assertIn("message_queue_size", summary)
        self.assertIn("symbol_entity_mappings", summary)
        self.assertIn("entity_symbol_mappings", summary)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.reasoner = NeuroSymbolicReasoner(symbol_dim=32, hidden_dim=64)
        self.world_model = WorldModelEngine(feature_dim=32, history_length=50)
        self.coordinator = NeuroSymbolicWorldCoordinator(
            neuro_symbolic_reasoner=self.reasoner,
            world_model_engine=self.world_model
        )

    def test_symbol_to_entity_flow(self):
        symbol = self.reasoner.add_symbol("flow_symbol", SymbolType.ACTION, np.random.rand(32))
        
        self.coordinator.synchronize_knowledge()
        
        self.assertIn("flow_symbol", self.coordinator.symbol_entity_mapping)
        entity_id = self.coordinator.symbol_entity_mapping["flow_symbol"]
        self.assertIn(entity_id, self.world_model.entities)

    def test_entity_to_symbol_flow(self):
        entity = self.world_model.add_entity("flow_entity", EntityCategory.AGENT, np.random.rand(32))
        
        self.coordinator.synchronize_knowledge()
        
        self.assertIn(entity.entity_id, self.coordinator.entity_symbol_mapping)
        symbol_id = self.coordinator.entity_symbol_mapping[entity.entity_id]
        self.assertIn(symbol_id, self.reasoner.symbol_memory)

    def test_causal_to_rule_flow(self):
        cause_entity = self.world_model.add_entity("cause", EntityCategory.ACTION, np.random.rand(32))
        effect_entity = self.world_model.add_entity("effect", EntityCategory.EVENT, np.random.rand(32))
        self.world_model.add_causal_relation("cause", "effect", strength=0.8)
        
        self.coordinator.synchronize_knowledge()
        
        self.coordinator.add_message(
            protocol=InteractionProtocol.CAUSAL_TO_RULE,
            sender="world_model",
            receiver="neuro_symbolic",
            payload={
                "causal_data": {
                    "cause_id": "cause",
                    "effect_id": "effect",
                    "strength": 0.8
                }
            },
            priority=2
        )
        
        self.coordinator.process_messages()
        
        self.assertGreater(len(self.reasoner.inference_rules), 0)

    def test_simulation_to_reasoning_flow(self):
        entity = self.world_model.add_entity("sim_entity", EntityCategory.AGENT, np.random.rand(32))
        
        self.coordinator.synchronize_knowledge()
        
        initial_state = {"sim_entity": {"activated": True, "confidence": 0.9}}
        actions = [{"entity_id": "sim_entity"}]
        
        simulation = self.world_model.simulate_scenario("integration_test", initial_state, actions, max_steps=3)
        
        self.coordinator.add_message(
            protocol=InteractionProtocol.SIMULATION_TO_REASONING,
            sender="world_model",
            receiver="neuro_symbolic",
            payload={"simulation_result": simulation.to_dict()},
            priority=2
        )
        
        self.coordinator.process_messages()
        
        symbol_id = self.coordinator.entity_symbol_mapping.get("sim_entity")
        if symbol_id:
            symbol = self.reasoner.symbol_memory.get(symbol_id)
            self.assertGreaterEqual(symbol.confidence, 0.5)


def calculate_integration_metrics():
    reasoner = NeuroSymbolicReasoner(symbol_dim=32, hidden_dim=64)
    world_model = WorldModelEngine(feature_dim=32, history_length=50)
    coordinator = NeuroSymbolicWorldCoordinator(
        neuro_symbolic_reasoner=reasoner,
        world_model_engine=world_model
    )

    symbols_added = 0
    for i in range(10):
        reasoner.add_symbol(f"symbol_{i}", SymbolType.CONCEPT, np.random.rand(32))
        symbols_added += 1

    entities_added = 0
    for i in range(5):
        world_model.add_entity(f"entity_{i}", EntityCategory.AGENT, np.random.rand(32))
        entities_added += 1

    world_model.add_causal_relation("entity_0", "entity_1", strength=0.8)
    world_model.add_causal_relation("entity_2", "entity_3", strength=0.9)

    coordinator.synchronize_knowledge()

    symbol_a = reasoner.add_symbol("A", SymbolType.PREDICATE, np.array([0.9] * 32))
    symbol_a.confidence = 0.9
    symbol_b = reasoner.add_symbol("B", SymbolType.PREDICATE, np.array([0.5] * 32))
    
    condition = SymbolicExpression("AND", [symbol_a])
    conclusion = SymbolicExpression("IMPLIES", [symbol_a, symbol_b])
    reasoner.add_inference_rule("test_rule", condition, conclusion, confidence=0.95)

    query = SymbolicExpression("IMPLIES", [symbol_a, symbol_b])
    reasoning_result = reasoner.reason(query)

    initial_state = {"entity_0": {"activated": True, "confidence": 0.8}}
    actions = [{"entity_id": "entity_0"}]
    simulation_result = world_model.simulate_scenario("metrics_test", initial_state, actions, max_steps=5)

    plan_result = coordinator.plan_with_world_model(
        goal={"entity_1": {"activated": True}},
        current_state={"entity_1": {"activated": False}}
    )

    metrics = {
        "neuro_symbolic": {
            "symbol_count": reasoner.get_summary()["symbol_count"],
            "relation_count": reasoner.get_summary()["relation_count"],
            "rule_count": reasoner.get_summary()["rule_count"],
            "reasoning_confidence": reasoning_result.get("confidence", 0.0),
            "reasoning_steps": len(reasoning_result.get("steps", []))
        },
        "world_model": {
            "entity_count": world_model.get_summary()["entity_count"],
            "causal_relation_count": world_model.get_summary()["causal_relation_count"],
            "social_rule_count": world_model.get_summary()["social_rule_count"],
            "simulation_confidence": simulation_result.confidence,
            "simulation_steps": len(simulation_result.steps)
        },
        "coordination": {
            "symbol_entity_mappings": coordinator.get_coordination_summary()["symbol_entity_mappings"],
            "entity_symbol_mappings": coordinator.get_coordination_summary()["entity_symbol_mappings"],
            "coordination_history_length": coordinator.get_coordination_summary()["coordination_history_length"],
            "plan_length": len(plan_result.get("plan", [])),
            "plan_confidence": plan_result.get("confidence", 0.0)
        }
    }

    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Running Neuro-Symbolic Reasoning & World Model Test Suite")
    print("=" * 60)
    
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("Integration Metrics")
    print("=" * 60)
    
    metrics = calculate_integration_metrics()
    
    print("\n--- Neuro-Symbolic Reasoner Metrics ---")
    for key, value in metrics["neuro_symbolic"].items():
        print(f"  {key}: {value}")
    
    print("\n--- World Model Metrics ---")
    for key, value in metrics["world_model"].items():
        print(f"  {key}: {value}")
    
    print("\n--- Coordination Metrics ---")
    for key, value in metrics["coordination"].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All integration metrics calculated successfully!")
    print("=" * 60)