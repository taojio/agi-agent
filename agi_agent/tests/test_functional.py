import numpy as np
import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.perception import GrowingAutoEncoder, MultimodalFusion
from agi_agent.cognitive import CognitiveInferenceLayer
from agi_agent.learning import MetaLearningLayer, KnowledgeGraph
from agi_agent.evolution import EvolutionEngine
from agi_agent.execution import ActionExecutionLayer
from agi_agent.metacognition import MetaCognitionLayer
from agi_agent.storage import PersistenceManager
from agi_agent.security import SafetyMonitor, ComplianceChecker
from agi_agent.evaluation import PerformanceEvaluator
from agi_agent.agent import SelfEvolvingAGI


class TestPerception(unittest.TestCase):
    def test_autoencoder_initialization(self):
        ae = GrowingAutoEncoder(input_dim=16, hidden_dim=32)
        self.assertEqual(ae.input_dim, 16)
        self.assertEqual(ae.hidden_dim, 32)
    
    def test_autoencoder_forward(self):
        ae = GrowingAutoEncoder(input_dim=16, hidden_dim=32)
        x = torch.randn(1, 16)
        z, recon = ae(x)
        self.assertEqual(z.shape[-1], 16)
        self.assertEqual(recon.shape[-1], 16)
    
    def test_autoencoder_update(self):
        ae = GrowingAutoEncoder(input_dim=16, hidden_dim=32)
        x = torch.randn(1, 16)
        z, fe, changed = ae.update(x)
        self.assertIsInstance(fe, float)
        self.assertGreaterEqual(fe, 0)
    
    def test_multimodal_fusion(self):
        modalities = {"vision": 10, "audio": 8}
        fusion = MultimodalFusion(modalities, output_dim=16)
        inputs = {
            "vision": np.random.rand(1, 10),
            "audio": np.random.rand(1, 8)
        }
        result = fusion(inputs)
        self.assertEqual(result.shape[-1], 16)
    
    def test_add_remove_modality(self):
        modalities = {"vision": 10}
        fusion = MultimodalFusion(modalities, output_dim=16)
        fusion.add_modality("audio", 8)
        self.assertIn("audio", fusion.modalities)
        fusion.remove_modality("audio")
        self.assertNotIn("audio", fusion.modalities)


class TestCognitive(unittest.TestCase):
    def test_cognitive_inference_init(self):
        ci = CognitiveInferenceLayer(feat_dim=16)
        self.assertEqual(len(ci.memory_buffer), 0)
    
    def test_autonomous_thinking(self):
        ci = CognitiveInferenceLayer(feat_dim=16)
        feat = torch.randn(1, 16)
        pred_seq = ci.autonomous_thinking(feat)
        self.assertEqual(len(pred_seq), 5)
    
    def test_knowledge_deposit(self):
        ci = CognitiveInferenceLayer(feat_dim=16)
        feat = torch.randn(1, 16)
        pred = torch.randn(1, 16)
        ci.deposit_knowledge(feat, pred)
        self.assertEqual(len(ci.memory_buffer), 1)
    
    def test_knowledge_summary(self):
        ci = CognitiveInferenceLayer(feat_dim=16)
        summary = ci.get_knowledge_summary()
        self.assertIn("count", summary)
        self.assertIn("avg_confidence", summary)


class TestLearning(unittest.TestCase):
    def test_meta_learning_init(self):
        ml = MetaLearningLayer()
        self.assertEqual(len(ml.lr_pool), 4)
    
    def test_adaptive_hyper_update(self):
        ml = MetaLearningLayer()
        lr = ml.adaptive_hyper_update(0.01, 0.1)
        self.assertIn(lr, ml.lr_pool)
    
    def test_meta_stats(self):
        ml = MetaLearningLayer()
        stats = ml.get_meta_stats()
        self.assertIn("best_lr", stats)
        self.assertIn("exploration_rate", stats)
    
    def test_knowledge_graph_init(self):
        kg = KnowledgeGraph()
        self.assertEqual(len(kg.nodes), 0)
    
    def test_knowledge_graph_add_node(self):
        kg = KnowledgeGraph()
        feat = torch.randn(16)
        node_id = kg.add_node(feat)
        self.assertEqual(len(kg.nodes), 1)
    
    def test_knowledge_graph_add_edge(self):
        kg = KnowledgeGraph()
        feat1 = torch.randn(16)
        feat2 = torch.randn(16)
        node1 = kg.add_node(feat1)
        node2 = kg.add_node(feat2)
        kg.add_edge(node1, node2)
        self.assertEqual(len(kg.edges[node1]), 1)


class TestEvolution(unittest.TestCase):
    def test_evolution_engine_init(self):
        ee = EvolutionEngine()
        self.assertIsNone(ee.population)
    
    def test_evolution_stats(self):
        ee = EvolutionEngine()
        stats = ee.get_evolution_stats()
        self.assertIn("step", stats)
        self.assertIn("fitness", stats)


class TestExecution(unittest.TestCase):
    def test_action_executor_init(self):
        ae = ActionExecutionLayer(action_dim=8, feature_dim=16)
        self.assertEqual(ae.action_dim, 8)
    
    def test_autonomous_action(self):
        ae = ActionExecutionLayer(action_dim=8, feature_dim=16)
        feat = torch.randn(1, 16)
        pred = torch.randn(1, 16)
        action = ae.autonomous_action(feat, pred)
        self.assertEqual(action.shape[-1], 8)
    
    def test_hardware_adapt(self):
        ae = ActionExecutionLayer(action_dim=8, feature_dim=16)
        ae.hardware_adapt(20)
        self.assertEqual(ae.feature_dim, 20)


class TestMetaCognition(unittest.TestCase):
    def test_meta_cognition_init(self):
        mc = MetaCognitionLayer()
        self.assertEqual(mc.system_status, "healthy")
    
    def test_monitor(self):
        mc = MetaCognitionLayer()
        mc.monitor(0.01, 0.5, 0.1, 10.0)
        self.assertAlmostEqual(mc.cognitive_metrics["free_energy"], 0.01)
    
    def test_resource_schedule(self):
        mc = MetaCognitionLayer()
        schedule = mc.resource_schedule()
        self.assertIn("perception", schedule)
        self.assertIn("cognitive", schedule)
    
    def test_trend_analysis(self):
        mc = MetaCognitionLayer()
        for _ in range(15):
            mc.monitor(0.01, 0.5, 0.1, 10.0)
        trend = mc.get_trend_analysis()
        self.assertIn("fe_trend", trend)


class TestStorage(unittest.TestCase):
    def test_persistence_init(self):
        pm = PersistenceManager()
        self.assertEqual(pm.backup_count, 0)
    
    def test_save_load_state(self):
        pm = PersistenceManager()
        state = {"test_key": "test_value", "step": 100}
        pm.save_state(state, "test_state")
        loaded = pm.load_state("test_state")
        self.assertIsNotNone(loaded)


class TestSecurity(unittest.TestCase):
    def test_safety_monitor_init(self):
        sm = SafetyMonitor()
        self.assertEqual(len(sm.violations), 0)
    
    def test_resource_check(self):
        sm = SafetyMonitor()
        resources = sm.check_resource_usage()
        self.assertIn("memory_gb", resources)
        self.assertIn("cpu_percent", resources)
    
    def test_risk_level(self):
        sm = SafetyMonitor()
        level = sm.assess_risk_level()
        self.assertEqual(level, "low")
    
    def test_compliance_checker_init(self):
        cc = ComplianceChecker()
        self.assertTrue(cc.compliance_rules["bias_detection"])
    
    def test_data_privacy_check(self):
        cc = ComplianceChecker()
        result = cc.check_data_privacy("normal data")
        self.assertTrue(result["compliant"])
    
    def test_bias_check(self):
        cc = ComplianceChecker()
        features = torch.randn(10, 16)
        actions = np.random.randn(10, 8)
        result = cc.check_bias(features, actions)
        self.assertIn("compliant", result)


class TestEvaluation(unittest.TestCase):
    def test_evaluator_init(self):
        pe = PerformanceEvaluator()
        self.assertEqual(len(pe.evaluation_history), 0)
    
    def test_evaluate_step(self):
        pe = PerformanceEvaluator()
        metrics = {"free_energy": 0.01, "confidence": 0.9, "novelty": 0.1, "latency": 10.0}
        eval_result = pe.evaluate_step(1, metrics)
        self.assertEqual(eval_result["step"], 1)
    
    def test_performance_score(self):
        pe = PerformanceEvaluator()
        for i in range(60):
            metrics = {"free_energy": 0.01, "confidence": 0.9, "novelty": 0.1, "latency": 10.0}
            pe.evaluate_step(i, metrics)
        score = pe.calculate_performance_score()
        self.assertGreater(score["total_score"], 0.5)


class TestAgentIntegration(unittest.TestCase):
    def test_agent_init(self):
        agent = SelfEvolvingAGI(input_dim=16)
        self.assertEqual(agent.input_dim, 16)
    
    def test_agent_step(self):
        agent = SelfEvolvingAGI(input_dim=16)
        obs = np.random.uniform(-1, 1, 16)
        result = agent.step(obs)
        self.assertIn("free_energy", result)
        self.assertIn("confidence", result)
    
    def test_agent_hardware_expand(self):
        agent = SelfEvolvingAGI(input_dim=16)
        agent.hardware_self_expand(20)
        self.assertEqual(agent.input_dim, 20)
    
    def test_agent_generate_report(self):
        agent = SelfEvolvingAGI(input_dim=16)
        for _ in range(20):
            obs = np.random.uniform(-1, 1, 16)
            agent.step(obs)
        report = agent.generate_report()
        self.assertIn("agent_info", report)
        self.assertIn("performance", report)


if __name__ == "__main__":
    unittest.main()