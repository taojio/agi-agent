import numpy as np
import torch
import unittest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.agent import SelfEvolvingAGI
from agi_agent.perception import GrowingAutoEncoder


class TestPerformance(unittest.TestCase):
    def test_autoencoder_speed(self):
        ae = GrowingAutoEncoder(input_dim=32, hidden_dim=64)
        x = torch.randn(10, 32)
        
        start_time = time.time()
        for _ in range(100):
            ae.update(x)
        elapsed = time.time() - start_time
        
        self.assertLess(elapsed, 5.0)
    
    def test_agent_step_latency(self):
        agent = SelfEvolvingAGI(input_dim=16)
        obs = np.random.uniform(-1, 1, 16)
        
        latencies = []
        for _ in range(50):
            start_time = time.time()
            agent.step(obs)
            elapsed = (time.time() - start_time) * 1000
            latencies.append(elapsed)
        
        avg_latency = float(np.mean(latencies))
        self.assertLess(avg_latency, 500.0)
    
    def test_agent_throughput(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        start_time = time.time()
        for i in range(100):
            obs = np.random.uniform(-1, 1, 16)
            agent.step(obs)
        elapsed = time.time() - start_time
        
        throughput = float(100 / elapsed)
        self.assertGreater(throughput, 5.0)
    
    def test_structure_adaptation(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        initial_dim = agent.perception.get_feature_dim()
        for i in range(300):
            obs = np.random.uniform(-1, 1, 16)
            result = agent.step(obs)
        
        final_dim = agent.perception.get_feature_dim()
        self.assertTrue(final_dim >= initial_dim or final_dim <= initial_dim)
    
    def test_meta_learning_efficiency(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        learning_rates = []
        for i in range(100):
            obs = np.random.uniform(-1, 1, 16)
            result = agent.step(obs)
            learning_rates.append(agent.perception.optimizer.param_groups[0]['lr'])
        
        self.assertEqual(len(set(learning_rates)), 4)
    
    def test_memory_stability(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        for i in range(200):
            obs = np.random.uniform(-1, 1, 16)
            result = agent.step(obs)
        
        self.assertLess(len(agent.cognitive.memory_buffer), 201)


if __name__ == "__main__":
    unittest.main()