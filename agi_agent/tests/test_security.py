import numpy as np
import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.security import SafetyMonitor, ComplianceChecker
from agi_agent.agent import SelfEvolvingAGI


class TestSecurity(unittest.TestCase):
    def test_safety_monitor_risk_level(self):
        sm = SafetyMonitor()
        level = sm.assess_risk_level()
        self.assertEqual(level, "low")
    
    def test_resource_usage_check(self):
        sm = SafetyMonitor()
        resources = sm.check_resource_usage()
        self.assertIn("memory_gb", resources)
        self.assertIn("cpu_percent", resources)
        self.assertGreaterEqual(resources["memory_gb"], 0)
        self.assertGreaterEqual(resources["cpu_percent"], 0)
    
    def test_compliance_rules_enabled(self):
        cc = ComplianceChecker()
        self.assertTrue(cc.compliance_rules["bias_detection"])
        self.assertTrue(cc.compliance_rules["data_privacy"])
        self.assertTrue(cc.compliance_rules["transparency"])
        self.assertTrue(cc.compliance_rules["accountability"])
    
    def test_data_privacy_compliance(self):
        cc = ComplianceChecker()
        
        normal_result = cc.check_data_privacy("normal data")
        self.assertTrue(normal_result["compliant"])
        
        sensitive_result = cc.check_data_privacy("credit card number: 1234-5678-9012-3456")
        self.assertFalse(sensitive_result["compliant"])
    
    def test_bias_detection(self):
        cc = ComplianceChecker()
        
        features = torch.randn(50, 16)
        actions = np.random.randn(50, 8)
        
        result = cc.check_bias(features, actions)
        self.assertIn("compliant", result)
        self.assertIn("bias_score", result)
    
    def test_agent_safety_monitoring(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        for i in range(50):
            obs = np.random.uniform(-1, 1, 16)
            result = agent.step(obs)
        
        risk_level = agent.safety_monitor.assess_risk_level()
        self.assertEqual(risk_level, "low")
    
    def test_agent_compliance_rate(self):
        agent = SelfEvolvingAGI(input_dim=16)
        
        for i in range(50):
            obs = np.random.uniform(-1, 1, 16)
            result = agent.step(obs)
        
        report = agent.generate_report()
        self.assertGreaterEqual(report["compliance"]["compliance_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()