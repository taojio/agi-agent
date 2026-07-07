import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.utils.logger import setup_logger


class ComplianceChecker:
    def __init__(self):
        self.logger = setup_logger("compliance_checker")
        self.compliance_rules = {
            "bias_detection": True,
            "data_privacy": True,
            "transparency": True,
            "accountability": True
        }
        self.compliance_history = []

    def check_bias(self, features, actions):
        feature_mean = float(features.mean(dim=-1).detach().cpu().numpy())
        action_mean = float(actions.mean(axis=-1))
        
        bias_score = abs(feature_mean - action_mean)
        is_compliant = bias_score < 0.3
        
        return {
            "compliant": is_compliant,
            "bias_score": bias_score,
            "details": {"feature_mean": feature_mean, "action_mean": action_mean}
        }

    def check_data_privacy(self, data):
        sensitive_patterns = ["password", "token", "secret", "api_key", "credit card", "cc number", "bank account", "ssn", "social security"]
        data_str = str(data).lower()
        
        for pattern in sensitive_patterns:
            if pattern in data_str:
                return {
                    "compliant": False,
                    "issue": f"Sensitive data pattern detected: {pattern}",
                    "severity": "critical"
                }
        
        return {"compliant": True, "issue": None, "severity": "none"}

    def check_transparency(self, decision_trace):
        if len(decision_trace) < 3:
            return {
                "compliant": False,
                "issue": "Insufficient decision trace",
                "severity": "warning"
            }
        
        return {"compliant": True, "issue": None, "severity": "none"}

    def run_compliance_check(self, features=None, actions=None, data=None, decision_trace=None):
        results = {}
        
        if features is not None and actions is not None:
            results["bias_detection"] = self.check_bias(features, actions)
        
        if data is not None:
            results["data_privacy"] = self.check_data_privacy(data)
        
        if decision_trace is not None:
            results["transparency"] = self.check_transparency(decision_trace)
        
        all_compliant = all(result["compliant"] for result in results.values())
        self.compliance_history.append({
            "timestamp": len(self.compliance_history),
            "compliant": all_compliant,
            "results": results
        })
        
        if not all_compliant:
            self.logger.warning(f"[COMPLIANCE] Non-compliant: {results}")
        
        return results

    def get_compliance_report(self):
        recent_checks = self.compliance_history[-10:]
        compliance_rate = float(sum(1 for c in recent_checks if c["compliant"]) / len(recent_checks) if recent_checks else 1.0)
        
        return {
            "compliance_rate": compliance_rate,
            "total_checks": len(self.compliance_history),
            "recent_results": recent_checks,
            "rules_enabled": self.compliance_rules
        }