import os
import psutil
import agi_agent.utils.metrics as metrics
import torch
import time
from ..config.settings import SAFETY_MAX_ENERGY, SAFETY_MAX_MEMORY_GB, SAFETY_MAX_GPU_UTIL, SAFETY_MAX_LATENCY_MS
from ..utils.logger import setup_logger


class SafetyMonitor:
    def __init__(self):
        self.logger = setup_logger("safety_monitor")
        self.violations = []
        self.safety_rules = {
            "max_free_energy": SAFETY_MAX_ENERGY,
            "max_memory_gb": SAFETY_MAX_MEMORY_GB,
            "max_gpu_util": SAFETY_MAX_GPU_UTIL,
            "max_latency_ms": SAFETY_MAX_LATENCY_MS
        }
        self.alerts_enabled = True
        self.emergency_shutdown_threshold = 3

    def check_resource_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        
        gpu_util = 0.0
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization() / 100.0
        
        return {
            "memory_gb": memory_gb,
            "cpu_percent": process.cpu_percent(),
            "gpu_util": gpu_util,
            "num_threads": process.num_threads()
        }

    def check_safety_constraints(self, free_energy: float, latency_ms: float):
        violations = []
        
        if free_energy > self.safety_rules["max_free_energy"]:
            violations.append({
                "type": "free_energy_violation",
                "value": free_energy,
                "threshold": self.safety_rules["max_free_energy"],
                "severity": "critical"
            })
        
        resources = self.check_resource_usage()
        if resources["memory_gb"] > self.safety_rules["max_memory_gb"]:
            violations.append({
                "type": "memory_violation",
                "value": resources["memory_gb"],
                "threshold": self.safety_rules["max_memory_gb"],
                "severity": "warning"
            })
        
        if resources["gpu_util"] > self.safety_rules["max_gpu_util"]:
            violations.append({
                "type": "gpu_violation",
                "value": resources["gpu_util"],
                "threshold": self.safety_rules["max_gpu_util"],
                "severity": "warning"
            })
        
        if latency_ms > self.safety_rules["max_latency_ms"]:
            violations.append({
                "type": "latency_violation",
                "value": latency_ms,
                "threshold": self.safety_rules["max_latency_ms"],
                "severity": "info"
            })
        
        if violations and self.alerts_enabled:
            for violation in violations:
                self.logger.warning(f"[SAFETY VIOLATION] {violation['type']}: {violation['value']:.4f} > {violation['threshold']}")
            
            self.violations.extend(violations)
            if len(self.violations) > 100:
                self.violations = self.violations[-100:]
        
        return violations

    def assess_risk_level(self):
        critical_count = sum(1 for v in self.violations[-10:] if v["severity"] == "critical")
        warning_count = sum(1 for v in self.violations[-10:] if v["severity"] == "warning")
        
        if critical_count >= self.emergency_shutdown_threshold:
            return "emergency"
        elif critical_count > 0 or warning_count >= 3:
            return "high"
        elif warning_count > 0:
            return "medium"
        else:
            return "low"

    def enforce_safety_protocols(self, agent):
        risk_level = self.assess_risk_level()
        
        if risk_level == "emergency":
            self.logger.critical("[SAFETY] Emergency shutdown protocol activated!")
            return {"action": "shutdown", "reason": "critical safety violations exceeded threshold"}
        
        if risk_level == "high":
            self.logger.warning("[SAFETY] High risk detected. Reducing computational load.")
            return {"action": "throttle", "reason": "high resource usage"}
        
        return {"action": "continue", "reason": "safe operation"}

    def get_safety_report(self):
        recent_violations = self.violations[-20:]
        return {
            "total_violations": len(self.violations),
            "recent_violations": recent_violations,
            "risk_level": self.assess_risk_level(),
            "current_resources": self.check_resource_usage(),
            "safety_rules": self.safety_rules
        }

    def reset_violations(self):
        self.violations = []
        self.logger.info("[SAFETY] Violations log reset")