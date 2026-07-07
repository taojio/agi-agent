import numpy as np
from collections import deque
from enum import Enum


class SafetyBoundary:
    def __init__(self):
        self.red_lines = []
        self.warning_lines = []
        self.violation_history = deque(maxlen=100)

    def add_red_line(self, name, condition, action="block"):
        self.red_lines.append({"name": name, "condition": condition, "action": action})

    def add_warning_line(self, name, condition):
        self.warning_lines.append({"name": name, "condition": condition})

    def check(self, action_plan):
        violations = []
        warnings = []

        for red_line in self.red_lines:
            if red_line["condition"](action_plan):
                violations.append({"type": "red_line", "name": red_line["name"], "action": red_line["action"]})

        for warning_line in self.warning_lines:
            if warning_line["condition"](action_plan):
                warnings.append({"type": "warning", "name": warning_line["name"]})

        if violations:
            for v in violations:
                self.violation_history.append({
                    "type": v["type"],
                    "name": v["name"],
                    "action_plan": str(action_plan)[:200],
                    "timestamp": np.random.randint(1000000)
                })

        return {"violations": violations, "warnings": warnings, "safe": len(violations) == 0}


class PermissionBoundary:
    def __init__(self):
        self.permissions = {}
        self.denied_history = deque(maxlen=100)

    def grant_permission(self, action_type, resource, level="read"):
        if action_type not in self.permissions:
            self.permissions[action_type] = {}
        self.permissions[action_type][resource] = level

    def check_permission(self, action_type, resource):
        if action_type in self.permissions and resource in self.permissions[action_type]:
            return {"allowed": True, "level": self.permissions[action_type][resource]}
        
        self.denied_history.append({
            "action_type": action_type,
            "resource": resource,
            "timestamp": np.random.randint(1000000)
        })
        
        return {"allowed": False, "level": None}

    def get_permissions(self):
        return self.permissions

    def get_denied_count(self):
        return len(self.denied_history)


class EthicalBoundary:
    def __init__(self):
        self.principles = []
        self.ethical_violations = deque(maxlen=50)

    def add_principle(self, name, check_func, weight=1.0):
        self.principles.append({"name": name, "check": check_func, "weight": weight})

    def evaluate(self, action_plan):
        violations = []
        total_score = 0.0

        for principle in self.principles:
            result = principle["check"](action_plan)
            if not result["compliant"]:
                violations.append({"principle": principle["name"], "reason": result["reason"]})
                total_score -= principle["weight"] * 0.5
            else:
                total_score += principle["weight"] * 0.5

        ethical_score = max(0.0, min(1.0, total_score / len(self.principles))) if self.principles else 1.0

        if violations:
            self.ethical_violations.append({
                "violations": violations,
                "action_plan": str(action_plan)[:200],
                "ethical_score": ethical_score,
                "timestamp": np.random.randint(1000000)
            })

        return {"ethical_score": ethical_score, "violations": violations, "compliant": ethical_score >= 0.5}


class BoundaryGuardian:
    def __init__(self):
        self.safety_boundary = SafetyBoundary()
        self.permission_boundary = PermissionBoundary()
        self.ethical_boundary = EthicalBoundary()
        
        self._init_default_boundaries()

    def _init_default_boundaries(self):
        self.safety_boundary.add_red_line(
            "system_shutdown",
            lambda ap: ap.get("action") == "shutdown" and ap.get("force", False),
            action="block"
        )
        self.safety_boundary.add_red_line(
            "unauthorized_delete",
            lambda ap: ap.get("action") == "delete" and ap.get("target") == "system_files",
            action="block"
        )
        self.safety_boundary.add_red_line(
            "resource_exhaustion",
            lambda ap: ap.get("resources", 0) > 0.95,
            action="block"
        )

        self.safety_boundary.add_warning_line(
            "high_risk_action",
            lambda ap: ap.get("risk_level") == "high"
        )

        self.permission_boundary.grant_permission("read", "local_files", "read")
        self.permission_boundary.grant_permission("write", "workspace_files", "write")
        self.permission_boundary.grant_permission("execute", "approved_scripts", "execute")

        self.ethical_boundary.add_principle(
            "non_harm",
            lambda ap: {"compliant": ap.get("harmful", False) is False, "reason": "Action may cause harm"},
            weight=1.0
        )
        self.ethical_boundary.add_principle(
            "transparency",
            lambda ap: {"compliant": ap.get("transparent", True), "reason": "Action lacks transparency"},
            weight=0.8
        )
        self.ethical_boundary.add_principle(
            "fairness",
            lambda ap: {"compliant": True, "reason": "Fairness check passed"},
            weight=0.7
        )

    def check_all_boundaries(self, action_plan):
        safety_result = self.safety_boundary.check(action_plan)
        ethical_result = self.ethical_boundary.evaluate(action_plan)

        action_type = action_plan.get("action", "unknown")
        resource = action_plan.get("resource", "unknown")
        permission_result = self.permission_boundary.check_permission(action_type, resource)

        return {
            "safety": safety_result,
            "permission": permission_result,
            "ethical": ethical_result,
            "overall_allowed": safety_result["safe"] and permission_result["allowed"] and ethical_result["compliant"]
        }

    def check_safety(self, action_plan):
        return self.safety_boundary.check(action_plan)

    def check_permission(self, action_type, resource):
        return self.permission_boundary.check_permission(action_type, resource)

    def check_ethical(self, action_plan):
        return self.ethical_boundary.evaluate(action_plan)

    def get_boundary_stats(self):
        return {
            "safety_violations": len(self.safety_boundary.violation_history),
            "permission_denied": self.permission_boundary.get_denied_count(),
            "ethical_violations": len(self.ethical_boundary.ethical_violations),
            "safety_red_lines": len(self.safety_boundary.red_lines),
            "safety_warnings": len(self.safety_boundary.warning_lines),
            "ethical_principles": len(self.ethical_boundary.principles)
        }