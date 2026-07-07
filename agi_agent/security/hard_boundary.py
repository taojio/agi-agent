"""
hard_boundary.py - 硬边界冻结系统

实现SOUL.md中的安全红线、禁止事项、核心目标为不可篡改字段，
任何进化、思考、行动均不得突破。
"""
import time
import numpy as np
from collections import deque
from enum import Enum


class BoundaryType(Enum):
    SAFETY_RED_LINE = "safety_red_line"
    FORBIDDEN_ACTION = "forbidden_action"
    CORE_GOAL = "core_goal"
    PERMISSION_BOUNDARY = "permission_boundary"


class BoundaryRule:
    def __init__(self, rule_id, boundary_type, description, check_func, action="block"):
        self.rule_id = rule_id
        self.boundary_type = boundary_type
        self.description = description
        self.check_func = check_func
        self.action = action
        self.created_at = time.time()
        self.violation_count = 0

    def check(self, input_data):
        try:
            violation = self.check_func(input_data)
            if violation:
                self.violation_count += 1
                return {
                    "violation": True,
                    "rule_id": self.rule_id,
                    "boundary_type": self.boundary_type.value,
                    "description": self.description,
                    "action": self.action,
                    "reason": violation
                }
            return {"violation": False}
        except Exception as e:
            return {"violation": True, "rule_id": self.rule_id, "error": str(e), "reason": f"check_func_error: {e}"}

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "boundary_type": self.boundary_type.value,
            "description": self.description,
            "action": self.action,
            "violation_count": self.violation_count,
            "created_at": self.created_at
        }


class HardBoundarySystem:
    def __init__(self):
        self.boundary_rules = {}
        self.violation_history = deque(maxlen=200)
        self.frozen_fields = set()
        
        self._init_default_boundaries()

    def _init_default_boundaries(self):
        self.add_boundary_rule(
            rule_id="redline_system_shutdown",
            boundary_type=BoundaryType.SAFETY_RED_LINE,
            description="禁止强制系统关机",
            check_func=lambda d: "强制关机" in str(d.get("action", "")) or d.get("action") == "force_shutdown",
            action="block"
        )

        self.add_boundary_rule(
            rule_id="redline_data_delete",
            boundary_type=BoundaryType.SAFETY_RED_LINE,
            description="禁止删除系统关键数据",
            check_func=lambda d: (d.get("action") == "delete" and d.get("target") == "system_data") or 
                                ("system_data" in str(d.get("target", "")) and d.get("action") == "delete"),
            action="block"
        )

        self.add_boundary_rule(
            rule_id="redline_network_access",
            boundary_type=BoundaryType.SAFETY_RED_LINE,
            description="禁止未经授权的网络访问",
            check_func=lambda d: d.get("action") == "network_access" and not d.get("authorized", False),
            action="block"
        )

        self.add_boundary_rule(
            rule_id="forbidden_format_disk",
            boundary_type=BoundaryType.FORBIDDEN_ACTION,
            description="禁止格式化磁盘",
            check_func=lambda d: d.get("action") == "format_disk",
            action="block"
        )

        self.add_boundary_rule(
            rule_id="forbidden_system_modify",
            boundary_type=BoundaryType.FORBIDDEN_ACTION,
            description="禁止修改系统配置",
            check_func=lambda d: d.get("action") == "modify_system_config",
            action="block"
        )

        self.add_boundary_rule(
            rule_id="core_goal_preservation",
            boundary_type=BoundaryType.CORE_GOAL,
            description="核心目标不可修改",
            check_func=lambda d: d.get("action") == "modify_goal" and d.get("goal_type") == "core",
            action="block"
        )

        self.add_boundary_rule(
            rule_id="permission_escalation",
            boundary_type=BoundaryType.PERMISSION_BOUNDARY,
            description="禁止权限提升",
            check_func=lambda d: d.get("action") == "escalate_permissions",
            action="block"
        )

        self.freeze_field("core_goals")
        self.freeze_field("safety_red_lines")
        self.freeze_field("ethical_rules")
        self.freeze_field("identity_anchor")

    def add_boundary_rule(self, rule_id, boundary_type, description, check_func, action="block"):
        if rule_id in self.boundary_rules:
            return False
        
        rule = BoundaryRule(rule_id, boundary_type, description, check_func, action)
        self.boundary_rules[rule_id] = rule
        return True

    def remove_boundary_rule(self, rule_id):
        if rule_id in self.boundary_rules:
            del self.boundary_rules[rule_id]
            return True
        return False

    def freeze_field(self, field_name):
        self.frozen_fields.add(field_name)

    def unfreeze_field(self, field_name):
        if field_name in self.frozen_fields:
            self.frozen_fields.remove(field_name)
            return True
        return False

    def is_field_frozen(self, field_name):
        return field_name in self.frozen_fields

    def check_all_boundaries(self, input_data):
        violations = []
        for rule_id, rule in self.boundary_rules.items():
            result = rule.check(input_data)
            if result.get("violation"):
                violations.append(result)
                self.violation_history.append({
                    "timestamp": time.time(),
                    "rule_id": rule_id,
                    "input_data": str(input_data)[:200],
                    "violation": result
                })

        return {
            "violations": violations,
            "allowed": len(violations) == 0,
            "blocked_by": [v["rule_id"] for v in violations]
        }

    def check_boundary(self, rule_id, input_data):
        if rule_id not in self.boundary_rules:
            return {"error": "Rule not found"}
        
        return self.boundary_rules[rule_id].check(input_data)

    def get_boundary_stats(self):
        violations_by_type = {}
        for rule in self.boundary_rules.values():
            violations_by_type[rule.boundary_type.value] = violations_by_type.get(rule.boundary_type.value, 0) + rule.violation_count

        return {
            "total_rules": len(self.boundary_rules),
            "frozen_fields": list(self.frozen_fields),
            "violations_by_type": violations_by_type,
            "total_violations": sum(r.violation_count for r in self.boundary_rules.values()),
            "recent_violations": len(self.violation_history)
        }

    def get_violation_history(self, limit=20):
        return list(self.violation_history)[-limit:]

    def get_status(self):
        return self.get_boundary_stats()