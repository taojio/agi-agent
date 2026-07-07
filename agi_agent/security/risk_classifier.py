"""
risk_classifier.py - 风险分级管控系统

实现四级风险分级：
- 低风险操作：自主执行
- 中风险操作：留痕审计，执行后同步通知
- 高风险操作：强制人工二次确认
- 危险操作：直接拦截，拒绝执行
"""
import numpy as np
import time
from collections import deque
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DANGEROUS = "dangerous"


class RiskAction(Enum):
    EXECUTE = "execute"
    AUDIT = "audit"
    CONFIRM = "confirm"
    BLOCK = "block"


class RiskRule:
    def __init__(self, action_pattern, risk_level, description, keywords=None):
        self.action_pattern = action_pattern
        self.risk_level = risk_level
        self.description = description
        self.keywords = keywords or []
        self.match_count = 0

    def matches(self, action_description):
        desc = str(action_description).lower()
        
        if self.action_pattern.lower() in desc:
            self.match_count += 1
            return True
        
        for keyword in self.keywords:
            if keyword.lower() in desc:
                self.match_count += 1
                return True
        
        return False

    def to_dict(self):
        return {
            "action_pattern": self.action_pattern,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "keywords": self.keywords,
            "match_count": self.match_count
        }


class RiskClassifier:
    def __init__(self):
        self.risk_rules = []
        self.audit_log = deque(maxlen=500)
        self.pending_confirmations = {}
        
        self._init_default_rules()

    def _init_default_rules(self):
        self.risk_rules.extend([
            RiskRule(
                action_pattern="file整理",
                risk_level=RiskLevel.LOW,
                description="文件整理操作",
                keywords=["整理", "排序", "分类"]
            ),
            RiskRule(
                action_pattern="数据统计",
                risk_level=RiskLevel.LOW,
                description="数据统计操作",
                keywords=["统计", "分析", "报表"]
            ),
            RiskRule(
                action_pattern="信息查询",
                risk_level=RiskLevel.LOW,
                description="信息查询操作",
                keywords=["查询", "搜索", "查找"]
            ),
            RiskRule(
                action_pattern="配置修改",
                risk_level=RiskLevel.MEDIUM,
                description="配置修改操作",
                keywords=["修改配置", "更新设置"]
            ),
            RiskRule(
                action_pattern="批量删除",
                risk_level=RiskLevel.MEDIUM,
                description="批量删除操作",
                keywords=["批量删除", "批量移除"]
            ),
            RiskRule(
                action_pattern="文件写入",
                risk_level=RiskLevel.MEDIUM,
                description="文件写入操作",
                keywords=["写入", "保存", "更新"]
            ),
            RiskRule(
                action_pattern="系统命令",
                risk_level=RiskLevel.HIGH,
                description="系统级命令执行",
                keywords=["system", "sudo", "cmd", "terminal"]
            ),
            RiskRule(
                action_pattern="外部数据写入",
                risk_level=RiskLevel.HIGH,
                description="外部数据写入操作",
                keywords=["外部写入", "网络写入"]
            ),
            RiskRule(
                action_pattern="权限修改",
                risk_level=RiskLevel.HIGH,
                description="权限修改操作",
                keywords=["权限", "permission", "access"]
            ),
            RiskRule(
                action_pattern="系统关机",
                risk_level=RiskLevel.DANGEROUS,
                description="系统关机操作",
                keywords=["关机", "shutdown", "poweroff"]
            ),
            RiskRule(
                action_pattern="磁盘格式化",
                risk_level=RiskLevel.DANGEROUS,
                description="磁盘格式化操作",
                keywords=["格式化", "format"]
            ),
            RiskRule(
                action_pattern="数据销毁",
                risk_level=RiskLevel.DANGEROUS,
                description="数据销毁操作",
                keywords=["销毁", "清除", "wipe"]
            )
        ])

    def classify_risk(self, action_description, context=None):
        context = context or {}
        
        matched_rule = None
        for rule in self.risk_rules:
            if rule.matches(action_description):
                matched_rule = rule
                break

        if matched_rule:
            risk_level = matched_rule.risk_level
        else:
            risk_level = self._infer_risk_level(action_description, context)

        action = self._get_risk_action(risk_level)

        self.audit_log.append({
            "timestamp": time.time(),
            "action_description": str(action_description)[:200],
            "risk_level": risk_level.value,
            "recommended_action": action.value,
            "context": str(context)[:100]
        })

        return {
            "risk_level": risk_level.value,
            "recommended_action": action.value,
            "matched_rule": matched_rule.to_dict() if matched_rule else None,
            "context": context
        }

    def _infer_risk_level(self, action_description, context):
        desc = str(action_description).lower()
        
        dangerous_keywords = ["destroy", "delete all", "wipe", "format", "shutdown", "kill"]
        if any(k in desc for k in dangerous_keywords):
            return RiskLevel.DANGEROUS
        
        high_keywords = ["system", "sudo", "root", "admin", "execute", "run command"]
        if any(k in desc for k in high_keywords):
            return RiskLevel.HIGH
        
        medium_keywords = ["modify", "update", "write", "save", "delete", "remove"]
        if any(k in desc for k in medium_keywords):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW

    def _get_risk_action(self, risk_level):
        actions = {
            RiskLevel.LOW: RiskAction.EXECUTE,
            RiskLevel.MEDIUM: RiskAction.AUDIT,
            RiskLevel.HIGH: RiskAction.CONFIRM,
            RiskLevel.DANGEROUS: RiskAction.BLOCK
        }
        return actions[risk_level]

    def request_confirmation(self, action_id, action_description, risk_level):
        confirmation_id = f"conf_{int(time.time() * 1000)}_{np.random.randint(1000)}"
        self.pending_confirmations[confirmation_id] = {
            "confirmation_id": confirmation_id,
            "action_id": action_id,
            "action_description": action_description,
            "risk_level": risk_level,
            "status": "pending",
            "created_at": time.time()
        }
        return confirmation_id

    def confirm_action(self, confirmation_id):
        if confirmation_id in self.pending_confirmations:
            self.pending_confirmations[confirmation_id]["status"] = "confirmed"
            return {"confirmed": True, "confirmation_id": confirmation_id}
        return {"confirmed": False, "error": "Confirmation not found"}

    def reject_action(self, confirmation_id):
        if confirmation_id in self.pending_confirmations:
            self.pending_confirmations[confirmation_id]["status"] = "rejected"
            return {"rejected": True, "confirmation_id": confirmation_id}
        return {"rejected": False, "error": "Confirmation not found"}

    def get_pending_confirmations(self):
        return [c for c in self.pending_confirmations.values() if c["status"] == "pending"]

    def get_audit_log(self, limit=20):
        return list(self.audit_log)[-limit:]

    def get_risk_stats(self):
        risk_distribution = {}
        for log in self.audit_log:
            level = log["risk_level"]
            risk_distribution[level] = risk_distribution.get(level, 0) + 1

        return {
            "total_actions_classified": len(self.audit_log),
            "risk_distribution": risk_distribution,
            "pending_confirmations": len(self.get_pending_confirmations()),
            "rules_count": len(self.risk_rules)
        }

    def get_stats(self):
        return self.get_risk_stats()