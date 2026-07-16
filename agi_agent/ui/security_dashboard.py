"""
ui/security_dashboard.py - 安全仪表板

实时监控安全状态、告警和风险评估
"""
import time
from typing import Dict, Any, Optional, List
from collections import deque


class SecurityDashboard:
    def __init__(self):
        self._security_event_bridge = None
        self._alert_history: deque = deque(maxlen=500)
        self._security_status_cache: Dict[str, Any] = {}
        self._last_update = 0.0

    def set_security_event_bridge(self, bridge):
        self._security_event_bridge = bridge

    def get_security_status(self) -> Dict[str, Any]:
        if self._security_event_bridge:
            status = self._security_event_bridge.get_security_status()
            self._security_status_cache = status
            self._last_update = time.time()
            return status
        return self._security_status_cache

    def get_active_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self._security_event_bridge:
            alerts = self._security_event_bridge.get_active_alerts(limit)
            self._alert_history.extend(alerts)
            return alerts
        return list(self._alert_history)[-limit:]

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._alert_history)[-limit:]

    def get_risk_distribution(self) -> Dict[str, Any]:
        if self._security_event_bridge:
            stats = self._security_event_bridge.risk_classifier.get_risk_stats()
            return {
                "total_actions": stats.get("total_actions_classified", 0),
                "risk_distribution": stats.get("risk_distribution", {}),
                "pending_confirmations": stats.get("pending_confirmations", 0),
            }
        return {}

    def get_safety_report(self) -> Dict[str, Any]:
        if self._security_event_bridge:
            return self._security_event_bridge.safety_monitor.get_safety_report()
        return {}

    def classify_risk(self, action_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._security_event_bridge:
            return self._security_event_bridge.classify_action_risk(action_description, context)
        return {
            "risk_level": "unknown",
            "requires_confirmation": False,
        }

    def scan_input(self, data: Any, input_type: str = "unknown") -> Dict[str, Any]:
        if self._security_event_bridge:
            return self._security_event_bridge.scan_input(data, input_type)
        return {"safe": True, "vulnerabilities": [], "malware_detection": None}

    def get_overview(self) -> Dict[str, Any]:
        status = self.get_security_status()
        alerts = self.get_active_alerts(5)
        risk_dist = self.get_risk_distribution()
        safety_report = self.get_safety_report()

        return {
            "current_risk_level": status.get("current_risk_level", "unknown"),
            "active_alerts_count": status.get("active_alerts_count", 0),
            "active_alerts": alerts,
            "risk_distribution": risk_dist.get("risk_distribution", {}),
            "total_actions_classified": risk_dist.get("total_actions", 0),
            "pending_confirmations": risk_dist.get("pending_confirmations", 0),
            "resource_usage": safety_report.get("current_resources", {}),
            "safety_rules": safety_report.get("safety_rules", {}),
            "last_update": self._last_update,
        }