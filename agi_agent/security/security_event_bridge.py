"""
security/security_event_bridge.py - 安全事件桥接器

实现安全模块与业务逻辑的实时联动：
- 安全事件订阅与分发
- 动态安全策略调整
- 上下文感知安全决策
- 安全告警与业务流程联动
"""
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import time
import logging

from ..core.event_bus import EventBus, get_event_bus, Event, EventCategory, EventSeverity
from .risk_classifier import RiskClassifier, RiskLevel, RiskAction
from .safety_monitor import SafetyMonitor
from .vulnerability_scanner import VulnerabilityScanner
from .malware_detector import MalwareDetector

logger = logging.getLogger("agi_agent.security")


class SecurityAlertType(Enum):
    VULNERABILITY_DETECTED = "vulnerability_detected"
    MALWARE_DETECTED = "malware_detected"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    SAFETY_VIOLATION = "safety_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"


class SecurityEventBridge:
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 risk_classifier: Optional[RiskClassifier] = None,
                 safety_monitor: Optional[SafetyMonitor] = None,
                 vulnerability_scanner: Optional[VulnerabilityScanner] = None,
                 malware_detector: Optional[MalwareDetector] = None):
        self.event_bus = event_bus or get_event_bus()
        self.risk_classifier = risk_classifier or RiskClassifier()
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.vulnerability_scanner = vulnerability_scanner
        self.malware_detector = malware_detector
        
        self._security_policies: Dict[str, Dict[str, Any]] = {}
        self._current_risk_level = "low"
        self._active_alerts: List[Dict[str, Any]] = []
        self._subscriptions = []
        
        self._init_security_policies()
        self._setup_event_subscriptions()
        
        logger.info("[SECURITY] SecurityEventBridge initialized")

    def _init_security_policies(self):
        self._security_policies = {
            "default": {
                "risk_threshold": "medium",
                "max_concurrent_actions": 10,
                "action_timeout_ms": 30000,
                "enable_vulnerability_scan": True,
                "enable_malware_detection": True,
                "require_confirmation": ["high", "dangerous"],
            },
            "high_risk": {
                "risk_threshold": "low",
                "max_concurrent_actions": 3,
                "action_timeout_ms": 10000,
                "enable_vulnerability_scan": True,
                "enable_malware_detection": True,
                "require_confirmation": ["medium", "high", "dangerous"],
            },
            "emergency": {
                "risk_threshold": "low",
                "max_concurrent_actions": 1,
                "action_timeout_ms": 5000,
                "enable_vulnerability_scan": True,
                "enable_malware_detection": True,
                "require_confirmation": ["low", "medium", "high", "dangerous"],
            },
        }

    def _setup_event_subscriptions(self):
        self._subscriptions.append(
            self.event_bus.subscribe("agent.action.requested", self._handle_action_request)
        )
        self._subscriptions.append(
            self.event_bus.subscribe("agent.action.executed", self._handle_action_executed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe("cognitive.decision.made", self._handle_cognitive_decision)
        )
        self._subscriptions.append(
            self.event_bus.subscribe("perception.input.received", self._handle_perception_input)
        )
        self._subscriptions.append(
            self.event_bus.subscribe("security.*", self._handle_security_event)
        )
        self._subscriptions.append(
            self.event_bus.subscribe("system.resource.usage", self._handle_resource_usage)
        )

    def _handle_action_request(self, event: Event):
        action_description = event.payload.get("action_description", "")
        context = event.payload.get("context", {})
        
        risk_result = self.risk_classifier.classify_risk(action_description, context)
        risk_level = risk_result.get("risk_level")
        recommended_action = risk_result.get("recommended_action")
        
        policy = self._get_current_policy()
        needs_confirmation = risk_level in policy.get("require_confirmation", [])
        
        if needs_confirmation:
            self.publish_security_event(
                SecurityAlertType.RISK_LEVEL_CHANGED,
                {
                    "risk_level": risk_level,
                    "action_description": action_description,
                    "needs_confirmation": True,
                    "recommended_action": recommended_action,
                },
                EventSeverity.WARNING
            )
            
            self.event_bus.publish(
                "security.action.requires_confirmation",
                source="security_event_bridge",
                payload={
                    "action_id": event.payload.get("action_id"),
                    "action_description": action_description,
                    "risk_level": risk_level,
                    "recommended_action": recommended_action,
                },
                severity=EventSeverity.WARNING,
                category=EventCategory.SECURITY,
            )
        else:
            self.event_bus.publish(
                "security.action.approved",
                source="security_event_bridge",
                payload={
                    "action_id": event.payload.get("action_id"),
                    "action_description": action_description,
                    "risk_level": risk_level,
                },
                severity=EventSeverity.INFO,
                category=EventCategory.SECURITY,
            )

    def _handle_action_executed(self, event: Event):
        action_description = event.payload.get("action_description", "")
        result = event.payload.get("result", "success")
        
        if result == "failed":
            self.risk_classifier.classify_risk(f"failed_action: {action_description}")

    def _handle_cognitive_decision(self, event: Event):
        decision = event.payload.get("decision", {})
        confidence = event.payload.get("confidence", 0.5)
        
        if confidence < 0.3:
            self.publish_security_event(
                SecurityAlertType.RISK_LEVEL_CHANGED,
                {
                    "reason": "low_confidence_decision",
                    "confidence": confidence,
                    "decision": decision,
                },
                EventSeverity.WARNING,
            )

    def _handle_perception_input(self, event: Event):
        input_type = event.payload.get("input_type", "")
        data = event.payload.get("data", "")
        
        if self.vulnerability_scanner:
            scan_result = self.vulnerability_scanner.scan_content(str(data))
            if scan_result and scan_result.get("vulnerabilities"):
                self.publish_security_event(
                    SecurityAlertType.VULNERABILITY_DETECTED,
                    {
                        "input_type": input_type,
                        "vulnerabilities": scan_result.get("vulnerabilities"),
                    },
                    EventSeverity.ERROR,
                )
        
        if self.malware_detector:
            detection = self.malware_detector.detect(data)
            if detection and detection.get("threat_level") in ["high", "critical"]:
                self.publish_security_event(
                    SecurityAlertType.MALWARE_DETECTED,
                    {
                        "input_type": input_type,
                        "threat_level": detection.get("threat_level"),
                        "malware_type": detection.get("malware_type"),
                    },
                    EventSeverity.CRITICAL,
                )

    def _handle_security_event(self, event: Event):
        alert_type = event.event_type.replace("security.", "")
        self._add_alert(alert_type, event.payload, event.severity)
        
        if event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
            self._update_risk_level("high")

    def _handle_resource_usage(self, event: Event):
        resources = event.payload.get("resources", {})
        free_energy = event.payload.get("free_energy", 0.0)
        latency_ms = event.payload.get("latency_ms", 0.0)
        
        violations = self.safety_monitor.check_safety_constraints(free_energy, latency_ms)
        
        if violations:
            for violation in violations:
                self.publish_security_event(
                    SecurityAlertType.SAFETY_VIOLATION,
                    {
                        "type": violation.get("type"),
                        "value": violation.get("value"),
                        "threshold": violation.get("threshold"),
                        "severity": violation.get("severity"),
                    },
                    self._map_severity(violation.get("severity", "info")),
                )

            risk_level = self.safety_monitor.assess_risk_level()
            self._update_risk_level(risk_level)

    def _get_current_policy(self) -> Dict[str, Any]:
        return self._security_policies.get(self._current_risk_level, self._security_policies["default"])

    def _update_risk_level(self, level: str):
        old_level = self._current_risk_level
        self._current_risk_level = level
        
        if old_level != level:
            self.publish_security_event(
                SecurityAlertType.RISK_LEVEL_CHANGED,
                {
                    "old_level": old_level,
                    "new_level": level,
                },
                EventSeverity.WARNING,
            )
            
            self.event_bus.publish(
                "security.risk_level.changed",
                source="security_event_bridge",
                payload={
                    "old_level": old_level,
                    "new_level": level,
                    "policy": self._get_current_policy(),
                },
                severity=EventSeverity.WARNING,
                category=EventCategory.SECURITY,
            )

    def _add_alert(self, alert_type: str, payload: Dict[str, Any], severity: EventSeverity):
        alert = {
            "alert_type": alert_type,
            "payload": payload,
            "severity": severity.value,
            "timestamp": time.time(),
            "alert_id": f"alert_{int(time.time() * 1000)}",
        }
        self._active_alerts.append(alert)
        
        if len(self._active_alerts) > 100:
            self._active_alerts = self._active_alerts[-100:]

    def _map_severity(self, severity_str: str) -> EventSeverity:
        mapping = {
            "critical": EventSeverity.CRITICAL,
            "error": EventSeverity.ERROR,
            "warning": EventSeverity.WARNING,
            "info": EventSeverity.INFO,
            "debug": EventSeverity.DEBUG,
        }
        return mapping.get(severity_str, EventSeverity.INFO)

    def publish_security_event(self, alert_type: SecurityAlertType, 
                               payload: Dict[str, Any], 
                               severity: EventSeverity = EventSeverity.INFO):
        self.event_bus.publish(
            f"security.{alert_type.value}",
            source="security_event_bridge",
            payload=payload,
            severity=severity,
            category=EventCategory.SECURITY,
        )

    def classify_action_risk(self, action_description: str, 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = self.risk_classifier.classify_risk(action_description, context)
        policy = self._get_current_policy()
        result["requires_confirmation"] = result.get("risk_level") in policy.get("require_confirmation", [])
        result["current_policy"] = policy
        return result

    def assess_safety_constraints(self, free_energy: float, latency_ms: float) -> Dict[str, Any]:
        violations = self.safety_monitor.check_safety_constraints(free_energy, latency_ms)
        risk_level = self.safety_monitor.assess_risk_level()
        
        return {
            "violations": violations,
            "risk_level": risk_level,
            "current_resources": self.safety_monitor.check_resource_usage(),
        }

    def scan_input(self, data: Any, input_type: str = "unknown") -> Dict[str, Any]:
        result = {
            "vulnerabilities": [],
            "malware_detection": None,
            "safe": True,
        }
        
        if self.vulnerability_scanner:
            scan_result = self.vulnerability_scanner.scan_content(str(data))
            if scan_result and scan_result.get("vulnerabilities"):
                result["vulnerabilities"] = scan_result.get("vulnerabilities")
                result["safe"] = False
        
        if self.malware_detector:
            detection = self.malware_detector.detect(data)
            result["malware_detection"] = detection
            if detection and detection.get("threat_level") in ["high", "critical"]:
                result["safe"] = False
        
        return result

    def get_active_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._active_alerts[-limit:]

    def get_security_status(self) -> Dict[str, Any]:
        return {
            "current_risk_level": self._current_risk_level,
            "active_alerts_count": len(self._active_alerts),
            "current_policy": self._get_current_policy(),
            "safety_report": self.safety_monitor.get_safety_report(),
            "risk_stats": self.risk_classifier.get_risk_stats(),
        }

    def shutdown(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()
        logger.info("[SECURITY] SecurityEventBridge shutdown")


def get_security_event_bridge(event_bus: Optional[EventBus] = None) -> SecurityEventBridge:
    return SecurityEventBridge(event_bus=event_bus)