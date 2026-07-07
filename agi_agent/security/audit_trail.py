"""
audit_trail.py - 思考审计追溯系统

所有自主思考过程全链路留痕，可回溯每一步假设、推演、决策依据，
支持事后审计与问题定位。
"""
import numpy as np
import time
from collections import deque
from enum import Enum


class AuditCategory(Enum):
    THINKING_START = "thinking_start"
    PROBLEM_FORMULATION = "problem_formulation"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LOGICAL_DEDUCTION = "logical_deduction"
    SIMULATION = "simulation"
    DECISION = "decision"
    ACTION_EXECUTION = "action_execution"
    REFLECTION = "reflection"
    EVOLUTION = "evolution"
    SAFETY_CHECK = "safety_check"


class AuditEntry:
    def __init__(self, entry_id, category, thinking_id, details, timestamp=None):
        self.entry_id = entry_id
        self.category = category
        self.thinking_id = thinking_id
        self.details = details
        self.timestamp = timestamp or time.time()
        self.validation_status = "pending"
        self.validation_details = None

    def validate(self, status, details=None):
        self.validation_status = status
        self.validation_details = details

    def to_dict(self):
        return {
            "entry_id": self.entry_id,
            "category": self.category.value,
            "thinking_id": self.thinking_id,
            "details": self.details,
            "timestamp": self.timestamp,
            "validation_status": self.validation_status,
            "validation_details": self.validation_details
        }


class AuditTrail:
    def __init__(self):
        self.audit_entries = deque(maxlen=1000)
        self.thinking_sessions = {}
        self.entry_counter = 0

    def _generate_entry_id(self):
        self.entry_counter += 1
        return f"audit_{self.entry_counter:06d}"

    def start_thinking_session(self, thinking_id, context=None):
        self.thinking_sessions[thinking_id] = {
            "thinking_id": thinking_id,
            "start_time": time.time(),
            "end_time": None,
            "status": "in_progress",
            "entries": [],
            "context": context or {}
        }

        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            category=AuditCategory.THINKING_START,
            thinking_id=thinking_id,
            details={"context": context}
        )
        self.audit_entries.append(entry)
        self.thinking_sessions[thinking_id]["entries"].append(entry.entry_id)

        return entry

    def record_audit_entry(self, thinking_id, category, details):
        if thinking_id not in self.thinking_sessions:
            self.start_thinking_session(thinking_id)

        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            category=category,
            thinking_id=thinking_id,
            details=details
        )
        self.audit_entries.append(entry)
        self.thinking_sessions[thinking_id]["entries"].append(entry.entry_id)

        return entry

    def end_thinking_session(self, thinking_id, status, outcome):
        if thinking_id not in self.thinking_sessions:
            return None

        self.thinking_sessions[thinking_id]["end_time"] = time.time()
        self.thinking_sessions[thinking_id]["status"] = status
        self.thinking_sessions[thinking_id]["outcome"] = outcome

        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            category=AuditCategory.REFLECTION,
            thinking_id=thinking_id,
            details={"status": status, "outcome": outcome}
        )
        entry.validate(status)
        self.audit_entries.append(entry)
        self.thinking_sessions[thinking_id]["entries"].append(entry.entry_id)

        return entry

    def get_thinking_session(self, thinking_id):
        if thinking_id not in self.thinking_sessions:
            return None

        session = self.thinking_sessions[thinking_id].copy()
        session["entries"] = []
        for entry_id in self.thinking_sessions[thinking_id]["entries"]:
            entry = self._find_entry(entry_id)
            if entry:
                session["entries"].append(entry.to_dict())

        return session

    def _find_entry(self, entry_id):
        for entry in self.audit_entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def search_entries(self, thinking_id=None, category=None, start_time=None, end_time=None):
        results = []
        
        for entry in self.audit_entries:
            match = True
            
            if thinking_id and entry.thinking_id != thinking_id:
                match = False
            if category and entry.category != category:
                match = False
            if start_time and entry.timestamp < start_time:
                match = False
            if end_time and entry.timestamp > end_time:
                match = False
            
            if match:
                results.append(entry.to_dict())
        
        return results

    def validate_entries(self, thinking_id):
        session = self.get_thinking_session(thinking_id)
        if not session:
            return {"validated": False, "error": "Session not found"}

        entries = session.get("entries", [])
        validated_count = 0
        issues = []

        for entry in entries:
            if entry["category"] == AuditCategory.DECISION.value:
                if "decision_reason" not in entry["details"]:
                    issues.append(f"Decision entry {entry['entry_id']} missing reason")
                else:
                    validated_count += 1
            elif entry["category"] == AuditCategory.SAFETY_CHECK.value:
                if not entry["details"].get("passed", False):
                    issues.append(f"Safety check {entry['entry_id']} failed")
                else:
                    validated_count += 1
            else:
                validated_count += 1

        return {
            "validated": len(issues) == 0,
            "validated_entries": validated_count,
            "total_entries": len(entries),
            "issues": issues,
            "thinking_id": thinking_id
        }

    def get_audit_summary(self):
        category_counts = {}
        status_counts = {}
        
        for entry in self.audit_entries:
            category_counts[entry.category.value] = category_counts.get(entry.category.value, 0) + 1
            status_counts[entry.validation_status] = status_counts.get(entry.validation_status, 0) + 1

        session_status = {}
        for session in self.thinking_sessions.values():
            status = session["status"]
            session_status[status] = session_status.get(status, 0) + 1

        return {
            "total_entries": len(self.audit_entries),
            "total_sessions": len(self.thinking_sessions),
            "entries_by_category": category_counts,
            "entries_by_status": status_counts,
            "sessions_by_status": session_status
        }

    def export_audit(self, thinking_id=None):
        if thinking_id:
            session = self.get_thinking_session(thinking_id)
            return {"type": "single_session", "thinking_id": thinking_id, "data": session}
        
        return {
            "type": "full_export",
            "summary": self.get_audit_summary(),
            "recent_entries": [e.to_dict() for e in list(self.audit_entries)[-50:]]
        }

    def log_entry(self, category, action, details):
        thinking_id = f"session_{self.entry_counter}"
        if thinking_id not in self.thinking_sessions:
            self.start_thinking_session(thinking_id)
        
        audit_category = None
        for cat in AuditCategory:
            if cat.value.lower() == category.lower():
                audit_category = cat
                break
        
        if audit_category is None:
            audit_category = AuditCategory.THINKING_START
        
        self.record_audit_entry(thinking_id, audit_category, {
            "action": action,
            **details
        })

    def get_recent_entries(self, limit=20):
        return [e.to_dict() for e in list(self.audit_entries)[-limit:]]