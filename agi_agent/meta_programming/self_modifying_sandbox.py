import os
import sys
import time
import json
import hashlib
import traceback
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


class SandboxMode(Enum):
    READ_ONLY = "read_only"
    ANALYSIS_ONLY = "analysis_only"
    SIMULATION = "simulation"
    LIMITED_MODIFICATION = "limited_modification"
    FULL_MODIFICATION = "full_modification"


class PermissionLevel(Enum):
    DENY = 0
    READ = 1
    EXECUTE = 2
    WRITE = 3
    ADMIN = 4


class ModificationType(Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class AuditAction(Enum):
    ACCESS = "access"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    MODIFY = "modify"
    ROLLBACK = "rollback"
    APPROVAL = "approval"
    DENIED = "denied"


@dataclass
class ResourceQuota:
    max_memory_mb: int = 256
    max_cpu_percent: float = 50.0
    max_execution_time_sec: float = 30.0
    max_files: int = 100
    max_file_size_mb: int = 10
    max_network_bytes: int = 10 * 1024 * 1024
    max_processes: int = 5


@dataclass
class PermissionRule:
    resource_pattern: str
    permission: PermissionLevel
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)

    def matches(self, resource: str) -> bool:
        if self.resource_pattern == "*":
            return True
        if self.resource_pattern.endswith("*"):
            prefix = self.resource_pattern[:-1]
            return resource.startswith(prefix)
        if self.resource_pattern.startswith("*"):
            suffix = self.resource_pattern[1:]
            return resource.endswith(suffix)
        return resource == self.resource_pattern


@dataclass
class ModificationRequest:
    request_id: str
    type: ModificationType
    target_path: str
    content: str = ""
    old_content: str = ""
    reason: str = ""
    proposed_by: str = "system"
    timestamp: float = 0.0
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str = ""
    approval_time: float = 0.0
    rollback_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "type": self.type.value,
            "target_path": self.target_path,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "reason": self.reason,
            "proposed_by": self.proposed_by,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approval_time": self.approval_time,
            "rollback_id": self.rollback_id
        }


@dataclass
class VersionSnapshot:
    snapshot_id: str
    timestamp: float
    files: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_file_hash(self, file_path: str) -> str:
        content = self.files.get(file_path, "")
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AuditLogEntry:
    entry_id: str
    timestamp: float
    action: AuditAction
    resource: str
    user: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "action": self.action.value,
            "resource": self.resource,
            "user": self.user,
            "success": self.success,
            "details": self.details
        }


class SecurityBoundary:
    def __init__(self):
        self.forbidden_patterns: Set[str] = set()
        self.protected_files: Set[str] = set()
        self.trusted_domains: Set[str] = set()
        self.max_modification_size: int = 10000
        self.min_approval_level: PermissionLevel = PermissionLevel.WRITE

        self._init_default_boundaries()

    def _init_default_boundaries(self):
        self.forbidden_patterns.update([
            "/etc/*",
            "/bin/*",
            "/usr/*",
            "/system/*",
            "*.key",
            "*.pem",
            "*.crt",
            ".env",
            "*.sqlite",
        ])

        self.protected_files.update([
            "core/*.py",
            "security/*.py",
            "bootstrap/*.py",
        ])

        self.trusted_domains.update([
            "localhost",
            "127.0.0.1",
        ])

    def is_forbidden(self, path: str) -> bool:
        for pattern in self.forbidden_patterns:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if path.startswith(prefix) or "/" + path.startswith("/" + prefix):
                    return True
            elif path == pattern or "/" + path == "/" + pattern:
                return True
        return False

    def is_protected(self, path: str) -> bool:
        for pattern in self.protected_files:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if path.startswith(prefix):
                    return True
            elif path == pattern:
                return True
        return False

    def is_allowed_domain(self, domain: str) -> bool:
        return domain in self.trusted_domains or domain.startswith("127.") or domain.startswith("192.168.")

    def check_modification_size(self, content: str) -> bool:
        return len(content) <= self.max_modification_size


class PermissionManager:
    def __init__(self):
        self.rules: List[PermissionRule] = []
        self.user_permissions: Dict[str, PermissionLevel] = {}

        self._init_default_rules()

    def _init_default_rules(self):
        self.add_rule(PermissionRule("*", PermissionLevel.READ, "Default read access"))
        self.add_rule(PermissionRule("tests/*", PermissionLevel.WRITE, "Test files can be modified"))
        self.add_rule(PermissionRule("temp/*", PermissionLevel.WRITE, "Temporary files"))
        self.add_rule(PermissionRule("output/*", PermissionLevel.WRITE, "Output files"))
        self.add_rule(PermissionRule("core/*", PermissionLevel.READ, "Core files are read-only"))
        self.add_rule(PermissionRule("security/*", PermissionLevel.READ, "Security files are read-only"))
        self.add_rule(PermissionRule("*.py", PermissionLevel.EXECUTE, "Python files can be executed"))

        self.user_permissions["system"] = PermissionLevel.ADMIN
        self.user_permissions["analyzer"] = PermissionLevel.READ
        self.user_permissions["optimizer"] = PermissionLevel.WRITE
        self.user_permissions["tester"] = PermissionLevel.EXECUTE

    def add_rule(self, rule: PermissionRule):
        self.rules.append(rule)

    def get_effective_permission(self, resource: str, user: str = "system") -> PermissionLevel:
        user_level = self.user_permissions.get(user, PermissionLevel.READ)

        matched_rules = [r for r in self.rules if r.matches(resource)]
        if not matched_rules:
            return user_level

        max_permission = PermissionLevel.DENY
        for rule in matched_rules:
            if rule.permission.value > max_permission.value:
                max_permission = rule.permission

        if max_permission.value > user_level.value:
            return user_level

        return max_permission

    def can_read(self, resource: str, user: str = "system") -> bool:
        return self.get_effective_permission(resource, user).value >= PermissionLevel.READ.value

    def can_write(self, resource: str, user: str = "system") -> bool:
        return self.get_effective_permission(resource, user).value >= PermissionLevel.WRITE.value

    def can_execute(self, resource: str, user: str = "system") -> bool:
        return self.get_effective_permission(resource, user).value >= PermissionLevel.EXECUTE.value

    def check_permission(self, resource: str, required: PermissionLevel, user: str = "system") -> bool:
        return self.get_effective_permission(resource, user).value >= required.value


class VersionControl:
    def __init__(self, max_snapshots: int = 50):
        self.snapshots: List[VersionSnapshot] = []
        self.max_snapshots = max_snapshots
        self._lock = threading.Lock()

    def create_snapshot(self, files: Dict[str, str], metadata: Dict[str, Any] = None) -> str:
        snapshot_id = f"snapshot_{int(time.time() * 1000)}_{hashlib.md5(json.dumps(files).encode()).hexdigest()[:8]}"
        snapshot = VersionSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            files=files,
            metadata=metadata or {}
        )

        with self._lock:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[VersionSnapshot]:
        with self._lock:
            for snapshot in reversed(self.snapshots):
                if snapshot.snapshot_id == snapshot_id:
                    return snapshot
        return None

    def get_latest_snapshot(self) -> Optional[VersionSnapshot]:
        with self._lock:
            return self.snapshots[-1] if self.snapshots else None

    def rollback_to_snapshot(self, snapshot_id: str) -> Optional[Dict[str, str]]:
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot:
            return snapshot.files.copy()
        return None

    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{
                "snapshot_id": s.snapshot_id,
                "timestamp": s.timestamp,
                "file_count": len(s.files),
                "metadata": s.metadata
            } for s in self.snapshots]

    def diff_snapshots(self, from_id: str, to_id: str) -> Dict[str, Dict[str, str]]:
        from_snap = self.get_snapshot(from_id)
        to_snap = self.get_snapshot(to_id)

        if not from_snap or not to_snap:
            return {}

        diff = {}
        all_files = set(from_snap.files.keys()) | set(to_snap.files.keys())

        for file in all_files:
            from_content = from_snap.files.get(file, "")
            to_content = to_snap.files.get(file, "")

            if from_content != to_content:
                diff[file] = {
                    "from": from_content[:500] + "..." if len(from_content) > 500 else from_content,
                    "to": to_content[:500] + "..." if len(to_content) > 500 else to_content
                }

        return diff


class AuditLogger:
    def __init__(self, max_logs: int = 1000):
        self.logs: List[AuditLogEntry] = []
        self.max_logs = max_logs
        self._lock = threading.Lock()

    def log(self, action: AuditAction, resource: str, user: str, success: bool, details: Dict[str, Any] = None):
        entry = AuditLogEntry(
            entry_id=f"audit_{int(time.time() * 1000)}_{hashlib.md5(f'{action}{resource}{user}'.encode()).hexdigest()[:8]}",
            timestamp=time.time(),
            action=action,
            resource=resource,
            user=user,
            success=success,
            details=details or {}
        )

        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

    def get_logs(self, limit: int = 100, action: AuditAction = None, user: str = None) -> List[Dict[str, Any]]:
        with self._lock:
            filtered = self.logs
            if action:
                filtered = [l for l in filtered if l.action == action]
            if user:
                filtered = [l for l in filtered if l.user == user]
            return [l.to_dict() for l in filtered[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self.logs)
            by_action = {}
            by_user = {}
            denied_count = 0

            for log in self.logs:
                by_action[log.action.value] = by_action.get(log.action.value, 0) + 1
                by_user[log.user] = by_user.get(log.user, 0) + 1
                if log.action == AuditAction.DENIED:
                    denied_count += 1

            return {
                "total_logs": total,
                "by_action": by_action,
                "by_user": by_user,
                "denied_count": denied_count,
                "success_rate": (total - denied_count) / max(total, 1)
            }


class SelfModifyingSandbox:
    def __init__(self, mode: SandboxMode = SandboxMode.ANALYSIS_ONLY):
        self.mode = mode
        self.security_boundary = SecurityBoundary()
        self.permission_manager = PermissionManager()
        self.version_control = VersionControl()
        self.audit_logger = AuditLogger()
        self.resource_quota = ResourceQuota()
        self._lock = threading.Lock()
        self._execution_count = 0
        self._memory_usage = 0
        self._cpu_usage = 0.0

    def set_mode(self, mode: SandboxMode):
        self.audit_logger.log(
            AuditAction.MODIFY,
            "sandbox_mode",
            "system",
            True,
            {"old_mode": self.mode.value, "new_mode": mode.value}
        )
        self.mode = mode

    def check_access(self, resource: str, action: AuditAction, user: str = "system") -> bool:
        if self.security_boundary.is_forbidden(resource):
            self.audit_logger.log(action, resource, user, False, {"reason": "forbidden"})
            return False

        if self.mode == SandboxMode.READ_ONLY and action in (AuditAction.WRITE, AuditAction.MODIFY):
            self.audit_logger.log(action, resource, user, False, {"reason": "read_only_mode"})
            return False

        if self.mode == SandboxMode.ANALYSIS_ONLY and action in (AuditAction.WRITE, AuditAction.MODIFY):
            self.audit_logger.log(action, resource, user, False, {"reason": "analysis_only_mode"})
            return False

        if not self.permission_manager.check_permission(resource, self._action_to_permission(action), user):
            self.audit_logger.log(action, resource, user, False, {"reason": "permission_denied"})
            return False

        return True

    def _action_to_permission(self, action: AuditAction) -> PermissionLevel:
        mapping = {
            AuditAction.ACCESS: PermissionLevel.READ,
            AuditAction.READ: PermissionLevel.READ,
            AuditAction.EXECUTE: PermissionLevel.EXECUTE,
            AuditAction.WRITE: PermissionLevel.WRITE,
            AuditAction.MODIFY: PermissionLevel.WRITE,
            AuditAction.ROLLBACK: PermissionLevel.ADMIN,
        }
        return mapping.get(action, PermissionLevel.READ)

    def read_file(self, file_path: str, user: str = "system") -> Optional[str]:
        if not self.check_access(file_path, AuditAction.READ, user):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.audit_logger.log(AuditAction.READ, file_path, user, True, {"size": len(content)})
            return content
        except Exception as e:
            self.audit_logger.log(AuditAction.READ, file_path, user, False, {"error": str(e)})
            return None

    def write_file(self, file_path: str, content: str, user: str = "system", reason: str = "") -> bool:
        if not self.check_access(file_path, AuditAction.WRITE, user):
            return False

        if not self.security_boundary.check_modification_size(content):
            self.audit_logger.log(AuditAction.WRITE, file_path, user, False, {"reason": "size_exceeded"})
            return False

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.audit_logger.log(AuditAction.WRITE, file_path, user, True, {"size": len(content), "reason": reason})
            return True
        except Exception as e:
            self.audit_logger.log(AuditAction.WRITE, file_path, user, False, {"error": str(e)})
            return False

    def execute_code(self, code: str, user: str = "system") -> Dict[str, Any]:
        if not self.check_access("dynamic_code", AuditAction.EXECUTE, user):
            return {"success": False, "error": "Permission denied"}

        if self._execution_count >= self.resource_quota.max_processes:
            self.audit_logger.log(AuditAction.EXECUTE, "dynamic_code", user, False, {"reason": "max_processes_exceeded"})
            return {"success": False, "error": "Max processes exceeded"}

        self._execution_count += 1
        start_time = time.time()

        try:
            result = self._safe_execute(code)
            execution_time = time.time() - start_time

            if execution_time > self.resource_quota.max_execution_time_sec:
                self.audit_logger.log(AuditAction.EXECUTE, "dynamic_code", user, False, {"reason": "timeout"})
                return {"success": False, "error": "Execution timeout"}

            self.audit_logger.log(AuditAction.EXECUTE, "dynamic_code", user, True, {
                "execution_time": execution_time,
                "result_type": type(result).__name__
            })

            return {
                "success": True,
                "result": str(result),
                "execution_time": execution_time
            }

        except Exception as e:
            self.audit_logger.log(AuditAction.EXECUTE, "dynamic_code", user, False, {
                "error": str(e),
                "traceback": traceback.format_exc()[:500]
            })
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()[:500]
            }
        finally:
            self._execution_count -= 1

    def _safe_execute(self, code: str):
        safe_globals = {
            "__builtins__": {
                'abs': abs, 'all': all, 'any': any, 'bool': bool, 'float': float,
                'int': int, 'len': len, 'max': max, 'min': min, 'pow': pow,
                'range': range, 'round': round, 'sum': sum, 'str': str, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set, 'type': type, 'isinstance': isinstance,
                'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
                'reversed': reversed, 'sorted': sorted,
            },
            "__name__": "__sandbox__",
            "__doc__": "",
        }
        local_vars = {}
        exec(code, safe_globals, local_vars)
        return local_vars.get("_result")

    def propose_modification(self, modification_type: ModificationType, target_path: str,
                            content: str = "", old_content: str = "",
                            reason: str = "", proposed_by: str = "system") -> ModificationRequest:
        request = ModificationRequest(
            request_id=f"mod_{int(time.time() * 1000)}",
            type=modification_type,
            target_path=target_path,
            content=content,
            old_content=old_content,
            reason=reason,
            proposed_by=proposed_by,
            timestamp=time.time()
        )

        if self.security_boundary.is_forbidden(target_path):
            request.status = ApprovalStatus.REJECTED
            self.audit_logger.log(AuditAction.DENIED, target_path, proposed_by, False, {
                "reason": "forbidden_path",
                "request_id": request.request_id
            })
            return request

        if self.mode == SandboxMode.SIMULATION:
            request.status = ApprovalStatus.AUTO_APPROVED
            request.approved_by = "simulation"
            request.approval_time = time.time()
            self.audit_logger.log(AuditAction.APPROVAL, target_path, proposed_by, True, {
                "request_id": request.request_id,
                "reason": "simulation_mode"
            })
            return request

        if self.permission_manager.get_effective_permission(target_path, proposed_by) >= PermissionLevel.ADMIN:
            request.status = ApprovalStatus.AUTO_APPROVED
            request.approved_by = proposed_by
            request.approval_time = time.time()
            self.audit_logger.log(AuditAction.APPROVAL, target_path, proposed_by, True, {
                "request_id": request.request_id,
                "reason": "admin_auto_approval"
            })
            return request

        self.audit_logger.log(AuditAction.APPROVAL, target_path, proposed_by, False, {
            "request_id": request.request_id,
            "reason": "pending_approval"
        })
        return request

    def approve_modification(self, request_id: str, approver: str) -> bool:
        return False

    def execute_modification(self, request: ModificationRequest) -> bool:
        if request.status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
            self.audit_logger.log(AuditAction.DENIED, request.target_path, request.proposed_by, False, {
                "reason": "not_approved",
                "request_id": request.request_id
            })
            return False

        if not self.check_access(request.target_path, AuditAction.MODIFY, request.proposed_by):
            return False

        old_content = ""
        if os.path.exists(request.target_path):
            old_content = self.read_file(request.target_path, "system") or ""

        snapshot_id = self.version_control.create_snapshot({
            request.target_path: old_content
        }, {"request_id": request.request_id})
        request.rollback_id = snapshot_id

        try:
            if request.type == ModificationType.ADD or request.type == ModificationType.MODIFY:
                success = self.write_file(request.target_path, request.content, request.proposed_by, request.reason)
            elif request.type == ModificationType.DELETE:
                os.remove(request.target_path)
                success = True
            elif request.type == ModificationType.RENAME:
                os.rename(request.target_path, request.content)
                success = True
            else:
                success = False

            if success:
                self.audit_logger.log(AuditAction.MODIFY, request.target_path, request.proposed_by, True, {
                    "request_id": request.request_id,
                    "type": request.type.value,
                    "snapshot_id": snapshot_id
                })
            return success

        except Exception as e:
            self.audit_logger.log(AuditAction.MODIFY, request.target_path, request.proposed_by, False, {
                "request_id": request.request_id,
                "error": str(e),
                "rollback_id": snapshot_id
            })
            self.rollback_modification(snapshot_id, request.proposed_by)
            return False

    def rollback_modification(self, snapshot_id: str, user: str = "system") -> bool:
        if not self.check_access("rollback", AuditAction.ROLLBACK, user):
            return False

        files = self.version_control.rollback_to_snapshot(snapshot_id)
        if not files:
            self.audit_logger.log(AuditAction.ROLLBACK, snapshot_id, user, False, {"reason": "snapshot_not_found"})
            return False

        try:
            for file_path, content in files.items():
                if content == "":
                    if os.path.exists(file_path):
                        os.remove(file_path)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)

            self.audit_logger.log(AuditAction.ROLLBACK, snapshot_id, user, True, {"files_rolled_back": len(files)})
            return True
        except Exception as e:
            self.audit_logger.log(AuditAction.ROLLBACK, snapshot_id, user, False, {"error": str(e)})
            return False

    def get_sandbox_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "resource_quota": {
                "max_memory_mb": self.resource_quota.max_memory_mb,
                "max_cpu_percent": self.resource_quota.max_cpu_percent,
                "max_execution_time_sec": self.resource_quota.max_execution_time_sec,
                "max_processes": self.resource_quota.max_processes,
                "current_executions": self._execution_count
            },
            "security_boundary": {
                "forbidden_pattern_count": len(self.security_boundary.forbidden_patterns),
                "protected_file_count": len(self.security_boundary.protected_files),
            },
            "audit_stats": self.audit_logger.get_stats(),
            "version_snapshots": len(self.version_control.snapshots)
        }

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.audit_logger.get_logs(limit)

    def get_version_history(self) -> List[Dict[str, Any]]:
        return self.version_control.get_snapshot_history()