"""
security/audit_logger.py - 安全审计日志系统

记录所有安全相关事件，包括认证、授权、配置变更、安全边界违规等。
"""
import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .models import SecurityStore, get_security_store


class AuditSeverity(Enum):
    """审计日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditEventType(Enum):
    """审计事件类型"""
    AUTH_LOGIN = "auth.login"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_PASSWORD_CHANGE = "auth.password.change"
    AUTH_PASSWORD_RESET = "auth.password.reset"

    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ROLE_CHANGED = "user.role.changed"
    USER_DEACTIVATED = "user.deactivated"

    PERMISSION_DENIED = "permission.denied"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    SECURITY_BOUNDARY_VIOLATION = "security.boundary.violation"
    SECURITY_CONFIG_CHANGED = "security.config.changed"

    SYSTEM_CONFIG_CHANGED = "system.config.changed"

    DATA_EXPORT = "data.export"
    DATA_DELETE = "data.delete"
    DATA_IMPORT = "data.import"

    FILE_UPLOAD = "file.upload"
    FILE_DOWNLOAD = "file.download"
    FILE_DELETE = "file.delete"

    PLUGIN_INSTALLED = "plugin.installed"
    PLUGIN_REMOVED = "plugin.removed"
    PLUGIN_ENABLED = "plugin.enabled"
    PLUGIN_DISABLED = "plugin.disabled"

    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_STEPPED = "agent.stepped"

    EVOLUTION_RUN = "evolution.run"
    EVOLUTION_PROPOSAL_APPROVED = "evolution.proposal.approved"


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_dir: Optional[str] = None, max_file_size: int = 10 * 1024 * 1024):
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "data", "audit_logs")
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        os.makedirs(log_dir, exist_ok=True)

    def _get_log_file(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"audit_{date_str}.log")

    def _rotate_if_needed(self, filepath: str):
        if not os.path.exists(filepath):
            return
        if os.path.getsize(filepath) >= self.max_file_size:
            ts = int(time.time())
            rotated = filepath.replace(".log", f"_{ts}.log")
            os.rename(filepath, rotated)

    def log(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        记录审计日志

        Args:
            event_type: 事件类型
            severity: 严重级别
            user_id: 用户 ID
            username: 用户名
            ip_address: IP 地址
            user_agent: 用户代理
            resource: 访问资源
            method: 请求方法
            status_code: 状态码
            duration_ms: 耗时（毫秒）
            details: 详细信息
            request_id: 请求 ID

        Returns:
            事件 ID
        """
        event_id = str(uuid.uuid4())
        entry = {
            "event_id": event_id,
            "event_type": event_type.value,
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timestamp_unix": time.time(),
            "user_id": user_id,
            "username": username,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "resource": resource,
            "method": method,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "details": details or {},
            "request_id": request_id,
        }

        try:
            filepath = self._get_log_file()
            self._rotate_if_needed(filepath)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, IOError):
            pass

        return event_id

    def query(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        查询审计日志

        Args:
            event_type: 事件类型过滤
            user_id: 用户 ID 过滤
            severity: 级别过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            审计日志条目列表
        """
        results: List[Dict[str, Any]] = []
        count = 0
        skipped = 0

        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith("audit_") and f.endswith(".log")],
            reverse=True,
        )

        for filename in log_files:
            filepath = os.path.join(self.log_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if user_id and entry.get("user_id") != user_id:
                            continue
                        if severity and entry.get("severity") != severity:
                            continue
                        if start_time and entry.get("timestamp_unix", 0) < start_time:
                            continue
                        if end_time and entry.get("timestamp_unix", 0) > end_time:
                            continue

                        if skipped < offset:
                            skipped += 1
                            continue

                        results.append(entry)
                        count += 1

                        if count >= limit:
                            return results
            except (OSError, IOError):
                continue

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取审计统计信息"""
        stats = {
            "total_events": 0,
            "by_severity": {s.value: 0 for s in AuditSeverity},
            "by_type": {},
            "today_events": 0,
            "log_files": 0,
        }

        today = datetime.now().strftime("%Y%m%d")

        try:
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith(".log")]
            stats["log_files"] = len(log_files)

            for filename in log_files:
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            stats["total_events"] += 1

                            sev = entry.get("severity")
                            if sev in stats["by_severity"]:
                                stats["by_severity"][sev] += 1

                            etype = entry.get("event_type")
                            if etype:
                                stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1

                            if today in filename:
                                stats["today_events"] += 1
                except (OSError, IOError):
                    continue
        except OSError:
            pass

        return stats


_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger
