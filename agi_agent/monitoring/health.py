"""
monitoring/health.py - 健康检查模块

提供应用健康状态检查、指标收集等功能
"""
import os
import time
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass

from agi_agent.core import BaseModule


@dataclass
class HealthStatus:
    """健康状态"""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, Any]
    timestamp: float
    uptime: float


class HealthChecker(BaseModule):
    """健康检查器"""

    name = "health_checker"
    version = "1.0.0"
    description = "系统健康状态检查"

    def __init__(self):
        super().__init__()
        self._start_time = time.time()

    def health_check(self) -> HealthStatus:
        """执行全面健康检查"""
        checks = {}
        status = "healthy"

        checks["system"] = self._check_system()
        if checks["system"]["ok"] is False:
            status = "degraded"

        checks["memory"] = self._check_memory()
        if checks["memory"]["ok"] is False:
            status = "degraded"

        checks["storage"] = self._check_storage()
        if checks["storage"]["ok"] is False:
            status = "degraded"

        return HealthStatus(
            status=status,
            checks=checks,
            timestamp=time.time(),
            uptime=self._start_time,
        )

    def _check_system(self) -> Dict[str, Any]:
        """检查系统状态"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_cores = psutil.cpu_count(logical=True) or 1

        return {
            "ok": cpu_percent < 95,
            "cpu_percent": cpu_percent,
            "cpu_cores": cpu_cores,
            "load_avg": list(os.getloadavg()) if hasattr(os, 'getloadavg') else [],
        }

    def _check_memory(self) -> Dict[str, Any]:
        """检查内存使用"""
        mem = psutil.virtual_memory()
        return {
            "ok": mem.percent < 90,
            "percent": mem.percent,
            "available": mem.available,
            "total": mem.total,
        }

    def _check_storage(self) -> Dict[str, Any]:
        """检查存储状态"""
        disk = psutil.disk_usage('/')
        return {
            "ok": disk.percent < 90,
            "percent": disk.percent,
            "free": disk.free,
            "total": disk.total,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标（Prometheus 格式）"""
        health = self.health_check()
        return {
            "uptime_seconds": int(time.time() - self._start_time),
            "cpu_percent": health.checks["system"]["cpu_percent"],
            "memory_percent": health.checks["memory"]["percent"],
            "storage_percent": health.checks["storage"]["percent"],
            "health_status": 1 if health.status == "healthy" else 0,
        }

    def format_prometheus(self) -> str:
        """格式化指标为 Prometheus 文本格式"""
        metrics = self.get_metrics()
        lines = []
        lines.append(f"# HELP agi_agent_uptime_seconds Agent uptime in seconds")
        lines.append(f"# TYPE agi_agent_uptime_seconds gauge")
        lines.append(f"agi_agent_uptime_seconds {metrics['uptime_seconds']}")
        lines.append(f"# HELP agi_agent_cpu_percent CPU usage percentage")
        lines.append(f"# TYPE agi_agent_cpu_percent gauge")
        lines.append(f"agi_agent_cpu_percent {metrics['cpu_percent']}")
        lines.append(f"# HELP agi_agent_memory_percent Memory usage percentage")
        lines.append(f"# TYPE agi_agent_memory_percent gauge")
        lines.append(f"agi_agent_memory_percent {metrics['memory_percent']}")
        lines.append(f"# HELP agi_agent_storage_percent Storage usage percentage")
        lines.append(f"# TYPE agi_agent_storage_percent gauge")
        lines.append(f"agi_agent_storage_percent {metrics['storage_percent']}")
        lines.append(f"# HELP agi_agent_health_status Health status (1=healthy, 0=degraded)")
        lines.append(f"# TYPE agi_agent_health_status gauge")
        lines.append(f"agi_agent_health_status {metrics['health_status']}")
        return "\n".join(lines) + "\n"
