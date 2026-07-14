"""
heartbeat_scheduler.py - 心跳调度器

内置主动心跳轮询引擎，无需用户触发，智能体可按配置的周期自动执行数据监控、
系统巡检、数据同步、内容更新等周期性任务。
支持 Cron 表达式与自然语言配置调度规则，可设置执行窗口、重试策略与异常告警。
"""
import time
import uuid
import threading
import re
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field


class ScheduleType(Enum):
    """调度类型"""
    INTERVAL = "interval"
    CRON = "cron"
    NATURAL_LANGUAGE = "natural_language"


@dataclass
class ScheduledTask:
    """定时任务"""
    task_id: str
    name: str
    executor: Callable
    schedule_type: ScheduleType
    schedule_config: str
    enabled: bool = True
    last_run_at: Optional[float] = None
    next_run_at: Optional[float] = None
    run_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "schedule_config": self.schedule_config,
            "enabled": self.enabled,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "last_error": self.last_error
        }


class HeartbeatScheduler:
    """心跳调度器"""

    def __init__(self, tick_interval: float = 1.0):
        """
        初始化心跳调度器

        Args:
            tick_interval: 心跳间隔（秒）
        """
        self._tasks: Dict[str, ScheduledTask] = {}
        self._lock = threading.RLock()
        self._running = False
        self._tick_interval = tick_interval
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

    def add_task(self, name: str, executor: Callable, schedule_config: str,
                 schedule_type: ScheduleType = ScheduleType.INTERVAL,
                 max_retries: int = 3, timeout: float = 300.0,
                 metadata: Dict[str, Any] = None) -> ScheduledTask:
        """
        添加定时任务

        Args:
            name: 任务名称
            executor: 执行函数
            schedule_config: 调度配置
            schedule_type: 调度类型
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            metadata: 元数据

        Returns:
            定时任务对象
        """
        task_id = f"hb_{uuid.uuid4().hex[:8]}"

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            executor=executor,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {}
        )

        task.next_run_at = self._calculate_next_run(task)

        with self._lock:
            self._tasks[task_id] = task

        if self._running:
            self._notify("task_added", task.to_dict())

        return task

    def _calculate_next_run(self, task: ScheduledTask) -> float:
        """计算下次运行时间"""
        now = time.time()

        if task.schedule_type == ScheduleType.INTERVAL:
            try:
                interval = float(task.schedule_config)
                return now + interval
            except ValueError:
                return now + 60

        if task.schedule_type == ScheduleType.CRON:
            return self._parse_cron(task.schedule_config, now)

        if task.schedule_type == ScheduleType.NATURAL_LANGUAGE:
            return self._parse_natural_language(task.schedule_config, now)

        return now + 60

    def _parse_cron(self, cron_expr: str, now: float) -> float:
        """解析 Cron 表达式"""
        parts = cron_expr.split()
        if len(parts) < 5:
            return now + 60

        try:
            minute = parts[0]
            hour = parts[1]
            day = parts[2]
            month = parts[3]
            weekday = parts[4]

            from datetime import datetime, timedelta
            dt = datetime.fromtimestamp(now)

            if minute != "*":
                dt = dt.replace(minute=int(minute), second=0)
                if dt.timestamp() <= now:
                    dt += timedelta(hours=1)
            else:
                dt = dt.replace(second=0) + timedelta(minutes=1)

            return dt.timestamp()
        except Exception:
            return now + 60

    def _parse_natural_language(self, config: str, now: float) -> float:
        """解析自然语言调度配置"""
        config = config.lower()

        if "minute" in config or "分钟" in config:
            match = re.search(r'(\d+)\s*(minute|分钟)', config)
            if match:
                return now + int(match.group(1)) * 60
            return now + 60

        if "hour" in config or "小时" in config:
            match = re.search(r'(\d+)\s*(hour|小时)', config)
            if match:
                return now + int(match.group(1)) * 3600
            return now + 3600

        if "day" in config or "天" in config:
            match = re.search(r'(\d+)\s*(day|天)', config)
            if match:
                return now + int(match.group(1)) * 86400
            return now + 86400

        if "week" in config or "周" in config:
            match = re.search(r'(\d+)\s*(week|周)', config)
            if match:
                return now + int(match.group(1)) * 604800
            return now + 604800

        return now + 60

    def remove_task(self, task_id: str) -> bool:
        """移除定时任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._notify("task_removed", {"task_id": task_id})
                return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """启用任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.enabled = True
                task.next_run_at = self._calculate_next_run(task)
                self._notify("task_enabled", task.to_dict())
                return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """禁用任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.enabled = False
                self._notify("task_disabled", task.to_dict())
                return True
        return False

    def start(self):
        """启动调度器"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop)
        self._thread.daemon = True
        self._thread.start()

    def _run_loop(self):
        """调度循环"""
        while self._running:
            now = time.time()

            with self._lock:
                tasks_to_run = [
                    task for task in self._tasks.values()
                    if task.enabled and task.next_run_at and task.next_run_at <= now
                ]

            for task in tasks_to_run:
                self._execute_task(task)

            time.sleep(self._tick_interval)

    def _execute_task(self, task: ScheduledTask):
        """执行定时任务"""
        task.last_run_at = time.time()
        task.run_count += 1

        try:
            task.executor(task)
            task.success_count += 1
            task.retry_count = 0
            self._notify("task_success", task.to_dict())
        except Exception as e:
            task.fail_count += 1
            task.last_error = str(e)
            self._notify("task_failure", task.to_dict())

            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.next_run_at = time.time() + 60 * task.retry_count
                return

        task.next_run_at = self._calculate_next_run(task)

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[ScheduledTask]:
        """列出所有任务"""
        return list(self._tasks.values())

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def on_event(self, callback: Callable):
        """注册事件回调"""
        self._callbacks.append(callback)

    def _notify(self, event_type: str, task_data: Dict[str, Any]):
        """触发事件"""
        for cb in self._callbacks:
            try:
                cb(event_type, task_data)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            enabled_count = sum(1 for t in self._tasks.values() if t.enabled)
            disabled_count = len(self._tasks) - enabled_count
            total_runs = sum(t.run_count for t in self._tasks.values())
            total_success = sum(t.success_count for t in self._tasks.values())
            total_fail = sum(t.fail_count for t in self._tasks.values())

            return {
                "total_tasks": len(self._tasks),
                "enabled_tasks": enabled_count,
                "disabled_tasks": disabled_count,
                "total_runs": total_runs,
                "total_success": total_success,
                "total_fail": total_fail,
                "success_rate": total_success / total_runs if total_runs > 0 else 0.0
            }
