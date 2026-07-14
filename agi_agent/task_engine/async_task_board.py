"""
async_task_board.py - 异步任务看板

支持长周期、多依赖的复杂任务后台运行，解放前端对话窗口，实现无人值守执行。
"""
import time
import uuid
import threading
import json
import sqlite3
import os
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque
from dataclasses import dataclass, field


class AsyncTaskStatus(Enum):
    """异步任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class AsyncTask:
    """异步任务"""
    task_id: str
    name: str
    description: str = ""
    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    executor: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
            "logs": self.logs[-20:]
        }

    def add_log(self, message: str):
        """添加日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]


class AsyncTaskBoard:
    """异步任务看板"""

    def __init__(self, db_path: str = None):
        """
        初始化任务看板

        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = os.path.join(os.path.expanduser("~"), ".agi_tasks", "tasks.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._db_path = db_path
        self._tasks: Dict[str, AsyncTask] = {}
        self._task_queue = deque()
        self._lock = threading.RLock()
        self._workers: Dict[str, threading.Thread] = {}
        self._notifications: List[Callable] = []

        self._init_db()
        self._load_tasks()

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                progress REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                error TEXT,
                result TEXT,
                metadata TEXT,
                logs TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _load_tasks(self):
        """从数据库加载任务"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tasks')
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            task = AsyncTask(
                task_id=row[0],
                name=row[1],
                description=row[2] or "",
                status=AsyncTaskStatus(row[3]),
                priority=TaskPriority(row[4]),
                progress=row[5],
                created_at=row[6],
                started_at=row[7],
                completed_at=row[8],
                error=row[9],
                result=json.loads(row[10]) if row[10] else None,
                metadata=json.loads(row[11]) if row[11] else {},
                logs=json.loads(row[12]) if row[12] else []
            )
            self._tasks[task.task_id] = task
            if task.status == AsyncTaskStatus.PENDING:
                self._task_queue.append(task.task_id)

    def _save_task(self, task: AsyncTask):
        """保存任务到数据库"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO tasks
            (task_id, name, description, status, priority, progress,
             created_at, started_at, completed_at, error, result, metadata, logs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id,
            task.name,
            task.description,
            task.status.value,
            task.priority.value,
            task.progress,
            task.created_at,
            task.started_at,
            task.completed_at,
            task.error,
            json.dumps(task.result) if task.result else None,
            json.dumps(task.metadata) if task.metadata else None,
            json.dumps(task.logs) if task.logs else None
        ))
        conn.commit()
        conn.close()

    def submit_task(self, name: str, executor: Callable = None, description: str = "",
                    priority: TaskPriority = TaskPriority.NORMAL,
                    metadata: Dict[str, Any] = None) -> AsyncTask:
        """
        提交异步任务

        Args:
            name: 任务名称
            executor: 执行函数
            description: 任务描述
            priority: 优先级
            metadata: 元数据

        Returns:
            异步任务对象
        """
        task_id = f"async_{uuid.uuid4().hex[:8]}"
        task = AsyncTask(
            task_id=task_id,
            name=name,
            description=description,
            priority=priority,
            executor=executor,
            metadata=metadata or {}
        )

        with self._lock:
            self._tasks[task_id] = task
            self._task_queue.append(task_id)
            self._save_task(task)

        self._notify("task_created", task.to_dict())
        self._start_worker()

        return task

    def _start_worker(self):
        """启动工作线程"""
        if self._workers:
            return

        def worker():
            while True:
                with self._lock:
                    if not self._task_queue:
                        break
                    task_id = self._task_queue.popleft()
                    task = self._tasks.get(task_id)
                    if task is None or task.status != AsyncTaskStatus.PENDING:
                        continue
                    task.status = AsyncTaskStatus.RUNNING
                    task.started_at = time.time()
                    self._save_task(task)
                    self._notify("task_started", task.to_dict())

                self._execute_task(task)

        t = threading.Thread(target=worker)
        t.start()

    def _execute_task(self, task: AsyncTask):
        """执行任务"""
        try:
            if task.executor:
                result = task.executor(task)
                task.result = {"output": result} if result else {}
                task.progress = 1.0
                task.status = AsyncTaskStatus.COMPLETED
            else:
                task.result = {"message": "Task completed"}
                task.progress = 1.0
                task.status = AsyncTaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = AsyncTaskStatus.FAILED

        task.completed_at = time.time()

        with self._lock:
            self._save_task(task)
            self._notify("task_completed", task.to_dict())

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def list_tasks(self, status: AsyncTaskStatus = None, priority: TaskPriority = None) -> List[AsyncTask]:
        """列出任务"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == AsyncTaskStatus.RUNNING:
                task.status = AsyncTaskStatus.PAUSED
                self._save_task(task)
                self._notify("task_paused", task.to_dict())
                return True
        return False

    def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == AsyncTaskStatus.PAUSED:
                task.status = AsyncTaskStatus.PENDING
                self._task_queue.append(task_id)
                self._save_task(task)
                self._notify("task_resumed", task.to_dict())
                self._start_worker()
                return True
        return False

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in (AsyncTaskStatus.PENDING, AsyncTaskStatus.RUNNING):
                task.status = AsyncTaskStatus.CANCELLED
                task.completed_at = time.time()
                self._save_task(task)
                self._notify("task_cancelled", task.to_dict())
                return True
        return False

    def update_progress(self, task_id: str, progress: float):
        """更新任务进度"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = max(0.0, min(1.0, progress))
                self._save_task(task)

    def add_task_log(self, task_id: str, message: str):
        """添加任务日志"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.add_log(message)
                self._save_task(task)

    def on_notification(self, callback: Callable):
        """注册通知回调"""
        self._notifications.append(callback)

    def _notify(self, event_type: str, task_data: Dict[str, Any]):
        """触发通知"""
        for cb in self._notifications:
            try:
                cb(event_type, task_data)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            status_counts = {}
            for task in self._tasks.values():
                s = task.status.value
                status_counts[s] = status_counts.get(s, 0) + 1

            return {
                "total_tasks": len(self._tasks),
                "status_counts": status_counts,
                "pending_tasks": len(self._task_queue)
            }
