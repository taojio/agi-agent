"""
working_memory/cleanup.py - 短时记忆清理

实现任务清单：
- T047: 短时记忆清空（事件触发）—— ShortTermMemoryCleaner

设计要点：
- 接收会话结束/任务终止/用户退出指令，清空当前会话所有短时上下文、任务缓存、临时数据
- 支持与 DialogContextWriter、TaskStateSaver 协作：构造时可注入二者的实例引用（可选）
- 若未注入则只清理自身注册的资源
- 配置统一使用 dataclass，参数有默认值
- 完整类型注解，中文 docstring
- 模块可独立 import，无副作用实例化
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core import BaseModule

logger = logging.getLogger("agi_agent.working_memory")


# ========================================================
# 数据结构
# ========================================================
@dataclass
class ClearResult:
    """清理结果（T047）"""

    success: bool
    cleared_items: List[str] = field(default_factory=list)
    freed_bytes: int = 0
    errors: List[str] = field(default_factory=list)


# ========================================================
# 配置
# ========================================================
@dataclass
class CleanerConfig:
    """短时记忆清理器配置（T047）"""

    default_max_age_seconds: int = 3600  # 默认会话最大存活秒数
    enable_resource_cleanup: bool = True  # 是否启用注册资源清理


# ========================================================
# 资源注册条目类型：(resource_handle, cleanup_fn, estimated_bytes)
# ========================================================
_ResourceEntry = Tuple[Any, Callable[[], None], int]


# ========================================================
# T047: ShortTermMemoryCleaner
# ========================================================
class ShortTermMemoryCleaner(BaseModule):
    """短时记忆清理器（T047 - 事件触发）

    任务 T047：接收会话结束 / 任务终止 / 用户退出指令，清空当前会话所有短时上下文、
    任务缓存、临时数据，释放资源，防止会话串扰。

    触发机制：事件触发——外部调用 ``clear_session`` / ``clear_task`` / ``clear_all``。

    协作机制：
        - 构造时可注入 ``DialogContextWriter`` 与 ``TaskStateSaver`` 实例引用（可选）
        - 清理时调用其清理方法（``clear_session`` / ``clear_task``）
        - 若未注入则只清理自身注册的资源
    """

    name = "short_term_memory_cleaner"
    version = "1.0.0"
    description = "T047 短时记忆清空（事件触发）"

    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        context_writer: Optional[Any] = None,
        task_state_saver: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config: CleanerConfig = config or CleanerConfig()
        # 注入的协作者
        self._context_writer: Optional[Any] = context_writer
        self._task_state_saver: Optional[Any] = task_state_saver
        # 资源注册表：session_id -> List[(resource_handle, cleanup_fn, estimated_bytes)]
        self._resources: Dict[str, List[_ResourceEntry]] = {}
        # 会话最后活跃时间：session_id -> timestamp（用于过期清理）
        self._session_last_active: Dict[str, float] = {}
        # 清理前回调列表：(target_id, scope) -> None
        self._pre_cleanup_handlers: List[Callable[[str, str], None]] = []

    # ---------- 生命周期 ----------
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    def _shutdown(self) -> None:
        try:
            self.clear_all()
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("shutdown clear_all failed: %s", e)

    # ---------- 协作注入 ----------
    def set_context_writer(self, writer: Any) -> None:
        """注入 DialogContextWriter 实例

        Args:
            writer: DialogContextWriter 实例
        """
        self._context_writer = writer

    def set_task_state_saver(self, saver: Any) -> None:
        """注入 TaskStateSaver 实例

        Args:
            saver: TaskStateSaver 实例
        """
        self._task_state_saver = saver

    # ---------- 资源注册 ----------
    def register_resource(
        self,
        session_id: str,
        resource_handle: Any,
        cleanup_fn: Callable[[], None],
        estimated_bytes: int = 0,
    ) -> None:
        """注册需清理的额外资源（如文件句柄 / 连接）

        Args:
            session_id: 关联的会话 ID
            resource_handle: 资源句柄
            cleanup_fn: 清理函数（无参 callable）
            estimated_bytes: 预估占用字节数（用于统计释放量）
        """
        self._resources.setdefault(session_id, []).append(
            (resource_handle, cleanup_fn, estimated_bytes)
        )
        self._session_last_active[session_id] = time.time()

    def subscribe_cleanup(self, handler: Callable[[str, str], None]) -> None:
        """订阅清理前通知（便于其他模块在清理前保存数据）

        Args:
            handler: 回调函数，签名 ``(target_id: str, scope: str) -> None``
                     scope 取值：``"session"`` / ``"task"`` / ``"all"``
        """
        self._pre_cleanup_handlers.append(handler)

    def _notify_handlers(self, target_id: str, scope: str) -> None:
        for h in self._pre_cleanup_handlers:
            try:
                h(target_id, scope)
            except Exception as e:  # pragma: no cover - 防御性
                logger.warning("pre-cleanup handler error: %s", e)

    def touch_session(self, session_id: str) -> None:
        """更新会话活跃时间（用于过期判断）

        Args:
            session_id: 会话 ID
        """
        self._session_last_active[session_id] = time.time()

    # ---------- 核心方法 ----------
    def clear_session(self, session_id: str) -> ClearResult:
        """清空指定会话的所有短时上下文

        Args:
            session_id: 会话 ID

        Returns:
            ClearResult
        """
        cleared: List[str] = []
        errors: List[str] = []
        freed = 0

        self._notify_handlers(session_id, "session")

        # 清理 DialogContextWriter 中的会话上下文
        if self._context_writer is not None:
            try:
                n = self._context_writer.clear_session(session_id)
                cleared.append(f"dialog_messages:{n}")
            except Exception as e:
                errors.append(f"context_writer: {e}")

        # 清理已注册资源
        if self.config.enable_resource_cleanup:
            res_list = self._resources.pop(session_id, [])
            for handle, fn, est in res_list:
                try:
                    fn()
                    freed += est
                    cleared.append(f"resource:{type(handle).__name__}")
                except Exception as e:
                    errors.append(f"resource_cleanup: {e}")

        self._session_last_active.pop(session_id, None)
        logger.info(
            "clear_session %s cleared=%d errors=%d",
            session_id,
            len(cleared),
            len(errors),
        )
        return ClearResult(
            success=len(errors) == 0,
            cleared_items=cleared,
            freed_bytes=freed,
            errors=errors,
        )

    def clear_task(self, task_id: str) -> ClearResult:
        """清空指定任务的所有缓存

        Args:
            task_id: 任务 ID

        Returns:
            ClearResult
        """
        cleared: List[str] = []
        errors: List[str] = []
        freed = 0

        self._notify_handlers(task_id, "task")

        if self._task_state_saver is not None:
            try:
                n = self._task_state_saver.clear_task(task_id)
                cleared.append(f"task_snapshots:{n}")
            except Exception as e:
                errors.append(f"task_state_saver: {e}")

        logger.info(
            "clear_task %s cleared=%d errors=%d",
            task_id,
            len(cleared),
            len(errors),
        )
        return ClearResult(
            success=len(errors) == 0,
            cleared_items=cleared,
            freed_bytes=freed,
            errors=errors,
        )

    def clear_all(self) -> ClearResult:
        """清空所有短时记忆

        Returns:
            ClearResult
        """
        cleared: List[str] = []
        errors: List[str] = []
        freed = 0

        self._notify_handlers("*", "all")

        # 收集所有会话 ID：注册资源 + writer 中的会话
        session_ids = list(self._resources.keys())
        if self._context_writer is not None:
            try:
                sessions = self._context_writer.list_sessions()
                for sid in sessions:
                    if sid not in session_ids:
                        session_ids.append(sid)
            except Exception as e:
                errors.append(f"context_writer.list: {e}")

        for sid in session_ids:
            try:
                res = self.clear_session(sid)
                cleared.extend(res.cleared_items)
                freed += res.freed_bytes
                errors.extend(res.errors)
            except Exception as e:
                errors.append(f"clear_session[{sid}]: {e}")

        # 清理所有任务快照
        if self._task_state_saver is not None:
            try:
                snapshots_map = getattr(self._task_state_saver, "_snapshots", None)
                if isinstance(snapshots_map, dict):
                    for tid in list(snapshots_map.keys()):
                        try:
                            n = self._task_state_saver.clear_task(tid)
                            cleared.append(f"task_snapshots[{tid}]:{n}")
                        except Exception as e:
                            errors.append(f"clear_task[{tid}]: {e}")
            except Exception as e:
                errors.append(f"task_state_saver: {e}")

        logger.info("clear_all cleared=%d errors=%d", len(cleared), len(errors))
        return ClearResult(
            success=len(errors) == 0,
            cleared_items=cleared,
            freed_bytes=freed,
            errors=errors,
        )

    def cleanup_expired(self, max_age_seconds: Optional[int] = None) -> ClearResult:
        """清理超时会话

        Args:
            max_age_seconds: 最大存活秒数（None 则用配置默认值）

        Returns:
            ClearResult
        """
        if max_age_seconds is None:
            max_age_seconds = self.config.default_max_age_seconds
        now = time.time()
        cleared: List[str] = []
        errors: List[str] = []
        freed = 0

        expired = [
            sid
            for sid, ts in self._session_last_active.items()
            if now - ts > max_age_seconds
        ]
        for sid in expired:
            try:
                res = self.clear_session(sid)
                cleared.extend(res.cleared_items)
                freed += res.freed_bytes
                errors.extend(res.errors)
            except Exception as e:
                errors.append(f"expired[{sid}]: {e}")

        logger.info("cleanup_expired cleared=%d sessions", len(expired))
        return ClearResult(
            success=len(errors) == 0,
            cleared_items=cleared,
            freed_bytes=freed,
            errors=errors,
        )
