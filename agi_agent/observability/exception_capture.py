"""
observability/exception_capture.py - 异常捕获记录 (T019)

全局监听异常，抓取堆栈，生成带故障标签的告警日志。
基于 sys.excepthook + threading.excepthook 全局兜底，并提供
@exception_capture.guard 装饰器用法。
"""
import functools
import logging
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.observability")


@dataclass
class ExceptionRecord:
    """单条异常记录"""
    capture_id: str
    exception_type: str
    message: str
    traceback_str: str
    context: Dict[str, Any]
    timestamp: float
    tags: List[str] = field(default_factory=list)


@dataclass
class ExceptionCaptureConfig:
    """异常捕获配置"""
    buffer_size: int = 5000
    install_global_hook: bool = False  # 默认不在构造时安装，避免副作用


class ExceptionCapture(BaseModule):
    """异常捕获记录器 (T019)

    提供 capture / decorate / get_recent / subscribe_alerts 方法，以及
    guard 装饰器。全局兜底钩子通过 install_global_hooks() 显式安装，
    以避免导入时的副作用。
    """

    name = "exception_capture"
    version = "1.0.0"
    description = "异常捕获记录 (T019)"

    def __init__(self, config: Optional[ExceptionCaptureConfig] = None):
        super().__init__()
        self._cfg = config or ExceptionCaptureConfig()
        self._buffer: Deque[ExceptionRecord] = deque(maxlen=self._cfg.buffer_size)
        self._alert_handlers: List[Callable[[ExceptionRecord], None]] = []
        self._prev_excepthook: Optional[Callable] = None
        self._prev_threading_excepthook: Optional[Callable] = None
        self._hooks_installed: bool = False

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._cfg.install_global_hook:
            self.install_global_hooks()
        logger.info("ExceptionCapture 初始化完成")

    def _shutdown(self) -> None:
        self.uninstall_global_hooks()

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def capture(
        self,
        exception: BaseException,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ExceptionRecord:
        """捕获一个异常

        Args:
            exception: 异常对象
            context: 上下文信息
            tags: 故障标签

        Returns:
            ExceptionRecord: 异常记录
        """
        record = ExceptionRecord(
            capture_id=uuid.uuid4().hex,
            exception_type=type(exception).__name__,
            message=str(exception),
            traceback_str="".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
            context=dict(context or {}),
            timestamp=time.time(),
            tags=list(tags or [type(exception).__name__]),
        )
        self._buffer.append(record)
        logger.error(
            "异常捕获 [%s] %s | tags=%s",
            record.exception_type,
            record.message,
            record.tags,
        )
        self._dispatch_alert(record)
        return record

    def decorate(self, func: Callable) -> Callable:
        """装饰器：捕获被装饰函数抛出的异常并继续抛出

        Args:
            func: 被装饰函数

        Returns:
            Callable: 包装后的函数
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.capture(
                    exc,
                    context={"function": getattr(func, "__qualname__", func.__name__)},
                    tags=["decorated"],
                )
                raise

        return wrapper

    # 提供别名 guard，支持 @instance.guard 用法
    guard = decorate

    def get_recent(self, limit: int = 100) -> List[ExceptionRecord]:
        """获取最近的异常记录

        Args:
            limit: 返回数量上限

        Returns:
            List[ExceptionRecord]: 异常记录列表（最新在后）
        """
        items = list(self._buffer)
        return items[-limit:] if limit < len(items) else items

    def subscribe_alerts(self, handler: Callable[[ExceptionRecord], None]) -> None:
        """订阅异常告警

        Args:
            handler: 告警处理函数，接收 ExceptionRecord
        """
        self._alert_handlers.append(handler)

    # ====== 全局钩子 ======
    def install_global_hooks(self) -> None:
        """安装全局异常兜底钩子（sys.excepthook + threading.excepthook）"""
        if self._hooks_installed:
            return
        self._prev_excepthook = sys.excepthook

        def _excepthook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
            try:
                if exc_value is not None:
                    self.capture(
                        exc_value,
                        context={"source": "sys.excepthook"},
                        tags=["global", "main_thread"],
                    )
            except Exception:  # noqa: BLE001
                pass
            if self._prev_excepthook is not None:
                try:
                    self._prev_excepthook(exc_type, exc_value, exc_tb)
                except Exception:  # noqa: BLE001
                    pass

        sys.excepthook = _excepthook

        self._prev_threading_excepthook = getattr(threading, "excepthook", None)

        def _threading_excepthook(args):  # type: ignore[no-untyped-def]
            try:
                if args.exc_value is not None:
                    self.capture(
                        args.exc_value,
                        context={
                            "source": "threading.excepthook",
                            "thread": getattr(args.thread, "name", "?"),
                        },
                        tags=["global", "thread"],
                    )
            except Exception:  # noqa: BLE001
                pass
            if self._prev_threading_excepthook is not None:
                try:
                    self._prev_threading_excepthook(args)
                except Exception:  # noqa: BLE001
                    pass

        try:
            threading.excepthook = _threading_excepthook
        except Exception:  # noqa: BLE001
            pass
        self._hooks_installed = True
        logger.info("已安装全局异常兜底钩子")

    def uninstall_global_hooks(self) -> None:
        """卸载全局异常钩子"""
        if not self._hooks_installed:
            return
        try:
            if self._prev_excepthook is not None:
                sys.excepthook = self._prev_excepthook
            if self._prev_threading_excepthook is not None:
                threading.excepthook = self._prev_threading_excepthook  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        self._hooks_installed = False

    # ====== 内部 ======
    def _dispatch_alert(self, record: ExceptionRecord) -> None:
        for handler in list(self._alert_handlers):
            try:
                handler(record)
            except Exception as e:  # noqa: BLE001
                logger.warning("告警处理函数异常: %s", e)


# 模块级默认实例，支持 @exception_capture.guard 用法
exception_capture = ExceptionCapture()
