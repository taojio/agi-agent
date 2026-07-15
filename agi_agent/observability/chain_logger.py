"""
observability/chain_logger.py - 全链路日志采集 (T017)

无侵入采集各模块输入输出、调用记录、执行耗时、工具调用、用户交互。
输出结构化 JSON 日志（含 trace_id, span_id, module, input, output,
duration_ms, timestamp）。基于内存 deque 环形缓冲 + 文件落盘
(./logs/chain.jsonl)。
"""
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.observability")


@dataclass
class ChainLogConfig:
    """全链路日志配置"""
    buffer_size: int = 10000
    log_file: str = "./logs/chain.jsonl"
    flush_on_write: bool = False  # True 则每条立即落盘
    max_field_len: int = 4096  # 单字段最大字符数（超出截断）


@dataclass
class Span:
    """单个 span 上下文"""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    input: Any = None
    output: Any = None
    duration_ms: Optional[float] = None
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FullChainLogger(BaseModule):
    """全链路日志采集器 (T017)

    通过 start_trace / start_span / end_span / log_call 方法采集结构化
    调用链日志。日志同时存入内存环形缓冲与可选的 JSONL 文件。
    """

    name = "full_chain_logger"
    version = "1.0.0"
    description = "全链路日志采集 (T017)"

    def __init__(self, config: Optional[ChainLogConfig] = None):
        super().__init__()
        self._cfg = config or ChainLogConfig()
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self._cfg.buffer_size)
        self._spans: Dict[str, Span] = {}
        self._traces: Dict[str, List[str]] = {}
        self._file_handle = None

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._cfg.flush_on_write:
            try:
                os.makedirs(os.path.dirname(self._cfg.log_file) or ".", exist_ok=True)
                self._file_handle = open(self._cfg.log_file, "a", encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                logger.warning("无法打开链路日志文件 %s: %s", self._cfg.log_file, e)
                self._file_handle = None
        logger.info("FullChainLogger 初始化完成 (buffer=%d, file=%s)", self._cfg.buffer_size, self._cfg.log_file)

    def _shutdown(self) -> None:
        self.flush()
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._file_handle = None

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def start_trace(self, name: str) -> str:
        """开启一条新的调用链

        Args:
            name: trace 名称

        Returns:
            str: trace_id
        """
        trace_id = uuid.uuid4().hex
        self._traces[trace_id] = []
        # 同时开启根 span
        self.start_span(trace_id, name, parent_id=None)
        return trace_id

    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_id: Optional[str] = None,
        input: Any = None,
    ) -> str:
        """开启一个 span

        Args:
            trace_id: 所属 trace id
            name: span 名称
            parent_id: 父 span id
            input: span 输入

        Returns:
            str: span_id
        """
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        if parent_id is None and self._traces[trace_id]:
            # 默认挂到最后一个根 span
            parent_id = self._traces[trace_id][0]
        span_id = uuid.uuid4().hex
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            input=input,
        )
        self._spans[span_id] = span
        self._traces[trace_id].append(span_id)
        return span_id

    def end_span(self, span_id: str, output: Any = None, status: str = "ok") -> Optional[Span]:
        """结束一个 span

        Args:
            span_id: span id
            output: span 输出
            status: 结束状态

        Returns:
            Span: 已结束的 span（不存在返回 None）
        """
        span = self._spans.get(span_id)
        if span is None:
            return None
        span.output = output
        span.status = status
        span.duration_ms = (time.time() - span.start_time) * 1000.0
        self._emit(span)
        return span

    def log_call(
        self,
        module: str,
        input: Any,
        output: Any,
        duration: float,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录一次模块调用

        Args:
            module: 模块名称
            input: 调用输入
            output: 调用输出
            duration: 耗时（秒）
            trace_id: 关联 trace id
            span_id: 关联 span id
            metadata: 附加元数据
        """
        record = {
            "trace_id": trace_id or "-",
            "span_id": span_id or "-",
            "module": module,
            "input": self._truncate(input),
            "output": self._truncate(output),
            "duration_ms": float(duration) * 1000.0,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._append_record(record)

    # ====== 查询 ======
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """获取某条 trace 下的全部日志记录"""
        return [r for r in self._buffer if r.get("trace_id") == trace_id]

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """返回最近 limit 条日志记录"""
        items = list(self._buffer)
        return items[-limit:] if limit < len(items) else items

    def flush(self) -> None:
        """将缓冲区日志落盘"""
        if not self._buffer:
            return
        try:
            os.makedirs(os.path.dirname(self._cfg.log_file) or ".", exist_ok=True)
            mode = "a" if self._file_handle else "a"
            fh = self._file_handle or open(self._cfg.log_file, mode, encoding="utf-8")
            try:
                for record in self._buffer:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fh.flush()
            finally:
                if not self._file_handle:
                    fh.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("链路日志落盘失败: %s", e)

    # ====== 内部 ======
    def _emit(self, span: Span) -> None:
        record = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "module": span.name,
            "input": self._truncate(span.input),
            "output": self._truncate(span.output),
            "duration_ms": span.duration_ms,
            "timestamp": span.start_time,
            "status": span.status,
            "metadata": span.metadata,
        }
        self._append_record(record)

    def _append_record(self, record: Dict[str, Any]) -> None:
        self._buffer.append(record)
        if self._cfg.flush_on_write and self._file_handle is not None:
            try:
                self._file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._file_handle.flush()
            except Exception:  # noqa: BLE001
                pass

    def _truncate(self, value: Any) -> Any:
        max_len = self._cfg.max_field_len
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "...<truncated>"
        return value
