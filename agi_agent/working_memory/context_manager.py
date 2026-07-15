"""
working_memory/context_manager.py - 对话上下文管理

实现任务清单：
- T044: 对话上下文写入（事件触发）—— DialogContextWriter
- T045: 上下文窗口截断压缩（动态调度）—— ContextWindowTruncator

设计要点：
- redis 可选导入，不可用时降级为内存 dict + 过期时间戳实现
- 配置统一使用 dataclass，参数有默认值
- 完整类型注解，中文 docstring
- 模块可独立 import，无副作用实例化
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core import BaseModule

logger = logging.getLogger("agi_agent.working_memory")

# ====== redis 可选导入：不可用时降级为内存实现 ======
try:  # pragma: no cover - 环境相关分支
    import redis as _redis  # type: ignore

    _HAS_REDIS = True
except ImportError:  # pragma: no cover
    _redis = None  # type: ignore
    _HAS_REDIS = False


# ========================================================
# 数据结构
# ========================================================
@dataclass
class DialogMessage:
    """单条对话消息（T044）"""

    message_id: str
    role: str  # "user" / "assistant" / "system"
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TruncatedContext:
    """截断后的上下文（T045）"""

    messages: List[DialogMessage]
    summary: str
    dropped_count: int
    original_tokens: int
    final_tokens: int


@dataclass
class CompressedContext:
    """压缩后的上下文（T045）"""

    summary: str
    kept_messages: List[DialogMessage]
    compression_ratio: float


# ========================================================
# 配置
# ========================================================
@dataclass
class DialogContextConfig:
    """对话上下文写入器配置（T044）"""

    max_turns: int = 20  # 默认每会话最大保留轮数（1 轮 = 1 条消息）
    enable_redis: bool = False  # 是否启用 redis 后端
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "working_memory:dialog:"
    redis_ttl: int = 3600  # redis key 过期秒数


@dataclass
class TruncatorConfig:
    """上下文窗口截断器配置（T045）"""

    strategy: str = "hybrid"  # truncate / summarize / hybrid
    summary_keep_head: int = 1  # 摘要保留首部消息数
    summary_keep_tail: int = 3  # 摘要保留尾部消息数
    token_approx_ratio: float = 1.0  # 字符/token 近似比（中文约 1.0，英文约 0.25）
    enable_tiktoken: bool = False  # 是否启用 tiktoken 精确计数
    tiktoken_model: str = "cl100k_base"
    hybrid_threshold_ratio: float = 0.7  # hybrid 模式触发摘要的阈值占比


# ========================================================
# T044: DialogContextWriter
# ========================================================
class DialogContextWriter(BaseModule):
    """对话上下文写入器（T044 - 事件触发）

    任务 T044：每轮交互后写入用户输入、智能体回复、交互时间，更新会话上下文缓存。

    触发机制：事件触发——每轮交互由外部调用 ``append`` 写入消息。

    存储实现：
        - 内存：``dict[session_id -> deque[DialogMessage]]``
        - 可选 redis 后端（``enable_redis=True`` 且 redis 可用时同步写入；
          redis 不可用则自动降级为纯内存实现）。
    """

    name = "dialog_context_writer"
    version = "1.0.0"
    description = "T044 对话上下文写入（事件触发）"

    def __init__(self, config: Optional[DialogContextConfig] = None) -> None:
        super().__init__()
        self.config: DialogContextConfig = config or DialogContextConfig()
        # session_id -> deque[DialogMessage]
        self._sessions: Dict[str, deque] = {}
        # session_id -> 自定义最大轮数
        self._max_turns_map: Dict[str, int] = {}
        # redis 后端（可选）
        self._redis_client: Optional[Any] = None
        if self.config.enable_redis and _HAS_REDIS:
            try:
                self._redis_client = _redis.Redis.from_url(self.config.redis_url)
                self._redis_client.ping()
                logger.info("DialogContextWriter redis backend connected")
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning(
                    "DialogContextWriter redis unavailable, fallback to memory: %s", e
                )
                self._redis_client = None

    # ---------- 生命周期 ----------
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    def _shutdown(self) -> None:
        try:
            self._sessions.clear()
            self._max_turns_map.clear()
            if self._redis_client is not None:
                try:
                    self._redis_client.close()
                except Exception:
                    pass
                self._redis_client = None
        except Exception:
            pass

    # ---------- 核心方法 ----------
    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """追加一条对话消息

        Args:
            session_id: 会话 ID
            role: 角色，取值 ``"user"`` / ``"assistant"`` / ``"system"``
            content: 消息内容
            metadata: 附加元数据（可选）

        Returns:
            生成的 message_id
        """
        if role not in ("user", "assistant", "system"):
            logger.warning("Unknown role %r, accepting anyway", role)

        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        msg = DialogMessage(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=dict(metadata) if metadata else None,
        )

        dq = self._sessions.setdefault(session_id, deque())
        dq.append(msg)

        # 应用 max_turns 截断（保留最新 N 条）
        max_turns = self._max_turns_map.get(session_id, self.config.max_turns)
        if max_turns > 0:
            while len(dq) > max_turns:
                dq.popleft()

        # 可选 redis 同步
        if self._redis_client is not None:
            try:
                key = f"{self.config.redis_prefix}{session_id}:{message_id}"
                payload = json.dumps(
                    {
                        "message_id": message_id,
                        "role": role,
                        "content": content,
                        "timestamp": msg.timestamp,
                        "metadata": metadata or {},
                    },
                    ensure_ascii=False,
                )
                self._redis_client.setex(key, self.config.redis_ttl, payload)
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning("redis append failed: %s", e)

        logger.debug(
            "append session=%s role=%s len=%d", session_id, role, len(dq)
        )
        return message_id

    def get_context(self, session_id: str) -> List[DialogMessage]:
        """获取会话上下文消息列表（按时间正序）

        Args:
            session_id: 会话 ID

        Returns:
            消息列表；会话不存在时返回空列表
        """
        dq = self._sessions.get(session_id)
        if not dq:
            return []
        return list(dq)

    def get_session_count(self) -> int:
        """获取活跃会话总数"""
        return len(self._sessions)

    def list_sessions(self) -> List[str]:
        """列出所有会话 ID"""
        return list(self._sessions.keys())

    def set_max_turns(self, session_id: str, n: int) -> None:
        """设置会话最大保留轮数

        Args:
            session_id: 会话 ID
            n: 最大轮数（必须大于 0）
        """
        if n <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns_map[session_id] = n
        dq = self._sessions.get(session_id)
        if dq:
            while len(dq) > n:
                dq.popleft()

    def clear_session(self, session_id: str) -> int:
        """清理指定会话上下文（供 ShortTermMemoryCleaner 协作调用）

        Args:
            session_id: 会话 ID

        Returns:
            被清理的消息数
        """
        dq = self._sessions.pop(session_id, None)
        self._max_turns_map.pop(session_id, None)
        cleared = len(dq) if dq else 0

        if self._redis_client is not None:
            try:
                pattern = f"{self.config.redis_prefix}{session_id}:*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning("redis clear_session failed: %s", e)

        return cleared


# ========================================================
# T045: ContextWindowTruncator
# ========================================================
class ContextWindowTruncator(BaseModule):
    """上下文窗口截断压缩器（T045 - 动态调度）

    任务 T045：监控上下文长度，超阈值时对早期对话摘要压缩 + 截断，保留核心逻辑。

    触发机制：动态调度——在 ``truncate`` / ``compress`` 调用时按 token 阈值触发。

    支持三种策略：
        - ``truncate``：尾部按 token 保留，丢弃早期消息
        - ``summarize``：首尾保留，中间合并为摘要
        - ``hybrid``：先按阈值截断，再对被丢弃部分生成摘要
    """

    name = "context_window_truncator"
    version = "1.0.0"
    description = "T045 上下文窗口截断压缩（动态调度）"

    def __init__(self, config: Optional[TruncatorConfig] = None) -> None:
        super().__init__()
        self.config: TruncatorConfig = config or TruncatorConfig()
        self._strategy: str = self.config.strategy
        # 可选注入 DialogContextWriter，便于按 session_id 取上下文
        self._writer: Optional[Any] = None
        # tiktoken 可选
        self._tiktoken_enc: Optional[Any] = None
        if self.config.enable_tiktoken:
            try:
                import tiktoken  # type: ignore

                self._tiktoken_enc = tiktoken.get_encoding(self.config.tiktoken_model)
                logger.info(
                    "tiktoken encoding loaded: %s", self.config.tiktoken_model
                )
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning(
                    "tiktoken unavailable, fallback to char count: %s", e
                )
                self._tiktoken_enc = None

    # ---------- 生命周期 ----------
    def _initialize(self, config: Dict[str, Any]) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ---------- 协作注入 ----------
    def set_writer(self, writer: Any) -> None:
        """注入 DialogContextWriter 实例，便于按 session_id 获取上下文

        Args:
            writer: DialogContextWriter 实例
        """
        self._writer = writer

    # ---------- 核心方法 ----------
    def count_tokens(self, text: str) -> int:
        """估算 token 数

        优先使用 tiktoken；不可用时按字符数近似（乘以 ``token_approx_ratio``）。

        Args:
            text: 待计数文本

        Returns:
            token 数（至少为 0）
        """
        if not text:
            return 0
        if self._tiktoken_enc is not None:
            try:
                return len(self._tiktoken_enc.encode(text))
            except Exception:
                pass
        return max(0, int(len(text) * self.config.token_approx_ratio))

    def set_strategy(self, strategy: str) -> None:
        """设置截断策略

        Args:
            strategy: ``"truncate"`` / ``"summarize"`` / ``"hybrid"``
        """
        if strategy not in ("truncate", "summarize", "hybrid"):
            raise ValueError(f"unknown strategy: {strategy}")
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        """当前策略"""
        return self._strategy

    def summarize(self, messages: List[DialogMessage]) -> str:
        """简单摘要：取首尾消息 + 中间合并摘要

        Args:
            messages: 待摘要的消息列表

        Returns:
            摘要字符串
        """
        if not messages:
            return ""
        head = messages[: self.config.summary_keep_head]
        tail_keep = self.config.summary_keep_tail
        if tail_keep > 0 and len(messages) > self.config.summary_keep_head + tail_keep:
            tail = messages[-tail_keep:]
            middle = messages[
                self.config.summary_keep_head : len(messages) - tail_keep
            ]
        else:
            tail = []
            middle = messages[self.config.summary_keep_head :]

        parts: List[str] = []
        for m in head:
            parts.append(f"[{m.role}] {m.content}")
        if middle:
            merged = "; ".join(
                f"{m.role}:{m.content[:20]}" for m in middle
            )
            parts.append(f"(中间{len(middle)}条摘要: {merged})")
        for m in tail:
            parts.append(f"[{m.role}] {m.content}")
        return " | ".join(parts)

    def truncate(
        self,
        session_id: str,
        max_tokens: int,
        messages: Optional[List[DialogMessage]] = None,
    ) -> TruncatedContext:
        """对会话上下文执行截断/压缩

        Args:
            session_id: 会话 ID（当 ``messages`` 为 None 时从注入的 writer 取上下文）
            max_tokens: 最大 token 数
            messages: 可选，直接传入消息列表（优先于 session_id 取数）

        Returns:
            TruncatedContext
        """
        if messages is None:
            if self._writer is not None:
                try:
                    messages = self._writer.get_context(session_id)
                except Exception as e:
                    logger.warning("fetch context from writer failed: %s", e)
                    messages = []
            else:
                messages = []
        messages = list(messages)

        original_tokens = sum(self.count_tokens(m.content) for m in messages)

        if original_tokens <= max_tokens:
            return TruncatedContext(
                messages=messages,
                summary="",
                dropped_count=0,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
            )

        kept: List[DialogMessage] = []
        summary = ""
        dropped_count = 0
        final_tokens = 0

        if self._strategy == "truncate":
            # 尾部按 token 保留
            token_acc = 0
            for m in reversed(messages):
                t = self.count_tokens(m.content)
                if token_acc + t > max_tokens:
                    break
                kept.insert(0, m)
                token_acc += t
            dropped_count = len(messages) - len(kept)
            final_tokens = token_acc

        elif self._strategy == "summarize":
            head = messages[: self.config.summary_keep_head]
            tail_keep = self.config.summary_keep_tail
            if tail_keep > 0 and len(messages) > self.config.summary_keep_head + tail_keep:
                tail = messages[-tail_keep:]
                middle = messages[
                    self.config.summary_keep_head : len(messages) - tail_keep
                ]
            else:
                tail = []
                middle = messages[self.config.summary_keep_head :]
            summary = self.summarize(middle) if middle else ""
            kept = list(head) + list(tail)
            # 若仍超长，从尾部按 token 截断
            token_acc = self.count_tokens(summary)
            kept_final: List[DialogMessage] = []
            for m in reversed(kept):
                t = self.count_tokens(m.content)
                if token_acc + t > max_tokens:
                    break
                kept_final.insert(0, m)
                token_acc += t
            kept = kept_final
            dropped_count = len(messages) - len(kept)
            final_tokens = token_acc

        else:  # hybrid
            threshold = int(max_tokens * self.config.hybrid_threshold_ratio)
            threshold = max(0, threshold)
            token_acc = 0
            for m in reversed(messages):
                t = self.count_tokens(m.content)
                if token_acc + t > threshold:
                    break
                kept.insert(0, m)
                token_acc += t
            dropped = messages[: len(messages) - len(kept)]
            if dropped:
                summary = self.summarize(dropped)
            final_tokens = token_acc + self.count_tokens(summary)
            dropped_count = len(dropped)

        return TruncatedContext(
            messages=kept,
            summary=summary,
            dropped_count=dropped_count,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

    def compress(self, messages: List[DialogMessage]) -> CompressedContext:
        """压缩消息列表：保留首尾，中间合并为摘要

        Args:
            messages: 待压缩消息列表

        Returns:
            CompressedContext
        """
        if not messages:
            return CompressedContext(
                summary="", kept_messages=[], compression_ratio=0.0
            )
        head = messages[: self.config.summary_keep_head]
        tail_keep = self.config.summary_keep_tail
        if tail_keep > 0 and len(messages) > self.config.summary_keep_head + tail_keep:
            tail = messages[-tail_keep:]
            middle = messages[
                self.config.summary_keep_head : len(messages) - tail_keep
            ]
        else:
            tail = []
            middle = messages[self.config.summary_keep_head :]

        summary = self.summarize(middle) if middle else ""
        kept = list(head) + list(tail)

        original_tokens = sum(self.count_tokens(m.content) for m in messages)
        kept_tokens = sum(self.count_tokens(m.content) for m in kept) + self.count_tokens(
            summary
        )
        ratio = (kept_tokens / original_tokens) if original_tokens > 0 else 0.0
        return CompressedContext(
            summary=summary,
            kept_messages=kept,
            compression_ratio=ratio,
        )
