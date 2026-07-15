"""
working_memory/__init__.py - 瞬时短时记忆子模块

实现任务清单 T044-T047：
- T044: 对话上下文写入（事件触发）—— DialogContextWriter
- T045: 上下文窗口截断压缩（动态调度）—— ContextWindowTruncator
- T046: 当前任务状态保存（动态调度）—— TaskStateSaver
- T047: 短时记忆清空（事件触发）—— ShortTermMemoryCleaner

本包专用于短时上下文，与已有的 ``agi_agent.memory``（长期记忆）相互独立。

设计约定：
- 继承 ``agi_agent.core.BaseModule``，实现必要生命周期钩子
- 日志统一使用 ``logging.getLogger("agi_agent.working_memory")``
- redis 可选导入，降级为内存 dict + 过期时间戳实现
- 配置用 dataclass，参数有默认值
- 每个类可无参实例化不抛异常
- 模块可独立 import，无副作用实例化
"""
from dataclasses import dataclass
from typing import Any

from .context_manager import (
    CompressedContext,
    ContextWindowTruncator,
    DialogContextConfig,
    DialogContextWriter,
    DialogMessage,
    TruncatedContext,
    TruncatorConfig,
)
from .task_state import (
    TaskSnapshot,
    TaskState,
    TaskStateConfig,
    TaskStateSaver,
)
from .cleanup import (
    CleanerConfig,
    ClearResult,
    ShortTermMemoryCleaner,
)
from .enhanced_memory import (
    MemoryImportance,
    MemoryStatus,
    MemoryUnit,
    MemoryAssociationDiscovery,
    MemoryCompressor,
    MemoryConflictResolver,
    ActiveForgettingManager,
    EnhancedMemorySystem,
)

__all__ = [
    # T044
    "DialogContextWriter",
    "DialogMessage",
    "DialogContextConfig",
    # T045
    "ContextWindowTruncator",
    "TruncatorConfig",
    "TruncatedContext",
    "CompressedContext",
    # T046
    "TaskStateSaver",
    "TaskState",
    "TaskSnapshot",
    "TaskStateConfig",
    # T047
    "ShortTermMemoryCleaner",
    "ClearResult",
    "CleanerConfig",
    # 聚合包
    "WorkingMemoryBundle",
    # 增强记忆模块
    "MemoryImportance",
    "MemoryStatus",
    "MemoryUnit",
    "MemoryAssociationDiscovery",
    "MemoryCompressor",
    "MemoryConflictResolver",
    "ActiveForgettingManager",
    "EnhancedMemorySystem",
]


@dataclass
class WorkingMemoryBundle:
    """短时记忆模块聚合包

    组合四个短时记忆子模块实例，方便一次性创建并互相注入引用。

    Attributes:
        context_writer: 对话上下文写入器（T044）
        truncator: 上下文窗口截断压缩器（T045）
        task_state_saver: 任务状态保存器（T046）
        cleaner: 短时记忆清理器（T047）
    """

    context_writer: DialogContextWriter
    truncator: ContextWindowTruncator
    task_state_saver: TaskStateSaver
    cleaner: ShortTermMemoryCleaner

    @classmethod
    def create_default(cls) -> "WorkingMemoryBundle":
        """创建默认配置的短时记忆模块包

        创建四个实例并自动注入相互引用：
        - ``cleaner`` 注入 ``context_writer`` 与 ``task_state_saver``，便于清理时联动
        - ``truncator`` 注入 ``context_writer``，便于按 session_id 取上下文

        Returns:
            WorkingMemoryBundle 实例
        """
        writer = DialogContextWriter()
        truncator = ContextWindowTruncator()
        saver = TaskStateSaver()
        cleaner = ShortTermMemoryCleaner(
            context_writer=writer,
            task_state_saver=saver,
        )
        # truncator 注入 writer，便于按 session_id 获取上下文
        truncator.set_writer(writer)
        return cls(
            context_writer=writer,
            truncator=truncator,
            task_state_saver=saver,
            cleaner=cleaner,
        )
