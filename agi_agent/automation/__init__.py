"""
automation/__init__.py - 自动化处理模块

提供 Pipeline 编排、调度触发、预置处理步骤。
"""
from .engine import (
    AutomationEngine, Pipeline, PipelineStep, PipelineResult,
    PipelineStatus, StepStatus, StepResult,
    TriggerType, TriggerConfig,
)

__all__ = [
    "AutomationEngine", "Pipeline", "PipelineStep", "PipelineResult",
    "PipelineStatus", "StepStatus", "StepResult",
    "TriggerType", "TriggerConfig",
]
