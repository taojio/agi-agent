"""
automation/engine.py - 自动化处理引擎

提供 Pipeline 编排、调度触发、预置处理步骤。
符合 UPG-015 规格的 AutomationEngine 设计。

预置步骤：
- 数据清洗 (clean)
- 特征提取 (feature_extract)
- 异常检测 (anomaly_detect)
- 聚类分析 (cluster)
- 预测分析 (predict)
- 报告生成 (report)
"""
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


class PipelineStatus(Enum):
    """Pipeline 状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TriggerType(Enum):
    """触发类型"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"     # 定时
    EVENT = "event"             # 事件触发
    THRESHOLD = "threshold"     # 阈值触发
    DEPENDENCY = "dependency"   # 依赖完成触发


@dataclass
class PipelineStep:
    """Pipeline 步骤"""
    name: str
    step_type: str                  # clean / feature_extract / anomaly_detect / ...
    handler: Optional[Callable] = None  # 自定义处理函数
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable] = None  # 执行条件
    retries: int = 0
    retry_count: int = 0
    timeout: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step_type": self.step_type,
            "params": self.params,
            "retries": self.retries,
            "timeout": self.timeout,
        }


@dataclass
class StepResult:
    """步骤执行结果"""
    step_name: str
    status: StepStatus
    started_at: float
    completed_at: Optional[float] = None
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": float(self.duration),
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """Pipeline 执行结果"""
    pipeline_id: str
    run_id: str
    status: PipelineStatus
    started_at: float
    completed_at: Optional[float] = None
    steps: List[StepResult] = field(default_factory=list)
    output: Any = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return (self.completed_at or time.time()) - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": float(self.duration),
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
        }


@dataclass
class TriggerConfig:
    """触发配置"""
    trigger_type: TriggerType
    interval: float = 3600.0         # 定时间隔（秒）
    event_name: Optional[str] = None  # 事件名
    threshold_metric: Optional[str] = None  # 阈值指标
    threshold_value: Optional[float] = None
    depends_on: Optional[str] = None  # 依赖的 pipeline_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "interval": float(self.interval),
            "event_name": self.event_name,
            "threshold_metric": self.threshold_metric,
            "threshold_value": self.threshold_value,
            "depends_on": self.depends_on,
        }


class Pipeline:
    """Pipeline

    由一系列步骤组成的数据处理流水线。
    """

    def __init__(self, pipeline_id: str, name: str, steps: List[PipelineStep]):
        self.pipeline_id = pipeline_id
        self.name = name
        self.steps = steps
        self.created_at = time.time()
        self.run_count = 0
        self._last_run: Optional[PipelineResult] = None

    def add_step(self, step: PipelineStep) -> "Pipeline":
        self.steps.append(step)
        return self

    def remove_step(self, step_name: str) -> bool:
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                self.steps.pop(i)
                return True
        return False

    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        for step in self.steps:
            if step.name == step_name:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "run_count": self.run_count,
        }


class AutomationEngine:
    """自动化处理引擎

    管理 Pipeline 的创建、执行、调度。

    Usage:
        engine = AutomationEngine()
        pipeline = engine.create_pipeline("my_pipeline", [step1, step2])
        result = engine.run_pipeline(pipeline.pipeline_id, input_data)
    """

    # 预置步骤处理器
    PRESET_HANDLERS = {
        "clean", "feature_extract", "anomaly_detect",
        "cluster", "predict", "report",
    }

    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.runs: deque = deque(maxlen=200)
        self.schedules: Dict[str, TriggerConfig] = {}
        self._run_counter = 0
        self._pipeline_counter = 0
        # 自定义步骤处理器
        self._custom_handlers: Dict[str, Callable] = {}
        # 预置处理器
        self._preset_handlers: Dict[str, Callable] = {
            "clean": self._handle_clean,
            "feature_extract": self._handle_feature_extract,
            "anomaly_detect": self._handle_anomaly_detect,
            "cluster": self._handle_cluster,
            "predict": self._handle_predict,
            "report": self._handle_report,
        }

    def create_pipeline(self, name: str, steps: List[PipelineStep]) -> Pipeline:
        """创建 Pipeline"""
        self._pipeline_counter += 1
        pipeline_id = f"pipe_{self._pipeline_counter}"
        pipeline = Pipeline(pipeline_id, name, steps)
        self.pipelines[pipeline_id] = pipeline
        return pipeline

    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        return self.pipelines.get(pipeline_id)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.pipelines.values()]

    def delete_pipeline(self, pipeline_id: str) -> bool:
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
            if pipeline_id in self.schedules:
                del self.schedules[pipeline_id]
            return True
        return False

    def register_handler(self, step_type: str,
                          handler: Callable[[Any, Dict[str, Any]], Any]) -> None:
        """注册自定义步骤处理器"""
        self._custom_handlers[step_type] = handler

    def run_pipeline(self, pipeline_id: str, input_data: Any) -> PipelineResult:
        """执行 Pipeline

        Args:
            pipeline_id: Pipeline ID
            input_data: 输入数据

        Returns:
            PipelineResult: 执行结果
        """
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        self._run_counter += 1
        run_id = f"run_{self._run_counter}"
        result = PipelineResult(
            pipeline_id=pipeline_id,
            run_id=run_id,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
        )

        current_data = input_data
        for step in pipeline.steps:
            step_result = StepResult(
                step_name=step.name,
                status=StepStatus.RUNNING,
                started_at=time.time(),
            )

            # 检查执行条件
            if step.condition and not step.condition(current_data):
                step_result.status = StepStatus.SKIPPED
                step_result.completed_at = time.time()
                step_result.duration = step_result.completed_at - step_result.started_at
                result.steps.append(step_result)
                continue

            # 执行步骤（带重试）
            success = False
            for attempt in range(step.retries + 1):
                try:
                    current_data = self._execute_step(step, current_data)
                    step_result.output = current_data
                    step_result.status = StepStatus.SUCCESS
                    success = True
                    break
                except Exception as e:
                    step_result.error = str(e)
                    if attempt < step.retries:
                        continue

            step_result.completed_at = time.time()
            step_result.duration = step_result.completed_at - step_result.started_at
            result.steps.append(step_result)

            if not success:
                result.status = PipelineStatus.FAILED
                result.error = f"Step '{step.name}' failed: {step_result.error}"
                result.completed_at = time.time()
                pipeline.run_count += 1
                pipeline._last_run = result
                self.runs.append(result)
                return result

        result.status = PipelineStatus.COMPLETED
        result.output = current_data
        result.completed_at = time.time()
        pipeline.run_count += 1
        pipeline._last_run = result
        self.runs.append(result)
        return result

    def _execute_step(self, step: PipelineStep, data: Any) -> Any:
        """执行单个步骤"""
        if step.handler:
            return step.handler(data, step.params)

        # 自定义处理器
        if step.step_type in self._custom_handlers:
            return self._custom_handlers[step.step_type](data, step.params)

        # 预置处理器
        if step.step_type in self._preset_handlers:
            return self._preset_handlers[step.step_type](data, step.params)

        raise ValueError(f"Unknown step type: {step.step_type}")

    # 预置步骤处理器实现
    def _handle_clean(self, data: Any, params: Dict[str, Any]) -> Any:
        """数据清洗"""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        # 去除 NaN 和 Inf
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        # 去除重复行
        if data.ndim == 2:
            unique_data = np.unique(data, axis=0)
            return unique_data
        return data

    def _handle_feature_extract(self, data: Any, params: Dict[str, Any]) -> Any:
        """特征提取"""
        from agi_agent.ai_algorithms import FeatureEngineer
        engineer = FeatureEngineer()
        n_components = params.get("n_components", 2)
        return engineer.auto_pipeline(data, target_components=n_components)

    def _handle_anomaly_detect(self, data: Any, params: Dict[str, Any]) -> Any:
        """异常检测"""
        from agi_agent.ai_algorithms import AnomalyDetector
        detector = AnomalyDetector()
        method = params.get("method", "auto")
        report = detector.detect(data, method=method)
        return {
            "data": data,
            "anomaly_report": report.to_dict(),
        }

    def _handle_cluster(self, data: Any, params: Dict[str, Any]) -> Any:
        """聚类分析"""
        from agi_agent.ai_algorithms import AutoClusterer
        clusterer = AutoClusterer()
        method = params.get("method", "auto")
        k = params.get("k")
        result = clusterer.cluster(data, method=method, k=k)
        return {
            "data": data,
            "cluster_result": result.to_dict(),
        }

    def _handle_predict(self, data: Any, params: Dict[str, Any]) -> Any:
        """预测分析"""
        from agi_agent.ai_algorithms import TimeSeriesForecaster
        forecaster = TimeSeriesForecaster()
        horizon = params.get("horizon", 5)
        method = params.get("method", "auto")
        result = forecaster.forecast(data, horizon=horizon, method=method)
        return {
            "data": data,
            "forecast": result.to_dict(),
        }

    def _handle_report(self, data: Any, params: Dict[str, Any]) -> Any:
        """报告生成"""
        report = {
            "generated_at": time.time(),
            "data_shape": np.asarray(data).shape if data is not None else None,
            "data_summary": {},
        }
        if isinstance(data, np.ndarray):
            report["data_summary"] = {
                "mean": float(np.mean(data)) if data.size > 0 else 0,
                "std": float(np.std(data)) if data.size > 0 else 0,
                "min": float(np.min(data)) if data.size > 0 else 0,
                "max": float(np.max(data)) if data.size > 0 else 0,
            }
        elif isinstance(data, dict):
            # 保留原始 dict 的键，并添加类型摘要
            report["data_summary"] = {k: str(type(v).__name__) for k, v in data.items()}
            # 合并原始数据到报告中
            report.update(data)
        return report

    # 调度功能
    def schedule_pipeline(self, pipeline_id: str,
                            trigger: TriggerConfig) -> str:
        """调度 Pipeline

        Args:
            pipeline_id: Pipeline ID
            trigger: 触发配置

        Returns:
            schedule_id
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        self.schedules[pipeline_id] = trigger
        return pipeline_id

    def get_schedule(self, pipeline_id: str) -> Optional[TriggerConfig]:
        return self.schedules.get(pipeline_id)

    def unschedule(self, pipeline_id: str) -> bool:
        if pipeline_id in self.schedules:
            del self.schedules[pipeline_id]
            return True
        return False

    def check_triggers(self, current_metrics: Optional[Dict[str, float]] = None) -> List[str]:
        """检查所有触发条件，返回应执行的 pipeline_id 列表"""
        triggered = []
        for pipeline_id, trigger in self.schedules.items():
            if trigger.trigger_type == TriggerType.THRESHOLD and current_metrics:
                metric = trigger.threshold_metric
                value = trigger.threshold_value
                if metric and value is not None and metric in current_metrics:
                    if current_metrics[metric] > value:
                        triggered.append(pipeline_id)
        return triggered

    def get_run_history(self, limit: int = 20) -> List[PipelineResult]:
        return list(self.runs)[-limit:]

    def get_pipeline_status(self, run_id: str) -> Optional[PipelineResult]:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None

    def get_stats(self) -> Dict[str, Any]:
        success_count = sum(1 for r in self.runs
                           if r.status == PipelineStatus.COMPLETED)
        fail_count = sum(1 for r in self.runs
                        if r.status == PipelineStatus.FAILED)
        return {
            "total_pipelines": len(self.pipelines),
            "total_runs": len(self.runs),
            "successful_runs": success_count,
            "failed_runs": fail_count,
            "success_rate": success_count / len(self.runs) if self.runs else 0.0,
            "scheduled_pipelines": len(self.schedules),
            "avg_run_duration": (float(np.mean([r.duration for r in self.runs]))
                                if self.runs else 0.0),
        }

    def create_preset_pipeline(self, preset_type: str,
                                 name: str = "") -> Pipeline:
        """创建预置 Pipeline"""
        if not name:
            name = f"preset_{preset_type}"

        if preset_type == "anomaly_detection":
            steps = [
                PipelineStep(name="clean", step_type="clean"),
                PipelineStep(name="detect", step_type="anomaly_detect",
                            params={"method": "isolation_forest"}),
                PipelineStep(name="report", step_type="report"),
            ]
        elif preset_type == "clustering":
            steps = [
                PipelineStep(name="clean", step_type="clean"),
                PipelineStep(name="feature_extract", step_type="feature_extract",
                            params={"n_components": 2}),
                PipelineStep(name="cluster", step_type="cluster",
                            params={"method": "kmeans"}),
                PipelineStep(name="report", step_type="report"),
            ]
        elif preset_type == "forecasting":
            steps = [
                PipelineStep(name="clean", step_type="clean"),
                PipelineStep(name="predict", step_type="predict",
                            params={"horizon": 10, "method": "auto"}),
                PipelineStep(name="report", step_type="report"),
            ]
        elif preset_type == "full_analysis":
            steps = [
                PipelineStep(name="clean", step_type="clean"),
                PipelineStep(name="feature_extract", step_type="feature_extract"),
                PipelineStep(name="anomaly_detect", step_type="anomaly_detect"),
                PipelineStep(name="cluster", step_type="cluster"),
                PipelineStep(name="predict", step_type="predict"),
                PipelineStep(name="report", step_type="report"),
            ]
        else:
            raise ValueError(f"Unknown preset type: {preset_type}")

        return self.create_pipeline(name, steps)
