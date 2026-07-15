"""
infrastructure/compute_monitor.py - 算力监控与异常回收子模块

包含任务：
- T001 算力资源巡检（ResourceInspectionTask）
- T004 算力异常回收（AnomalyRecoveryTask）

设计原则：
1. 重型依赖（psutil/torch/pynvml）可选导入，失败降级为纯 Python
2. GPU 通过 nvidia-smi 子进程解析 或 pynvml 探测，失败降级为仅 CPU
3. NPU 使用占位探测，无硬件时返回 0
4. 模块可独立 import 且无副作用实例化
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.infrastructure")

# ====== 可选依赖探测 ======
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - 依赖缺失时的降级路径
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

try:
    import torch  # type: ignore
    _HAS_TORCH = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import pynvml  # type: ignore
    try:
        pynvml.nvmlInit()
        _HAS_PYNVML = True
    except Exception:
        _HAS_PYNVML = False
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _HAS_PYNVML = False


# ====== 数据结构 ======

@dataclass
class ResourceReport:
    """算力资源巡检报告（T001）"""
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    temperature: float = 0.0
    npu_percent: float = 0.0
    load_avg: List[float] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class AnomalyEvent:
    """算力异常事件（T004）"""
    anomaly_id: str
    event_type: str  # process_stuck / gpu_oom / node_offline / load_burst
    node_id: str
    description: str = ""
    severity: str = "warning"  # info / warning / critical
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    recovered: bool = False


@dataclass
class ResourceInspectionConfig:
    """T001 配置"""
    collect_interval: float = 5.0          # 采集间隔（秒）
    enable_gpu: bool = True                # 是否探测 GPU
    enable_npu: bool = True                # 是否探测 NPU
    nvidia_smi_path: str = "nvidia-smi"    # nvidia-smi 可执行路径
    history_size: int = 100                # 历史快照保留数量


@dataclass
class AnomalyRecoveryConfig:
    """T004 配置"""
    isolate_on_critical: bool = True       # critical 级别自动隔离节点
    max_recovery_attempts: int = 3         # 单次异常最大恢复尝试次数
    recovery_cooldown: float = 5.0         # 恢复冷却时间（秒）
    auto_cleanup_processes: bool = True    # 是否自动清理卡死进程


# ====== T001: 算力资源巡检 ======

class ResourceInspectionTask(BaseModule):
    """T001 算力资源巡检

    常驻后台采集 CPU/GPU/NPU 负载、显存占用、温度。
    GPU 优先使用 pynvml，其次 nvidia-smi 子进程解析，失败降级为仅 CPU。
    NPU 使用占位探测，无硬件时返回 0。
    """

    name = "resource_inspection_task"
    version = "1.0.0"
    description = "T001 算力资源巡检：常驻后台采集 CPU/GPU/NPU 负载、显存占用、温度"

    def __init__(self, config: Optional[ResourceInspectionConfig] = None) -> None:
        super().__init__()
        self.config: ResourceInspectionConfig = config or ResourceInspectionConfig()
        self._history: List[ResourceReport] = []
        self._latest: Optional[ResourceReport] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 探测可用后端
        self._gpu_backend: str = self._detect_gpu_backend()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("ResourceInspectionTask 初始化完成，GPU 后端=%s", self._gpu_backend)

    def _start(self) -> None:
        self._stop_event.clear()
        # 先采集一次，避免快照为空
        try:
            self.collect()
        except Exception as e:  # pragma: no cover
            logger.warning("首次采集失败: %s", e)
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="ResourceInspectionLoop",
            daemon=True,
        )
        self._thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _shutdown(self) -> None:
        self._stop()
        with self._lock:
            self._history.clear()

    def _health_check(self) -> bool:
        return self._latest is not None

    # ---- 公共方法 ----
    def collect(self) -> ResourceReport:
        """采集一次资源快照并返回报告"""
        report = ResourceReport(timestamp=time.time())
        report.cpu_percent = self._collect_cpu()
        report.load_avg = self._collect_load_avg()
        if self.config.enable_gpu:
            gpu_info = self._collect_gpu()
            report.gpu_percent = gpu_info.get("gpu_percent", 0.0)
            report.gpu_memory_used_mb = gpu_info.get("gpu_memory_used_mb", 0.0)
            report.gpu_memory_total_mb = gpu_info.get("gpu_memory_total_mb", 0.0)
            report.temperature = gpu_info.get("temperature", 0.0)
        if self.config.enable_npu:
            report.npu_percent = self._collect_npu()
        with self._lock:
            self._latest = report
            self._history.append(report)
            if len(self._history) > self.config.history_size:
                self._history = self._history[-self.config.history_size:]
        self.update_metric("last_collect_ts", report.timestamp)
        self.update_metric("gpu_backend", self._gpu_backend)
        return report

    def get_snapshot(self) -> Optional[ResourceReport]:
        """获取最近一次资源快照"""
        with self._lock:
            return self._latest

    # ---- 内部：CPU ----
    def _collect_cpu(self) -> float:
        if _HAS_PSUTIL:
            try:
                return float(psutil.cpu_percent(interval=None))
            except Exception:  # pragma: no cover
                pass
        return 0.0

    def _collect_load_avg(self) -> List[float]:
        if hasattr(os, "getloadavg"):
            try:
                return [float(x) for x in os.getloadavg()]
            except Exception:  # pragma: no cover
                pass
        return []

    # ---- 内部：GPU ----
    def _detect_gpu_backend(self) -> str:
        if not self.config.enable_gpu:
            return "none"
        if _HAS_TORCH and torch is not None:
            try:
                # 触发一次 utilization 探测，验证可用
                _ = torch.cuda.utilization()
                return "torch"
            except Exception:
                pass
        if _HAS_PYNVML and pynvml is not None:
            return "pynvml"
        # 尝试 nvidia-smi
        try:
            subprocess.run(
                [self.config.nvidia_smi_path, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                timeout=2.0,
                check=False,
            )
            return "nvidia_smi"
        except Exception:
            return "none"

    def _collect_gpu(self) -> Dict[str, float]:
        if self._gpu_backend == "torch" and torch is not None:
            return self._collect_gpu_torch()
        if self._gpu_backend == "pynvml" and pynvml is not None:
            return self._collect_gpu_pynvml()
        if self._gpu_backend == "nvidia_smi":
            return self._collect_gpu_nvidia_smi()
        return {}

    def _collect_gpu_torch(self) -> Dict[str, float]:
        try:
            gpu_percent = float(torch.cuda.utilization())
            used = float(torch.cuda.memory_allocated()) / (1024 * 1024)
            total = float(torch.cuda.get_device_properties(0).total_memory) / (1024 * 1024)
            temp = 0.0
            return {
                "gpu_percent": gpu_percent,
                "gpu_memory_used_mb": used,
                "gpu_memory_total_mb": total,
                "temperature": temp,
            }
        except Exception as e:  # pragma: no cover
            logger.debug("torch GPU 采集失败: %s", e)
            return {}

    def _collect_gpu_pynvml(self) -> Dict[str, float]:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = 0.0
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            return {
                "gpu_percent": float(util.gpu),
                "gpu_memory_used_mb": float(mem.used) / (1024 * 1024),
                "gpu_memory_total_mb": float(mem.total) / (1024 * 1024),
                "temperature": temp,
            }
        except Exception as e:  # pragma: no cover
            logger.debug("pynvml GPU 采集失败: %s", e)
            return {}

    def _collect_gpu_nvidia_smi(self) -> Dict[str, float]:
        try:
            result = subprocess.run(
                [
                    self.config.nvidia_smi_path,
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                timeout=2.0,
                check=False,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_percent = float(parts[0]) if len(parts) > 0 else 0.0
            mem_used = float(parts[1]) if len(parts) > 1 else 0.0
            mem_total = float(parts[2]) if len(parts) > 2 else 0.0
            temp = float(parts[3]) if len(parts) > 3 else 0.0
            return {
                "gpu_percent": gpu_percent,
                "gpu_memory_used_mb": mem_used,
                "gpu_memory_total_mb": mem_total,
                "temperature": temp,
            }
        except Exception as e:  # pragma: no cover
            logger.debug("nvidia-smi 采集失败: %s", e)
            return {}

    # ---- 内部：NPU 占位 ----
    def _collect_npu(self) -> float:
        # NPU 占位探测：当前环境无 NPU SDK，返回 0.0
        return 0.0

    # ---- 内部：采集循环 ----
    def _collect_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.collect()
            except Exception as e:  # pragma: no cover
                logger.warning("资源采集循环异常: %s", e)
            self._stop_event.wait(self.config.collect_interval)


# ====== T004: 算力异常回收 ======

class AnomalyRecoveryTask(BaseModule):
    """T004 算力异常回收

    事件触发，监听异常事件（进程卡死、显存溢出、节点离线、负载爆表），
    执行进程清理、显存释放、故障隔离。
    """

    name = "anomaly_recovery_task"
    version = "1.0.0"
    description = "T004 算力异常回收：进程卡死/显存溢出/节点离线/负载爆表的恢复与隔离"

    def __init__(self, config: Optional[AnomalyRecoveryConfig] = None) -> None:
        super().__init__()
        self.config: AnomalyRecoveryConfig = config or AnomalyRecoveryConfig()
        self._anomalies: Dict[str, AnomalyEvent] = {}
        self._isolated_nodes: set = set()
        self._recovery_attempts: Dict[str, int] = {}
        self._last_recovery_ts: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._fault_handlers: List[Callable[[AnomalyEvent], None]] = []

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("AnomalyRecoveryTask 初始化完成")

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        with self._lock:
            self._anomalies.clear()
            self._isolated_nodes.clear()
            self._recovery_attempts.clear()
            self._last_recovery_ts.clear()
            self._fault_handlers.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def register_anomaly(self, event: AnomalyEvent) -> str:
        """注册一个异常事件，触发后续恢复流程"""
        if not event.anomaly_id:
            event.anomaly_id = f"anom_{uuid.uuid4().hex[:8]}"
        if not event.timestamp:
            event.timestamp = time.time()
        with self._lock:
            self._anomalies[event.anomaly_id] = event
        logger.warning(
            "注册异常事件 id=%s type=%s node=%s severity=%s desc=%s",
            event.anomaly_id, event.event_type, event.node_id,
            event.severity, event.description,
        )
        # 通知订阅者
        for handler in list(self._fault_handlers):
            try:
                handler(event)
            except Exception as e:  # pragma: no cover
                logger.warning("故障订阅处理器异常: %s", e)
        # critical 级别自动隔离
        if (event.severity == "critical"
                and self.config.isolate_on_critical
                and event.node_id):
            try:
                self.isolate_node(event.node_id)
            except Exception as e:  # pragma: no cover
                logger.warning("自动隔离节点失败 node=%s: %s", event.node_id, e)
        return event.anomaly_id

    def recover(self, anomaly_id: str) -> bool:
        """执行异常恢复动作"""
        with self._lock:
            event = self._anomalies.get(anomaly_id)
        if event is None:
            logger.warning("未找到异常事件 id=%s", anomaly_id)
            return False
        # 冷却检查
        now = time.time()
        last = self._last_recovery_ts.get(anomaly_id, 0.0)
        if now - last < self.config.recovery_cooldown:
            logger.info("异常 %s 处于冷却中，跳过恢复", anomaly_id)
            return False
        # 尝试次数检查
        attempts = self._recovery_attempts.get(anomaly_id, 0)
        if attempts >= self.config.max_recovery_attempts:
            logger.warning("异常 %s 超过最大恢复尝试次数 %d", anomaly_id, self.config.max_recovery_attempts)
            return False
        self._recovery_attempts[anomaly_id] = attempts + 1
        self._last_recovery_ts[anomaly_id] = now

        ok = False
        try:
            ok = self._dispatch_recovery(event)
        except Exception as e:  # pragma: no cover
            logger.error("恢复异常 %s 时出错: %s", anomaly_id, e)
        if ok:
            with self._lock:
                event.recovered = True
            logger.info("异常 %s 恢复成功", anomaly_id)
        return ok

    def isolate_node(self, node_id: str) -> bool:
        """隔离故障节点，标记为不再分配任务"""
        if not node_id:
            return False
        with self._lock:
            self._isolated_nodes.add(node_id)
        logger.warning("节点 %s 已隔离", node_id)
        self.update_metric("isolated_nodes", list(self._isolated_nodes))
        return True

    def get_anomalies(self) -> List[AnomalyEvent]:
        """获取全部异常事件"""
        with self._lock:
            return list(self._anomalies.values())

    def get_isolated_nodes(self) -> List[str]:
        """获取已隔离的节点列表"""
        with self._lock:
            return sorted(self._isolated_nodes)

    # ---- 内部 ----
    def _dispatch_recovery(self, event: AnomalyEvent) -> bool:
        etype = event.event_type
        if etype == "process_stuck":
            return self._recover_process_stuck(event)
        if etype == "gpu_oom":
            return self._recover_gpu_oom(event)
        if etype == "node_offline":
            return self._recover_node_offline(event)
        if etype == "load_burst":
            return self._recover_load_burst(event)
        logger.info("未知的异常类型 %s，仅记录", etype)
        return True

    def _recover_process_stuck(self, event: AnomalyEvent) -> bool:
        if not self.config.auto_cleanup_processes:
            return True
        pid = event.payload.get("pid")
        if pid is None or not _HAS_PSUTIL:
            return True
        try:
            proc = psutil.Process(int(pid))
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
            logger.info("已清理卡死进程 pid=%s", pid)
        except Exception as e:  # pragma: no cover
            logger.warning("清理进程 pid=%s 失败: %s", pid, e)
        return True

    def _recover_gpu_oom(self, event: AnomalyEvent) -> bool:
        # 释放 GPU 显存缓存
        if _HAS_TORCH and torch is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("已释放 GPU 显存缓存")
            except Exception as e:  # pragma: no cover
                logger.warning("释放 GPU 显存失败: %s", e)
        return True

    def _recover_node_offline(self, event: AnomalyEvent) -> bool:
        if event.node_id:
            self.isolate_node(event.node_id)
        return True

    def _recover_load_burst(self, event: AnomalyEvent) -> bool:
        # 负载爆表：仅记录，等待调度器降级
        logger.info("节点 %s 负载爆表，建议降级调度", event.node_id)
        return True
