"""
infrastructure/load_balancer.py - 推理负载均衡子模块

包含任务：
- T002 推理负载均衡（InferenceLoadBalancer）

设计原则：
1. 重型依赖（torch）可选导入，无 GPU 时降级为单 CPU 设备
2. numpy 可选导入，缺失时使用纯 Python 实现
3. 接收 InferenceRequest（含 model_id, batch_size, priority），
   结合 ResourceInspectionTask 数据分配分片到设备
4. 维护多设备负载表，避免单设备过载/显存溢出
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.infrastructure")

# ====== 可选依赖 ======
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


# ====== 数据结构 ======

@dataclass
class InferenceRequest:
    """推理请求（T002）"""
    model_id: str
    batch_size: int = 1
    priority: int = 0                  # 数值越大优先级越高
    payload: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    created_at: float = 0.0


@dataclass
class AllocationPlan:
    """分配方案（T002）"""
    request_id: str
    device_id: str
    shard_size: int
    estimated_latency_ms: float = 0.0
    memory_required_mb: float = 0.0
    accepted: bool = True
    reason: str = ""
    timestamp: float = 0.0


@dataclass
class DeviceInfo:
    """设备负载表条目"""
    device_id: str
    device_type: str  # "cpu" / "gpu"
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    load_percent: float = 0.0          # 0-100
    max_batch_size: int = 32
    enabled: bool = True


@dataclass
class LoadBalancerConfig:
    """T002 配置"""
    cpu_device_id: str = "cpu:0"
    cpu_max_batch_size: int = 8
    cpu_total_memory_mb: float = 4096.0
    gpu_max_batch_size: int = 32
    gpu_memory_safety_margin_mb: float = 512.0  # 显存安全余量
    overload_threshold: float = 90.0   # 负载超过此阈值视为过载
    enable_gpu: bool = True


# ====== T002: 推理负载均衡 ======

class InferenceLoadBalancer(BaseModule):
    """T002 推理负载均衡

    动态调度：接收推理请求，结合 ResourceInspectionTask 数据分配分片到设备。
    维护多设备负载表，避免单设备过载/显存溢出。
    无 GPU 时降级为单 CPU 设备。
    """

    name = "inference_load_balancer"
    version = "1.0.0"
    description = "T002 推理负载均衡：动态调度，多设备负载表，避免过载/显存溢出"

    def __init__(self, config: Optional[LoadBalancerConfig] = None) -> None:
        super().__init__()
        self.config: LoadBalancerConfig = config or LoadBalancerConfig()
        self._devices: Dict[str, DeviceInfo] = {}
        self._pending: List[InferenceRequest] = []
        self._plans: Dict[str, AllocationPlan] = {}
        self._lock = threading.Lock()
        # 初始化设备表
        self._init_devices()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        self._init_devices()
        logger.info("InferenceLoadBalancer 初始化完成，设备数=%d", len(self._devices))

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        with self._lock:
            self._devices.clear()
            self._pending.clear()
            self._plans.clear()

    def _health_check(self) -> bool:
        with self._lock:
            return len(self._devices) > 0

    # ---- 公共方法 ----
    def submit(self, request: InferenceRequest) -> str:
        """提交推理请求，返回 task_id"""
        if not request.request_id:
            request.request_id = f"req_{uuid.uuid4().hex[:8]}"
        if not request.created_at:
            request.created_at = time.time()
        plan = self.allocate(request)
        with self._lock:
            self._plans[request.request_id] = plan
            if plan.accepted:
                # 占用设备资源
                dev = self._devices.get(plan.device_id)
                if dev is not None:
                    dev.used_memory_mb += plan.memory_required_mb
                    dev.load_percent = self._compute_load(dev)
            else:
                self._pending.append(request)
        logger.info(
            "提交推理请求 id=%s model=%s batch=%d -> 设备=%s 接受=%s",
            request.request_id, request.model_id, request.batch_size,
            plan.device_id, plan.accepted,
        )
        return request.request_id

    def allocate(self, request: InferenceRequest) -> AllocationPlan:
        """根据当前设备负载表为请求生成分配方案"""
        with self._lock:
            # 拷贝一份设备视图，避免在锁外修改
            devices = [d for d in self._devices.values() if d.enabled]
        if not devices:
            return AllocationPlan(
                request_id=request.request_id,
                device_id="",
                shard_size=0,
                accepted=False,
                reason="无可用设备",
                timestamp=time.time(),
            )
        # 选择负载最低且能容纳的设备
        candidates: List[DeviceInfo] = sorted(
            devices, key=lambda d: (d.load_percent, d.used_memory_mb)
        )
        chosen: Optional[DeviceInfo] = None
        for dev in candidates:
            if dev.load_percent >= self.config.overload_threshold:
                continue
            if dev.device_type == "gpu":
                free_mb = (dev.total_memory_mb - dev.used_memory_mb
                           - self.config.gpu_memory_safety_margin_mb)
                if free_mb <= 0:
                    continue
            chosen = dev
            break
        if chosen is None:
            # 所有设备过载，挑选负载最低的强制接受（CPU 兜底）
            cpu_dev = next((d for d in candidates if d.device_type == "cpu"), None)
            chosen = cpu_dev or candidates[0]
        # 分片大小：受设备 max_batch_size 限制
        shard = max(1, min(request.batch_size, chosen.max_batch_size))
        # 估算显存/内存占用（简单线性模型）
        mem_per_item = self._estimate_memory_per_item(chosen, request.model_id)
        mem_required = mem_per_item * shard
        # 校验显存余量
        accepted = True
        reason = ""
        if chosen.device_type == "gpu":
            free_mb = (chosen.total_memory_mb - chosen.used_memory_mb
                       - self.config.gpu_memory_safety_margin_mb)
            if mem_required > free_mb:
                # 降级：缩小分片
                shard = max(1, int(free_mb / max(mem_per_item, 1e-6)))
                mem_required = mem_per_item * shard
                if shard <= 0:
                    accepted = False
                    reason = "GPU 显存不足"
        latency = self._estimate_latency(chosen, shard, request.model_id)
        return AllocationPlan(
            request_id=request.request_id,
            device_id=chosen.device_id,
            shard_size=shard,
            estimated_latency_ms=latency,
            memory_required_mb=mem_required,
            accepted=accepted,
            reason=reason,
            timestamp=time.time(),
        )

    def get_device_load(self, device_id: str) -> Optional[DeviceInfo]:
        """获取指定设备的负载信息"""
        with self._lock:
            dev = self._devices.get(device_id)
            return DeviceInfo(**dev.__dict__) if dev else None

    def get_device_loads(self) -> List[DeviceInfo]:
        """获取全部设备负载"""
        with self._lock:
            return [DeviceInfo(**d.__dict__) for d in self._devices.values()]

    def release(self, request_id: str) -> bool:
        """释放已分配资源"""
        with self._lock:
            plan = self._plans.pop(request_id, None)
            if plan is None:
                return False
            dev = self._devices.get(plan.device_id)
            if dev is not None:
                dev.used_memory_mb = max(0.0, dev.used_memory_mb - plan.memory_required_mb)
                dev.load_percent = self._compute_load(dev)
            return True

    def update_device_resource(self, device_id: str,
                               load_percent: Optional[float] = None,
                               used_memory_mb: Optional[float] = None,
                               total_memory_mb: Optional[float] = None) -> bool:
        """供 ResourceInspectionTask 回写设备实时负载"""
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                return False
            if load_percent is not None:
                dev.load_percent = float(load_percent)
            if used_memory_mb is not None:
                dev.used_memory_mb = float(used_memory_mb)
            if total_memory_mb is not None:
                dev.total_memory_mb = float(total_memory_mb)
            return True

    # ---- 内部 ----
    def _init_devices(self) -> None:
        self._devices.clear()
        # CPU 设备始终存在
        self._devices[self.config.cpu_device_id] = DeviceInfo(
            device_id=self.config.cpu_device_id,
            device_type="cpu",
            total_memory_mb=self.config.cpu_total_memory_mb,
            used_memory_mb=0.0,
            load_percent=0.0,
            max_batch_size=self.config.cpu_max_batch_size,
            enabled=True,
        )
        # GPU 设备：torch.cuda 可用时挂载
        if self.config.enable_gpu and _HAS_TORCH and torch is not None:
            try:
                count = torch.cuda.device_count()
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    total_mb = float(props.total_memory) / (1024 * 1024)
                    self._devices[f"gpu:{i}"] = DeviceInfo(
                        device_id=f"gpu:{i}",
                        device_type="gpu",
                        total_memory_mb=total_mb,
                        used_memory_mb=0.0,
                        load_percent=0.0,
                        max_batch_size=self.config.gpu_max_batch_size,
                        enabled=True,
                    )
            except Exception as e:  # pragma: no cover
                logger.warning("GPU 设备探测失败，仅使用 CPU: %s", e)

    def _compute_load(self, dev: DeviceInfo) -> float:
        if dev.total_memory_mb <= 0:
            return dev.load_percent
        mem_load = (dev.used_memory_mb / dev.total_memory_mb) * 100.0
        # 取内存负载与已有负载的最大值，避免低估
        return max(dev.load_percent, min(100.0, mem_load))

    def _estimate_memory_per_item(self, dev: DeviceInfo, model_id: str) -> float:
        """简化估算：CPU/GPU 每条样本占用内存"""
        # 简单哈希让不同模型有差异
        base = 64.0 if dev.device_type == "gpu" else 128.0
        try:
            bias = float(hash(model_id) % 64)
        except Exception:
            bias = 0.0
        return base + bias

    def _estimate_latency(self, dev: DeviceInfo, shard: int, model_id: str) -> float:
        """简化估算：latency = shard * per_item_latency * (1 + load/100)"""
        per_item = 5.0 if dev.device_type == "gpu" else 20.0
        load_factor = 1.0 + dev.load_percent / 100.0
        return shard * per_item * load_factor
