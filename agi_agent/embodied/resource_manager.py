import os
import sys
import gc
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ResourceUsage:
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    disk_io_mb: float = 0.0


class ResourceOptimizer:
    def __init__(self, target_memory_gb: float = 2.0,
                 target_cpu_percent: float = 80.0):
        self.target_memory_gb = target_memory_gb
        self.target_cpu_percent = target_cpu_percent

        self.current_usage = ResourceUsage()
        self.usage_history = []

        self._monitoring = False
        self._monitor_thread = None

        self._optimization_level = "balanced"
        self._gc_interval = 300
        self._last_gc_time = 0

        self._optimization_strategies = {
            "minimal": self._strategy_minimal,
            "balanced": self._strategy_balanced,
            "aggressive": self._strategy_aggressive,
        }

    def start_monitoring(self, interval: float = 5.0):
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._monitoring = False

    def _monitor_loop(self, interval: float):
        while self._monitoring:
            try:
                self._update_usage()
                self._check_and_optimize()
            except Exception:
                pass
            time.sleep(interval)

    def _update_usage(self):
        usage = ResourceUsage()

        try:
            import psutil
            process = psutil.Process(os.getpid())
            usage.cpu_percent = process.cpu_percent(interval=0.1)
            usage.memory_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            usage.memory_mb = self._estimate_memory()

        try:
            import torch
            if torch.cuda.is_available():
                usage.gpu_percent = torch.cuda.utilization()
                usage.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass

        self.current_usage = usage
        self.usage_history.append({
            "timestamp": time.time(),
            **usage.__dict__
        })
        if len(self.usage_history) > 100:
            self.usage_history.pop(0)

    def _estimate_memory(self) -> float:
        import gc
        total = 0
        for obj in gc.get_objects():
            try:
                if hasattr(obj, '__sizeof__'):
                    total += obj.__sizeof__()
            except Exception:
                pass
        return total / (1024 * 1024)

    def _check_and_optimize(self):
        memory_gb = self.current_usage.memory_mb / 1024

        if memory_gb > self.target_memory_gb * 1.2:
            self._trigger_optimization("memory_high")
        elif self.current_usage.cpu_percent > self.target_cpu_percent:
            self._trigger_optimization("cpu_high")

        now = time.time()
        if now - self._last_gc_time > self._gc_interval:
            self._run_gc()
            self._last_gc_time = now

    def _trigger_optimization(self, reason: str):
        strategy = self._optimization_strategies.get(
            self._optimization_level,
            self._strategy_balanced
        )
        strategy(reason)

    def _strategy_minimal(self, reason: str):
        self._run_gc()

    def _strategy_balanced(self, reason: str):
        self._run_gc()
        self._clear_caches()

    def _strategy_aggressive(self, reason: str):
        self._run_gc()
        self._clear_caches()
        self._reduce_torch_cache()

    def _run_gc(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _clear_caches(self):
        for obj in gc.get_objects():
            try:
                if hasattr(obj, 'cache_clear'):
                    obj.cache_clear()
            except Exception:
                pass

    def _reduce_torch_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def set_optimization_level(self, level: str):
        if level in self._optimization_strategies:
            self._optimization_level = level

    def get_usage_report(self) -> Dict[str, Any]:
        return {
            "current": {
                "cpu_percent": self.current_usage.cpu_percent,
                "memory_mb": self.current_usage.memory_mb,
                "gpu_percent": self.current_usage.gpu_percent,
                "gpu_memory_mb": self.current_usage.gpu_memory_mb,
            },
            "optimization_level": self._optimization_level,
            "target_memory_gb": self.target_memory_gb,
            "target_cpu_percent": self.target_cpu_percent,
            "monitoring": self._monitoring,
        }
