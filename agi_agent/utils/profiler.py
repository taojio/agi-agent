"""
utils/profiler.py - 性能分析工具

提供函数级、模块级的性能分析和计时功能，支持：
- 函数执行时间统计
- 性能热点追踪
- 内存使用监控
- 调用频率统计

使用方式：
    @profile_function
    def my_function():
        pass
    
    with profile_context("task_name"):
        execute_task()
"""
import time
import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


class FunctionProfiler:
    """
    函数性能分析器
    
    统计函数执行时间、调用次数、平均耗时等指标。
    """
    
    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "min_time": float("inf"), "max_time": 0.0}
        )
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def profile(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._enabled:
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                func_name = f"{func.__module__}.{func.__qualname__}"
                stats = self.stats[func_name]
                stats["count"] += 1
                stats["total_time"] += elapsed
                stats["min_time"] = min(stats["min_time"], elapsed)
                stats["max_time"] = max(stats["max_time"], elapsed)
                
                if elapsed > 0.1:
                    logger.warning(f"Slow function: {func_name} took {elapsed:.3f}s")
        
        return wrapper

    def get_stats(self, sort_by: str = "total_time", limit: int = 20) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            sort_by: 排序字段 (total_time | count | avg_time)
            limit: 返回数量限制
            
        Returns:
            统计信息字典
        """
        results = []
        for func_name, stats in self.stats.items():
            count = stats["count"]
            total_time = stats["total_time"]
            avg_time = total_time / count if count > 0 else 0.0
            results.append({
                "function": func_name,
                "count": count,
                "total_time": total_time,
                "avg_time": avg_time,
                "min_time": stats["min_time"],
                "max_time": stats["max_time"],
            })
        
        sort_key_map = {
            "total_time": lambda x: x["total_time"],
            "count": lambda x: x["count"],
            "avg_time": lambda x: x["avg_time"],
        }
        sort_key = sort_key_map.get(sort_by, lambda x: x["total_time"])
        results.sort(key=sort_key, reverse=True)
        
        return {
            "total_functions": len(self.stats),
            "total_calls": sum(s["count"] for s in self.stats.values()),
            "total_time": sum(s["total_time"] for s in self.stats.values()),
            "top_functions": results[:limit],
        }

    def reset(self):
        """重置统计信息"""
        self.stats.clear()


_global_profiler = FunctionProfiler()


def profile_function(func: Callable) -> Callable:
    """
    装饰器：性能分析单个函数
    
    Args:
        func: 待分析的函数
        
    Returns:
        包装后的函数
    """
    return _global_profiler.profile(func)


def get_profiler() -> FunctionProfiler:
    """
    获取全局性能分析器实例
    
    Returns:
        FunctionProfiler 实例
    """
    return _global_profiler


class ProfileContext:
    """
    上下文管理器：性能分析代码块
    
    使用方式：
        with ProfileContext("task_name"):
            execute_task()
    """
    
    def __init__(self, name: str, log_threshold: float = 0.1):
        self.name = name
        self.log_threshold = log_threshold
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.log_threshold:
            logger.info(f"Profile [{self.name}]: {elapsed:.3f}s")
        return False


def profile_context(name: str, log_threshold: float = 0.1):
    """
    创建性能分析上下文管理器
    
    Args:
        name: 上下文名称
        log_threshold: 日志输出阈值（秒）
        
    Returns:
        ProfileContext 实例
    """
    return ProfileContext(name, log_threshold)


class ExecutionTimer:
    """
    执行计时器
    
    用于测量代码执行时间，支持暂停/继续。
    """
    
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0
        self.running = False

    def start(self):
        """开始计时"""
        if not self.running:
            self.start_time = time.perf_counter()
            self.running = True

    def stop(self):
        """停止计时"""
        if self.running:
            self.elapsed += time.perf_counter() - self.start_time
            self.running = False

    def pause(self):
        """暂停计时"""
        self.stop()

    def resume(self):
        """继续计时"""
        self.start()

    def reset(self):
        """重置计时器"""
        self.start_time = 0.0
        self.elapsed = 0.0
        self.running = False

    def get_elapsed(self) -> float:
        """获取已流逝时间"""
        if self.running:
            return self.elapsed + (time.perf_counter() - self.start_time)
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False