"""
utils/async_utils.py - 异步工具

提供异步执行、并发控制、线程池管理等工具函数，支持：
- 异步执行器（AsyncExecutor）
- 并发任务管理（TaskPool）
- 超时控制（timeout decorator）
- 线程安全缓存（ThreadSafeCache）
- 异步锁管理（AsyncLockManager）

使用方式：
    from agi_agent.utils import AsyncExecutor, TaskPool, timeout
    
    executor = AsyncExecutor()
    result = await executor.submit(func, arg1, arg2)
    
    @timeout(seconds=5)
    async def my_task():
        pass
"""
import asyncio
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """
    异步执行器
    
    提供同步函数的异步包装、线程池执行、进程池执行等功能。
    """
    
    def __init__(self, max_workers: int = 4, use_process_pool: bool = False):
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._process_pool = None
        if use_process_pool:
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步提交任务到线程池
        
        Args:
            func: 待执行函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数执行结果
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._thread_pool, func, *args, **kwargs
        )
    
    async def submit_process(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步提交任务到进程池
        
        Args:
            func: 待执行函数（必须可序列化）
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数执行结果
        """
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return await asyncio.get_event_loop().run_in_executor(
            self._process_pool, func, *args, **kwargs
        )
    
    def submit_sync(self, func: Callable, *args, **kwargs) -> Future:
        """
        同步提交任务到线程池
        
        Args:
            func: 待执行函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            Future 对象
        """
        return self._thread_pool.submit(func, *args, **kwargs)
    
    async def gather(self, tasks: List[Callable], args_list: Optional[List[Tuple]] = None):
        """
        并发执行多个任务
        
        Args:
            tasks: 任务函数列表
            args_list: 参数列表（与tasks一一对应）
        
        Returns:
            结果列表
        """
        if args_list is None:
            args_list = [() for _ in tasks]
        
        async_tasks = [
            self.submit(task, *args)
            for task, args in zip(tasks, args_list)
        ]
        
        return await asyncio.gather(*async_tasks, return_exceptions=True)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        关闭执行器
        
        Args:
            wait: 是否等待所有任务完成
        """
        self._thread_pool.shutdown(wait=wait)
        if self._process_pool:
            self._process_pool.shutdown(wait=wait)


class TaskPool:
    """
    任务池
    
    管理一组异步任务，支持并发控制、超时管理、错误处理。
    """
    
    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: List[asyncio.Task] = []
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行单个任务（带并发控制）
        
        Args:
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            任务结果
        """
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs) if asyncio.iscoroutinefunction(func)
                    else AsyncExecutor().submit(func, *args, **kwargs),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Task timeout after {self.timeout}s")
                raise
            except Exception as e:
                logger.error(f"Task failed: {e}")
                raise
    
    async def execute_all(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """
        执行所有任务
        
        Args:
            tasks: 任务列表，每个任务为 (func, args, kwargs)
        
        Returns:
            结果列表
        """
        async_tasks = []
        for func, args, kwargs in tasks:
            async_tasks.append(self.execute(func, *args, **kwargs))
        
        return await asyncio.gather(*async_tasks, return_exceptions=True)
    
    def add_task(self, func: Callable, *args, **kwargs) -> asyncio.Task:
        """
        添加任务到任务池
        
        Args:
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            asyncio.Task 对象
        """
        task = asyncio.create_task(self.execute(func, *args, **kwargs))
        self._tasks.append(task)
        return task
    
    async def wait_all(self) -> List[Any]:
        """
        等待所有任务完成
        
        Returns:
            结果列表
        """
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        return results
    
    def get_pending_count(self) -> int:
        """
        获取待处理任务数
        
        Returns:
            待处理任务数量
        """
        return len([t for t in self._tasks if not t.done()])


def timeout(seconds: float = 30.0):
    """
    装饰器：超时控制
    
    Args:
        seconds: 超时时间（秒）
    
    Returns:
        装饰后的函数
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        asyncio.wait_for(
                            loop.run_in_executor(None, func, *args, **kwargs),
                            timeout=seconds
                        )
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")
                finally:
                    loop.close()
            return sync_wrapper
    return decorator


class ThreadSafeCache:
    """
    线程安全缓存
    
    提供线程安全的键值存储，支持过期时间、最大容量限制。
    """
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
        
        Returns:
            缓存值或默认值
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return default
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒，默认使用全局ttl）
        """
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict()
            
            self._cache[key] = (value, time.time())
    
    def delete(self, key: str) -> None:
        """
        删除缓存值
        
        Args:
            key: 缓存键
        """
        with self._lock:
            self._cache.pop(key, None)
    
    def _evict(self) -> None:
        """
        驱逐过期或最旧的缓存
        """
        now = time.time()
        expired = [k for k, (_, t) in self._cache.items() if now - t > self.ttl]
        for k in expired:
            del self._cache[k]
        
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
    
    def clear(self) -> None:
        """
        清空缓存
        """
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            now = time.time()
            expired_count = sum(1 for _, t in self._cache.values() if now - t > self.ttl)
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "expired_count": expired_count,
            }


class AsyncLockManager:
    """
    异步锁管理器
    
    提供命名锁管理，支持分布式锁模拟。
    """
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_lock = asyncio.Lock()
    
    async def acquire(self, name: str) -> asyncio.Lock:
        """
        获取命名锁
        
        Args:
            name: 锁名称
        
        Returns:
            锁对象（已获取）
        """
        async with self._lock_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
        
        lock = self._locks[name]
        await lock.acquire()
        return lock
    
    def release(self, name: str) -> None:
        """
        释放命名锁
        
        Args:
            name: 锁名称
        """
        if name in self._locks:
            try:
                self._locks[name].release()
            except RuntimeError:
                pass
    
    @contextmanager
    def sync_lock(self, name: str):
        """
        同步锁上下文管理器
        
        Args:
            name: 锁名称
        
        Yields:
            None
        """
        lock = self._locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[name] = lock
        
        try:
            lock.acquire()
            yield
        finally:
            try:
                lock.release()
            except RuntimeError:
                pass
    
    def get_lock(self, name: str) -> asyncio.Lock:
        """
        获取锁对象（不自动获取）
        
        Args:
            name: 锁名称
        
        Returns:
            锁对象
        """
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]
    
    def is_locked(self, name: str) -> bool:
        """
        检查锁是否被持有
        
        Args:
            name: 锁名称
        
        Returns:
            是否被持有
        """
        lock = self._locks.get(name)
        if lock is None:
            return False
        return lock.locked()


_global_executor: Optional[AsyncExecutor] = None
_global_cache: Optional[ThreadSafeCache] = None
_global_lock_manager: Optional[AsyncLockManager] = None


def get_executor(max_workers: int = 4) -> AsyncExecutor:
    """
    获取全局异步执行器
    
    Args:
        max_workers: 最大工作线程数
    
    Returns:
        AsyncExecutor 实例
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = AsyncExecutor(max_workers=max_workers)
    return _global_executor


def get_cache(max_size: int = 1000, ttl: float = 3600.0) -> ThreadSafeCache:
    """
    获取全局线程安全缓存
    
    Args:
        max_size: 最大容量
        ttl: 过期时间（秒）
    
    Returns:
        ThreadSafeCache 实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ThreadSafeCache(max_size=max_size, ttl=ttl)
    return _global_cache


def get_lock_manager() -> AsyncLockManager:
    """
    获取全局异步锁管理器
    
    Returns:
        AsyncLockManager 实例
    """
    global _global_lock_manager
    if _global_lock_manager is None:
        _global_lock_manager = AsyncLockManager()
    return _global_lock_manager


async def run_async(func: Callable, *args, **kwargs) -> Any:
    """
    便捷函数：异步执行同步函数
    
    Args:
        func: 函数
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        函数执行结果
    """
    return await AsyncExecutor().submit(func, *args, **kwargs)