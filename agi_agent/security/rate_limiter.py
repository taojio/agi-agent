"""
security/rate_limiter.py - 速率限制系统

基于内存（可扩展至 Redis）的滑动窗口速率限制器。
支持 IP 级和用户级双重限流。
"""
import time
import threading
from collections import defaultdict, deque
from typing import Callable, Dict, Optional, Tuple

from .exceptions import RateLimitException


class SlidingWindowLimiter:
    """滑动窗口速率限制器（内存版）"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, Dict[str, int]]:
        """
        检查是否超过速率限制

        Args:
            key: 限流键（如 IP 地址、用户 ID）

        Returns:
            (是否允许, 限流状态信息)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            requests = self._requests[key]

            while requests and requests[0] < window_start:
                requests.popleft()

            current_count = len(requests)
            remaining = self.max_requests - current_count - 1
            reset_time = int(requests[0] + self.window_seconds) if requests else int(now + self.window_seconds)

            if current_count >= self.max_requests:
                return False, {
                    "limit": self.max_requests,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": reset_time - int(now),
                }

            requests.append(now)

            return True, {
                "limit": self.max_requests,
                "remaining": max(0, remaining),
                "reset": reset_time,
                "retry_after": 0,
            }

    def reset(self, key: str):
        """重置某个键的限流计数"""
        with self._lock:
            self._requests.pop(key, None)

    def cleanup(self):
        """清理过期记录"""
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            keys_to_remove = []
            for key, requests in self._requests.items():
                while requests and requests[0] < window_start:
                    requests.popleft()
                if not requests:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._requests[key]


class RateLimiter:
    """综合速率限制器"""

    def __init__(self):
        self._limiters: Dict[str, SlidingWindowLimiter] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

    def _get_limiter(self, name: str, max_requests: int, window_seconds: int) -> SlidingWindowLimiter:
        if name not in self._limiters:
            with self._lock:
                if name not in self._limiters:
                    self._limiters[name] = SlidingWindowLimiter(max_requests, window_seconds)
        return self._limiters[name]

    def check(
        self,
        limiter_name: str,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> Tuple[bool, Dict[str, int]]:
        """
        检查速率限制

        Args:
            limiter_name: 限流器名称
            key: 限流键
            max_requests: 最大请求数
            window_seconds: 时间窗口（秒）

        Returns:
            (是否允许, 状态信息)
        """
        limiter = self._get_limiter(limiter_name, max_requests, window_seconds)
        return limiter.check(key)

    def check_login(self, ip: str) -> Tuple[bool, Dict[str, int]]:
        """登录限流：5次/分钟/IP"""
        return self.check("login", ip, 5, 60)

    def check_register(self, ip: str) -> Tuple[bool, Dict[str, int]]:
        """注册限流：3次/小时/IP"""
        return self.check("register", ip, 3, 3600)

    def check_api(self, user_id: str) -> Tuple[bool, Dict[str, int]]:
        """普通API限流：100次/分钟/用户"""
        return self.check("api_user", user_id, 100, 60)

    def check_api_ip(self, ip: str) -> Tuple[bool, Dict[str, int]]:
        """普通API限流（IP级）：200次/分钟/IP"""
        return self.check("api_ip", ip, 200, 60)

    def check_admin(self, user_id: str) -> Tuple[bool, Dict[str, int]]:
        """管理API限流：30次/分钟/用户"""
        return self.check("admin", user_id, 30, 60)

    def check_file_upload(self, user_id: str) -> Tuple[bool, Dict[str, int]]:
        """文件上传限流：10次/小时/用户"""
        return self.check("file_upload", user_id, 10, 3600)

    def check_password_reset(self, ip: str) -> Tuple[bool, Dict[str, int]]:
        """密码重置限流：5次/小时/IP"""
        return self.check("password_reset", ip, 5, 3600)

    def periodic_cleanup(self):
        """定期清理过期记录"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        with self._lock:
            for limiter in self._limiters.values():
                limiter.cleanup()
        self._last_cleanup = now


_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def reset_rate_limiter():
    """重置全局速率限制器（测试用）"""
    global _global_rate_limiter
    _global_rate_limiter = None
