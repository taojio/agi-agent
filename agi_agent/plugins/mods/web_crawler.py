from agi_agent.plugins.plugin_base import (
    PeripheralPlugin, PluginPriority, PluginHookPoint, PluginStatus
)
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import threading
import logging
import time
import hashlib
import random
import collections
import urllib.robotparser
import heapq
import re
import json
import uuid
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree


class AdvancedWebCrawlerPlugin(PeripheralPlugin):
    """高性能异步分布式爬虫插件，为智能体提供网页数据采集能力。

    增强版特性：
    - 优先级任务队列（heapq 实现，0=最高，10=最低）
    - BFS 深度爬取（max_depth 控制，仅同域名扩散）
    - 域名级限速（domain_delay 配置）
    - 结构化数据抽取（OpenGraph / JSON-LD / meta / canonical / RSS）
    - 站点地图解析（支持 sitemap index 递归）
    - 内容清洗与主内容区提取
    - 暂停/恢复控制
    - 单/批量结果查询
    - 分布式结果回写 Redis
    - 完善的异常恢复与任务重试
    """

    def __init__(self):
        super().__init__(
            name="advanced_web_crawler",
            version="2.0.0",
            description="企业级异步爬虫，支持增量去重、反爬规避、分布式扩展、多格式解析、BFS深度爬取、结构化数据抽取",
            plugin_type="processor",
            priority=PluginPriority.NORMAL,
            config={
                "max_concurrent": 30,                # 最大并发请求数
                "request_timeout": 15,                # 单请求超时时间(秒)
                "min_request_delay": 0.3,            # 最小请求间隔(秒)
                "max_request_delay": 1.5,             # 最大请求间隔(秒)
                "max_retry_times": 3,                 # 失败最大重试次数
                "respect_robots": True,               # 是否遵守robots协议
                "enable_bloom_filter": False,         # 是否启用布隆过滤器(大规模去重)
                "redis_url": None,                    # Redis连接地址，启用分布式模式需填写
                "max_task_queue": 20000,              # 任务队列最大长度
                "result_cache_ttl": 7200,             # 结果缓存时长(秒)
                "max_depth": 5,                       # BFS爬取最大深度
                "seed_urls": [],                      # 种子URL列表，激活后自动入队
                "domain_delay": 1.0,                  # 单域名最小请求间隔(秒)
                "max_content_length": 50000,          # 单页内容最大字符数
                "user_agent_pool": [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"
                ],
                "proxy_list": []                      # 代理池列表，格式: http://user:pass@ip:port
            },
            dependencies=[],
            compatible_versions=["1.0.0", "2.0.0"],
            hook_points=[PluginHookPoint.PERIODIC, PluginHookPoint.PRE_COGNITION]
        )

        # 核心运行态变量
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._session = None
        # 优先级队列：使用 heapq 维护 (priority, counter, task) 元组
        self._task_queue: List[Tuple[int, int, dict]] = []
        self._task_counter: int = 0
        self._seen_urls: set = set()
        self._content_hashes: set = set()
        self._result_cache: Dict[str, dict] = {}
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._running: bool = False
        self._paused: bool = False
        self._lock = threading.Lock()

        # 域名级限速与统计
        self._domain_last_request: Dict[str, float] = {}
        self._domain_stats: Dict[str, dict] = {}

        # Redis 连接复用（避免每次拉取任务都重建连接）
        self._redis_conn = None

        # 可选依赖在 on_load 中填充，这里预置默认值以保证 get_data 在加载前可调用
        self._has_redis = False
        self._redis_lib = None
        self._aiohttp = None
        self._BeautifulSoup = None
        # 布隆过滤器
        self._bloom_filter = None

        # 统计指标
        self._stats = {
            "total_requests": 0,
            "success_count": 0,
            "fail_count": 0,
            "pending_tasks": 0,
            "crawled_pages": 0,
            "total_bytes_downloaded": 0,
            "total_content_length": 0,
            "avg_response_time": 0.0
        }
        # 用于计算平均响应时间的滚动窗口
        self._response_times: List[float] = []

        self._logger = logging.getLogger("advanced_web_crawler")

    # ------------------------------ 生命周期方法 ------------------------------
    def on_load(self) -> bool:
        """插件加载：依赖检查与基础资源初始化"""
        # 核心依赖校验
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            self._aiohttp = aiohttp
            self._BeautifulSoup = BeautifulSoup
        except ImportError as e:
            self._last_error = f"核心依赖缺失: {str(e)}，请执行 pip install aiohttp beautifulsoup4 lxml"
            self._error_count += 1
            return False

        # 可选依赖检测：Redis
        self._has_redis = False
        try:
            import redis
            self._redis_lib = redis
            if self.config["redis_url"]:
                self._has_redis = True
        except ImportError:
            self._logger.warning("未安装redis库，分布式集群模式不可用")

        # 布隆过滤器可选依赖
        self._bloom_filter = None
        if self.config["enable_bloom_filter"]:
            try:
                from pybloom_live import ScalableBloomFilter
                self._bloom_filter = ScalableBloomFilter(
                    initial_capacity=100000, error_rate=0.001
                )
            except ImportError:
                self._logger.warning("未安装pybloom_live，布隆过滤器已禁用")

        self._logger.info("爬虫插件加载完成")
        return True

    def on_activate(self) -> bool:
        """插件激活：启动事件循环与爬虫引擎"""
        if self._running:
            return True

        try:
            # Redis 连接复用：在激活阶段建立一次连接，后续任务拉取与结果回写共享
            if self._has_redis and self._redis_conn is None:
                try:
                    self._redis_conn = self._redis_lib.from_url(
                        self.config["redis_url"], decode_responses=True
                    )
                    self._redis_conn.ping()
                    self._logger.info("Redis连接已建立")
                except Exception as e:
                    self._logger.warning(f"Redis连接失败，分布式模式降级: {str(e)}")
                    self._has_redis = False
                    self._redis_conn = None

            # 启动独立事件循环线程
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_event_loop, daemon=True, name="crawler-event-loop"
            )
            self._loop_thread.start()

            # 等待循环就绪
            time.sleep(0.1)

            # 异步初始化爬虫引擎
            asyncio.run_coroutine_threadsafe(self._init_crawler_engine(), self._loop)

            self._running = True
            self._logger.info("爬虫引擎已激活，开始处理任务")
            return True

        except Exception as e:
            self._last_error = f"激活失败: {str(e)}"
            self._error_count += 1
            return False

    def on_deactivate(self) -> bool:
        """插件停用：停止引擎，释放网络连接"""
        if not self._running:
            return True

        try:
            self._running = False

            # 关闭HTTP会话
            if self._session and self._loop:
                close_future = asyncio.run_coroutine_threadsafe(
                    self._session.close(), self._loop
                )
                try:
                    close_future.result(timeout=5)
                except Exception as e:
                    self._logger.warning(f"关闭HTTP会话异常: {str(e)}")

            # 停止事件循环
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            # 等待线程退出
            if self._loop_thread:
                self._loop_thread.join(timeout=5)

            # 关闭Redis连接
            if self._redis_conn is not None:
                try:
                    self._redis_conn.close()
                except Exception:
                    pass
                self._redis_conn = None

            self._logger.info("爬虫引擎已停用")
            return True

        except Exception as e:
            self._last_error = f"停用异常: {str(e)}"
            self._error_count += 1
            return False

    def on_unload(self) -> bool:
        """插件卸载：彻底释放所有资源，确保零泄漏"""
        try:
            # 先确保引擎停止
            if self._running:
                self.on_deactivate()

            # 清空所有内存资源
            with self._lock:
                self._task_queue.clear()
                self._task_counter = 0
                self._seen_urls.clear()
                self._content_hashes.clear()
                self._result_cache.clear()
                self._robots_cache.clear()
                self._domain_last_request.clear()
                self._domain_stats.clear()
                self._response_times.clear()
                self._bloom_filter = None
                self._redis_conn = None
                self._paused = False
                self._stats = {
                    "total_requests": 0,
                    "success_count": 0,
                    "fail_count": 0,
                    "pending_tasks": 0,
                    "crawled_pages": 0,
                    "total_bytes_downloaded": 0,
                    "total_content_length": 0,
                    "avg_response_time": 0.0
                }

            self._loop = None
            self._loop_thread = None
            self._session = None

            self._logger.info("爬虫插件已完全卸载，所有资源已释放")
            return True

        except Exception as e:
            self._last_error = f"卸载异常: {str(e)}"
            self._error_count += 1
            self._running = False
            return False

    # ------------------------------ 核心业务方法 ------------------------------
    def process(self, input_data: Any) -> Any:
        """统一入口：根据 action 字段分发到不同子任务
        默认行为（无 action 字段）等价于提交爬取任务，保留向后兼容
        """
        if not isinstance(input_data, dict):
            return {"status": "error", "error": "输入参数必须为字典"}

        action = input_data.get("action", "submit")

        # 写操作必须在引擎激活后执行；查询操作允许在未激活时调用
        write_actions = {"submit", "crawl_sitemap", "clear_cache", "pause", "resume", "reset_stats"}
        if not self._running and action in write_actions:
            self._last_error = "爬虫未激活，请先激活插件"
            self._error_count += 1
            return {"status": "error", "error": self._last_error}

        try:
            if action == "submit":
                return self._action_submit(input_data)
            elif action == "get_result":
                return self._action_get_result(input_data)
            elif action == "get_stats":
                return self._action_get_stats(input_data)
            elif action == "clear_cache":
                return self._action_clear_cache(input_data)
            elif action == "pause":
                return self._action_pause(input_data)
            elif action == "resume":
                return self._action_resume(input_data)
            elif action == "reset_stats":
                return self._action_reset_stats(input_data)
            elif action == "crawl_sitemap":
                return self._action_crawl_sitemap(input_data)
            else:
                return {"status": "error", "error": f"未知操作: {action}"}

        except Exception as e:
            self._last_error = f"处理失败: {str(e)}"
            self._error_count += 1
            return {"status": "error", "error": str(e)}

    def _action_submit(self, input_data: dict) -> dict:
        """提交爬取任务，异步执行，立即返回任务ID
        输入格式: {"action":"submit","urls":["https://xxx.com",...],"options":{"priority":0,"depth":0,"max_depth":2}}
        """
        if "urls" not in input_data:
            return {"status": "error", "error": "参数格式错误，必须包含 urls 列表"}

        urls = input_data["urls"]
        if not isinstance(urls, list):
            urls = [urls]

        options = input_data.get("options", {}) or {}
        priority = max(0, min(10, int(options.get("priority", 5))))
        depth = max(0, int(options.get("depth", 0)))
        max_depth = int(options.get("max_depth", self.config["max_depth"]))

        task_ids = []
        with self._lock:
            for url in urls:
                # URL去重
                if url in self._seen_urls:
                    continue
                if self._bloom_filter and url in self._bloom_filter:
                    continue

                if len(self._task_queue) >= self.config["max_task_queue"]:
                    self._logger.warning("任务队列已满，丢弃后续任务")
                    break

                task_id = hashlib.md5(f"{url}_{time.time_ns()}".encode()).hexdigest()[:10]
                task = {
                    "task_id": task_id,
                    "url": url,
                    "retry_count": 0,
                    "submit_time": time.time(),
                    "priority": priority,
                    "depth": depth,
                    "max_depth": max_depth
                }
                # 优先级队列：priority 小者优先，counter 用于稳定排序避免比较 dict
                heapq.heappush(self._task_queue, (priority, self._task_counter, task))
                self._task_counter += 1
                self._seen_urls.add(url)
                task_ids.append(task_id)

            self._stats["pending_tasks"] = len(self._task_queue)

        return {
            "status": "submitted",
            "task_ids": task_ids,
            "pending_count": len(self._task_queue)
        }

    def _action_get_result(self, input_data: dict) -> dict:
        """获取单个任务结果"""
        task_id = input_data.get("task_id")
        if not task_id:
            return {"status": "error", "error": "缺少 task_id 参数"}
        return self.get_result(task_id)

    def _action_get_stats(self, input_data: dict) -> dict:
        """获取运行统计"""
        return self.get_data()

    def _action_clear_cache(self, input_data: dict) -> dict:
        """清空结果缓存与已访问URL集合"""
        with self._lock:
            cache_count = len(self._result_cache)
            seen_count = len(self._seen_urls)
            self._result_cache.clear()
            self._seen_urls.clear()
            self._content_hashes.clear()
            if self._bloom_filter is not None:
                try:
                    from pybloom_live import ScalableBloomFilter
                    self._bloom_filter = ScalableBloomFilter(
                        initial_capacity=100000, error_rate=0.001
                    )
                except Exception:
                    self._bloom_filter = None
        return {
            "status": "ok",
            "cleared_results": cache_count,
            "cleared_seen_urls": seen_count
        }

    def _action_pause(self, input_data: dict) -> dict:
        """暂停爬虫工作协程（保留已采集结果，停止消费新任务）"""
        self._paused = True
        return {"status": "paused", "pending_count": len(self._task_queue)}

    def _action_resume(self, input_data: dict) -> dict:
        """恢复爬虫工作协程"""
        self._paused = False
        return {"status": "resumed", "pending_count": len(self._task_queue)}

    def _action_reset_stats(self, input_data: dict) -> dict:
        """重置所有统计计数器"""
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "success_count": 0,
                "fail_count": 0,
                "pending_tasks": len(self._task_queue),
                "crawled_pages": 0,
                "total_bytes_downloaded": 0,
                "total_content_length": 0,
                "avg_response_time": 0.0
            }
            self._response_times.clear()
            self._domain_stats.clear()
        return {"status": "ok"}

    def _action_crawl_sitemap(self, input_data: dict) -> dict:
        """解析站点地图并自动入队所有URL
        输入格式: {"action":"crawl_sitemap","sitemap_url":"https://xxx.com/sitemap.xml","options":{...}}
        """
        sitemap_url = input_data.get("sitemap_url")
        if not sitemap_url:
            return {"status": "error", "error": "缺少 sitemap_url 参数"}

        if not self._loop:
            return {"status": "error", "error": "事件循环未启动"}

        if self._session is None:
            return {"status": "error", "error": "爬虫会话未初始化"}

        future = asyncio.run_coroutine_threadsafe(
            self._parse_sitemap(sitemap_url), self._loop
        )
        try:
            urls = future.result(timeout=30)
        except Exception as e:
            return {"status": "error", "error": f"sitemap解析失败: {str(e)}"}

        # 复用 submit 路径将URL入队
        if not urls:
            return {"status": "ok", "discovered_urls": 0, "task_ids": []}

        submit_result = self._action_submit({
            "action": "submit",
            "urls": urls,
            "options": input_data.get("options", {})
        })
        submit_result["discovered_urls"] = len(urls)
        return submit_result

    def get_result(self, task_id: str) -> dict:
        """获取单个任务结果。结果未就绪时返回 pending 状态"""
        with self._lock:
            result = self._result_cache.get(task_id)
            if result is None:
                return {
                    "status": "pending",
                    "task_id": task_id,
                    "message": "任务结果尚未就绪"
                }
            return result

    def get_results_batch(self, task_ids: List[str]) -> List[dict]:
        """批量获取任务结果"""
        with self._lock:
            return [
                self._result_cache.get(tid, {
                    "status": "pending",
                    "task_id": tid,
                    "message": "任务结果尚未就绪"
                })
                for tid in task_ids
            ]

    def get_data(self) -> Dict[str, Any]:
        """获取插件运行状态与统计数据，可JSON序列化
        增强：包含域名级统计、平均响应时间、吞吐量等指标
        """
        with self._lock:
            stats = self._stats.copy()
            stats["queue_size"] = len(self._task_queue)
            stats["cache_count"] = len(self._result_cache)
            stats["seen_url_count"] = len(self._seen_urls)
            stats["paused"] = self._paused
            # 实时计算平均响应时间
            if self._response_times:
                stats["avg_response_time"] = sum(self._response_times) / len(self._response_times)
            else:
                stats["avg_response_time"] = 0.0
            # 吞吐量：每秒成功爬取页数
            uptime = 0.0
            if self.activate_time:
                uptime = max(time.time() - self.activate_time, 0.001)
            stats["throughput_pages_per_sec"] = (
                stats["crawled_pages"] / uptime if uptime else 0.0
            )
            # 域名级统计快照
            domain_stats = {
                domain: {
                    "requests": info.get("requests", 0),
                    "success": info.get("success", 0),
                    "fail": info.get("fail", 0),
                    "success_rate": (
                        info.get("success", 0) / info.get("requests", 1)
                        if info.get("requests", 0) > 0 else 0.0
                    ),
                    "avg_response_time": (
                        info.get("total_response_time", 0.0) / info.get("success", 1)
                        if info.get("success", 0) > 0 else 0.0
                    )
                }
                for domain, info in self._domain_stats.items()
            }

        return {
            "plugin_name": self.name,
            "version": self.version,
            "running_status": (
                "paused" if (self._running and self._paused)
                else ("active" if self._running else "loaded")
            ),
            "statistics": stats,
            "domain_stats": domain_stats,
            "config_snapshot": {
                "max_concurrent": self.config["max_concurrent"],
                "respect_robots": self.config["respect_robots"],
                "distributed_mode": self._has_redis,
                "max_depth": self.config["max_depth"],
                "domain_delay": self.config["domain_delay"],
                "max_content_length": self.config["max_content_length"]
            },
            "last_error": self._last_error,
            "error_count": self._error_count
        }

    # ------------------------------ 钩子方法 ------------------------------
    def hook_periodic(self, input_data: dict) -> dict:
        """周期钩子：清理过期缓存、同步分布式任务、更新状态"""
        # 清理过期结果缓存
        now = time.time()
        with self._lock:
            expired_keys = [
                tid for tid, res in self._result_cache.items()
                if now - res.get("timestamp", 0) > self.config["result_cache_ttl"]
            ]
            for key in expired_keys:
                del self._result_cache[key]

        # 分布式模式下从Redis拉取任务
        if self._has_redis and self._running:
            self._pull_redis_tasks()

        return input_data

    def hook_pre_cognition(self, input_data: dict) -> dict:
        """认知前钩子：将最近爬取结果注入认知上下文，供推理层使用"""
        try:
            recent_results = []
            with self._lock:
                # 取最近 5 条结果用于上下文
                items = list(self._result_cache.values())[-5:]
            for res in items:
                if res.get("status") == "success":
                    recent_results.append({
                        "url": res.get("url"),
                        "title": res.get("title"),
                        "summary": (res.get("main_text") or "")[:200],
                        "timestamp": res.get("timestamp")
                    })
            if isinstance(input_data, dict):
                input_data.setdefault("context", {})["recent_crawl_results"] = recent_results
            else:
                input_data = {
                    "original": input_data,
                    "context": {"recent_crawl_results": recent_results}
                }
        except Exception as e:
            self._logger.warning(f"hook_pre_cognition 注入失败: {str(e)}")
        return input_data

    # ------------------------------ 内部引擎实现 ------------------------------
    def _run_event_loop(self):
        """事件循环线程入口"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_crawler_engine(self):
        """初始化异步爬虫引擎与工作协程池
        修复点：移除 TraceConfig 中间件 hack，改为在每次请求时显式注入 User-Agent
        """
        timeout = self._aiohttp.ClientTimeout(total=self.config["request_timeout"])

        self._session = self._aiohttp.ClientSession(
            timeout=timeout,
            connector=self._aiohttp.TCPConnector(
                limit=self.config["max_concurrent"],
                ttl_dns_cache=300
            )
        )

        # 启动工作协程
        for _ in range(self.config["max_concurrent"]):
            self._loop.create_task(self._crawler_worker())

        # 种子URL自动入队：支持配置后启动即爬取
        seed_urls = self.config.get("seed_urls") or []
        if seed_urls:
            self._logger.info(f"自动入队 {len(seed_urls)} 个种子URL")
            # 复用提交逻辑，绕开 process 的写操作前置检查
            self._action_submit({
                "action": "submit",
                "urls": seed_urls,
                "options": {"priority": 5, "depth": 0, "max_depth": self.config["max_depth"]}
            })

    async def _crawler_worker(self):
        """爬虫工作协程，循环消费优先级任务队列"""
        while self._running:
            try:
                # 暂停态：保留结果，停止消费新任务
                if self._paused:
                    await asyncio.sleep(0.5)
                    continue

                # 获取任务
                task = None
                with self._lock:
                    if self._task_queue:
                        _, _, task = heapq.heappop(self._task_queue)
                        self._stats["pending_tasks"] = len(self._task_queue)

                if not task:
                    await asyncio.sleep(0.1)
                    continue

                # 任务处理交由专门方法，确保异常时任务不会丢失
                await self._process_task(task)

            except Exception as e:
                self._logger.error(f"工作协程异常: {str(e)}")
                await asyncio.sleep(1)

    async def _process_task(self, task: dict):
        """处理单个爬取任务：校验、限速、抓取、解析、入BFS队列、回写缓存
        异常时进行重试或标记失败，确保任务不会丢失
        """
        url = task["url"]
        task_id = task["task_id"]
        retry = task.get("retry_count", 0)

        try:
            # robots协议校验
            if self.config["respect_robots"] and not self._check_robots_allowed(url):
                with self._lock:
                    self._stats["fail_count"] += 1
                    self._update_domain_stats(url, success=False)
                    self._result_cache[task_id] = {
                        "status": "forbidden",
                        "url": url,
                        "reason": "robots.txt 禁止爬取",
                        "timestamp": time.time()
                    }
                return

            # 执行页面爬取
            result = await self._fetch_and_parse(url, retry)
            result["task_id"] = task_id
            result["timestamp"] = time.time()

            # 缓存结果并分布式回写
            with self._lock:
                self._result_cache[task_id] = result
                # 分布式模式：把结果推送回 Redis 供其他节点消费
                if self._has_redis and self._redis_conn is not None:
                    try:
                        self._redis_conn.rpush(
                            f"crawler:results:{task_id}",
                            json.dumps(result)
                        )
                    except Exception as e:
                        self._logger.warning(f"结果回写Redis失败: {str(e)}")

            # BFS 深度爬取：解析成功且未达最大深度时，将外链入队
            depth = task.get("depth", 0)
            max_depth = task.get("max_depth", self.config["max_depth"])
            if result.get("status") == "success" and depth < max_depth:
                base_domain = urlparse(url).netloc
                new_links = result.get("out_links", []) or []
                self._enqueue_bfs_links(
                    new_links, base_domain, depth + 1, max_depth,
                    task.get("priority", 5)
                )

        except Exception as e:
            # 任务处理异常：尝试重试或标记失败，确保不丢失
            self._logger.error(f"任务处理异常 task={task_id} url={url}: {str(e)}")
            if retry < self.config["max_retry_times"]:
                task["retry_count"] = retry + 1
                with self._lock:
                    if len(self._task_queue) < self.config["max_task_queue"]:
                        heapq.heappush(
                            self._task_queue,
                            (task.get("priority", 5), self._task_counter, task)
                        )
                        self._task_counter += 1
                        self._stats["pending_tasks"] = len(self._task_queue)
                    else:
                        self._mark_task_failed(task_id, url, "队列已满，重试任务被丢弃")
                return
            self._mark_task_failed(task_id, url, str(e))

    def _mark_task_failed(self, task_id: str, url: str, error: str):
        """将任务标记为失败并写入缓存"""
        with self._lock:
            self._stats["fail_count"] += 1
            self._result_cache[task_id] = {
                "status": "failed",
                "url": url,
                "error": error,
                "timestamp": time.time()
            }

    def _enqueue_bfs_links(self, links: List[str], base_domain: str,
                            depth: int, max_depth: int, priority: int):
        """将BFS外链入队，仅爬取同域名链接以避免爬虫陷阱"""
        if not links:
            return
        with self._lock:
            for link in links:
                # 仅入队未访问过的同域链接
                if link in self._seen_urls:
                    continue
                parsed = urlparse(link)
                if parsed.netloc != base_domain:
                    continue
                if not parsed.scheme.startswith("http"):
                    continue
                if len(self._task_queue) >= self.config["max_task_queue"]:
                    break
                task_id = hashlib.md5(f"{link}_{time.time_ns()}".encode()).hexdigest()[:10]
                task = {
                    "task_id": task_id,
                    "url": link,
                    "retry_count": 0,
                    "submit_time": time.time(),
                    "priority": priority,
                    "depth": depth,
                    "max_depth": max_depth
                }
                heapq.heappush(self._task_queue, (priority, self._task_counter, task))
                self._task_counter += 1
                self._seen_urls.add(link)
            self._stats["pending_tasks"] = len(self._task_queue)

    def _update_domain_stats(self, url: str, success: bool, response_time: Optional[float] = None):
        """更新域名级统计（调用方需持有 _lock）"""
        domain = urlparse(url).netloc
        if not domain:
            return
        info = self._domain_stats.setdefault(domain, {
            "requests": 0,
            "success": 0,
            "fail": 0,
            "total_response_time": 0.0
        })
        info["requests"] += 1
        if success:
            info["success"] += 1
            if response_time is not None:
                info["total_response_time"] = info.get("total_response_time", 0.0) + response_time
        else:
            info["fail"] += 1

    async def _enforce_domain_delay(self, url: str):
        """域名级限速：保证对同一域名的最小请求间隔"""
        domain = urlparse(url).netloc
        if not domain:
            return
        delay = self.config["domain_delay"]
        now = time.time()
        last = self._domain_last_request.get(domain, 0.0)
        wait = delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._domain_last_request[domain] = time.time()

    async def _fetch_and_parse(self, url: str, retry: int) -> dict:
        """执行单页爬取与解析，包含重试、反爬、限速、解析全流程
        修复点：
        - UA按请求显式注入，避免 TraceConfig hack
        - 使用 time.time() 手动计时，aiohttp response 不含 .elapsed
        - 域名级限速
        """
        # 域名级限速
        await self._enforce_domain_delay(url)

        # 请求间隔抖动，模拟人类行为
        delay = random.uniform(
            self.config["min_request_delay"],
            self.config["max_request_delay"]
        )
        await asyncio.sleep(delay)

        # 代理选择
        proxy = random.choice(self.config["proxy_list"]) if self.config["proxy_list"] else None
        # 每请求显式注入 UA
        headers = {
            "User-Agent": random.choice(self.config["user_agent_pool"]),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        }

        # 手动计时起点
        start_ts = time.time()
        try:
            async with self._session.get(
                url, proxy=proxy, allow_redirects=True, headers=headers
            ) as resp:
                self._stats["total_requests"] += 1
                resp.raise_for_status()

                raw_content = await resp.text(errors="replace")
                content_type = resp.content_type
                # 计算响应耗时（aiohttp response 无 .elapsed 属性）
                elapsed = time.time() - start_ts
                # 滚动窗口维护平均响应时间
                self._response_times.append(elapsed)
                if len(self._response_times) > 1000:
                    self._response_times = self._response_times[-1000:]
                self._stats["avg_response_time"] = (
                    sum(self._response_times) / len(self._response_times)
                )
                # 统计下载字节数
                content_bytes = len(raw_content.encode("utf-8", errors="ignore"))
                self._stats["total_bytes_downloaded"] += content_bytes
                self._stats["total_content_length"] += content_bytes

                # 内容哈希去重
                content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
                if content_hash in self._content_hashes:
                    self._update_domain_stats(url, success=True, response_time=elapsed)
                    return {
                        "status": "duplicate",
                        "url": url,
                        "content_hash": content_hash,
                        "status_code": resp.status,
                        "response_time": elapsed
                    }
                self._content_hashes.add(content_hash)

                # 智能解析
                parsed_data = self._smart_parse(raw_content, content_type, url)

                self._stats["success_count"] += 1
                self._stats["crawled_pages"] += 1
                self._update_domain_stats(url, success=True, response_time=elapsed)

                return {
                    "status": "success",
                    "url": url,
                    "status_code": resp.status,
                    "content_type": content_type,
                    "title": parsed_data["title"],
                    "main_text": parsed_data["main_text"],
                    "out_links": parsed_data["links"],
                    "meta": parsed_data.get("meta", {}),
                    "structured_data": parsed_data.get("structured_data", {}),
                    "content_hash": content_hash,
                    "response_time": elapsed,
                    "content_length": content_bytes
                }

        except Exception as e:
            self._update_domain_stats(url, success=False)
            # 指数退避重试
            if retry < self.config["max_retry_times"]:
                await asyncio.sleep(2 ** retry)
                return await self._fetch_and_parse(url, retry + 1)
            self._stats["fail_count"] += 1
            return {
                "status": "failed",
                "url": url,
                "error": str(e),
                "retry_count": retry
            }

    def _smart_parse(self, content: str, content_type: str, base_url: str) -> dict:
        """智能内容解析，自动适配不同文档类型
        增强：OpenGraph / JSON-LD / meta description / canonical / RSS-Atom / 内容清洗 / 主内容提取
        """
        result = {
            "title": "",
            "main_text": "",
            "links": [],
            "meta": {},
            "structured_data": {}
        }

        if "text/html" in content_type:
            try:
                soup = self._BeautifulSoup(content, "lxml")

                # 提取标题
                if soup.title and soup.title.string:
                    result["title"] = soup.title.string.strip()

                # 结构化数据抽取：OpenGraph / meta / canonical / RSS
                result["meta"] = self._extract_meta_tags(soup)
                result["structured_data"] = self._extract_structured_data(soup)

                # 正文提取（降噪版）：移除无关元素
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
                    tag.decompose()

                # 尝试提取主内容区：<main> / <article>，否则取文本最长的 <div>
                main_area = soup.find("main") or soup.find("article")
                if main_area is None:
                    best_div = None
                    best_len = 0
                    for div in soup.find_all("div"):
                        text = div.get_text(strip=True)
                        if len(text) > best_len:
                            best_len = len(text)
                            best_div = div
                    main_area = best_div or soup

                raw_text = main_area.get_text(separator="\n", strip=True)
                # 清洗：合并多余空白、去除样板噪声、按上限截断
                result["main_text"] = self._clean_text(raw_text)

                # 提取并补全链接
                result["links"] = list({
                    urljoin(base_url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if a["href"].startswith(("http", "/", "#"))
                })

            except Exception as e:
                self._logger.warning(f"HTML解析异常: {str(e)}")
                result["main_text"] = self._clean_text(content)

        elif "application/json" in content_type:
            result["main_text"] = content[: self.config["max_content_length"]]
            try:
                result["structured_data"]["json"] = json.loads(content)
            except Exception:
                pass

        elif "application/xml" in content_type or "text/xml" in content_type:
            result["main_text"] = content[: self.config["max_content_length"]]

        else:
            result["main_text"] = self._clean_text(content)

        return result

    def _extract_meta_tags(self, soup) -> dict:
        """抽取 OpenGraph / 标准 meta / canonical / RSS 等元信息"""
        meta = {}
        try:
            # OpenGraph 标签（og:title / og:description / og:image 等）
            for tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
                key = tag.get("property")
                val = tag.get("content")
                if key and val:
                    meta[key] = val
            # 标准 meta 标签（description / keywords / author 等）
            for tag in soup.find_all("meta", attrs={"name": True}):
                key = tag.get("name")
                val = tag.get("content")
                if key and val and key not in meta:
                    meta[key] = val
            # canonical URL
            canon = soup.find("link", rel="canonical")
            if canon and canon.get("href"):
                meta["canonical"] = canon["href"]
            # RSS / Atom feed 链接
            feeds = []
            for link in soup.find_all("link", rel="alternate"):
                link_type = link.get("type", "")
                if "rss" in link_type or "atom" in link_type:
                    feeds.append({
                        "type": link_type,
                        "href": link.get("href"),
                        "title": link.get("title")
                    })
            if feeds:
                meta["feeds"] = feeds
        except Exception as e:
            self._logger.warning(f"meta抽取异常: {str(e)}")
        return meta

    def _extract_structured_data(self, soup) -> dict:
        """抽取 JSON-LD 结构化数据（schema.org）"""
        structured = {}
        try:
            json_ld_items = []
            for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
                try:
                    data = json.loads(tag.string or "{}")
                    if isinstance(data, list):
                        json_ld_items.extend(data)
                    else:
                        json_ld_items.append(data)
                except Exception:
                    continue
            if json_ld_items:
                structured["json_ld"] = json_ld_items
        except Exception as e:
            self._logger.warning(f"JSON-LD抽取异常: {str(e)}")
        return structured

    def _clean_text(self, text: str) -> str:
        """内容清洗：合并空白、去除样板噪声、按上限截断"""
        if not text:
            return ""
        # 合并多余空白
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        # 移除常见样板噪声（Cookie提示、版权声明、跳转链接等）
        boilerplate_patterns = [
            r"Cookie\s*Consent.*?(?=\n|$)",
            r"We use cookies.*?(?=\n|$)",
            r"©\s*\d{4}.*?(?=\n|$)",
            r"Skip to content.*?(?=\n|$)",
            r"Sign in|Sign up|Subscribe now",
        ]
        for pat in boilerplate_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        # 截断到上限
        max_len = self.config["max_content_length"]
        if len(text) > max_len:
            text = text[:max_len]
        return text.strip()

    def _check_robots_allowed(self, url: str) -> bool:
        """检查robots协议是否允许爬取"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                # 读取失败默认放行
                return True
            self._robots_cache[domain] = rp

        return self._robots_cache[domain].can_fetch("*", url)

    def _pull_redis_tasks(self):
        """分布式模式：从Redis队列拉取任务（复用持久连接）"""
        if not self._has_redis or self._redis_conn is None:
            return
        try:
            r = self._redis_conn
            pulled = 0
            max_pull = 100  # 单次最多拉取100条，避免长时间持锁
            while pulled < max_pull:
                task_raw = r.lpop("crawler:task_queue")
                if not task_raw:
                    break
                try:
                    task = json.loads(task_raw)
                except Exception as e:
                    self._logger.warning(f"任务反序列化失败: {str(e)}")
                    continue
                priority = max(0, min(10, int(task.get("priority", 5))))
                with self._lock:
                    if len(self._task_queue) < self.config["max_task_queue"]:
                        heapq.heappush(self._task_queue, (priority, self._task_counter, task))
                        self._task_counter += 1
                pulled += 1
            if pulled:
                with self._lock:
                    self._stats["pending_tasks"] = len(self._task_queue)
        except Exception as e:
            self._logger.warning(f"Redis任务拉取失败: {str(e)}")

    async def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """解析站点地图（含 sitemap index 递归），返回去重后的URL列表"""
        urls: List[str] = []
        if self._session is None:
            return urls
        try:
            headers = {
                "User-Agent": random.choice(self.config["user_agent_pool"])
            }
            async with self._session.get(sitemap_url, headers=headers, allow_redirects=True) as resp:
                if resp.status != 200:
                    self._logger.warning(f"sitemap抓取失败 status={resp.status} url={sitemap_url}")
                    return urls
                xml_text = await resp.text(errors="replace")
        except Exception as e:
            self._logger.warning(f"sitemap请求异常: {str(e)}")
            return urls

        try:
            root = ElementTree.fromstring(xml_text)
        except Exception as e:
            self._logger.warning(f"sitemap XML解析失败: {str(e)}")
            return urls

        # 命名空间处理
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0].strip("{")

        def _find(root_elem, tag_local):
            if ns:
                return root_elem.findall(f".//{{{ns}}}{tag_local}")
            return root_elem.findall(f".//{tag_local}")

        # 区分 sitemap index 与普通 urlset
        sitemap_locs: List[str] = []
        url_locs: List[str] = []
        for loc in _find(root, "loc"):
            if loc.text:
                text = loc.text.strip()
                if "sitemap" in text.lower():
                    sitemap_locs.append(text)
                else:
                    url_locs.append(text)

        # 若为 sitemap index，递归抓取子 sitemap（限制最多10个，避免爆炸）
        if sitemap_locs and not url_locs:
            for sub_url in sitemap_locs[:10]:
                sub_urls = await self._parse_sitemap(sub_url)
                urls.extend(sub_urls)
        else:
            urls.extend(url_locs)

        # 去重保序
        seen = set()
        deduped = []
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped


# ------------------------------ 工厂函数 ------------------------------
def create_plugin():
    """插件入口，推荐使用工厂函数模式"""
    return AdvancedWebCrawlerPlugin()
