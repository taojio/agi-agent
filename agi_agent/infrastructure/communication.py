"""
infrastructure/communication.py - 通信传输子模块

包含任务：
- T009 IPC 本地消息转发（IPCBus）
- T010 网络连接维持（NetworkLinkManager）
- T011 消息队列分发（MessageQueueDispatcher）
- T012 跨设备数据加密传输（EncryptedTransport）

设计原则：
1. 重型依赖（websocket/cryptography/pika）可选导入，失败降级为标准库实现
2. Windows 下 Named Pipe 可选，降级用 threading.Queue
3. 加密降级为 hmac/hashlib+base64 简易实现
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.infrastructure")

# ====== 可选依赖 ======
try:
    import websocket  # type: ignore  # websocket-client
    _HAS_WEBSOCKET = True
except Exception:  # pragma: no cover
    websocket = None  # type: ignore
    _HAS_WEBSOCKET = False

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore
    from cryptography.hazmat.primitives import padding as sym_padding  # type: ignore
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding  # type: ignore
    from cryptography.hazmat.primitives import hashes, serialization  # type: ignore
    _HAS_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover
    Cipher = None  # type: ignore
    algorithms = None  # type: ignore
    modes = None  # type: ignore
    _HAS_CRYPTOGRAPHY = False

try:
    import pika  # type: ignore
    _HAS_PIKA = True
except Exception:  # pragma: no cover
    pika = None  # type: ignore
    _HAS_PIKA = False


# ====== T009: IPC 本地消息转发 ======

@dataclass
class IPCConfig:
    """T009 配置"""
    queue_max_size: int = 1000
    use_named_pipe: bool = False        # Windows 下使用 Named Pipe（占位，降级用 Queue）
    pipe_name: str = "agi_agent_ipc"
    receive_timeout: float = 1.0


class IPCBus(BaseModule):
    """T009 IPC 本地消息转发

    常驻后台：基于 threading.Queue + 路由表。
    Windows 下 Named Pipe 可选，降级用 Queue。
    """

    name = "ipc_bus"
    version = "1.0.0"
    description = "T009 IPC 本地消息转发：threading.Queue + 路由表，Windows Named Pipe 可选降级"

    def __init__(self, config: Optional[IPCConfig] = None) -> None:
        super().__init__()
        self.config: IPCConfig = config or IPCConfig()
        self._queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("IPCBus 初始化完成")

    def _start(self) -> None:
        self._running.set()

    def _stop(self) -> None:
        self._running.clear()

    def _shutdown(self) -> None:
        self._running.clear()
        with self._lock:
            self._queues.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def register(self, module_name: str) -> bool:
        """注册模块到 IPC 总线"""
        with self._lock:
            if module_name in self._queues:
                return False
            self._queues[module_name] = queue.Queue(maxsize=self.config.queue_max_size)
        logger.info("IPC 注册模块 %s", module_name)
        return True

    def unregister(self, module_name: str) -> bool:
        with self._lock:
            return self._queues.pop(module_name, None) is not None

    def send(self, target: str, message: Any) -> bool:
        """向目标模块发送消息"""
        with self._lock:
            q = self._queues.get(target)
        if q is None:
            logger.warning("IPC 目标模块 %s 未注册", target)
            return False
        try:
            q.put({"target": target, "message": message, "ts": time.time()}, block=False)
            return True
        except queue.Full:
            logger.warning("IPC 队列满，丢弃消息 target=%s", target)
            return False

    def receive(self, module_name: str, timeout: Optional[float] = None) -> Optional[Any]:
        """接收发往本模块的消息"""
        with self._lock:
            q = self._queues.get(module_name)
        if q is None:
            return None
        if timeout is None:
            timeout = self.config.receive_timeout
        try:
            envelope = q.get(timeout=timeout)
            return envelope.get("message")
        except queue.Empty:
            return None

    def broadcast(self, message: Any) -> int:
        """广播消息到所有已注册模块，返回成功投递数"""
        with self._lock:
            targets = list(self._queues.items())
        count = 0
        for name, q in targets:
            try:
                q.put({"target": name, "message": message, "ts": time.time()}, block=False)
                count += 1
            except queue.Full:
                logger.warning("IPC 队列满，丢弃广播消息 target=%s", name)
        return count

    def list_modules(self) -> List[str]:
        with self._lock:
            return list(self._queues.keys())


# ====== T010: 网络连接维持 ======

@dataclass
class NetworkEndpoint:
    """网络端点"""
    endpoint_id: str
    host: str
    port: int
    protocol: str = "tcp"           # tcp / websocket
    path: str = ""                  # websocket 路径
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkStatus:
    """链路状态"""
    endpoint_id: str
    connected: bool = False
    last_active: float = 0.0
    reconnect_count: int = 0
    error: str = ""


@dataclass
class NetworkLinkConfig:
    """T010 配置"""
    keepalive_interval: float = 30.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 5
    connect_timeout: float = 5.0
    enable_websocket: bool = True


class NetworkLinkManager(BaseModule):
    """T010 网络连接维持

    常驻后台：维持 TCP/WebSocket 长连接，心跳保活、断线重连、链路切换。
    用 socket 标准库实现，websocket 可选导入。
    """

    name = "network_link_manager"
    version = "1.0.0"
    description = "T010 网络连接维持：TCP/WebSocket 长连接、心跳保活、断线重连"

    def __init__(self, config: Optional[NetworkLinkConfig] = None) -> None:
        super().__init__()
        self.config: NetworkLinkConfig = config or NetworkLinkConfig()
        self._endpoints: Dict[str, NetworkEndpoint] = {}
        self._sockets: Dict[str, Any] = {}
        # 注意：基类 BaseModule 已使用 self._status 表示 ModuleStatus，
        # 这里使用 _link_status 避免覆盖。
        self._link_status: Dict[str, LinkStatus] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("NetworkLinkManager 初始化完成")

    def _start(self) -> None:
        self._stop_event.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="NetworkKeepalive", daemon=True
        )
        self._keepalive_thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=2.0)
        self._keepalive_thread = None
        self._close_all_sockets()

    def _shutdown(self) -> None:
        self._stop()
        with self._lock:
            self._endpoints.clear()
            self._link_status.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def connect(self, endpoint: NetworkEndpoint) -> bool:
        """连接到指定端点"""
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            self._link_status[endpoint.endpoint_id] = LinkStatus(
                endpoint_id=endpoint.endpoint_id
            )
        return self._do_connect(endpoint)

    def disconnect(self, endpoint_id: str) -> bool:
        with self._lock:
            sock = self._sockets.pop(endpoint_id, None)
            self._link_status.get(endpoint_id, LinkStatus(endpoint_id=endpoint_id)).connected = False
        if sock is not None:
            self._safe_close(sock)
        return True

    def keepalive(self) -> int:
        """对所有连接执行一次心跳保活，返回活跃连接数"""
        with self._lock:
            endpoints = list(self._endpoints.values())
        alive = 0
        for ep in endpoints:
            ok = self._send_keepalive(ep)
            if ok:
                alive += 1
            else:
                # 触发重连
                self.reconnect(ep.endpoint_id)
        return alive

    def reconnect(self, endpoint_id: Optional[str] = None) -> bool:
        """重连指定端点；不指定则重连所有断开的端点"""
        with self._lock:
            if endpoint_id is not None:
                targets = [self._endpoints.get(endpoint_id)]
                targets = [t for t in targets if t is not None]
            else:
                targets = [ep for ep in self._endpoints.values()
                           if not self._link_status.get(ep.endpoint_id, LinkStatus(endpoint_id="")).connected]
        if not targets:
            return False
        ok_any = False
        for ep in targets:
            st = self._link_status.get(ep.endpoint_id)
            if st is not None and st.reconnect_count >= self.config.max_reconnect_attempts:
                logger.warning("端点 %s 达到最大重连次数", ep.endpoint_id)
                continue
            ok = self._do_connect(ep)
            ok_any = ok_any or ok
            if st is not None and ok:
                st.reconnect_count = 0
            elif st is not None:
                st.reconnect_count += 1
        return ok_any

    def get_link_status(self, endpoint_id: Optional[str] = None) -> Optional[LinkStatus]:
        """获取链路状态"""
        with self._lock:
            if endpoint_id is None:
                # 返回任一活跃状态
                for s in self._link_status.values():
                    if s.connected:
                        return LinkStatus(**s.__dict__)
                return None
            st = self._link_status.get(endpoint_id)
            return LinkStatus(**st.__dict__) if st else None

    def get_all_status(self) -> List[LinkStatus]:
        with self._lock:
            return [LinkStatus(**s.__dict__) for s in self._link_status.values()]

    def send_data(self, endpoint_id: str, data: bytes) -> bool:
        """向指定端点发送原始数据"""
        with self._lock:
            sock = self._sockets.get(endpoint_id)
        if sock is None:
            return False
        try:
            if _HAS_WEBSOCKET and isinstance(sock, websocket.WebSocket):
                sock.send_binary(data)
            else:
                sock.sendall(data)
            self._touch(endpoint_id)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("发送数据失败 endpoint=%s: %s", endpoint_id, e)
            self._mark_error(endpoint_id, str(e))
            return False

    # ---- 内部 ----
    def _do_connect(self, ep: NetworkEndpoint) -> bool:
        try:
            if ep.protocol == "websocket":
                if not (self.config.enable_websocket and _HAS_WEBSOCKET):
                    raise RuntimeError("websocket 不可用，降级失败")
                url = f"ws://{ep.host}:{ep.port}{ep.path}"
                ws = websocket.create_connection(url, timeout=self.config.connect_timeout)
                with self._lock:
                    self._sockets[ep.endpoint_id] = ws
                    st = self._link_status.setdefault(ep.endpoint_id, LinkStatus(endpoint_id=ep.endpoint_id))
                    st.connected = True
                    st.last_active = time.time()
                    st.error = ""
                logger.info("WebSocket 连接成功 %s", url)
                return True
            # 默认 TCP
            sock = socket.create_connection(
                (ep.host, ep.port), timeout=self.config.connect_timeout
            )
            with self._lock:
                self._sockets[ep.endpoint_id] = sock
                st = self._link_status.setdefault(ep.endpoint_id, LinkStatus(endpoint_id=ep.endpoint_id))
                st.connected = True
                st.last_active = time.time()
                st.error = ""
            logger.info("TCP 连接成功 %s:%d", ep.host, ep.port)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("连接端点 %s 失败: %s", ep.endpoint_id, e)
            self._mark_error(ep.endpoint_id, str(e))
            return False

    def _send_keepalive(self, ep: NetworkEndpoint) -> bool:
        with self._lock:
            sock = self._sockets.get(ep.endpoint_id)
        if sock is None:
            return False
        try:
            if _HAS_WEBSOCKET and isinstance(sock, websocket.WebSocket):
                sock.send(b"PING")
            else:
                sock.sendall(b"PING")
            self._touch(ep.endpoint_id)
            return True
        except Exception:  # pragma: no cover
            self._mark_error(ep.endpoint_id, "keepalive 失败")
            return False

    def _touch(self, endpoint_id: str) -> None:
        with self._lock:
            st = self._link_status.get(endpoint_id)
            if st is not None:
                st.last_active = time.time()

    def _mark_error(self, endpoint_id: str, err: str) -> None:
        with self._lock:
            st = self._link_status.get(endpoint_id)
            if st is not None:
                st.connected = False
                st.error = err
            sock = self._sockets.pop(endpoint_id, None)
        if sock is not None:
            self._safe_close(sock)

    def _safe_close(self, sock: Any) -> None:
        try:
            sock.close()
        except Exception:  # pragma: no cover
            pass

    def _close_all_sockets(self) -> None:
        with self._lock:
            socks = list(self._sockets.values())
            self._sockets.clear()
            for st in self._link_status.values():
                st.connected = False
        for s in socks:
            self._safe_close(s)

    def _keepalive_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.keepalive()
            except Exception as e:  # pragma: no cover
                logger.warning("keepalive 循环异常: %s", e)
            self._stop_event.wait(self.config.keepalive_interval)


# ====== T011: 消息队列分发 ======

@dataclass
class QueueMessage:
    """队列消息"""
    message_id: str
    topic: str
    content: Any
    content_hash: str = ""
    priority: int = 0                # 数值越大优先级越高
    timestamp: float = 0.0


@dataclass
class MessageQueueConfig:
    """T011 配置"""
    max_queue_size: int = 10000
    dedup_window: int = 4096         # 去重窗口大小
    enable_rabbitmq: bool = False    # 是否启用 RabbitMQ（可选）
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    consumer_poll_interval: float = 0.1


class MessageQueueDispatcher(BaseModule):
    """T011 消息队列分发

    常驻后台：入队、去重（基于 message_id+content_hash）、优先级排序、异步消费分发。
    内置基于 PriorityQueue 的实现，RabbitMQ 可选导入降级。
    """

    name = "message_queue_dispatcher"
    version = "1.0.0"
    description = "T011 消息队列分发：入队、去重、优先级排序、异步消费分发"

    def __init__(self, config: Optional[MessageQueueConfig] = None) -> None:
        super().__init__()
        self.config: MessageQueueConfig = config or MessageQueueConfig()
        # 优先级队列：(-priority, seq, QueueMessage)
        self._pq: "queue.PriorityQueue[Tuple[int, int, QueueMessage]]" = queue.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        self._seq = 0
        self._seen: Dict[str, float] = {}   # hash -> 入队时间
        self._consumers: Dict[str, List[Callable[[QueueMessage], None]]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self._rabbitmq_channel = None

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        self._pq.maxsize = self.config.max_queue_size
        if self.config.enable_rabbitmq and _HAS_PIKA:
            try:
                params = pika.URLParameters(self.config.rabbitmq_url)
                connection = pika.BlockingConnection(params)
                self._rabbitmq_channel = connection.channel()
                logger.info("RabbitMQ 通道已建立")
            except Exception as e:  # pragma: no cover
                logger.warning("RabbitMQ 连接失败，降级为内置队列: %s", e)
                self._rabbitmq_channel = None
        logger.info("MessageQueueDispatcher 初始化完成")

    def _start(self) -> None:
        self._stop_event.clear()
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, name="MQConsumer", daemon=True
        )
        self._consumer_thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=2.0)
        self._consumer_thread = None

    def _shutdown(self) -> None:
        self._stop()
        with self._lock:
            self._seen.clear()
            self._consumers.clear()
        if self._rabbitmq_channel is not None:
            try:
                self._rabbitmq_channel.close()
            except Exception:  # pragma: no cover
                pass
            self._rabbitmq_channel = None

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def enqueue(self, message: Any, priority: int = 0,
                topic: str = "default", message_id: str = "") -> bool:
        """入队：基于 message_id+content_hash 去重"""
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        content_hash = self._hash_content(message)
        dedup_key = f"{message_id}:{content_hash}"
        now = time.time()
        with self._lock:
            if dedup_key in self._seen:
                logger.debug("消息去重命中 key=%s", dedup_key)
                return False
            self._seen[dedup_key] = now
            self._evict_expired(now)
            self._seq += 1
            seq = self._seq
        msg = QueueMessage(
            message_id=message_id,
            topic=topic,
            content=message,
            content_hash=content_hash,
            priority=priority,
            timestamp=now,
        )
        # RabbitMQ 优先
        if self._rabbitmq_channel is not None:
            try:
                self._rabbitmq_channel.basic_publish(
                    exchange="",
                    routing_key=topic,
                    body=json.dumps({"message_id": message_id, "content": str(message)}).encode(),
                )
                return True
            except Exception as e:  # pragma: no cover
                logger.warning("RabbitMQ 投递失败，降级内置队列: %s", e)
        try:
            self._pq.put((-priority, seq, msg), block=False)
            return True
        except queue.Full:
            logger.warning("消息队列满，丢弃 message_id=%s", message_id)
            return False

    def dequeue(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        """同步出队一条消息"""
        try:
            _, _, msg = self._pq.get(timeout=timeout or 0.1)
            return msg
        except queue.Empty:
            return None

    def register_consumer(self, topic: str,
                          handler: Callable[[QueueMessage], None]) -> None:
        """注册主题消费者"""
        with self._lock:
            self._consumers.setdefault(topic, []).append(handler)
        logger.info("注册消费者 topic=%s", topic)

    def get_queue_size(self) -> int:
        return self._pq.qsize()

    # ---- 内部 ----
    def _hash_content(self, content: Any) -> str:
        try:
            raw = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            raw = str(content).encode("utf-8", errors="ignore")
        return hashlib.md5(raw).hexdigest()

    def _evict_expired(self, now: float) -> None:
        # 简单 LRU：超过窗口大小时清理最早的
        if len(self._seen) <= self.config.dedup_window:
            return
        # 按时间排序，淘汰最早的 25%
        sorted_items = sorted(self._seen.items(), key=lambda x: x[1])
        drop_n = max(1, len(sorted_items) // 4)
        for k, _ in sorted_items[:drop_n]:
            self._seen.pop(k, None)

    def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            msg = self.dequeue(timeout=self.config.consumer_poll_interval)
            if msg is None:
                continue
            with self._lock:
                handlers = list(self._consumers.get(msg.topic, []))
                # 也分发给通配消费者
                handlers.extend(self._consumers.get("*", []))
            for h in handlers:
                try:
                    h(msg)
                except Exception as e:  # pragma: no cover
                    logger.warning("消费者处理消息异常 topic=%s: %s", msg.topic, e)


# ====== T012: 跨设备数据加密传输 ======

@dataclass
class EncryptedTransportConfig:
    """T012 配置"""
    aes_key: bytes = b""             # 留空则自动生成
    rsa_key_size: int = 2048
    use_rsa: bool = True
    # 降级实现的简易密钥（base64 编码）
    fallback_key: str = "agi_agent_fallback_key_v1"


class EncryptedTransport(BaseModule):
    """T012 跨设备数据加密传输

    动态调度：AES 加密、RSA 密钥交换、MD5 校验。
    用 cryptography（可选）或内置 hmac/hashlib+base64 简易实现降级。
    """

    name = "encrypted_transport"
    version = "1.0.0"
    description = "T012 跨设备数据加密传输：AES + RSA 密钥交换 + MD5 校验，可降级"

    def __init__(self, config: Optional[EncryptedTransportConfig] = None) -> None:
        super().__init__()
        self.config: EncryptedTransportConfig = config or EncryptedTransportConfig()
        self._aes_key: bytes = self.config.aes_key or self._gen_aes_key()
        self._rsa_private_key: Any = None
        self._rsa_public_key: Any = None
        self._lock = threading.Lock()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    if k in ("aes_key",):
                        setattr(self.config, k, v.encode() if isinstance(v, str) else v)
                    else:
                        setattr(self.config, k, v)
        if not self.config.aes_key:
            self._aes_key = self._gen_aes_key()
        else:
            self._aes_key = self.config.aes_key
        if self.config.use_rsa:
            self.generate_keypair()
        logger.info(
            "EncryptedTransport 初始化完成，cryptography=%s", _HAS_CRYPTOGRAPHY
        )

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        with self._lock:
            self._aes_key = b""
            self._rsa_private_key = None
            self._rsa_public_key = None

    def _health_check(self) -> bool:
        return bool(self._aes_key)

    # ---- 公共方法 ----
    def encrypt(self, data: bytes) -> bytes:
        """加密数据"""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data 必须是 bytes")
        if _HAS_CRYPTOGRAPHY:
            return self._encrypt_aes_cryptography(bytes(data))
        return self._encrypt_fallback(bytes(data))

    def decrypt(self, ciphertext: bytes) -> bytes:
        """解密数据"""
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise TypeError("ciphertext 必须是 bytes")
        if _HAS_CRYPTOGRAPHY:
            return self._decrypt_aes_cryptography(bytes(ciphertext))
        return self._decrypt_fallback(bytes(ciphertext))

    def verify(self, data: bytes, checksum: str) -> bool:
        """校验数据与 MD5 checksum 是否一致"""
        return hmac.compare_digest(self.checksum(data), checksum)

    @staticmethod
    def checksum(data: bytes) -> str:
        """计算 MD5 校验和"""
        return hashlib.md5(data).hexdigest()

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """生成 RSA 密钥对，返回 (public_pem, private_pem)"""
        if not _HAS_CRYPTOGRAPHY:
            # 降级：返回占位密钥（不安全，仅占位）
            logger.warning("cryptography 不可用，RSA 密钥对降级为占位实现")
            pub = base64.b64encode(self._aes_key).decode()
            priv = base64.b64encode(self._aes_key).decode()
            return pub.encode(), priv.encode()
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.rsa_key_size,
        )
        public_key = private_key.public_key()
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        priv_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with self._lock:
            self._rsa_private_key = private_key
            self._rsa_public_key = public_key
        return pub_pem, priv_pem

    def get_aes_key(self) -> bytes:
        """获取当前 AES 密钥（仅用于调试/密钥交换）"""
        return self._aes_key

    # ---- 内部：cryptography 实现 ----
    def _encrypt_aes_cryptography(self, data: bytes) -> bytes:
        iv = os.urandom(16)
        padder = sym_padding.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        cipher = Cipher(algorithms.AES(self._aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded) + encryptor.finalize()
        return iv + ct

    def _decrypt_aes_cryptography(self, ciphertext: bytes) -> bytes:
        if len(ciphertext) < 16:
            raise ValueError("密文长度不足")
        iv = ciphertext[:16]
        ct = ciphertext[16:]
        cipher = Cipher(algorithms.AES(self._aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ct) + decryptor.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    # ---- 内部：降级实现（XOR + base64，非安全等级） ----
    def _encrypt_fallback(self, data: bytes) -> bytes:
        key = self._aes_key or self.config.fallback_key.encode()
        iv = os.urandom(8)
        xored = self._xor(data, key, iv)
        payload = iv + xored
        return base64.b64encode(payload)

    def _decrypt_fallback(self, ciphertext: bytes) -> bytes:
        key = self._aes_key or self.config.fallback_key.encode()
        try:
            payload = base64.b64decode(ciphertext)
        except Exception as e:
            raise ValueError("降级密文解码失败") from e
        if len(payload) < 8:
            raise ValueError("降级密文长度不足")
        iv = payload[:8]
        xored = payload[8:]
        return self._xor(xored, key, iv)

    @staticmethod
    def _xor(data: bytes, key: bytes, salt: bytes) -> bytes:
        if not key:
            key = b"\x00"
        combined = (key + salt) or key
        kl = len(combined)
        return bytes(b ^ combined[i % kl] for i, b in enumerate(data))

    @staticmethod
    def _gen_aes_key() -> bytes:
        """生成 32 字节 AES-256 密钥"""
        if _HAS_CRYPTOGRAPHY:
            return os.urandom(32)
        # 降级：基于时间与 uuid 的伪随机
        seed = (str(time.time_ns()) + uuid.uuid4().hex).encode()
        return hashlib.sha256(seed).digest()
