"""
module_bus/data_stream.py - 数据流管理

提供流式数据传输能力，支持背压控制
"""
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BackPressureStrategy(Enum):
    """背压策略"""
    DROP_OLDEST = "drop_oldest"     # 丢弃最旧的数据
    DROP_NEWEST = "drop_newest"     # 丢弃最新的数据
    BLOCK = "block"                    # 阻塞生产者
    ERROR = "error"                    # 报错
    BUFFER_GROW = "buffer_grow"    # 扩容缓冲区（有上限）


class StreamStatus(Enum):
    """流状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class StreamConsumer:
    """流消费者

    Attributes:
        consumer_id: 消费者ID
        callback: 回调函数
        last_consumed: 最后消费时间
        processed_count: 已处理数量
        backlog: 待处理数据量
    """

    consumer_id: str
    callback: Callable
    last_consumed: float = 0.0
    processed_count: int = 0
    backlog: int = 0
    paused: bool = False

    def consume(self, data: Any) -> None:
        """消费数据"""
        self.last_consumed = time.time()
        self.processed_count += 1
        try:
            self.callback(data)
        except Exception:
            pass


class DataStream:
    """数据流

    支持：
    - 发布/订阅模式的流式数据传输
    - 多消费者支持
    - 背压控制
    - 缓冲区管理
    - 流控策略
    """

    def __init__(self, stream_id: str,
                 buffer_size: int = 1000,
                 back_pressure: BackPressureStrategy = BackPressureStrategy.DROP_OLDEST,
                 max_buffer_size: int = 10000):
        self.stream_id = stream_id
        self.buffer_size = buffer_size
        self.max_buffer_size = max_buffer_size
        self.back_pressure = back_pressure

        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._consumers: Dict[str, StreamConsumer] = {}
        self._status: StreamStatus = StreamStatus.ACTIVE
        self._lock = threading.Lock()

        self._total_published = 0
        self._total_dropped = 0
        self._created_at = time.time()

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def buffer_usage(self) -> float:
        """缓冲区使用率"""
        if self.buffer_size == 0:
            return 0.0
        return len(self._buffer) / self.buffer_size

    @property
    def consumer_count(self) -> int:
        """消费者数量"""
        return len(self._consumers)

    def publish(self, data: Any) -> bool:
        """发布数据到流

        Args:
            data: 待发布的数据

        Returns:
            是否成功发布
        """
        if self._status != StreamStatus.ACTIVE:
            return False

        with self._lock:
            if len(self._buffer) >= self.buffer_size:
                if not self._handle_back_pressure(data):
                    return False
            else:
                self._buffer.append(data)
                self._total_published += 1

            self._deliver_to_consumers()

        return True

    def _handle_back_pressure(self, data: Any) -> bool:
        """处理背压

        Returns:
            True 表示数据已处理（加入或丢弃），False 表示失败
        """
        if self.back_pressure == BackPressureStrategy.DROP_OLDEST:
            if self._buffer:
                self._buffer.popleft()
                self._total_dropped += 1
            self._buffer.append(data)
            self._total_published += 1
            return True

        elif self.back_pressure == BackPressureStrategy.DROP_NEWEST:
            self._total_dropped += 1
            return True

        elif self.back_pressure == BackPressureStrategy.BUFFER_GROW:
            new_size = min(self.buffer_size * 2, self.max_buffer_size)
            if new_size > self.buffer_size:
                self.buffer_size = new_size
                self._buffer = deque(self._buffer, maxlen=self.max_buffer_size)
                self._buffer.append(data)
                self._total_published += 1
                return True
            else:
                self._total_dropped += 1
                return True

        elif self.back_pressure == BackPressureStrategy.ERROR:
            raise BufferError(f"Stream '{self.stream_id}' buffer is full")

        elif self.back_pressure == BackPressureStrategy.BLOCK:
            return False

        return True

    def _deliver_to_consumers(self):
        """向所有活跃消费者递送数据"""
        if not self._buffer:
            return

        consumers = [
            c for c in self._consumers.values()
            if not c.paused
        ]

        if not consumers:
            return

        while self._buffer:
            data = self._buffer[0]
            all_consumed = True
            for consumer in consumers:
                try:
                    consumer.consume(data)
                except Exception:
                    pass
            if all_consumed:
                self._buffer.popleft()
            else:
                break

    def subscribe(self, callback: Callable, consumer_id: str = None) -> str:
        """订阅数据流

        Args:
            callback: 数据回调函数
            consumer_id: 消费者ID（自动生成）

        Returns:
            消费者ID
        """
        import uuid
        if consumer_id is None:
            consumer_id = f"consumer_{uuid.uuid4().hex[:8]}"

        with self._lock:
            self._consumers[consumer_id] = StreamConsumer(
                consumer_id=consumer_id,
                callback=callback,
            )

        return consumer_id

    def unsubscribe(self, consumer_id: str) -> bool:
        """取消订阅"""
        with self._lock:
            if consumer_id in self._consumers:
                del self._consumers[consumer_id]
                return True
        return False

    def pause_consumer(self, consumer_id: str) -> bool:
        """暂停消费者"""
        with self._lock:
            if consumer_id in self._consumers:
                self._consumers[consumer_id].paused = True
                return True
        return False

    def resume_consumer(self, consumer_id: str) -> bool:
        """恢复消费者"""
        with self._lock:
            if consumer_id in self._consumers:
                self._consumers[consumer_id].paused = False
                return True
        return False

    def pause(self) -> None:
        """暂停流"""
        self._status = StreamStatus.PAUSED

    def resume(self) -> None:
        """恢复流"""
        self._status = StreamStatus.ACTIVE

    def close(self) -> None:
        """关闭流"""
        self._status = StreamStatus.CLOSED
        with self._lock:
            self._buffer.clear()
            self._consumers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取流统计信息"""
        with self._lock:
            return {
                "stream_id": self.stream_id,
                "status": self._status.value,
                "buffer_size": self.buffer_size,
                "buffer_used": len(self._buffer),
                "buffer_usage": self.buffer_usage,
                "consumer_count": self.consumer_count,
                "total_published": self._total_published,
                "total_dropped": self._total_dropped,
                "created_at": self._created_at,
                "consumers": {
                    cid: {
                        "processed": c.processed_count,
                        "last_consumed": c.last_consumed,
                        "paused": c.paused,
                    }
                    for cid, c in self._consumers.items()
                }
            }


_stream_registry: Dict[str, DataStream] = {}


def create_stream(stream_id: str,
                  buffer_size: int = 1000,
                  back_pressure: BackPressureStrategy = BackPressureStrategy.DROP_OLDEST,
                  max_buffer_size: int = 10000) -> DataStream:
    """创建数据流"""
    if stream_id in _stream_registry:
        return _stream_registry[stream_id]

    stream = DataStream(
        stream_id=stream_id,
        buffer_size=buffer_size,
        back_pressure=back_pressure,
        max_buffer_size=max_buffer_size,
    )
    _stream_registry[stream_id] = stream
    return stream


def get_stream(stream_id: str) -> Optional[DataStream]:
    """获取已创建的数据流"""
    return _stream_registry.get(stream_id)


def remove_stream(stream_id: str) -> bool:
    """移除数据流"""
    if stream_id in _stream_registry:
        _stream_registry[stream_id].close()
        del _stream_registry[stream_id]
        return True
    return False
