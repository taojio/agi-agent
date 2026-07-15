"""
infrastructure/hardware_driver.py - 硬件驱动子模块（软件智能体降级实现）

包含任务：
- T005 传感器初始化（SensorInitializer）
- T006 外设数据读取（PeripheralDataReader）
- T007 运动驱动下发（MotionDriverDispatcher）
- T008 硬件故障自检（HardwareFaultDetector）

设计原则：
1. 重型依赖（cv2/serial/can）可选导入，失败降级为模拟数据
2. 无硬件时返回 DeviceStatus(status="simulated")
3. 纯 Python + 标准库即可实例化与运行
"""
from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.infrastructure")

# ====== 可选依赖 ======
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    import serial  # type: ignore  # pyserial
    _HAS_SERIAL = True
except Exception:  # pragma: no cover
    serial = None  # type: ignore
    _HAS_SERIAL = False

try:
    import can  # type: ignore  # python-can
    _HAS_CAN = True
except Exception:  # pragma: no cover
    can = None  # type: ignore
    _HAS_CAN = False


# ====== 数据结构 ======

@dataclass
class DeviceConfig:
    """设备配置"""
    device_id: str
    device_type: str = "camera"     # camera / radar / imu / serial / can / motor
    port: str = ""                  # 串口/CAN 端口
    baudrate: int = 115200
    camera_index: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceStatus:
    """设备状态"""
    device_id: str
    device_type: str
    status: str = "ok"              # ok / simulated / error / offline
    calibrated: bool = False
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReading:
    """传感器读数"""
    device_id: str
    device_type: str
    timestamp: float = 0.0
    values: Dict[str, Any] = field(default_factory=dict)
    raw: bytes = b""
    simulated: bool = False


@dataclass
class MotionCommand:
    """运动指令"""
    device_id: str
    command_type: str = "angle"     # angle / velocity / limit / point
    values: Dict[str, Any] = field(default_factory=dict)
    command_id: str = ""
    timestamp: float = 0.0


@dataclass
class MotionFeedback:
    """运动反馈"""
    device_id: str
    command_id: str = ""
    status: str = "accepted"        # accepted / executing / done / error
    position: Dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass
class FaultReport:
    """故障报告"""
    device_id: str
    fault_type: str                 # disconnect / overload / offline / packet_loss
    severity: str = "warning"       # info / warning / critical
    description: str = ""
    timestamp: float = 0.0


@dataclass
class HardwareDriverConfig:
    """硬件驱动通用配置"""
    poll_interval: float = 1.0
    simulate_when_unavailable: bool = True
    packet_loss_threshold: float = 0.2  # 丢包率阈值
    overload_load_threshold: float = 95.0


# ====== T005: 传感器初始化 ======

class SensorInitializer(BaseModule):
    """T005 传感器初始化

    单次触发：挂载摄像头/雷达/IMU（用 OpenCV/serial/占位降级）。
    无硬件时返回 DeviceStatus(status="simulated")。
    """

    name = "sensor_initializer"
    version = "1.0.0"
    description = "T005 传感器初始化：挂载摄像头/雷达/IMU，无硬件时降级为模拟"

    def __init__(self, config: Optional[HardwareDriverConfig] = None) -> None:
        super().__init__()
        self.config: HardwareDriverConfig = config or HardwareDriverConfig()
        self._devices: Dict[str, DeviceStatus] = {}
        self._handles: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("SensorInitializer 初始化完成")

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        self._release_all()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def init_device(self, device_config: DeviceConfig) -> DeviceStatus:
        """初始化一个传感器设备"""
        did = device_config.device_id
        dtype = device_config.device_type
        status = DeviceStatus(device_id=did, device_type=dtype)
        handle: Any = None
        try:
            if dtype == "camera":
                handle = self._init_camera(device_config)
            elif dtype == "radar":
                handle = self._init_radar(device_config)
            elif dtype == "imu":
                handle = self._init_imu(device_config)
            elif dtype in ("serial",):
                handle = self._init_serial(device_config)
            elif dtype in ("can",):
                handle = self._init_can(device_config)
            else:
                status.status = "simulated" if self.config.simulate_when_unavailable else "error"
                status.error = "" if status.status == "simulated" else f"未知设备类型 {dtype}"
        except Exception as e:  # pragma: no cover
            logger.warning("初始化设备 %s (%s) 失败: %s", did, dtype, e)
            if self.config.simulate_when_unavailable:
                status.status = "simulated"
                status.error = str(e)
            else:
                status.status = "error"
                status.error = str(e)
        with self._lock:
            self._devices[did] = status
            if handle is not None:
                self._handles[did] = handle
        logger.info("设备初始化 id=%s type=%s status=%s", did, dtype, status.status)
        return status

    def calibrate(self, device_id: str) -> bool:
        """校准设备"""
        with self._lock:
            status = self._devices.get(device_id)
        if status is None:
            logger.warning("校准失败：设备 %s 未初始化", device_id)
            return False
        # 简化：模拟校准流程
        time.sleep(0.01)
        status.calibrated = True
        logger.info("设备 %s 校准完成", device_id)
        return True

    def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        with self._lock:
            st = self._devices.get(device_id)
            return DeviceStatus(**st.__dict__) if st else None

    def list_devices(self) -> List[DeviceStatus]:
        with self._lock:
            return [DeviceStatus(**s.__dict__) for s in self._devices.values()]

    def release_device(self, device_id: str) -> bool:
        with self._lock:
            handle = self._handles.pop(device_id, None)
            self._devices.pop(device_id, None)
        if handle is None:
            return False
        try:
            if _HAS_CV2 and isinstance(handle, cv2.VideoCapture):
                handle.release()
        except Exception:  # pragma: no cover
            pass
        return True

    # ---- 内部：设备初始化 ----
    def _init_camera(self, cfg: DeviceConfig) -> Any:
        if not _HAS_CV2:
            if not self.config.simulate_when_unavailable:
                raise RuntimeError("OpenCV 不可用")
            return None  # 模拟
        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            cap.release()
            if not self.config.simulate_when_unavailable:
                raise RuntimeError(f"摄像头 {cfg.camera_index} 打开失败")
            return None
        return cap

    def _init_radar(self, cfg: DeviceConfig) -> Any:
        # 雷达通常通过串口/CAN 接入，此处占位
        if not self.config.simulate_when_unavailable:
            raise RuntimeError("雷达驱动未实现")
        return None

    def _init_imu(self, cfg: DeviceConfig) -> Any:
        if not _HAS_SERIAL and not self.config.simulate_when_unavailable:
            raise RuntimeError("pyserial 不可用")
        return None

    def _init_serial(self, cfg: DeviceConfig) -> Any:
        if not _HAS_SERIAL:
            if not self.config.simulate_when_unavailable:
                raise RuntimeError("pyserial 不可用")
            return None
        try:
            return serial.Serial(cfg.port, cfg.baudrate, timeout=1.0)
        except Exception as e:  # pragma: no cover
            if not self.config.simulate_when_unavailable:
                raise
            return None

    def _init_can(self, cfg: DeviceConfig) -> Any:
        if not _HAS_CAN:
            if not self.config.simulate_when_unavailable:
                raise RuntimeError("python-can 不可用")
            return None
        try:
            return can.interface.Bus(channel=cfg.port, interface="virtual")
        except Exception:  # pragma: no cover
            if not self.config.simulate_when_unavailable:
                raise
            return None

    def _release_all(self) -> None:
        with self._lock:
            handles = list(self._handles.items())
            self._handles.clear()
            self._devices.clear()
        for did, handle in handles:
            try:
                if _HAS_CV2 and isinstance(handle, cv2.VideoCapture):
                    handle.release()
            except Exception:  # pragma: no cover
                pass


# ====== T006: 外设数据读取 ======

class PeripheralDataReader(BaseModule):
    """T006 外设数据读取

    轮询执行：通过串口/CAN 读取传感器原始数据。
    降级返回模拟数据。
    """

    name = "peripheral_data_reader"
    version = "1.0.0"
    description = "T006 外设数据读取：轮询串口/CAN，降级返回模拟数据"

    def __init__(
        self,
        config: Optional[HardwareDriverConfig] = None,
        sensor_initializer: Optional[SensorInitializer] = None,
    ) -> None:
        super().__init__()
        self.config: HardwareDriverConfig = config or HardwareDriverConfig()
        self._sensor_initializer = sensor_initializer
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Dict[str, SensorReading] = {}

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("PeripheralDataReader 初始化完成")

    def _start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="PeripheralPollLoop", daemon=True
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
            self._latest.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def read(self, device_id: str) -> SensorReading:
        """读取指定设备数据"""
        device_status = self._get_device_status(device_id)
        dtype = device_status.device_type if device_status else "unknown"
        simulated = (device_status is None) or (device_status.status == "simulated")
        if simulated:
            reading = self._simulate_reading(device_id, dtype)
        else:
            reading = self._read_real(device_id, dtype, device_status)
        with self._lock:
            self._latest[device_id] = reading
        return reading

    def read_all(self) -> List[SensorReading]:
        """读取全部已注册设备数据"""
        device_ids = self._list_device_ids()
        return [self.read(did) for did in device_ids]

    def get_latest(self, device_id: str) -> Optional[SensorReading]:
        with self._lock:
            return self._latest.get(device_id)

    # ---- 内部 ----
    def _get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        if self._sensor_initializer is None:
            return None
        return self._sensor_initializer.get_device_status(device_id)

    def _list_device_ids(self) -> List[str]:
        if self._sensor_initializer is None:
            return []
        return [d.device_id for d in self._sensor_initializer.list_devices()]

    def _read_real(self, device_id: str, dtype: str,
                   status: DeviceStatus) -> SensorReading:
        # 简化：真实硬件读取流程在依赖可用时实现
        # 这里复用模拟路径，保证降级可用
        return self._simulate_reading(device_id, dtype)

    def _simulate_reading(self, device_id: str, dtype: str) -> SensorReading:
        rng = random.Random(device_id)
        values: Dict[str, Any] = {}
        if dtype == "camera":
            values = {"width": 640, "height": 480, "channels": 3}
        elif dtype == "radar":
            values = {"distance_m": rng.uniform(0.1, 50.0), "velocity_ms": rng.uniform(-5.0, 5.0)}
        elif dtype == "imu":
            values = {
                "ax": rng.uniform(-9.8, 9.8),
                "ay": rng.uniform(-9.8, 9.8),
                "az": rng.uniform(-9.8, 9.8),
                "gx": rng.uniform(-1.0, 1.0),
                "gy": rng.uniform(-1.0, 1.0),
                "gz": rng.uniform(-1.0, 1.0),
            }
        elif dtype in ("serial", "can"):
            values = {"raw_int": rng.randint(0, 1023)}
        else:
            values = {"value": rng.uniform(0.0, 1.0)}
        return SensorReading(
            device_id=device_id,
            device_type=dtype,
            timestamp=time.time(),
            values=values,
            raw=b"",
            simulated=True,
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.read_all()
            except Exception as e:  # pragma: no cover
                logger.warning("外设轮询异常: %s", e)
            self._stop_event.wait(self.config.poll_interval)


# ====== T007: 运动驱动下发 ======

class MotionDriverDispatcher(BaseModule):
    """T007 运动驱动下发

    事件触发：解析角度/速度/限位/点位，下发电机指令。
    降级仅记录指令。
    """

    name = "motion_driver_dispatcher"
    version = "1.0.0"
    description = "T007 运动驱动下发：解析角度/速度/限位/点位，降级仅记录"

    def __init__(self, config: Optional[HardwareDriverConfig] = None) -> None:
        super().__init__()
        self.config: HardwareDriverConfig = config or HardwareDriverConfig()
        self._lock = threading.Lock()
        self._history: List[MotionCommand] = []
        self._feedbacks: Dict[str, MotionFeedback] = {}

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("MotionDriverDispatcher 初始化完成")

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _shutdown(self) -> None:
        with self._lock:
            self._history.clear()
            self._feedbacks.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def dispatch(self, motion_command: MotionCommand) -> str:
        """下发运动指令"""
        if not motion_command.command_id:
            motion_command.command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        if not motion_command.timestamp:
            motion_command.timestamp = time.time()
        self._validate_command(motion_command)
        with self._lock:
            self._history.append(motion_command)
            self._feedbacks[motion_command.command_id] = MotionFeedback(
                device_id=motion_command.device_id,
                command_id=motion_command.command_id,
                status="accepted",
            )
        logger.info(
            "下发运动指令 id=%s device=%s type=%s values=%s",
            motion_command.command_id, motion_command.device_id,
            motion_command.command_type, motion_command.values,
        )
        # 模拟异步执行完成
        fb = self._feedbacks[motion_command.command_id]
        fb.status = "done"
        return motion_command.command_id

    def get_feedback(self, device_id: str) -> Optional[MotionFeedback]:
        """获取指定设备最近一次指令反馈"""
        with self._lock:
            for cmd in reversed(self._history):
                if cmd.device_id == device_id:
                    return self._feedbacks.get(cmd.command_id)
        return None

    def get_history(self) -> List[MotionCommand]:
        with self._lock:
            return list(self._history)

    # ---- 内部 ----
    def _validate_command(self, cmd: MotionCommand) -> None:
        ctype = cmd.command_type
        if ctype == "angle":
            # values: {"axis": str, "angle": float}
            angle = cmd.values.get("angle")
            if angle is None or not (-360.0 <= float(angle) <= 360.0):
                cmd.values["angle"] = max(-360.0, min(360.0, float(angle or 0.0)))
        elif ctype == "velocity":
            v = cmd.values.get("velocity")
            if v is None:
                cmd.values["velocity"] = 0.0
        elif ctype == "limit":
            # values: {"axis": str, "min": float, "max": float}
            lo = float(cmd.values.get("min", 0.0))
            hi = float(cmd.values.get("max", 0.0))
            if lo > hi:
                cmd.values["min"], cmd.values["max"] = hi, lo
        elif ctype == "point":
            # values: {"x": float, "y": float, "z": float}
            for k in ("x", "y", "z"):
                if k not in cmd.values:
                    cmd.values[k] = 0.0
        else:
            logger.debug("未知指令类型 %s，原样下发", ctype)


# ====== T008: 硬件故障自检 ======

class HardwareFaultDetector(BaseModule):
    """T008 硬件故障自检

    轮询执行：检测断线/过载/离线/丢包。
    """

    name = "hardware_fault_detector"
    version = "1.0.0"
    description = "T008 硬件故障自检：断线/过载/离线/丢包检测"

    def __init__(
        self,
        config: Optional[HardwareDriverConfig] = None,
        sensor_initializer: Optional[SensorInitializer] = None,
    ) -> None:
        super().__init__()
        self.config: HardwareDriverConfig = config or HardwareDriverConfig()
        self._sensor_initializer = sensor_initializer
        self._lock = threading.Lock()
        self._fault_handlers: List[Callable[[FaultReport], None]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_reports: List[FaultReport] = []
        # 模拟丢包率统计
        self._packet_loss: Dict[str, float] = {}

    # ---- 生命周期钩子 ----
    def _initialize(self, config: Dict[str, Any]) -> None:
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        logger.info("HardwareFaultDetector 初始化完成")

    def _start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._check_loop, name="HardwareFaultLoop", daemon=True
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
            self._fault_handlers.clear()
            self._last_reports.clear()

    def _health_check(self) -> bool:
        return True

    # ---- 公共方法 ----
    def check(self) -> List[FaultReport]:
        """执行一次硬件故障自检"""
        reports: List[FaultReport] = []
        devices = self._list_devices()
        for dev in devices:
            did = dev.device_id
            # 断线检测
            if dev.status == "error":
                reports.append(FaultReport(
                    device_id=did, fault_type="disconnect",
                    severity="critical",
                    description=f"设备 {did} 状态异常: {dev.error}",
                    timestamp=time.time(),
                ))
            # 离线检测
            if dev.status == "offline":
                reports.append(FaultReport(
                    device_id=did, fault_type="offline",
                    severity="critical",
                    description=f"设备 {did} 离线",
                    timestamp=time.time(),
                ))
            # 模拟丢包检测
            loss = self._sample_packet_loss(did)
            if loss > self.config.packet_loss_threshold:
                reports.append(FaultReport(
                    device_id=did, fault_type="packet_loss",
                    severity="warning",
                    description=f"设备 {did} 丢包率 {loss:.2%} 超阈值",
                    timestamp=time.time(),
                ))
            # 模拟过载检测
            load = self._sample_load(did)
            if load > self.config.overload_load_threshold:
                reports.append(FaultReport(
                    device_id=did, fault_type="overload",
                    severity="warning",
                    description=f"设备 {did} 负载 {load:.1f}% 过载",
                    timestamp=time.time(),
                ))
        with self._lock:
            self._last_reports = reports
        # 通知订阅者
        for r in reports:
            for handler in list(self._fault_handlers):
                try:
                    handler(r)
                except Exception as e:  # pragma: no cover
                    logger.warning("故障订阅处理器异常: %s", e)
        return reports

    def subscribe_faults(self, handler: Callable[[FaultReport], None]) -> None:
        """订阅故障事件"""
        with self._lock:
            self._fault_handlers.append(handler)

    def get_last_reports(self) -> List[FaultReport]:
        with self._lock:
            return list(self._last_reports)

    # ---- 内部 ----
    def _list_devices(self) -> List[DeviceStatus]:
        if self._sensor_initializer is None:
            return []
        return self._sensor_initializer.list_devices()

    def _sample_packet_loss(self, device_id: str) -> float:
        rng = random.Random(device_id + ":loss")
        loss = rng.uniform(0.0, 0.3)
        self._packet_loss[device_id] = loss
        return loss

    def _sample_load(self, device_id: str) -> float:
        rng = random.Random(device_id + ":load")
        return rng.uniform(0.0, 100.0)

    def _check_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.check()
            except Exception as e:  # pragma: no cover
                logger.warning("故障自检循环异常: %s", e)
            self._stop_event.wait(self.config.poll_interval)
