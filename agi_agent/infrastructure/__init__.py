"""
infrastructure/__init__.py - 底层算力&硬件支撑模块

实现智能体全模块原子任务清单中的「底层算力&硬件支撑模块」全部任务（T001-T012）。

子模块：
- compute_monitor:        T001 算力资源巡检 / T004 算力异常回收
- load_balancer:          T002 推理负载均衡
- cluster_scheduler:      T003 分布式节点调度
- hardware_driver:        T005 传感器初始化 / T006 外设数据读取 /
                          T007 运动驱动下发 / T008 硬件故障自检
- communication:          T009 IPC 本地消息转发 / T010 网络连接维持 /
                          T011 消息队列分发 / T012 跨设备数据加密传输
"""
from .compute_monitor import (
    AnomalyEvent,
    AnomalyRecoveryConfig,
    AnomalyRecoveryTask,
    ResourceInspectionConfig,
    ResourceInspectionTask,
    ResourceReport,
)
from .load_balancer import (
    AllocationPlan,
    DeviceInfo,
    InferenceLoadBalancer,
    InferenceRequest,
    LoadBalancerConfig,
)
from .cluster_scheduler import (
    ClusterSchedulerConfig,
    DistributedNodeScheduler,
    GlobalTask,
    NodeHeartbeat,
    NodeInfo,
    SubTaskAssignment,
)
from .hardware_driver import (
    DeviceConfig,
    DeviceStatus,
    FaultReport,
    HardwareDriverConfig,
    HardwareFaultDetector,
    MotionCommand,
    MotionDriverDispatcher,
    MotionFeedback,
    PeripheralDataReader,
    SensorInitializer,
    SensorReading,
)
from .communication import (
    EncryptedTransport,
    EncryptedTransportConfig,
    IPCBus,
    IPCConfig,
    LinkStatus,
    MessageQueueConfig,
    MessageQueueDispatcher,
    NetworkEndpoint,
    NetworkLinkConfig,
    NetworkLinkManager,
    QueueMessage,
)

__all__ = [
    # T001 / T004
    "ResourceInspectionTask",
    "ResourceInspectionConfig",
    "ResourceReport",
    "AnomalyRecoveryTask",
    "AnomalyRecoveryConfig",
    "AnomalyEvent",
    # T002
    "InferenceLoadBalancer",
    "LoadBalancerConfig",
    "InferenceRequest",
    "AllocationPlan",
    "DeviceInfo",
    # T003
    "DistributedNodeScheduler",
    "ClusterSchedulerConfig",
    "NodeHeartbeat",
    "NodeInfo",
    "GlobalTask",
    "SubTaskAssignment",
    # T005 - T008
    "SensorInitializer",
    "PeripheralDataReader",
    "MotionDriverDispatcher",
    "HardwareFaultDetector",
    "HardwareDriverConfig",
    "DeviceConfig",
    "DeviceStatus",
    "SensorReading",
    "MotionCommand",
    "MotionFeedback",
    "FaultReport",
    # T009 - T012
    "IPCBus",
    "IPCConfig",
    "NetworkLinkManager",
    "NetworkLinkConfig",
    "NetworkEndpoint",
    "LinkStatus",
    "MessageQueueDispatcher",
    "MessageQueueConfig",
    "QueueMessage",
    "EncryptedTransport",
    "EncryptedTransportConfig",
]
