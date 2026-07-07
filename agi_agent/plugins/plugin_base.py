from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
import numpy as np


class PluginStatus(Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"


class PluginPriority(Enum):
    LOW = 10
    NORMAL = 50
    HIGH = 80
    CRITICAL = 100


class PluginHookPoint(Enum):
    PRE_PERCEPTION = "pre_perception"
    POST_PERCEPTION = "post_perception"
    PRE_COGNITION = "pre_cognition"
    POST_COGNITION = "post_cognition"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    PERIODIC = "periodic"
    ON_STRUCTURE_CHANGE = "on_structure_change"


class PeripheralPlugin(ABC):
    """热拔插外设插件基类。

    所有外设MOD需继承此类并实现接口方法。
    插件生命周期: load -> activate -> [process/hooks] -> deactivate -> unload

    开发规范:
    - 必须实现 on_load, on_unload, process, get_data 四个核心方法
    - 可选实现钩子方法: on_activate, on_deactivate, on_structure_change
    - 所有状态变更必须通过 status 属性反映
    - 资源在 on_unload 中必须完全释放
    """

    def __init__(self, name: str, version: str = "1.0.0", description: str = "",
                 plugin_type: str = "generic", priority: PluginPriority = PluginPriority.NORMAL,
                 config: Optional[Dict] = None,
                 dependencies: Optional[List[str]] = None,
                 compatible_versions: Optional[List[str]] = None,
                 hook_points: Optional[List[PluginHookPoint]] = None):
        self.name = name
        self.version = version
        self.description = description
        self.plugin_type = plugin_type
        self.priority = priority
        self.config = config or {}
        self.dependencies = dependencies or []
        self.compatible_versions = compatible_versions or ["1.0.0"]
        self.hook_points = hook_points or []
        self.status = PluginStatus.UNLOADED
        self.data_buffer = []
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._error_count = 0
        self._last_error: Optional[str] = None
        self.load_time: Optional[float] = None
        self.activate_time: Optional[float] = None
        self.process_count = 0

    def on_event(self, event_name: str, callback: Callable):
        """注册事件回调。"""
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def _emit_event(self, event_name: str, data: Any = None):
        """触发事件。"""
        for callback in self._event_callbacks.get(event_name, []):
            try:
                callback(self, data)
            except Exception:
                pass

    @abstractmethod
    def on_load(self) -> bool:
        """插件加载时调用，返回是否成功。

        在此方法中初始化资源、读取配置、建立连接等。
        失败时返回 False，管理器会拒绝加载。
        """
        pass

    @abstractmethod
    def on_unload(self) -> bool:
        """插件卸载时调用，返回是否成功。

        在此方法中释放所有资源：关闭连接、清理缓存、
        销毁线程、释放内存等。必须保证完全释放。
        """
        pass

    def on_activate(self) -> bool:
        """插件激活前调用，返回是否成功激活。

        可选实现。用于启动后台线程、开始数据流等。
        """
        return True

    def on_deactivate(self) -> bool:
        """插件停用后调用，返回是否成功停用。

        可选实现。用于暂停后台线程、停止数据流等。
        """
        return True

    def on_structure_change(self, new_dim: int) -> bool:
        """系统结构变化时调用。

        可选实现。用于调整插件内部维度以适配新结构。
        """
        return True

    def activate(self) -> bool:
        """激活插件。"""
        if self.status == PluginStatus.ACTIVE:
            return True
        try:
            result = self.on_activate()
            if result:
                self.status = PluginStatus.ACTIVE
                import time
                self.activate_time = time.time()
                self._emit_event("activated")
            else:
                self.status = PluginStatus.ERROR
                self._last_error = "on_activate returned False"
            return result
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            self._error_count += 1
            return False

    def deactivate(self) -> bool:
        """停用插件。"""
        if self.status == PluginStatus.LOADED:
            return True
        try:
            result = self.on_deactivate()
            if result:
                self.status = PluginStatus.LOADED
                self.activate_time = None
                self._emit_event("deactivated")
            return result
        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1
            self.status = PluginStatus.LOADED
            return True

    def check_dependencies(self, loaded_plugins: Dict[str, Any]) -> bool:
        """检查依赖是否满足。"""
        for dep in self.dependencies:
            if dep not in loaded_plugins:
                return False
        return True

    def resize(self, new_dim: int) -> bool:
        """适配新的维度。"""
        return self.on_structure_change(new_dim)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理输入数据，返回处理结果。

        Args:
            input_data: 输入数据，可以是观测向量、传感器数据等

        Returns:
            处理结果，将被注入到智能体的感知或认知模块
        """
        self.process_count += 1
        return input_data

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """获取插件当前数据状态。"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息。"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.plugin_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "config": self.config,
            "dependencies": self.dependencies,
            "hook_points": [hp.value for hp in self.hook_points],
            "error_count": self._error_count,
            "last_error": self._last_error,
            "process_count": self.process_count,
            "compatible_versions": self.compatible_versions
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """获取插件状态摘要。"""
        return {
            "name": self.name,
            "status": self.status.value,
            "error_count": self._error_count,
            "process_count": self.process_count
        }
