import numpy as np
from typing import Dict, Any
from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint


class DataProcessorPlugin(PeripheralPlugin):
    """数据预处理插件。

    对输入数据进行滤波、归一化等预处理操作。
    """

    def __init__(self):
        super().__init__(
            name="data_processor",
            version="1.0.0",
            description="数据预处理插件，提供滤波和归一化功能",
            plugin_type="processor",
            priority=PluginPriority.HIGH,
            config={"filter_type": "lowpass", "cutoff": 0.8},
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[PluginHookPoint.PRE_PERCEPTION, PluginHookPoint.PRE_COGNITION]
        )
        self.filter_buffer = []
        self.buffer_size = 5

    def on_load(self) -> bool:
        self.filter_buffer = []
        return True

    def on_unload(self) -> bool:
        self.filter_buffer = []
        return True

    def hook_pre_perception(self, input_data):
        return self.process(input_data)

    def hook_pre_cognition(self, input_data):
        return input_data

    def process(self, input_data: Any) -> Any:
        if isinstance(input_data, np.ndarray):
            self.filter_buffer.append(input_data.copy())
            if len(self.filter_buffer) > self.buffer_size:
                self.filter_buffer.pop(0)

            if len(self.filter_buffer) >= 2:
                filtered = np.mean(self.filter_buffer, axis=0)
            else:
                filtered = input_data

            min_val = np.min(filtered)
            max_val = np.max(filtered)
            if max_val - min_val > 1e-8:
                normalized = (filtered - min_val) / (max_val - min_val)
            else:
                normalized = filtered

            return normalized
        return input_data

    def get_data(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.filter_buffer),
            "filter_type": self.config["filter_type"],
            "is_processing": len(self.filter_buffer) > 0
        }


def create_plugin():
    return DataProcessorPlugin()
