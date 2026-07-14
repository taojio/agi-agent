import numpy as np
from typing import Dict, Any
from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint


class TemperatureSensorPlugin(PeripheralPlugin):
    """温度传感器插件。

    模拟环境温度检测，将温度数据注入智能体感知。
    """

    def __init__(self):
        super().__init__(
            name="temperature_sensor",
            version="1.0.0",
            description="环境温度传感器，检测周围环境温度变化",
            plugin_type="sensor",
            priority=PluginPriority.NORMAL,
            config={"unit": "celsius", "range": [-40, 85]},
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[PluginHookPoint.POST_PERCEPTION]
        )
        self.temperature = 25.0
        self.history = []
        self.max_history = 100

    def on_load(self) -> bool:
        self.temperature = 25.0
        self.history = []
        return True

    def on_unload(self) -> bool:
        self.history = []
        return True

    def on_structure_change(self, new_dim: int) -> bool:
        return True

    def hook_post_perception(self, input_data):
        return self.process(input_data)

    def process(self, input_data: Any) -> Any:
        self.temperature += np.random.uniform(-0.5, 0.5)
        self.temperature = max(-40.0, min(85.0, self.temperature))

        self.history.append(self.temperature)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if isinstance(input_data, np.ndarray):
            flat = input_data.flatten()
            temp_signal = np.full(1, self.temperature / 85.0)
            return np.concatenate([flat, temp_signal])
        return input_data

    def get_data(self) -> Dict[str, Any]:
        return {
            "temperature": round(self.temperature, 2),
            "unit": "celsius",
            "trend": "rising" if len(self.history) >= 2 and self.history[-1] > self.history[-2] else "falling",
            "avg": round(float(np.mean(self.history)), 2) if self.history else 25.0
        }


def create_plugin():
    return TemperatureSensorPlugin()
