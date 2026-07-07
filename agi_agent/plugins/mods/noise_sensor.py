import numpy as np
from typing import Dict, Any
from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint


class NoiseSensorPlugin(PeripheralPlugin):
    """环境噪声传感器插件。

    模拟环境噪声检测，将噪声数据注入智能体感知。
    """

    def __init__(self):
        super().__init__(
            name="noise_sensor",
            version="1.0.0",
            description="环境噪声传感器，检测周围环境噪声水平",
            plugin_type="sensor",
            priority=PluginPriority.NORMAL,
            config={"sample_rate": 44100, "buffer_size": 1024},
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[PluginHookPoint.PRE_PERCEPTION]
        )
        self.noise_level = 0.0
        self.frequency_spectrum = None

    def on_load(self) -> bool:
        self.noise_level = 0.0
        return True

    def on_unload(self) -> bool:
        self.noise_level = 0.0
        self.frequency_spectrum = None
        return True

    def on_activate(self) -> bool:
        return True

    def on_deactivate(self) -> bool:
        return True

    def hook_pre_perception(self, input_data):
        return self.process(input_data)

    def process(self, input_data: Any) -> Any:
        if isinstance(input_data, np.ndarray):
            noise = np.random.normal(0, 0.05, size=input_data.shape)
            self.noise_level = float(np.mean(np.abs(noise)))
            return input_data + noise
        return input_data

    def get_data(self) -> Dict[str, Any]:
        return {
            "noise_level": self.noise_level,
            "frequency_spectrum": self.frequency_spectrum,
            "is_loud": self.noise_level > 0.5
        }


def create_plugin():
    return NoiseSensorPlugin()
