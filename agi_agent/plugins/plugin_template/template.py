import os
from typing import Dict, Any


PLUGIN_TEMPLATE = '''"""
__plugin_name__.py - __plugin_description__

__plugin_long_description__
"""
from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger("__plugin_name__")


class __plugin_class_name__(PeripheralPlugin):
    """__plugin_description__"""

    def __init__(self):
        super().__init__(
            name="__plugin_name__",
            version="__plugin_version__",
            description="__plugin_description__",
            plugin_type="__plugin_type__",
            priority=PluginPriority.__priority__,
            config=__config__,
            dependencies=__dependencies__,
            compatible_versions=["1.0.0"],
            hook_points=__hook_points__
        )
        self._lock = None

    def on_load(self) -> bool:
        """Initialize plugin resources."""
        try:
            logger.info(f"Loading {self.name} v{self.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return False

    def on_unload(self) -> bool:
        """Release all resources."""
        try:
            logger.info(f"Unloading {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload {self.name}: {e}")
            return False

    def on_activate(self) -> bool:
        """Activate the plugin."""
        logger.info(f"Activating {self.name}")
        return True

    def on_deactivate(self) -> bool:
        """Deactivate the plugin."""
        logger.info(f"Deactivating {self.name}")
        return True

    def process(self, input_data: Any) -> Any:
        """Process input data.
        
        Args:
            input_data: Input data from the system
            
        Returns:
            Processed result
        """
        return input_data

    def get_data(self) -> Dict[str, Any]:
        """Get current plugin state."""
        return {
            "status": "active" if self.status == "active" else "inactive",
            "config": self.config,
        }


def create_plugin():
    """Factory function for creating plugin instance."""
    return __plugin_class_name__()
'''


def create_plugin_template(plugin_name: str, 
                           plugin_description: str = "My Plugin",
                           plugin_type: str = "utility",
                           version: str = "1.0.0",
                           priority: str = "NORMAL",
                           config: Dict[str, Any] = None,
                           dependencies: list = None,
                           hook_points: list = None) -> str:
    """Generate a plugin template.
    
    Args:
        plugin_name: Name of the plugin
        plugin_description: Brief description
        plugin_type: Type of plugin (sensor/processor/analyzer/actuator/utility)
        version: Plugin version
        priority: Plugin priority (HIGH/NORMAL/LOW)
        config: Default configuration
        dependencies: List of dependent plugins
        hook_points: List of hook points to register
        
    Returns:
        Generated plugin code as string
    """
    plugin_class_name = ''.join(word.capitalize() for word in plugin_name.split('_')) + 'Plugin'
    
    config_str = str(config) if config else "{}"
    dependencies_str = str(dependencies) if dependencies else "[]"
    hook_points_str = str(hook_points) if hook_points else "[PluginHookPoint.PERIODIC]"
    
    template = PLUGIN_TEMPLATE
    template = template.replace("__plugin_name__", plugin_name)
    template = template.replace("__plugin_description__", plugin_description)
    template = template.replace("__plugin_long_description__", f"This is a {plugin_type} plugin for AGI Agent.")
    template = template.replace("__plugin_class_name__", plugin_class_name)
    template = template.replace("__plugin_version__", version)
    template = template.replace("__plugin_type__", plugin_type)
    template = template.replace("__priority__", priority)
    template = template.replace("__config__", config_str)
    template = template.replace("__dependencies__", dependencies_str)
    template = template.replace("__hook_points__", hook_points_str)
    
    return template


def write_plugin_file(plugin_name: str, output_dir: str = None, **kwargs) -> str:
    """Write plugin template to a file.
    
    Args:
        plugin_name: Name of the plugin
        output_dir: Directory to write the file to
        **kwargs: Additional parameters for template generation
        
    Returns:
        Path to the generated file
    """
    template = create_plugin_template(plugin_name, **kwargs)
    output_dir = output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mods"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{plugin_name}.py")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return filepath