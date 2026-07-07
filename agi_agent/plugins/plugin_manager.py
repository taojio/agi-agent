import os
import sys
import importlib
import importlib.util
import importlib.machinery
import threading
import time
import hashlib
import gc
from typing import Dict, List, Optional, Any, Callable
from .plugin_base import PeripheralPlugin, PluginStatus, PluginPriority, PluginHookPoint


class PluginEvent:
    PLUGIN_DISCOVERED = "plugin_discovered"
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"
    PLUGIN_ACTIVATED = "plugin_activated"
    PLUGIN_DEACTIVATED = "plugin_deactivated"
    PLUGIN_ERROR = "plugin_error"


class PluginManager:
    """热拔插插件管理器。

    完全动态加载机制，支持：
    - 实时检测：文件系统监视，自动发现新增/修改/删除的插件
    - 自动注册：发现新插件后自动注册到系统
    - 按需加载：首次使用时才加载，节省资源
    - 资源释放：卸载时完全清理模块引用和缓存
    - 依赖管理：自动解析插件间依赖关系
    - 热更新：支持插件代码热重载
    """

    SYSTEM_VERSION = "1.0.0"

    def __init__(self, plugins_dir: str = None, auto_watch: bool = True):
        self.plugins_dir = plugins_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "plugins", "mods"
        )
        self._plugins: Dict[str, PeripheralPlugin] = {}
        self._module_cache: Dict[str, Any] = {}
        self._file_hashes: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._lazy_load_queue: Dict[str, str] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._watching = False
        self._watch_interval = 2.0

        os.makedirs(self.plugins_dir, exist_ok=True)

        self._scan_initial_plugins()

        if auto_watch:
            self._start_file_watcher()

    def _scan_initial_plugins(self):
        """启动时扫描所有可用插件（仅注册元信息，不加载）。"""
        available = self._scan_plugin_files()
        for name, filepath in available.items():
            self._lazy_load_queue[name] = filepath

    def _scan_plugin_files(self) -> Dict[str, str]:
        """扫描插件目录，返回插件名到文件路径的映射。"""
        plugins = {}
        if not os.path.exists(self.plugins_dir):
            return plugins

        for filename in os.listdir(self.plugins_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                filepath = os.path.join(self.plugins_dir, filename)
                name = os.path.splitext(filename)[0]
                plugins[name] = filepath

        return plugins

    def _compute_file_hash(self, filepath: str) -> str:
        """计算文件哈希用于检测变更。"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _start_file_watcher(self):
        """启动文件系统监视线程。"""
        if self._watching:
            return
        self._watching = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="PluginFileWatcher"
        )
        self._watch_thread.start()

    def _stop_file_watcher(self):
        """停止文件系统监视。"""
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
            self._watch_thread = None

    def _watch_loop(self):
        """文件监视主循环。"""
        last_scan = set()

        while self._watching:
            try:
                current_files = self._scan_plugin_files()

                for name, filepath in current_files.items():
                    file_hash = self._compute_file_hash(filepath)
                    old_hash = self._file_hashes.get(name)

                    if name not in last_scan:
                        self._on_plugin_file_added(name, filepath)
                    elif file_hash != old_hash and old_hash is not None:
                        self._on_plugin_file_modified(name, filepath)

                    self._file_hashes[name] = file_hash
                    last_scan.add(name)

                removed = last_scan - set(current_files.keys())
                for name in removed:
                    self._on_plugin_file_removed(name)
                    self._file_hashes.pop(name, None)
                    last_scan.discard(name)

            except Exception:
                pass

            time.sleep(self._watch_interval)

    def _on_plugin_file_added(self, name: str, filepath: str):
        """插件文件新增时的处理。"""
        if name not in self._plugins:
            self._lazy_load_queue[name] = filepath
            self._emit_event(PluginEvent.PLUGIN_DISCOVERED, {
                "name": name,
                "filepath": filepath
            })

    def _on_plugin_file_modified(self, name: str, filepath: str):
        """插件文件修改时的处理（热重载）。"""
        if name in self._plugins:
            try:
                self.reload_plugin(name)
            except Exception:
                pass

    def _on_plugin_file_removed(self, name: str):
        """插件文件删除时的处理。"""
        if name in self._plugins:
            try:
                self.unload_plugin(name)
            except Exception:
                pass
        self._lazy_load_queue.pop(name, None)

    def on_event(self, event_name: str, callback: Callable):
        """注册全局事件回调。"""
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def _emit_event(self, event_name: str, data: Any = None):
        """触发全局事件。"""
        for callback in self._event_callbacks.get(event_name, []):
            try:
                callback(event_name, data)
            except Exception:
                pass

    def scan_available_plugins(self) -> List[Dict[str, Any]]:
        """扫描插件目录，返回所有可用插件信息。"""
        available = []
        files = self._scan_plugin_files()

        for name, filepath in files.items():
            if name in self._plugins:
                plugin = self._plugins[name]
                info = plugin.get_info()
                info["filepath"] = filepath
                info["loaded"] = True
                available.append(info)
            else:
                info = self._inspect_plugin_metadata(filepath)
                if info:
                    info["filepath"] = filepath
                    info["loaded"] = False
                    info["status"] = "unloaded"
                    available.append(info)

        return available

    def _inspect_plugin_metadata(self, filepath: str) -> Optional[Dict[str, Any]]:
        """轻量级插件元信息检查，不实例化插件。"""
        try:
            module_name = os.path.splitext(os.path.basename(filepath))[0]

            loader = importlib.machinery.SourceFileLoader(module_name, filepath)
            spec = importlib.util.spec_from_loader(module_name, loader)
            if spec is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name + "_meta"] = module

            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules[module_name + "_meta"]
                return None

            plugin_class = getattr(module, 'Plugin', None) or getattr(module, 'create_plugin', None)

            if plugin_class is None:
                del sys.modules[module_name + "_meta"]
                return None

            try:
                if callable(plugin_class) and not isinstance(plugin_class, type):
                    instance = plugin_class()
                elif isinstance(plugin_class, type):
                    instance = plugin_class()
                else:
                    del sys.modules[module_name + "_meta"]
                    return None

                if hasattr(instance, 'get_info') and hasattr(instance, 'on_load'):
                    info = instance.get_info()
                    del sys.modules[module_name + "_meta"]
                    del instance
                    gc.collect()
                    return info

            except Exception:
                pass

            del sys.modules[module_name + "_meta"]
            gc.collect()
            return None

        except Exception:
            return None

    def _inspect_plugin_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """检查插件文件信息（兼容旧接口）。"""
        return self._inspect_plugin_metadata(filepath)

    def load_plugin(self, filepath: str = None, plugin_name: str = None, auto_activate: bool = False) -> Dict[str, Any]:
        """加载插件。

        Args:
            filepath: 插件文件路径。如果为None，则从plugins_dir中按plugin_name查找。
            plugin_name: 插件名称。如果filepath为None，则用此参数查找文件。
            auto_activate: 是否自动激活。

        Returns:
            加载结果信息。
        """
        with self._lock:
            if filepath is None:
                if plugin_name is None:
                    return {"success": False, "error": "必须提供filepath或plugin_name"}
                filepath = os.path.join(self.plugins_dir, f"{plugin_name}.py")
                if not os.path.exists(filepath):
                    return {"success": False, "error": f"插件文件不存在: {filepath}"}

            module_name = os.path.splitext(os.path.basename(filepath))[0]

            if module_name in self._plugins:
                return {"success": True, "info": self._plugins[module_name].get_info(), "already_loaded": True}

            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    return {"success": False, "error": "无法创建模块规范"}

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                self._module_cache[module_name] = module
                spec.loader.exec_module(module)

                plugin_class = getattr(module, 'Plugin', None) or getattr(module, 'create_plugin', None)

                if plugin_class is None:
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {"success": False, "error": "插件文件中未找到Plugin类或create_plugin函数"}

                if callable(plugin_class) and not isinstance(plugin_class, type):
                    instance = plugin_class()
                elif isinstance(plugin_class, type):
                    instance = plugin_class()
                else:
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {"success": False, "error": "Plugin类无效"}

                if not hasattr(instance, 'on_load') or not hasattr(instance, 'on_unload'):
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {"success": False, "error": "实例不符合插件接口"}

                if not self._check_version_compatibility(instance):
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {
                        "success": False,
                        "error": f"版本不兼容。系统版本: {self.SYSTEM_VERSION}, 兼容版本: {instance.compatible_versions}"
                    }

                if not instance.check_dependencies(self._plugins):
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {"success": False, "error": f"依赖不满足: {instance.dependencies}"}

                success = instance.on_load()
                if not success:
                    instance.status = PluginStatus.ERROR
                    del sys.modules[module_name]
                    self._module_cache.pop(module_name, None)
                    return {"success": False, "error": "插件on_load返回失败"}

                instance.status = PluginStatus.LOADED
                import time
                instance.load_time = time.time()
                self._plugins[instance.name] = instance

                self._lazy_load_queue.pop(instance.name, None)

                if auto_activate:
                    instance.activate()

                self._emit_event(PluginEvent.PLUGIN_LOADED, {
                    "name": instance.name,
                    "info": instance.get_info()
                })

                return {"success": True, "info": instance.get_info()}

            except Exception as e:
                module_name = os.path.splitext(os.path.basename(filepath))[0] if filepath else "unknown"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                self._module_cache.pop(module_name, None)
                self._emit_event(PluginEvent.PLUGIN_ERROR, {
                    "name": module_name,
                    "error": str(e),
                    "stage": "load"
                })
                return {"success": False, "error": str(e)}

    def _check_version_compatibility(self, plugin_instance) -> bool:
        """检查版本兼容性。"""
        if not hasattr(plugin_instance, 'compatible_versions'):
            return True
        return self.SYSTEM_VERSION in plugin_instance.compatible_versions

    def unload_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """卸载插件，完全释放资源。"""
        with self._lock:
            if plugin_name not in self._plugins:
                return {"success": False, "error": f"插件未加载: {plugin_name}"}

            plugin = self._plugins[plugin_name]
            plugin_info = plugin.get_info()

            try:
                if plugin.status == PluginStatus.ACTIVE:
                    plugin.deactivate()

                plugin.on_unload()
                plugin.status = PluginStatus.UNLOADED

                del self._plugins[plugin_name]

                for attr_name in list(dir(plugin)):
                    if not attr_name.startswith('_'):
                        try:
                            delattr(plugin, attr_name)
                        except Exception:
                            pass

                self._cleanup_module(plugin_name)

                gc.collect()

                self._emit_event(PluginEvent.PLUGIN_UNLOADED, {
                    "name": plugin_name,
                    "info": plugin_info
                })

                return {"success": True, "name": plugin_name}
            except Exception as e:
                self._emit_event(PluginEvent.PLUGIN_ERROR, {
                    "name": plugin_name,
                    "error": str(e),
                    "stage": "unload"
                })
                return {"success": False, "error": str(e)}

    def _cleanup_module(self, module_name: str):
        """彻底清理模块引用。"""
        module = sys.modules.get(module_name)
        if module is not None:
            for attr_name in list(dir(module)):
                if not attr_name.startswith('__'):
                    try:
                        delattr(module, attr_name)
                    except Exception:
                        pass

            del sys.modules[module_name]

        self._module_cache.pop(module_name, None)

        suffixes = ["_meta"]
        for suffix in suffixes:
            key = module_name + suffix
            if key in sys.modules:
                del sys.modules[key]

    def activate_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """激活插件。"""
        with self._lock:
            if plugin_name not in self._plugins:
                if plugin_name in self._lazy_load_queue:
                    result = self.load_plugin(plugin_name=plugin_name, auto_activate=True)
                    return result
                return {"success": False, "error": f"插件未加载: {plugin_name}"}

            plugin = self._plugins[plugin_name]

            if plugin.status == PluginStatus.ACTIVE:
                return {"success": True, "info": plugin.get_info()}

            if not plugin.check_dependencies(self._plugins):
                return {"success": False, "error": f"依赖不满足: {plugin.dependencies}"}

            success = plugin.activate()
            if success:
                self._emit_event(PluginEvent.PLUGIN_ACTIVATED, {
                    "name": plugin_name,
                    "info": plugin.get_info()
                })
            return {"success": success, "info": plugin.get_info()}

    def deactivate_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """停用插件。"""
        with self._lock:
            if plugin_name not in self._plugins:
                return {"success": False, "error": f"插件未加载: {plugin_name}"}

            plugin = self._plugins[plugin_name]
            success = plugin.deactivate()
            if success:
                self._emit_event(PluginEvent.PLUGIN_DEACTIVATED, {
                    "name": plugin_name,
                    "info": plugin.get_info()
                })
            return {"success": success, "info": plugin.get_info()}

    def get_loaded_plugins(self) -> List[Dict[str, Any]]:
        """获取所有已加载插件信息。"""
        with self._lock:
            return [plugin.get_info() for plugin in self._plugins.values()]

    def get_active_plugins(self) -> List[PeripheralPlugin]:
        """获取所有活跃的插件实例，按优先级排序。"""
        with self._lock:
            active = [p for p in self._plugins.values() if p.status == PluginStatus.ACTIVE]
            active.sort(key=lambda p: p.priority.value, reverse=True)
            return active

    def get_plugin(self, plugin_name: str) -> Optional[PeripheralPlugin]:
        """获取指定插件实例。"""
        return self._plugins.get(plugin_name)

    def process_with_plugins(self, input_data: Any, hook_point: PluginHookPoint = None) -> Dict[str, Any]:
        """用所有活跃插件处理输入数据。

        Args:
            input_data: 输入数据
            hook_point: 可选，仅处理注册了特定钩子的插件

        Returns:
            各插件的处理结果字典
        """
        results = {}
        for plugin in self.get_active_plugins():
            if hook_point and hook_point not in plugin.hook_points:
                continue
            try:
                results[plugin.name] = plugin.process(input_data)
            except Exception as e:
                results[plugin.name] = {"error": str(e)}
                plugin._error_count += 1
                plugin._last_error = str(e)
        return results

    def invoke_hook(self, hook_point: PluginHookPoint, data: Any = None) -> Dict[str, Any]:
        """在指定钩子点调用所有注册了该钩子的活跃插件。"""
        results = {}
        for plugin in self.get_active_plugins():
            if hook_point in plugin.hook_points:
                try:
                    hook_method = getattr(plugin, f"hook_{hook_point.value}", None)
                    if hook_method and callable(hook_method):
                        results[plugin.name] = hook_method(data)
                except Exception as e:
                    results[plugin.name] = {"error": str(e)}
        return results

    def get_all_plugin_data(self) -> Dict[str, Any]:
        """获取所有活跃插件的数据。"""
        data = {}
        for plugin in self.get_active_plugins():
            try:
                data[plugin.name] = plugin.get_data()
            except Exception as e:
                data[plugin.name] = {"error": str(e)}
        return data

    def reload_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """热重载插件（先卸载再加载，保留配置）。"""
        with self._lock:
            was_active = False
            old_config = None

            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                was_active = plugin.status == PluginStatus.ACTIVE
                old_config = plugin.config.copy() if plugin.config else None
                self.unload_plugin(plugin_name)

            filepath = os.path.join(self.plugins_dir, f"{plugin_name}.py")
            if not os.path.exists(filepath):
                return {"success": False, "error": f"插件文件不存在: {filepath}"}

            result = self.load_plugin(filepath=filepath, auto_activate=was_active)

            if result.get("success") and old_config and plugin_name in self._plugins:
                self._plugins[plugin_name].config.update(old_config)

            return result

    def load_all_from_dir(self) -> List[Dict[str, Any]]:
        """加载插件目录中的所有插件。"""
        results = []
        available = self._scan_plugin_files()

        for name in sorted(available.keys()):
            if name not in self._plugins:
                result = self.load_plugin(plugin_name=name)
                results.append({"name": name, "result": result})

        return results

    def activate_all(self) -> List[Dict[str, Any]]:
        """激活所有已加载的插件。"""
        results = []
        for name in list(self._plugins.keys()):
            result = self.activate_plugin(name)
            results.append({"name": name, "result": result})
        return results

    def deactivate_all(self) -> List[Dict[str, Any]]:
        """停用所有活跃插件。"""
        results = []
        for name in list(self._plugins.keys()):
            result = self.deactivate_plugin(name)
            results.append({"name": name, "result": result})
        return results

    def unload_all(self) -> List[Dict[str, Any]]:
        """卸载所有插件。"""
        results = []
        for name in list(self._plugins.keys()):
            result = self.unload_plugin(name)
            results.append({"name": name, "result": result})
        return results

    def notify_structure_change(self, new_dim: int):
        """通知所有活跃插件结构变更。"""
        for plugin in self.get_active_plugins():
            try:
                plugin.resize(new_dim)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """获取插件管理器整体状态。"""
        return {
            "plugins_dir": self.plugins_dir,
            "system_version": self.SYSTEM_VERSION,
            "total_loaded": len(self._plugins),
            "total_active": len(self.get_active_plugins()),
            "total_discovered": len(self._lazy_load_queue) + len(self._plugins),
            "watching": self._watching,
            "loaded_plugins": self.get_loaded_plugins()
        }

    def shutdown(self):
        """关闭插件管理器，卸载所有插件并停止监视。"""
        self._stop_file_watcher()
        self.unload_all()
