"""
config_runtime/config_hot_reloader.py - 配置热更新 (T022)

事件触发的配置热更新：应用新配置、注册监听器、刷新通知。
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.config_runtime")


@dataclass
class HotReloaderConfig:
    """配置热更新配置"""
    notify_on_unchanged: bool = False  # 配置未变化时是否仍通知


class ConfigHotReloader(BaseModule):
    """配置热更新器 (T022)

    维护当前生效的配置与监听器列表，apply_update 应用新配置并按 key 通知
    关注该 key 的回调；reload 重新触发全量通知。
    """

    name = "config_hot_reloader"
    version = "1.0.0"
    description = "配置热更新 (T022)"

    def __init__(self, config: Optional[HotReloaderConfig] = None, initial: Optional[Any] = None):
        super().__init__()
        self._cfg = config or HotReloaderConfig()
        self._current: Any = initial
        self._previous: Any = None
        self._listeners: List[Tuple[str, Callable[[Any], None]]] = []
        self._lock = threading.Lock()
        self._reload_count: int = 0

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        logger.info("ConfigHotReloader 初始化完成 (listeners=%d)", len(self._listeners))

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    @property
    def current(self) -> Any:
        """当前生效配置"""
        return self._current

    @property
    def reload_count(self) -> int:
        return self._reload_count

    def register_listener(self, key: str, callback: Callable[[Any], None]) -> None:
        """注册监听器

        Args:
            key: 监听的配置 key（支持 "*" 通配全部）
            callback: 配置变更回调
        """
        with self._lock:
            self._listeners.append((key, callback))
        logger.info("注册配置监听器: key=%s", key)

    def apply_update(self, new_config: Any) -> Dict[str, Any]:
        """应用新配置并通知变更的监听器

        Args:
            new_config: 新配置对象（AgentConfig 或 dict）

        Returns:
            dict: 通知统计
        """
        with self._lock:
            self._previous = self._current
            self._current = new_config
            changed_keys = self._diff_keys(self._previous, new_config)
            notified = 0
            for key, cb in list(self._listeners):
                if key == "*" or key in changed_keys or (self._cfg.notify_on_unchanged):
                    try:
                        cb(new_config)
                        notified += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning("配置监听器回调异常 (key=%s): %s", key, e)
            self._reload_count += 1
        logger.info("配置热更新完成 (changed_keys=%d, notified=%d)", len(changed_keys), notified)
        return {"changed_keys": list(changed_keys), "notified": notified}

    def reload(self) -> Dict[str, Any]:
        """重新触发全量通知（基于当前配置）

        Returns:
            dict: 通知统计
        """
        if self._current is None:
            return {"notified": 0}
        notified = 0
        with self._lock:
            for key, cb in list(self._listeners):
                try:
                    cb(self._current)
                    notified += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("配置 reload 回调异常 (key=%s): %s", key, e)
            self._reload_count += 1
        logger.info("配置 reload 完成 (notified=%d)", notified)
        return {"notified": notified}

    def list_listeners(self) -> List[str]:
        """返回已注册监听器的 key 列表"""
        with self._lock:
            return [k for k, _ in self._listeners]

    # ====== 内部 ======
    @staticmethod
    def _diff_keys(old: Any, new: Any) -> List[str]:
        if old is None:
            if new is None:
                return []
            # 首次应用：新配置的全部顶层 key 视为变更
            return list(ConfigHotReloader._as_dict(new).keys())
        old_dict = ConfigHotReloader._as_dict(old)
        new_dict = ConfigHotReloader._as_dict(new)
        keys = set(old_dict.keys()) | set(new_dict.keys())
        changed = [k for k in keys if old_dict.get(k) != new_dict.get(k)]
        return changed

    @staticmethod
    def _as_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return dict(obj)
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}
