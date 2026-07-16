import importlib
import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict


logger = logging.getLogger(__name__)


class LoadStatus(Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class LoadTrigger(Enum):
    EVENT = "event"
    ROUTE = "route"
    CALL = "call"
    TIMER = "timer"
    CONDITION = "condition"


@dataclass
class LazyModule:
    name: str
    module_path: str
    class_name: Optional[str] = None
    trigger: LoadTrigger = LoadTrigger.CALL
    trigger_value: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: LoadStatus = LoadStatus.NOT_LOADED
    instance: Any = None
    load_time: float = 0.0
    error: Optional[str] = None
    priority: int = 0
    auto_unload_after: Optional[float] = None
    last_access_time: float = 0.0
    access_count: int = 0


@dataclass
class LoadMetrics:
    total_modules: int = 0
    loaded_modules: int = 0
    failed_modules: int = 0
    total_load_time: float = 0.0
    avg_load_time: float = 0.0
    peak_memory_mb: float = 0.0
    load_count: int = 0


class LazyLoader:
    def __init__(self):
        self._modules: Dict[str, LazyModule] = {}
        self._loading_tasks: Dict[str, asyncio.Future] = {}
        self._load_triggers: Dict[LoadTrigger, Dict[str, List[str]]] = defaultdict(dict)
        self._metrics = LoadMetrics()
        self._event_subscriptions: Dict[str, Callable] = {}
        self._auto_unload_task = None
    
    def register(self, name: str, module_path: str, class_name: Optional[str] = None,
                 trigger: LoadTrigger = LoadTrigger.CALL, trigger_value: Optional[str] = None,
                 dependencies: Optional[List[str]] = None, priority: int = 0,
                 auto_unload_after: Optional[float] = None) -> "LazyLoader":
        if name in self._modules:
            logger.warning(f"Module '{name}' already registered, overwriting")
        
        module = LazyModule(
            name=name,
            module_path=module_path,
            class_name=class_name,
            trigger=trigger,
            trigger_value=trigger_value,
            dependencies=dependencies or [],
            priority=priority,
            auto_unload_after=auto_unload_after,
        )
        
        self._modules[name] = module
        self._metrics.total_modules += 1
        
        if trigger_value:
            if trigger_value not in self._load_triggers[trigger]:
                self._load_triggers[trigger][trigger_value] = []
            self._load_triggers[trigger][trigger_value].append(name)
        
        return self
    
    def register_multiple(self, modules: List[Dict[str, Any]]) -> "LazyLoader":
        for mod in modules:
            self.register(**mod)
        return self
    
    async def load(self, name: str, **kwargs) -> Any:
        if name not in self._modules:
            raise ModuleNotFoundError(f"Module '{name}' not registered")
        
        module = self._modules[name]
        
        if module.status == LoadStatus.LOADED:
            module.last_access_time = time.time()
            module.access_count += 1
            return module.instance
        
        if module.status == LoadStatus.LOADING:
            if name in self._loading_tasks:
                await self._loading_tasks[name]
            return module.instance if module.status == LoadStatus.LOADED else None
        
        module.status = LoadStatus.LOADING
        start_time = time.time()
        
        try:
            for dep_name in module.dependencies:
                if dep_name in self._modules and self._modules[dep_name].status != LoadStatus.LOADED:
                    await self.load(dep_name)
            
            imported_module = importlib.import_module(module.module_path)
            
            if module.class_name:
                cls = getattr(imported_module, module.class_name)
                if hasattr(cls, '__init__'):
                    try:
                        module.instance = cls(**kwargs)
                    except TypeError:
                        module.instance = cls()
                else:
                    module.instance = cls()
            else:
                module.instance = imported_module
            
            module.status = LoadStatus.LOADED
            module.load_time = time.time() - start_time
            module.last_access_time = time.time()
            module.access_count += 1
            
            self._metrics.loaded_modules += 1
            self._metrics.total_load_time += module.load_time
            self._metrics.avg_load_time = self._metrics.total_load_time / self._metrics.loaded_modules
            self._metrics.load_count += 1
            
            logger.info(f"Module '{name}' loaded in {module.load_time:.3f}s")
            
            return module.instance
            
        except Exception as e:
            module.status = LoadStatus.FAILED
            module.error = str(e)
            self._metrics.failed_modules += 1
            
            logger.error(f"Failed to load module '{name}': {e}")
            raise
    
    def load_sync(self, name: str, **kwargs) -> Any:
        if asyncio.get_event_loop().is_running():
            return asyncio.ensure_future(self.load(name, **kwargs))
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.load(name, **kwargs))
        finally:
            loop.close()
    
    async def load_on_event(self, event_type: str, **kwargs) -> None:
        modules = self._load_triggers.get(LoadTrigger.EVENT, {}).get(event_type, [])
        for name in modules:
            await self.load(name, **kwargs)
    
    async def load_on_route(self, route: str, **kwargs) -> None:
        modules = self._load_triggers.get(LoadTrigger.ROUTE, {}).get(route, [])
        for name in modules:
            await self.load(name, **kwargs)
    
    def trigger_load(self, trigger: LoadTrigger, value: str, **kwargs) -> None:
        modules = self._load_triggers.get(trigger, {}).get(value, [])
        for name in modules:
            asyncio.create_task(self.load(name, **kwargs))
    
    def get_module(self, name: str) -> Optional[LazyModule]:
        return self._modules.get(name)
    
    def get_status(self, name: str) -> LoadStatus:
        module = self._modules.get(name)
        return module.status if module else LoadStatus.NOT_LOADED
    
    def is_loaded(self, name: str) -> bool:
        return self.get_status(name) == LoadStatus.LOADED
    
    def get_all_status(self) -> Dict[str, str]:
        return {name: mod.status.value for name, mod in self._modules.items()}
    
    def get_metrics(self) -> LoadMetrics:
        return self._metrics
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        return {
            "total_modules": self._metrics.total_modules,
            "loaded_modules": self._metrics.loaded_modules,
            "failed_modules": self._metrics.failed_modules,
            "total_load_time": self._metrics.total_load_time,
            "avg_load_time": self._metrics.avg_load_time,
            "peak_memory_mb": self._metrics.peak_memory_mb,
            "load_count": self._metrics.load_count,
        }
    
    def unload(self, name: str) -> bool:
        if name not in self._modules:
            return False
        
        module = self._modules[name]
        
        if module.status == LoadStatus.LOADING:
            if name in self._loading_tasks:
                self._loading_tasks[name].cancel()
                del self._loading_tasks[name]
        
        if hasattr(module.instance, 'shutdown'):
            try:
                result = module.instance.shutdown()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.warning(f"Error during shutdown of '{name}': {e}")
        
        module.instance = None
        module.status = LoadStatus.NOT_LOADED
        module.load_time = 0.0
        module.error = None
        
        self._metrics.loaded_modules -= 1
        
        logger.info(f"Module '{name}' unloaded")
        return True
    
    async def unload_unused(self, idle_threshold: float = 300.0) -> int:
        unloaded_count = 0
        now = time.time()
        
        for name, module in self._modules.items():
            if module.status == LoadStatus.LOADED and module.auto_unload_after:
                if now - module.last_access_time > module.auto_unload_after:
                    if self.unload(name):
                        unloaded_count += 1
        
        if unloaded_count > 0:
            logger.info(f"Unloaded {unloaded_count} unused modules")
        
        return unloaded_count
    
    def start_auto_unload(self, interval: float = 60.0) -> None:
        async def auto_unload_loop():
            while True:
                await asyncio.sleep(interval)
                await self.unload_unused()
        
        self._auto_unload_task = asyncio.create_task(auto_unload_loop())
    
    def stop_auto_unload(self) -> None:
        if self._auto_unload_task:
            self._auto_unload_task.cancel()
            self._auto_unload_task = None
    
    def shutdown(self) -> None:
        self.stop_auto_unload()
        for name in list(self._modules.keys()):
            self.unload(name)


_default_loader: Optional[LazyLoader] = None


def get_loader() -> LazyLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = LazyLoader()
    return _default_loader


def lazy_load(name: str, **kwargs):
    def decorator(cls: Type) -> Type:
        loader = get_loader()
        loader.register(
            name=name,
            module_path=cls.__module__,
            class_name=cls.__name__,
            **kwargs
        )
        return cls
    return decorator