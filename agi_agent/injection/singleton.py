import threading
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class SingletonMeta(type):
    _instances: Dict[Type, Any] = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class SingletonLazyMeta(type):
    _instances: Dict[Type, Any] = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


def singleton(cls: Type[T]) -> Type[T]:
    instances: Dict[Type, Any] = {}
    lock = threading.RLock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    get_instance.__name__ = cls.__name__
    get_instance.__doc__ = cls.__doc__
    return get_instance


class SingletonManager:
    _singletons: Dict[str, Any] = {}
    _lock = threading.RLock()
    _config = {
        "enable_lazy": True,
        "enable_cleanup": True,
        "max_instances": 100,
    }

    @classmethod
    def register(cls, name: str, instance: Any, factory: Optional[Callable] = None) -> None:
        with cls._lock:
            cls._singletons[name] = {
                "instance": instance,
                "factory": factory,
                "created": True,
            }

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        with cls._lock:
            entry = cls._singletons.get(name)
            if entry:
                if entry["created"]:
                    return entry["instance"]
                elif entry["factory"]:
                    entry["instance"] = entry["factory"]()
                    entry["created"] = True
                    return entry["instance"]
        return None

    @classmethod
    def lazy_register(cls, name: str, factory: Callable) -> None:
        with cls._lock:
            cls._singletons[name] = {
                "instance": None,
                "factory": factory,
                "created": False,
            }

    @classmethod
    def exists(cls, name: str) -> bool:
        return name in cls._singletons

    @classmethod
    def cleanup(cls, name: Optional[str] = None) -> bool:
        with cls._lock:
            if name:
                if name in cls._singletons:
                    instance = cls._singletons[name]["instance"]
                    if instance and hasattr(instance, "close"):
                        try:
                            instance.close()
                        except Exception:
                            pass
                    elif instance and hasattr(instance, "shutdown"):
                        try:
                            instance.shutdown()
                        except Exception:
                            pass
                    del cls._singletons[name]
                    return True
                return False
            else:
                count = 0
                for key in list(cls._singletons.keys()):
                    if cls.cleanup(key):
                        count += 1
                return count > 0

    @classmethod
    def list_singletons(cls) -> Dict[str, Dict[str, Any]]:
        with cls._lock:
            return {
                name: {
                    "created": entry["created"],
                    "class": type(entry["instance"]).__name__ if entry["instance"] else None,
                }
                for name, entry in cls._singletons.items()
            }

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        with cls._lock:
            total = len(cls._singletons)
            created = sum(1 for entry in cls._singletons.values() if entry["created"])
            return {
                "total_singletons": total,
                "created_instances": created,
                "lazy_instances": total - created,
            }


def with_singleton(name: str):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            instance = SingletonManager.get(name)
            if instance is None:
                raise RuntimeError(f"Singleton {name} not registered")
            return func(instance, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator