import asyncio
import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, overload
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from .singleton import SingletonManager


class Lifecycle(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceRegistration:
    service_type: Type
    implementation_type: Type
    factory: Optional[Callable] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    instance: Any = None
    dependencies: List[Type] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    priority: int = 0


T = TypeVar("T")


class DependencyInjectionContainer:
    def __init__(self):
        self._registrations: Dict[Type, List[ServiceRegistration]] = defaultdict(list)
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._registered_names: Dict[str, Type] = {}
        self._resolving: set = set()

    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
        tags: Optional[List[str]] = None,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> "DependencyInjectionContainer":
        impl_type = implementation_type or service_type
        
        deps = self._extract_dependencies(impl_type, factory)
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=impl_type,
            factory=factory,
            lifecycle=lifecycle,
            dependencies=deps,
            tags=tags or [],
            priority=priority,
        )
        
        self._registrations[service_type].append(registration)
        
        if name:
            self._registered_names[name] = service_type
        
        return self

    def register_singleton(
        self,
        service_type: Type[T],
        instance: Any,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "DependencyInjectionContainer":
        self._singletons[service_type] = instance
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=type(instance),
            instance=instance,
            lifecycle=Lifecycle.SINGLETON,
            tags=tags or [],
            priority=0,
        )
        
        self._registrations[service_type].append(registration)
        
        if name:
            self._registered_names[name] = service_type
        
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable,
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "DependencyInjectionContainer":
        deps = self._extract_dependencies(None, factory)
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=service_type,
            factory=factory,
            lifecycle=lifecycle,
            dependencies=deps,
            tags=tags or [],
            priority=0,
        )
        
        self._registrations[service_type].append(registration)
        
        if name:
            self._registered_names[name] = service_type
        
        return self

    def resolve(self, service_type: Type[T], name: Optional[str] = None) -> T:
        if name and name in self._registered_names:
            service_type = self._registered_names[name]
        
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        if service_type in self._resolving:
            raise RuntimeError(f"Circular dependency detected for {service_type}")
        
        registrations = self._registrations.get(service_type, [])
        if not registrations:
            raise RuntimeError(f"No registration found for {service_type}")
        
        registration = max(registrations, key=lambda r: r.priority)
        
        if registration.instance is not None:
            return registration.instance
        
        self._resolving.add(service_type)
        
        try:
            if registration.lifecycle == Lifecycle.SINGLETON:
                if service_type not in self._singletons:
                    instance = self._create_instance(registration)
                    self._singletons[service_type] = instance
                return self._singletons[service_type]
            elif registration.lifecycle == Lifecycle.TRANSIENT:
                return self._create_instance(registration)
            elif registration.lifecycle == Lifecycle.SCOPED:
                return self._get_scoped_instance(service_type, registration)
            else:
                return self._create_instance(registration)
        finally:
            self._resolving.discard(service_type)

    async def resolve_async(self, service_type: Type[T], name: Optional[str] = None) -> T:
        if name and name in self._registered_names:
            service_type = self._registered_names[name]
        
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        registrations = self._registrations.get(service_type, [])
        if not registrations:
            raise RuntimeError(f"No registration found for {service_type}")
        
        registration = max(registrations, key=lambda r: r.priority)
        
        if registration.instance is not None:
            return registration.instance
        
        self._resolving.add(service_type)
        
        try:
            if registration.lifecycle == Lifecycle.SINGLETON:
                if service_type not in self._singletons:
                    instance = await self._create_instance_async(registration)
                    self._singletons[service_type] = instance
                return self._singletons[service_type]
            elif registration.lifecycle == Lifecycle.TRANSIENT:
                return await self._create_instance_async(registration)
            elif registration.lifecycle == Lifecycle.SCOPED:
                return self._get_scoped_instance(service_type, registration)
            else:
                return await self._create_instance_async(registration)
        finally:
            self._resolving.discard(service_type)

    def resolve_all(self, service_type: Type[T]) -> List[T]:
        registrations = self._registrations.get(service_type, [])
        return [self.resolve(service_type) for _ in registrations]

    def inject(self, obj: Any) -> Any:
        if inspect.isclass(obj):
            return self._inject_into_class(obj)
        elif callable(obj):
            return self._inject_into_callable(obj)
        return obj

    def _inject_into_class(self, cls: Type) -> Type:
        original_init = cls.__init__
        
        def injected_init(self, *args, **kwargs):
            hints = typing.get_type_hints(original_init)
            hints.pop("return", None)
            hints.pop("self", None)
            
            for param_name, param_type in hints.items():
                if param_name not in kwargs:
                    try:
                        kwargs[param_name] = self.resolve(param_type)
                    except RuntimeError:
                        pass
            
            original_init(self, *args, **kwargs)
        
        cls.__init__ = injected_init
        return cls

    def _inject_into_callable(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            hints = typing.get_type_hints(func)
            hints.pop("return", None)
            
            for param_name, param_type in hints.items():
                if param_name not in kwargs:
                    try:
                        kwargs[param_name] = self.resolve(param_type)
                    except RuntimeError:
                        pass
            
            return func(*args, **kwargs)
        
        return wrapper

    def _create_instance(self, registration: ServiceRegistration) -> Any:
        if registration.factory:
            return self._call_factory(registration.factory)
        
        cls = registration.implementation_type
        hints = typing.get_type_hints(cls.__init__)
        hints.pop("return", None)
        hints.pop("self", None)
        
        kwargs = {}
        for param_name, param_type in hints.items():
            kwargs[param_name] = self.resolve(param_type)
        
        return cls(**kwargs)

    async def _create_instance_async(self, registration: ServiceRegistration) -> Any:
        if registration.factory:
            result = self._call_factory(registration.factory)
            if asyncio.iscoroutine(result):
                return await result
            return result
        
        cls = registration.implementation_type
        hints = typing.get_type_hints(cls.__init__)
        hints.pop("return", None)
        hints.pop("self", None)
        
        kwargs = {}
        for param_name, param_type in hints.items():
            try:
                resolved = self.resolve(param_type)
                if asyncio.iscoroutine(resolved):
                    kwargs[param_name] = await resolved
                else:
                    kwargs[param_name] = resolved
            except RuntimeError:
                pass
        
        instance = cls(**kwargs)
        
        if hasattr(instance, "initialize") and callable(instance.initialize):
            result = instance.initialize()
            if asyncio.iscoroutine(result):
                await result
        
        return instance

    def _call_factory(self, factory: Callable) -> Any:
        hints = typing.get_type_hints(factory)
        hints.pop("return", None)
        
        kwargs = {}
        for param_name, param_type in hints.items():
            kwargs[param_name] = self.resolve(param_type)
        
        return factory(**kwargs)

    def _get_scoped_instance(self, service_type: Type, registration: ServiceRegistration) -> Any:
        scope_id = self._get_current_scope_id()
        
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        
        if service_type not in self._scoped_instances[scope_id]:
            self._scoped_instances[scope_id][service_type] = self._create_instance(registration)
        
        return self._scoped_instances[scope_id][service_type]

    def _get_current_scope_id(self) -> str:
        return "default"

    def _extract_dependencies(self, cls: Optional[Type], factory: Optional[Callable]) -> List[Type]:
        deps = []
        
        if factory:
            hints = typing.get_type_hints(factory)
            hints.pop("return", None)
            deps = list(hints.values())
        elif cls and hasattr(cls, "__init__"):
            hints = typing.get_type_hints(cls.__init__)
            hints.pop("return", None)
            hints.pop("self", None)
            deps = list(hints.values())
        
        return deps

    def get_registrations(self) -> Dict[Type, List[ServiceRegistration]]:
        return dict(self._registrations)

    def get_singleton_count(self) -> int:
        return len(self._singletons)

    def get_stats(self) -> Dict[str, Any]:
        total_registrations = sum(len(v) for v in self._registrations.values())
        by_lifecycle = defaultdict(int)
        
        for regs in self._registrations.values():
            for reg in regs:
                by_lifecycle[reg.lifecycle.value] += 1
        
        return {
            "total_registrations": total_registrations,
            "total_services": len(self._registrations),
            "singleton_instances": len(self._singletons),
            "scoped_instances": sum(len(v) for v in self._scoped_instances.values()),
            "by_lifecycle": dict(by_lifecycle),
        }

    def clear_scoped(self, scope_id: str = None) -> None:
        scope_id = scope_id or "default"
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]

    def unregister(self, service_type: Type) -> None:
        if service_type in self._registrations:
            del self._registrations[service_type]
        if service_type in self._singletons:
            del self._singletons[service_type]

    def integrate_singleton_manager(self) -> None:
        for service_type, instance in self._singletons.items():
            name = service_type.__name__
            SingletonManager.register(name, instance)

    def register_to_singleton_manager(self, name: str, service_type: Type) -> None:
        if service_type in self._singletons:
            SingletonManager.register(name, self._singletons[service_type])
        else:
            def factory():
                return self.resolve(service_type)

            SingletonManager.lazy_register(name, factory)


_global_container: Optional[DependencyInjectionContainer] = None


def get_container() -> DependencyInjectionContainer:
    global _global_container
    if _global_container is None:
        _global_container = DependencyInjectionContainer()
    return _global_container


def register(service_type: Type[T], **kwargs) -> Callable:
    def decorator(impl_type: Type) -> Type:
        container = get_container()
        container.register(service_type, impl_type, **kwargs)
        return impl_type
    return decorator


def inject(func: Callable) -> Callable:
    container = get_container()
    return container._inject_into_callable(func)