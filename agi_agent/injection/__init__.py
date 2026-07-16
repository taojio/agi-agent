from .container import (
    DependencyInjectionContainer,
    Lifecycle,
    ServiceRegistration,
    get_container,
    register,
    inject,
)
from .singleton import (
    SingletonMeta,
    SingletonLazyMeta,
    singleton,
    SingletonManager,
    with_singleton,
)