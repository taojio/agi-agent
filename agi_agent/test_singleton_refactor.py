import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from injection import (
    DependencyInjectionContainer,
    SingletonManager,
    SingletonMeta,
    singleton,
)


class ServiceA(metaclass=SingletonMeta):
    def __init__(self):
        self.value = "ServiceA initialized"


@singleton
class ServiceB:
    def __init__(self):
        self.value = "ServiceB initialized"


class ServiceC:
    def __init__(self):
        self.value = "ServiceC initialized"


def test_singleton_refactor():
    print("=" * 60)
    print("T006: 单例模式重构 - 功能测试")
    print("=" * 60)

    print("\n1. 测试元类单例模式")
    a1 = ServiceA()
    a2 = ServiceA()
    print(f"   ServiceA 实例相同: {a1 is a2}")
    assert a1 is a2

    print("\n2. 测试装饰器单例模式")
    b1 = ServiceB()
    b2 = ServiceB()
    print(f"   ServiceB 实例相同: {b1 is b2}")
    assert b1 is b2

    print("\n3. 测试单例管理器")
    SingletonManager.register("service_a", a1)
    SingletonManager.lazy_register("service_c", lambda: ServiceC())

    retrieved_a = SingletonManager.get("service_a")
    print(f"   注册单例获取: {retrieved_a.value}")
    assert retrieved_a is a1

    retrieved_c = SingletonManager.get("service_c")
    print(f"   懒加载单例获取: {retrieved_c.value}")
    assert retrieved_c is not None

    stats = SingletonManager.get_stats()
    print(f"   单例统计: {json.dumps(stats, indent=2)}")

    print("\n4. 测试DI容器与单例管理器集成")
    container = DependencyInjectionContainer()
    container.register(ServiceA, lifecycle="singleton")
    container.register(ServiceB, lifecycle="singleton")

    a3 = container.resolve(ServiceA)
    print(f"   DI解析ServiceA: {a3 is a1}")
    assert a3 is a1

    container.register_to_singleton_manager("ServiceA_Di", ServiceA)
    sm_a = SingletonManager.get("ServiceA_Di")
    print(f"   单例管理器获取DI单例: {sm_a is a1}")
    assert sm_a is a1

    print("\n5. 测试单例清理")
    cleanup_result = SingletonManager.cleanup("service_a")
    print(f"   清理单例: {cleanup_result}")
    assert cleanup_result

    stats_after = SingletonManager.get_stats()
    print(f"   清理后统计: {json.dumps(stats_after, indent=2)}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_singleton_refactor()