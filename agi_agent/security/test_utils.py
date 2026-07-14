"""
security/test_utils.py - 安全模块测试工具

提供测试用的辅助函数，如重置全局单例等
"""
import os


def reset_security_singletons():
    """重置所有安全模块全局单例（测试用）

    用于集成测试中确保每个测试用例独立运行
    """
    from .models import _global_security_store
    from .jwt_auth import _global_jwt_auth, _global_jwt_config
    from .rbac import _global_rbac
    from .validation import _global_validator
    from .rate_limiter import _global_rate_limiter
    from .headers import _global_security_headers
    from .audit_logger import _global_audit_logger

    import sys
    this_module = sys.modules[__name__]

    for name, var in [
        ("_global_security_store", _global_security_store),
        ("_global_jwt_auth", _global_jwt_auth),
        ("_global_jwt_config", _global_jwt_config),
        ("_global_rbac", _global_rbac),
        ("_global_validator", _global_validator),
        ("_global_rate_limiter", _global_rate_limiter),
        ("_global_security_headers", _global_security_headers),
        ("_global_audit_logger", _global_audit_logger),
    ]:
        # 直接设置为 None
        pass

    # 使用更直接的方式：在各模块中重置
    import agi_agent.security.models as m
    m._global_security_store = None

    import agi_agent.security.jwt_auth as j
    j._global_jwt_auth = None
    j._global_jwt_config = None

    import agi_agent.security.rbac as r
    r._global_rbac = None

    import agi_agent.security.validation as v
    v._global_validator = None

    import agi_agent.security.rate_limiter as rl
    rl._global_rate_limiter = None

    import agi_agent.security.headers as h
    h._global_security_headers = None

    import agi_agent.security.audit_logger as al
    al._global_audit_logger = None
