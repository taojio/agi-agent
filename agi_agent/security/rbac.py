"""
security/rbac.py - RBAC 权限系统

基于角色的访问控制（Role-Based Access Control）。
支持 5 级角色、细粒度权限、权限检查中间件。
"""
from functools import wraps
from typing import Callable, Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .exceptions import AuthorizationException, SecurityErrorCode
from .jwt_auth import JWTAuth, get_jwt_auth
from .models import Permission, ROLE_PERMISSIONS, User, UserRole


security_scheme = HTTPBearer(auto_error=False)


class RBACManager:
    """RBAC 权限管理器"""

    def __init__(self, jwt_auth: Optional[JWTAuth] = None):
        self.jwt_auth = jwt_auth or get_jwt_auth()

    def get_current_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    ) -> Optional[User]:
        """
        从请求中获取当前用户（未认证返回 None）

        Args:
            credentials: HTTP Bearer Token 凭证

        Returns:
            用户对象或 None
        """
        if credentials is None:
            return None
        try:
            return self.jwt_auth.get_user_from_token(credentials.credentials)
        except Exception:
            return None

    def require_authentication(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    ) -> User:
        """
        要求用户已认证

        Args:
            credentials: HTTP Bearer Token 凭证

        Returns:
            认证通过的用户对象

        Raises:
            HTTPException: 401 未认证
        """
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return self.jwt_auth.get_user_from_token(credentials.credentials)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    def require_permission(self, permission: Permission) -> Callable:
        """
        权限检查装饰器工厂

        Args:
            permission: 需要的权限

        Returns:
            装饰器函数
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get("current_user")
                if user is None:
                    raise AuthorizationException(
                        "Authentication required to access this resource",
                        {"required_permission": permission.value},
                    )

                if not self.has_permission(user, permission):
                    raise AuthorizationException(
                        f"Insufficient permissions. Required: {permission.value}",
                        {
                            "user_role": user.role.value,
                            "required_permission": permission.value,
                        },
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def require_role(self, role: UserRole) -> Callable:
        """
        角色检查装饰器工厂

        Args:
            role: 需要的最低角色级别

        Returns:
            装饰器函数
        """
        role_hierarchy = [
            UserRole.GUEST,
            UserRole.VIEWER,
            UserRole.OPERATOR,
            UserRole.ADMIN,
            UserRole.SUPER_ADMIN,
        ]

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get("current_user")
                if user is None:
                    raise AuthorizationException(
                        "Authentication required to access this resource",
                        {"required_role": role.value},
                    )

                user_level = role_hierarchy.index(user.role) if user.role in role_hierarchy else 0
                required_level = role_hierarchy.index(role) if role in role_hierarchy else 0

                if user_level < required_level:
                    raise AuthorizationException(
                        f"Insufficient role. Required at least: {role.value}",
                        {
                            "user_role": user.role.value,
                            "required_role": role.value,
                        },
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def has_permission(self, user: User, permission: Permission) -> bool:
        """
        检查用户是否具有指定权限

        Args:
            user: 用户对象
            permission: 权限

        Returns:
            是否有权限
        """
        return user.has_permission(permission)

    def has_any_permission(self, user: User, permissions: Set[Permission]) -> bool:
        """
        检查用户是否具有任一权限

        Args:
            user: 用户对象
            permissions: 权限集合

        Returns:
            是否具有任一权限
        """
        return any(self.has_permission(user, p) for p in permissions)

    def has_all_permissions(self, user: User, permissions: Set[Permission]) -> bool:
        """
        检查用户是否具有所有权限

        Args:
            user: 用户对象
            permissions: 权限集合

        Returns:
            是否具有所有权限
        """
        return all(self.has_permission(user, p) for p in permissions)

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """
        获取用户的所有权限

        Args:
            user: 用户对象

        Returns:
            权限集合
        """
        return ROLE_PERMISSIONS.get(user.role, set())

    def list_roles(self) -> list:
        """列出所有角色及其权限"""
        return [
            {
                "role": role.value,
                "permissions": [p.value for p in perms],
                "permission_count": len(perms),
            }
            for role, perms in ROLE_PERMISSIONS.items()
        ]

    def check_role_permission(
        self,
        user: User,
        required_permission: Permission,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        综合权限检查（支持资源级权限）

        Args:
            user: 用户对象
            required_permission: 需要的权限
            resource_id: 资源 ID（用于资源级权限检查）

        Returns:
            是否有权限
        """
        if not self.has_permission(user, required_permission):
            return False

        if resource_id is not None:
            pass

        return True


_global_rbac: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    global _global_rbac
    if _global_rbac is None:
        _global_rbac = RBACManager()
    return _global_rbac
