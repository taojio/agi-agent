"""
security/jwt_auth.py - JWT 认证系统

基于 JWT 的用户认证，支持 access token / refresh token 双令牌机制。
"""
import os
import time
import uuid
import hashlib
import secrets
from typing import Any, Dict, Optional, Tuple

import jwt as pyjwt

from .exceptions import (
    AuthenticationException,
    SecurityErrorCode,
)
from .models import (
    User,
    UserRole,
    SecurityStore,
    get_security_store,
)


class JWTConfig:
    """JWT 配置"""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expiry: int = 900,
        refresh_token_expiry: int = 604800,
        issuer: str = "agi-agent",
    ):
        if secret_key is None:
            secret_key = os.environ.get("JWT_SECRET_KEY")
            if not secret_key:
                secret_key = self._load_or_generate_key()
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        self.issuer = issuer

    @staticmethod
    def _load_or_generate_key() -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_file = os.path.join(base_dir, "data", ".jwt_secret")
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                return f.read().strip()
        key = secrets.token_hex(64)
        try:
            with open(key_file, "w") as f:
                f.write(key)
            os.chmod(key_file, 0o600)
        except (OSError, PermissionError):
            pass
        return key


_global_jwt_config: Optional[JWTConfig] = None


def get_jwt_config() -> JWTConfig:
    global _global_jwt_config
    if _global_jwt_config is None:
        _global_jwt_config = JWTConfig()
    return _global_jwt_config


class JWTAuth:
    """JWT 认证管理器"""

    def __init__(
        self,
        config: Optional[JWTConfig] = None,
        store: Optional[SecurityStore] = None,
    ):
        self.config = config or get_jwt_config()
        self.store = store or get_security_store()

    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        使用 PBKDF2-HMAC-SHA256 哈希密码

        Args:
            password: 明文密码
            salt: 可选盐值，不传则自动生成

        Returns:
            (password_hash, salt) 元组
        """
        if salt is None:
            salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        ).hex()
        return password_hash, salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        验证密码

        Args:
            password: 明文密码
            password_hash: 存储的密码哈希
            salt: 盐值

        Returns:
            是否匹配
        """
        computed_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, password_hash)

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
    ) -> User:
        """
        注册新用户

        Args:
            username: 用户名
            email: 邮箱
            password: 明文密码
            role: 默认角色

        Returns:
            创建的用户对象

        Raises:
            ValueError: 用户已存在
        """
        if self.store.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")
        if self.store.get_user_by_email(email):
            raise ValueError(f"Email '{email}' already exists")

        password_hash, salt = self.hash_password(password)
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
            is_active=True,
            is_verified=False,
        )
        return self.store.create_user(user)

    def authenticate(self, username: str, password: str) -> Tuple[User, Dict[str, str]]:
        """
        用户认证，返回用户对象和 Token 对

        Args:
            username: 用户名或邮箱
            password: 明文密码

        Returns:
            (user, tokens) 元组，tokens 包含 access_token 和 refresh_token

        Raises:
            AuthenticationException: 认证失败
        """
        user = self.store.get_user_by_username(username)
        if user is None:
            user = self.store.get_user_by_email(username)

        if user is None:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_CREDENTIALS,
                "Invalid username or password",
            )

        if not user.is_active:
            raise AuthenticationException(
                SecurityErrorCode.ACCOUNT_DISABLED,
                "Account is disabled",
            )

        if not self.verify_password(password, user.password_hash, user.salt):
            raise AuthenticationException(
                SecurityErrorCode.INVALID_CREDENTIALS,
                "Invalid username or password",
            )

        user.last_login_at = time.time()
        self.store.update_user(user)

        tokens = self.generate_token_pair(user)
        return user, tokens

    def generate_token_pair(self, user: User) -> Dict[str, str]:
        """
        生成 access token 和 refresh token 对

        Args:
            user: 用户对象

        Returns:
            包含 access_token 和 refresh_token 的字典
        """
        now = time.time()
        access_jti = str(uuid.uuid4())
        refresh_jti = str(uuid.uuid4())

        access_payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "type": "access",
            "iss": self.config.issuer,
            "iat": int(now),
            "exp": int(now + self.config.access_token_expiry),
            "jti": access_jti,
        }

        refresh_payload = {
            "sub": user.user_id,
            "type": "refresh",
            "iss": self.config.issuer,
            "iat": int(now),
            "exp": int(now + self.config.refresh_token_expiry),
            "jti": refresh_jti,
        }

        access_token = pyjwt.encode(access_payload, self.config.secret_key, algorithm=self.config.algorithm)
        refresh_token = pyjwt.encode(refresh_payload, self.config.secret_key, algorithm=self.config.algorithm)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.access_token_expiry,
            "refresh_expires_in": self.config.refresh_token_expiry,
        }

    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        验证 JWT Token

        Args:
            token: JWT Token 字符串
            token_type: 期望的 token 类型 ("access" 或 "refresh")

        Returns:
            解码后的 payload

        Raises:
            AuthenticationException: Token 无效或过期
        """
        try:
            payload = pyjwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
            )
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationException(
                SecurityErrorCode.TOKEN_EXPIRED,
                "Token has expired",
            )
        except pyjwt.InvalidTokenError as e:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_TOKEN,
                f"Invalid token: {str(e)}",
            )

        if payload.get("type") != token_type:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_TOKEN,
                f"Expected {token_type} token, got {payload.get('type')}",
            )

        jti = payload.get("jti")
        if jti and self.store.is_token_revoked(jti):
            raise AuthenticationException(
                SecurityErrorCode.TOKEN_REVOKED,
                "Token has been revoked",
            )

        return payload

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        使用 refresh token 刷新 access token

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的 token 对

        Raises:
            AuthenticationException: Refresh token 无效
        """
        payload = self.verify_token(refresh_token, token_type="refresh")
        user_id = payload.get("sub")

        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_TOKEN,
                "User not found",
            )

        if not user.is_active:
            raise AuthenticationException(
                SecurityErrorCode.ACCOUNT_DISABLED,
                "Account is disabled",
            )

        return self.generate_token_pair(user)

    def logout(self, token: str, token_type: str = "access"):
        """
        登出，将 Token 加入黑名单

        Args:
            token: JWT Token
            token_type: Token 类型
        """
        try:
            payload = pyjwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            user_id = payload.get("sub")
            issued_at = payload.get("iat", 0)
            expires_at = payload.get("exp", 0)

            if jti:
                self.store.revoke_token(
                    jti=jti,
                    user_id=user_id,
                    token_type=token_type,
                    issued_at=issued_at,
                    expires_at=expires_at,
                    reason="logout",
                )
        except pyjwt.InvalidTokenError:
            pass

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """
        修改用户密码

        Args:
            user_id: 用户 ID
            old_password: 旧密码
            new_password: 新密码

        Returns:
            是否成功

        Raises:
            AuthenticationException: 旧密码错误
        """
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_TOKEN,
                "User not found",
            )

        if not self.verify_password(old_password, user.password_hash, user.salt):
            raise AuthenticationException(
                SecurityErrorCode.INVALID_CREDENTIALS,
                "Old password is incorrect",
            )

        new_hash, new_salt = self.hash_password(new_password)
        user.password_hash = new_hash
        user.salt = new_salt
        self.store.update_user(user)

        return True

    def get_user_from_token(self, token: str) -> User:
        """
        从 access token 获取用户对象

        Args:
            token: access token

        Returns:
            用户对象

        Raises:
            AuthenticationException: Token 无效或用户不存在
        """
        payload = self.verify_token(token, token_type="access")
        user_id = payload.get("sub")

        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationException(
                SecurityErrorCode.INVALID_TOKEN,
                "User not found",
            )

        if not user.is_active:
            raise AuthenticationException(
                SecurityErrorCode.ACCOUNT_DISABLED,
                "Account is disabled",
            )

        return user


_global_jwt_auth: Optional[JWTAuth] = None


def get_jwt_auth() -> JWTAuth:
    global _global_jwt_auth
    if _global_jwt_auth is None:
        _global_jwt_auth = JWTAuth()
    return _global_jwt_auth
