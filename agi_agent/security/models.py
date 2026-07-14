"""
security/models.py - 安全数据模型

用户、角色、权限、会话等安全相关数据模型。
基于 SQLite 实现，后续可迁移至 PostgreSQL。
"""
import os
import sqlite3
import time
import uuid
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class UserRole(Enum):
    """用户角色枚举"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """权限枚举"""
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"

    SOUL_READ = "soul:read"
    SOUL_WRITE = "soul:write"

    EVOLUTION_READ = "evolution:read"
    EVOLUTION_RUN = "evolution:run"

    AGENT_READ = "agent:read"
    AGENT_CONTROL = "agent:control"

    TASK_READ = "task:read"
    TASK_WRITE = "task:write"

    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"

    SECURITY_AUDIT = "security:audit"
    SECURITY_CONFIG = "security:config"

    ADMIN_USER_MANAGE = "admin:user:manage"
    ADMIN_SYSTEM_CONFIG = "admin:system:config"

    PLUGIN_MANAGE = "plugin:manage"
    SKILL_MANAGE = "skill:manage"

    FILE_READ = "file:read"
    FILE_UPLOAD = "file:upload"
    FILE_DELETE = "file:delete"


ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.SUPER_ADMIN: set(Permission),
    UserRole.ADMIN: {
        Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_DELETE,
        Permission.SOUL_READ, Permission.SOUL_WRITE,
        Permission.EVOLUTION_READ, Permission.EVOLUTION_RUN,
        Permission.AGENT_READ, Permission.AGENT_CONTROL,
        Permission.TASK_READ, Permission.TASK_WRITE,
        Permission.KNOWLEDGE_READ, Permission.KNOWLEDGE_WRITE,
        Permission.SECURITY_AUDIT,
        Permission.ADMIN_USER_MANAGE,
        Permission.PLUGIN_MANAGE, Permission.SKILL_MANAGE,
        Permission.FILE_READ, Permission.FILE_UPLOAD, Permission.FILE_DELETE,
    },
    UserRole.OPERATOR: {
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.SOUL_READ,
        Permission.EVOLUTION_READ,
        Permission.AGENT_READ, Permission.AGENT_CONTROL,
        Permission.TASK_READ, Permission.TASK_WRITE,
        Permission.KNOWLEDGE_READ, Permission.KNOWLEDGE_WRITE,
        Permission.FILE_READ, Permission.FILE_UPLOAD,
    },
    UserRole.VIEWER: {
        Permission.MEMORY_READ,
        Permission.SOUL_READ,
        Permission.EVOLUTION_READ,
        Permission.AGENT_READ,
        Permission.TASK_READ,
        Permission.KNOWLEDGE_READ,
        Permission.FILE_READ,
    },
    UserRole.GUEST: {
        Permission.AGENT_READ,
    },
}


@dataclass
class User:
    """用户模型"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    last_login_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "last_login_at": self.last_login_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_sensitive:
            data["password_hash"] = self.password_hash
            data["salt"] = self.salt
            data["metadata"] = self.metadata
        return data

    def has_permission(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class TokenInfo:
    """Token 信息"""
    jti: str
    user_id: str
    token_type: str  # "access" or "refresh"
    issued_at: float
    expires_at: float
    revoked: bool = False

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jti": self.jti,
            "user_id": self.user_id,
            "token_type": self.token_type,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
        }


class SecurityStore:
    """安全数据存储（SQLite 实现）"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "data", "security.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    is_verified INTEGER DEFAULT 0,
                    last_login_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS token_blacklist (
                    jti TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL NOT NULL,
                    reason TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_token_blacklist_user ON token_blacklist(user_id);
                CREATE INDEX IF NOT EXISTS idx_token_blacklist_expires ON token_blacklist(expires_at);
            """)
            conn.commit()
            self._ensure_default_admin(conn)
        finally:
            conn.close()

    def _ensure_default_admin(self, conn: sqlite3.Connection):
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM users")
        count = cursor.fetchone()["cnt"]
        if count == 0:
            default_admin = self._create_default_admin()
            conn.execute(
                """INSERT INTO users (user_id, username, email, password_hash, salt, role,
                   is_active, is_verified, last_login_at, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    default_admin.user_id,
                    default_admin.username,
                    default_admin.email,
                    default_admin.password_hash,
                    default_admin.salt,
                    default_admin.role.value,
                    1 if default_admin.is_active else 0,
                    1 if default_admin.is_verified else 0,
                    default_admin.last_login_at,
                    default_admin.created_at,
                    default_admin.updated_at,
                    "{}",
                ),
            )
            conn.commit()

    @staticmethod
    def _create_default_admin() -> User:
        salt = secrets.token_hex(16)
        password = "admin123"
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        ).hex()
        return User(
            user_id=str(uuid.uuid4()),
            username="admin",
            email="admin@localhost",
            password_hash=password_hash,
            salt=salt,
            role=UserRole.SUPER_ADMIN,
            is_active=True,
            is_verified=True,
        )

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[User]:
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def create_user(self, user: User) -> User:
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO users (user_id, username, email, password_hash, salt, role,
                   is_active, is_verified, last_login_at, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user.user_id,
                    user.username,
                    user.email,
                    user.password_hash,
                    user.salt,
                    user.role.value,
                    1 if user.is_active else 0,
                    1 if user.is_verified else 0,
                    user.last_login_at,
                    user.created_at,
                    user.updated_at,
                    "{}",
                ),
            )
            conn.commit()
            return user
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(f"User already exists: {e}") from e
        finally:
            conn.close()

    def update_user(self, user: User) -> User:
        user.updated_at = time.time()
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE users SET username=?, email=?, password_hash=?, salt=?, role=?,
                   is_active=?, is_verified=?, last_login_at=?, updated_at=?
                   WHERE user_id=?""",
                (
                    user.username,
                    user.email,
                    user.password_hash,
                    user.salt,
                    user.role.value,
                    1 if user.is_active else 0,
                    1 if user.is_verified else 0,
                    user.last_login_at,
                    user.updated_at,
                    user.user_id,
                ),
            )
            conn.commit()
            return user
        finally:
            conn.close()

    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [self._row_to_user(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_user(self, user_id: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def revoke_token(self, jti: str, user_id: str, token_type: str,
                     issued_at: float, expires_at: float, reason: str = "logout"):
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO token_blacklist
                   (jti, user_id, token_type, issued_at, expires_at, revoked_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (jti, user_id, token_type, issued_at, expires_at, time.time(), reason),
            )
            conn.commit()
        finally:
            conn.close()

    def is_token_revoked(self, jti: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT revoked_at FROM token_blacklist WHERE jti = ?",
                (jti,),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            return True
        finally:
            conn.close()

    def cleanup_expired_tokens(self) -> int:
        conn = self._get_conn()
        try:
            now = time.time()
            cursor = conn.execute(
                "DELETE FROM token_blacklist WHERE expires_at < ?",
                (now,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            salt=row["salt"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            is_verified=bool(row["is_verified"]),
            last_login_at=row["last_login_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata={},
        )


_global_security_store: Optional[SecurityStore] = None


def get_security_store() -> SecurityStore:
    global _global_security_store
    if _global_security_store is None:
        _global_security_store = SecurityStore()
    return _global_security_store
