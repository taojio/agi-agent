"""
workspace.py - 工作空间隔离模块

为每个 Agent 提供独立的文件目录、记忆库、配置参数与运行沙箱，
数据物理隔离，不同任务、不同智能体之间互不干扰。
"""
import os
import json
import time
import uuid
import shutil
import sqlite3
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class WorkspacePermission(Enum):
    """工作空间权限"""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ADMIN = "admin"


class WorkspaceType(Enum):
    """工作空间类型"""
    AGENT = "agent"
    TASK = "task"
    SHARED = "shared"
    SYSTEM = "system"


@dataclass
class WorkspaceAccess:
    """工作空间访问记录"""
    agent_id: str
    permission: WorkspacePermission
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_valid(self) -> bool:
        """检查访问权限是否有效"""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "permission": self.permission.value,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at
        }


class Workspace:
    """工作空间"""

    def __init__(self, workspace_id: str, workspace_type: WorkspaceType,
                 base_dir: str = None):
        """
        初始化工作空间

        Args:
            workspace_id: 工作空间ID
            workspace_type: 工作空间类型
            base_dir: 基础目录
        """
        self.workspace_id = workspace_id
        self.workspace_type = workspace_type

        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser("~"), ".agi_workspaces")
        self.base_dir = base_dir

        self._root_path = os.path.join(base_dir, workspace_type.value, workspace_id)
        self._files_dir = os.path.join(self._root_path, "files")
        self._memory_dir = os.path.join(self._root_path, "memory")
        self._config_dir = os.path.join(self._root_path, "config")
        self._sandbox_dir = os.path.join(self._root_path, "sandbox")

        self._access_list: Dict[str, WorkspaceAccess] = {}
        self._access_db_path = os.path.join(self._root_path, "access.db")

        self._init_structure()

    def _init_structure(self):
        """初始化工作空间目录结构"""
        os.makedirs(self._files_dir, exist_ok=True)
        os.makedirs(self._memory_dir, exist_ok=True)
        os.makedirs(self._config_dir, exist_ok=True)
        os.makedirs(self._sandbox_dir, exist_ok=True)

        conn = sqlite3.connect(self._access_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access (
                agent_id TEXT PRIMARY KEY,
                permission TEXT NOT NULL,
                granted_at REAL NOT NULL,
                expires_at REAL
            )
        ''')
        conn.commit()
        conn.close()

    def grant_access(self, agent_id: str, permission: WorkspacePermission,
                     duration_hours: float = None):
        """
        授予 Agent 访问权限

        Args:
            agent_id: Agent ID
            permission: 权限类型
            duration_hours: 有效期（小时），None 表示永久
        """
        expires_at = time.time() + duration_hours * 3600 if duration_hours else None

        access = WorkspaceAccess(
            agent_id=agent_id,
            permission=permission,
            expires_at=expires_at
        )
        self._access_list[agent_id] = access

        conn = sqlite3.connect(self._access_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO access
            (agent_id, permission, granted_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (agent_id, permission.value, access.granted_at, expires_at))
        conn.commit()
        conn.close()

    def revoke_access(self, agent_id: str):
        """撤销 Agent 访问权限"""
        self._access_list.pop(agent_id, None)

        conn = sqlite3.connect(self._access_db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM access WHERE agent_id = ?', (agent_id,))
        conn.commit()
        conn.close()

    def check_access(self, agent_id: str, required_permission: str = "read") -> bool:
        """
        检查 Agent 是否有指定权限

        Args:
            agent_id: Agent ID
            required_permission: 需要的权限 (read/write/execute/admin)

        Returns:
            是否有权限
        """
        access = self._access_list.get(agent_id)
        if access is None:
            conn = sqlite3.connect(self._access_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM access WHERE agent_id = ?', (agent_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                access = WorkspaceAccess(
                    agent_id=row[0],
                    permission=WorkspacePermission(row[1]),
                    granted_at=row[2],
                    expires_at=row[3]
                )
                self._access_list[agent_id] = access
            else:
                return False

        if not access.is_valid():
            self.revoke_access(agent_id)
            return False

        perm = access.permission
        if required_permission == "read" and perm in (WorkspacePermission.READ, WorkspacePermission.READ_WRITE, WorkspacePermission.ADMIN):
            return True
        if required_permission == "write" and perm in (WorkspacePermission.WRITE, WorkspacePermission.READ_WRITE, WorkspacePermission.ADMIN):
            return True
        if required_permission == "execute" and perm in (WorkspacePermission.READ_WRITE, WorkspacePermission.ADMIN):
            return True
        if required_permission == "admin" and perm == WorkspacePermission.ADMIN:
            return True
        return False

    def list_files(self, agent_id: str) -> List[str]:
        """列出工作空间文件"""
        if not self.check_access(agent_id, "read"):
            return []
        return [f for f in os.listdir(self._files_dir) if os.path.isfile(os.path.join(self._files_dir, f))]

    def read_file(self, agent_id: str, filename: str) -> Optional[str]:
        """读取文件内容"""
        if not self.check_access(agent_id, "read"):
            return None
        filepath = os.path.join(self._files_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def write_file(self, agent_id: str, filename: str, content: str) -> bool:
        """写入文件内容"""
        if not self.check_access(agent_id, "write"):
            return False
        filepath = os.path.join(self._files_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    def delete_file(self, agent_id: str, filename: str) -> bool:
        """删除文件"""
        if not self.check_access(agent_id, "write"):
            return False
        filepath = os.path.join(self._files_dir, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            return True
        return False

    def get_memory_path(self, agent_id: str) -> Optional[str]:
        """获取记忆库路径"""
        if not self.check_access(agent_id, "read"):
            return None
        return self._memory_dir

    def get_sandbox_path(self, agent_id: str) -> Optional[str]:
        """获取沙箱路径"""
        if not self.check_access(agent_id, "execute"):
            return None
        return self._sandbox_dir

    def get_config(self, agent_id: str, key: str = None) -> Dict[str, Any]:
        """获取配置"""
        if not self.check_access(agent_id, "read"):
            return {}
        config_path = os.path.join(self._config_dir, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if key:
                    return {key: config.get(key)}
                return config
        return {}

    def set_config(self, agent_id: str, key: str, value: Any) -> bool:
        """设置配置"""
        if not self.check_access(agent_id, "write"):
            return False
        config_path = os.path.join(self._config_dir, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
        config[key] = value
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True

    def get_access_list(self) -> List[Dict[str, Any]]:
        """获取所有访问记录"""
        result = []
        for agent_id, access in self._access_list.items():
            if access.is_valid():
                result.append(access.to_dict())
        return result

    def get_status(self) -> Dict[str, Any]:
        """获取工作空间状态"""
        file_count = len([f for f in os.listdir(self._files_dir) if os.path.isfile(os.path.join(self._files_dir, f))])
        return {
            "workspace_id": self.workspace_id,
            "type": self.workspace_type.value,
            "root_path": self._root_path,
            "file_count": file_count,
            "access_count": len([a for a in self._access_list.values() if a.is_valid()])
        }


class WorkspaceManager:
    """工作空间管理器"""

    def __init__(self, base_dir: str = None):
        """
        初始化工作空间管理器

        Args:
            base_dir: 基础目录
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser("~"), ".agi_workspaces")
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self._workspaces: Dict[str, Workspace] = {}
        self._shared_workspace: Optional[Workspace] = None

    def create_workspace(self, workspace_id: str = None,
                         workspace_type: WorkspaceType = WorkspaceType.AGENT) -> Workspace:
        """
        创建工作空间

        Args:
            workspace_id: 工作空间ID，默认自动生成
            workspace_type: 工作空间类型

        Returns:
            工作空间实例
        """
        workspace_id = workspace_id or str(uuid.uuid4())[:8]

        if workspace_id in self._workspaces:
            return self._workspaces[workspace_id]

        workspace = Workspace(workspace_id, workspace_type, self.base_dir)
        self._workspaces[workspace_id] = workspace
        return workspace

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """获取工作空间"""
        return self._workspaces.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> bool:
        """删除工作空间"""
        workspace = self._workspaces.pop(workspace_id, None)
        if workspace:
            shutil.rmtree(workspace._root_path, ignore_errors=True)
            return True
        return False

    def get_shared_workspace(self) -> Workspace:
        """获取共享工作空间"""
        if self._shared_workspace is None:
            self._shared_workspace = self.create_workspace("shared", WorkspaceType.SHARED)
        return self._shared_workspace

    def delegate_permission(self, from_agent_id: str, to_agent_id: str,
                            workspace_id: str, permission: WorkspacePermission,
                            duration_hours: float = 24):
        """
        权限委托：主 Agent 向子 Agent 临时授予指定工作空间的访问权限

        Args:
            from_agent_id: 授权者（主 Agent）
            to_agent_id: 被授权者（子 Agent）
            workspace_id: 工作空间ID
            permission: 权限类型
            duration_hours: 有效期（小时）

        Returns:
            是否成功
        """
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            return False

        if not workspace.check_access(from_agent_id, "admin"):
            return False

        workspace.grant_access(to_agent_id, permission, duration_hours)
        return True

    def revoke_delegated_permission(self, agent_id: str, workspace_id: str):
        """撤销委托的权限"""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            workspace.revoke_access(agent_id)

    def get_all_workspaces(self) -> List[Dict[str, Any]]:
        """获取所有工作空间列表"""
        result = []
        for workspace in self._workspaces.values():
            result.append(workspace.get_status())
        return result

    def cleanup_expired_access(self):
        """清理过期的访问权限"""
        for workspace in self._workspaces.values():
            expired = [aid for aid, access in workspace._access_list.items() if not access.is_valid()]
            for aid in expired:
                workspace.revoke_access(aid)
