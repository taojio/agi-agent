import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from collections import deque


class ChatRole(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    GUEST = "guest"
    BANNED = "banned"


class ChatPermission(Enum):
    SEND_MESSAGE = "send_message"
    DELETE_MESSAGE = "delete_message"
    PIN_MESSAGE = "pin_message"
    MANAGE_MEMBERS = "manage_members"
    MANAGE_CHANNEL = "manage_channel"
    UPLOAD_FILE = "upload_file"
    MENTION_ALL = "mention_all"
    CHANGE_TOPIC = "change_topic"


ROLE_PERMISSIONS = {
    ChatRole.OWNER: {p for p in ChatPermission},
    ChatRole.ADMIN: {
        ChatPermission.SEND_MESSAGE,
        ChatPermission.DELETE_MESSAGE,
        ChatPermission.PIN_MESSAGE,
        ChatPermission.MANAGE_MEMBERS,
        ChatPermission.UPLOAD_FILE,
        ChatPermission.MENTION_ALL,
        ChatPermission.CHANGE_TOPIC,
    },
    ChatRole.MODERATOR: {
        ChatPermission.SEND_MESSAGE,
        ChatPermission.DELETE_MESSAGE,
        ChatPermission.PIN_MESSAGE,
        ChatPermission.MANAGE_MEMBERS,
        ChatPermission.UPLOAD_FILE,
        ChatPermission.CHANGE_TOPIC,
    },
    ChatRole.MEMBER: {
        ChatPermission.SEND_MESSAGE,
        ChatPermission.UPLOAD_FILE,
    },
    ChatRole.GUEST: {
        ChatPermission.SEND_MESSAGE,
    },
    ChatRole.BANNED: set(),
}


class ChatPermissionManager:
    def __init__(self):
        self._global_roles: Dict[str, ChatRole] = {}
        self._channel_roles: Dict[str, Dict[str, ChatRole]] = {}
        self._banned_users: Set[str] = set()
        self._muted_users: Dict[str, float] = {}
        self._permission_overrides: Dict[str, Dict[str, Set[ChatPermission]]] = {}

        self._action_log: deque = deque(maxlen=200)

    def set_global_role(self, agent_id: str, role: ChatRole):
        self._global_roles[agent_id] = role
        self._log_action("set_global_role", agent_id, {"role": role.value})

    def set_channel_role(self, channel_id: str, agent_id: str, role: ChatRole):
        if channel_id not in self._channel_roles:
            self._channel_roles[channel_id] = {}
        self._channel_roles[channel_id][agent_id] = role
        self._log_action("set_channel_role", agent_id, {"channel_id": channel_id, "role": role.value})

    def get_role(self, agent_id: str, channel_id: str = None) -> ChatRole:
        if channel_id and channel_id in self._channel_roles:
            if agent_id in self._channel_roles[channel_id]:
                return self._channel_roles[channel_id][agent_id]

        return self._global_roles.get(agent_id, ChatRole.MEMBER)

    def has_permission(self, agent_id: str, permission: ChatPermission,
                       channel_id: str = None) -> bool:
        if agent_id in self._banned_users:
            return False

        if agent_id in self._muted_users:
            if time.time() < self._muted_users[agent_id]:
                if permission in (ChatPermission.SEND_MESSAGE,):
                    return False

        role = self.get_role(agent_id, channel_id)
        role_perms = ROLE_PERMISSIONS.get(role, set())

        if channel_id and channel_id in self._permission_overrides:
            overrides = self._permission_overrides[channel_id].get(agent_id, set())
            role_perms = role_perms | overrides

        return permission in role_perms

    def add_permission_override(self, channel_id: str, agent_id: str,
                                 permission: ChatPermission):
        if channel_id not in self._permission_overrides:
            self._permission_overrides[channel_id] = {}
        if agent_id not in self._permission_overrides[channel_id]:
            self._permission_overrides[channel_id][agent_id] = set()
        self._permission_overrides[channel_id][agent_id].add(permission)

    def ban_user(self, agent_id: str, reason: str = "", banned_by: str = "system"):
        self._banned_users.add(agent_id)
        self._log_action("ban", agent_id, {"reason": reason, "banned_by": banned_by})

    def unban_user(self, agent_id: str):
        self._banned_users.discard(agent_id)
        self._log_action("unban", agent_id, {})

    def mute_user(self, agent_id: str, duration_seconds: float = 600,
                   reason: str = "", muted_by: str = "system"):
        self._muted_users[agent_id] = time.time() + duration_seconds
        self._log_action("mute", agent_id, {
            "duration": duration_seconds,
            "reason": reason,
            "muted_by": muted_by
        })

    def unmute_user(self, agent_id: str):
        self._muted_users.pop(agent_id, None)
        self._log_action("unmute", agent_id, {})

    def is_banned(self, agent_id: str) -> bool:
        return agent_id in self._banned_users

    def is_muted(self, agent_id: str) -> bool:
        if agent_id not in self._muted_users:
            return False
        return time.time() < self._muted_users[agent_id]

    def can_send_message(self, agent_id: str, channel_id: str = None) -> bool:
        return self.has_permission(agent_id, ChatPermission.SEND_MESSAGE, channel_id)

    def can_delete_message(self, agent_id: str, channel_id: str = None,
                            sender_id: str = None) -> bool:
        if sender_id == agent_id:
            return True
        return self.has_permission(agent_id, ChatPermission.DELETE_MESSAGE, channel_id)

    def can_manage_members(self, agent_id: str, channel_id: str = None) -> bool:
        return self.has_permission(agent_id, ChatPermission.MANAGE_MEMBERS, channel_id)

    def can_manage_channel(self, agent_id: str, channel_id: str = None) -> bool:
        return self.has_permission(agent_id, ChatPermission.MANAGE_CHANNEL, channel_id)

    def can_upload_file(self, agent_id: str, channel_id: str = None) -> bool:
        return self.has_permission(agent_id, ChatPermission.UPLOAD_FILE, channel_id)

    def _log_action(self, action: str, target_id: str, details: Dict[str, Any]):
        self._action_log.append({
            "action": action,
            "target_id": target_id,
            "details": details,
            "timestamp": time.time()
        })

    def get_permission_stats(self) -> Dict[str, Any]:
        return {
            "total_global_roles": len(self._global_roles),
            "total_channels_with_roles": len(self._channel_roles),
            "banned_count": len(self._banned_users),
            "muted_count": sum(1 for t in self._muted_users.values() if time.time() < t),
            "available_roles": [r.value for r in ChatRole],
            "available_permissions": [p.value for p in ChatPermission],
            "total_admin_actions": len(self._action_log)
        }
