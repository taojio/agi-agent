import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from enum import Enum


class MessageType(Enum):
    TEXT = "text"
    FILE = "file"
    SYSTEM = "system"
    TASK = "task"
    COMMAND = "command"
    STATUS = "status"


class ChatChannelType(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    GROUP = "group"
    TASK = "task"


class ChatMessage:
    def __init__(self, message_id: str, sender_id: str, channel_id: str,
                 message_type: MessageType, content: Any,
                 timestamp: float = None, metadata: Dict[str, Any] = None):
        self.message_id = message_id
        self.sender_id = sender_id
        self.channel_id = channel_id
        self.message_type = message_type
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.read_by: set = set()
        self.reactions: Dict[str, List[str]] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "channel_id": self.channel_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "read_count": len(self.read_by)
        }


class ChatChannel:
    def __init__(self, channel_id: str, name: str,
                 channel_type: ChatChannelType = ChatChannelType.GROUP,
                 created_by: str = "system"):
        self.channel_id = channel_id
        self.name = name
        self.channel_type = channel_type
        self.created_by = created_by
        self.created_at = time.time()
        self.members: Dict[str, Dict[str, Any]] = {}
        self.messages: deque = deque(maxlen=1000)
        self.pinned_messages: List[str] = []
        self.topic: str = ""

    def add_member(self, agent_id: str, role: str = "member") -> bool:
        if agent_id in self.members:
            return False
        self.members[agent_id] = {
            "joined_at": time.time(),
            "role": role,
            "last_read": time.time()
        }
        return True

    def remove_member(self, agent_id: str) -> bool:
        if agent_id not in self.members:
            return False
        del self.members[agent_id]
        return True

    def has_member(self, agent_id: str) -> bool:
        return agent_id in self.members

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "name": self.name,
            "channel_type": self.channel_type.value,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "member_count": len(self.members),
            "message_count": len(self.messages),
            "topic": self.topic
        }


class AgentChatServer:
    def __init__(self, server_id: str = "default"):
        self.server_id = server_id
        self.channels: Dict[str, ChatChannel] = {}
        self._lock = threading.RLock()

        self._message_callbacks: List[Callable] = []
        self._channel_callbacks: List[Callable] = []
        self._presence_callbacks: List[Callable] = []

        self.online_agents: Dict[str, float] = {}

        self._create_default_channels()

    def _create_default_channels(self):
        self.create_channel(
            channel_id="general",
            name="综合讨论",
            channel_type=ChatChannelType.PUBLIC,
            created_by="system"
        )
        self.create_channel(
            channel_id="tasks",
            name="任务协作",
            channel_type=ChatChannelType.TASK,
            created_by="system"
        )
        self.create_channel(
            channel_id="random",
            name="自由交流",
            channel_type=ChatChannelType.PUBLIC,
            created_by="system"
        )

    def create_channel(self, channel_id: str = None, name: str = "",
                       channel_type: ChatChannelType = ChatChannelType.GROUP,
                       created_by: str = "system") -> ChatChannel:
        with self._lock:
            cid = channel_id or str(uuid.uuid4())[:8]
            channel = ChatChannel(
                channel_id=cid,
                name=name or cid,
                channel_type=channel_type,
                created_by=created_by
            )
            self.channels[cid] = channel

            self._emit_channel_event("created", channel.to_dict())

            return channel

    def delete_channel(self, channel_id: str) -> bool:
        with self._lock:
            if channel_id not in self.channels:
                return False
            channel = self.channels.pop(channel_id)
            self._emit_channel_event("deleted", {"channel_id": channel_id, "name": channel.name})
            return True

    def join_channel(self, channel_id: str, agent_id: str, role: str = "member") -> bool:
        with self._lock:
            if channel_id not in self.channels:
                return False

            channel = self.channels[channel_id]
            result = channel.add_member(agent_id, role)

            if result:
                self._send_system_message(
                    channel_id,
                    f"智能体 {agent_id} 加入了频道"
                )
                self._emit_presence_event("join", {"channel_id": channel_id, "agent_id": agent_id})

            return result

    def leave_channel(self, channel_id: str, agent_id: str) -> bool:
        with self._lock:
            if channel_id not in self.channels:
                return False

            channel = self.channels[channel_id]
            result = channel.remove_member(agent_id)

            if result:
                self._send_system_message(
                    channel_id,
                    f"智能体 {agent_id} 离开了频道"
                )
                self._emit_presence_event("leave", {"channel_id": channel_id, "agent_id": agent_id})

            return result

    def send_message(self, channel_id: str, sender_id: str,
                     message_type: MessageType = MessageType.TEXT,
                     content: Any = None,
                     metadata: Dict[str, Any] = None) -> Optional[ChatMessage]:
        with self._lock:
            if channel_id not in self.channels:
                return None

            channel = self.channels[channel_id]

            if channel.channel_type != ChatChannelType.PUBLIC:
                if not channel.has_member(sender_id):
                    return None

            msg = ChatMessage(
                message_id=str(uuid.uuid4())[:8],
                sender_id=sender_id,
                channel_id=channel_id,
                message_type=message_type,
                content=content,
                metadata=metadata
            )

            channel.messages.append(msg)
            msg.read_by.add(sender_id)

            self._emit_message_event(msg)

            return msg

    def _send_system_message(self, channel_id: str, content: str):
        msg = ChatMessage(
            message_id=str(uuid.uuid4())[:8],
            sender_id="system",
            channel_id=channel_id,
            message_type=MessageType.SYSTEM,
            content=content
        )
        if channel_id in self.channels:
            self.channels[channel_id].messages.append(msg)

    def get_messages(self, channel_id: str, since: float = 0.0,
                     limit: int = 100, agent_id: str = None) -> List[Dict[str, Any]]:
        with self._lock:
            if channel_id not in self.channels:
                return []

            channel = self.channels[channel_id]

            if agent_id and channel.channel_type != ChatChannelType.PUBLIC:
                if not channel.has_member(agent_id):
                    return []

            msgs = [m for m in channel.messages if m.timestamp > since]
            msgs = msgs[-limit:]

            return [m.to_dict() for m in msgs]

    def mark_read(self, channel_id: str, agent_id: str, message_id: str = None) -> bool:
        with self._lock:
            if channel_id not in self.channels:
                return False

            channel = self.channels[channel_id]
            now = time.time()

            if message_id:
                for msg in channel.messages:
                    if msg.message_id == message_id:
                        msg.read_by.add(agent_id)
                        break
            else:
                for msg in channel.messages:
                    msg.read_by.add(agent_id)

            if agent_id in channel.members:
                channel.members[agent_id]["last_read"] = now

            return True

    def get_unread_count(self, channel_id: str, agent_id: str) -> int:
        with self._lock:
            if channel_id not in self.channels:
                return 0

            channel = self.channels[channel_id]
            if agent_id not in channel.members:
                return len(channel.messages)

            last_read = channel.members[agent_id]["last_read"]
            unread = [m for m in channel.messages if m.timestamp > last_read and m.sender_id != agent_id]

            return len(unread)

    def agent_online(self, agent_id: str):
        with self._lock:
            self.online_agents[agent_id] = time.time()

    def agent_offline(self, agent_id: str):
        with self._lock:
            self.online_agents.pop(agent_id, None)

    def get_online_agents(self) -> List[str]:
        with self._lock:
            now = time.time()
            return [aid for aid, t in self.online_agents.items() if now - t < 60]

    def list_channels(self, agent_id: str = None) -> List[Dict[str, Any]]:
        with self._lock:
            channels = []
            for ch in self.channels.values():
                if ch.channel_type == ChatChannelType.PUBLIC:
                    channels.append(ch.to_dict())
                elif agent_id and ch.has_member(agent_id):
                    channels.append(ch.to_dict())
            return channels

    def on_message(self, callback: Callable):
        self._message_callbacks.append(callback)

    def on_channel_event(self, callback: Callable):
        self._channel_callbacks.append(callback)

    def on_presence(self, callback: Callable):
        self._presence_callbacks.append(callback)

    def _emit_message_event(self, message: ChatMessage):
        for cb in self._message_callbacks:
            try:
                cb(message.to_dict())
            except Exception:
                pass

    def _emit_channel_event(self, event: str, data: Dict[str, Any]):
        for cb in self._channel_callbacks:
            try:
                cb({"event": event, **data})
            except Exception:
                pass

    def _emit_presence_event(self, event: str, data: Dict[str, Any]):
        for cb in self._presence_callbacks:
            try:
                cb({"event": event, **data})
            except Exception:
                pass

    def get_chat_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_msgs = sum(len(ch.messages) for ch in self.channels.values())
            return {
                "server_id": self.server_id,
                "total_channels": len(self.channels),
                "total_messages": total_msgs,
                "online_agents": len(self.get_online_agents()),
                "channel_list": [ch.to_dict() for ch in self.channels.values()]
            }
