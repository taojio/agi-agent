"""
ui/chat_handler.py - 聊天消息处理器

处理多轮对话、上下文管理和消息路由
"""
import time
import uuid
from typing import Dict, Any, Optional, List
from collections import deque

class ChatMessage:
    def __init__(self, message_id: str, sender: str, content: str, 
                 timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        self.message_id = message_id
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ChatSession:
    def __init__(self, session_id: str, max_history: int = 100):
        self.session_id = session_id
        self.messages: deque = deque(maxlen=max_history)
        self.created_at = time.time()
        self.last_active = time.time()
        self.context: Dict[str, Any] = {}

    def add_message(self, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        message = ChatMessage(message_id, sender, content, metadata=metadata)
        self.messages.append(message)
        self.last_active = time.time()
        return message

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        messages = list(self.messages)[-limit:]
        return [msg.to_dict() for msg in messages]

    def update_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self) -> Dict[str, Any]:
        return self.context.copy()


class ChatHandler:
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._max_history = 100
        self._agent_ref = None

    def set_agent_ref(self, agent_ref):
        self._agent_ref = agent_ref

    def create_session(self) -> str:
        session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self._sessions[session_id] = ChatSession(session_id, self._max_history)
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def send_message(self, session_id: str, sender: str, content: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if session_id not in self._sessions:
            session_id = self.create_session()

        session = self._sessions[session_id]
        message = session.add_message(sender, content, metadata)

        if sender == "user" and self._agent_ref:
            response = self._generate_agent_response(session, content)
            session.add_message("agent", response.get("content", ""), response.get("metadata"))
            return {
                "session_id": session_id,
                "user_message": message.to_dict(),
                "agent_response": response,
                "history": session.get_history(),
            }

        return {
            "session_id": session_id,
            "message": message.to_dict(),
            "history": session.get_history(),
        }

    def _generate_agent_response(self, session: ChatSession, content: str) -> Dict[str, Any]:
        try:
            if hasattr(self._agent_ref, 'process_message'):
                response = self._agent_ref.process_message(content)
                return {
                    "content": response,
                    "metadata": {"source": "agent.process_message"},
                }
            elif hasattr(self._agent_ref, 'think'):
                result = self._agent_ref.think(content)
                return {
                    "content": str(result),
                    "metadata": {"source": "agent.think"},
                }
            else:
                return {
                    "content": "Agent received your message.",
                    "metadata": {"source": "default"},
                }
        except Exception as e:
            return {
                "content": f"Error processing message: {str(e)}",
                "metadata": {"error": str(e)},
            }

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.get_history(limit)

    def update_session_context(self, session_id: str, context: Dict[str, Any]):
        session = self._sessions.get(session_id)
        if session:
            for key, value in context.items():
                session.update_context(key, value)

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if not session:
            return {}
        return session.get_context()

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [{
            "session_id": sid,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "last_active": session.last_active,
        } for sid, session in self._sessions.items()]

    def cleanup_inactive_sessions(self, timeout_hours: float = 24):
        cutoff = time.time() - timeout_hours * 3600
        inactive = [sid for sid, session in self._sessions.items() if session.last_active < cutoff]
        for sid in inactive:
            del self._sessions[sid]
        return len(inactive)

    def get_session_count(self) -> int:
        return len(self._sessions)