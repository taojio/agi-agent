"""
ui/websocket_manager.py - WebSocket连接管理器

管理客户端连接、消息路由和实时通信
"""
import json
import logging
from typing import Dict, Any, Optional, Callable
from collections import defaultdict

logger = logging.getLogger("agi_agent.ui")


class WebSocketManager:
    def __init__(self):
        self._connections: Dict[str, Any] = {}
        self._connection_metadata: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._broadcast_channel_subscriptions: Dict[str, set] = defaultdict(set)

    def register_connection(self, connection_id: str, connection: Any, metadata: Optional[Dict[str, Any]] = None):
        self._connections[connection_id] = connection
        self._connection_metadata[connection_id] = metadata or {}
        logger.info(f"WebSocket connection registered: {connection_id}")

    def unregister_connection(self, connection_id: str):
        if connection_id in self._connections:
            del self._connections[connection_id]
            del self._connection_metadata[connection_id]
            for channel in self._broadcast_channel_subscriptions:
                self._broadcast_channel_subscriptions[channel].discard(connection_id)
            logger.info(f"WebSocket connection unregistered: {connection_id}")

    def send_message(self, connection_id: str, message: Dict[str, Any]):
        if connection_id in self._connections:
            try:
                connection = self._connections[connection_id]
                if hasattr(connection, 'send_text'):
                    connection.send_text(json.dumps(message))
                elif hasattr(connection, 'send'):
                    connection.send(json.dumps(message))
                logger.debug(f"Message sent to {connection_id}: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.unregister_connection(connection_id)

    def broadcast(self, message: Dict[str, Any], channel: Optional[str] = None):
        if channel:
            connection_ids = self._broadcast_channel_subscriptions.get(channel, set())
        else:
            connection_ids = set(self._connections.keys())

        for conn_id in connection_ids:
            self.send_message(conn_id, message)
        logger.debug(f"Broadcast message to {len(connection_ids)} connections: {message.get('type', 'unknown')}")

    def subscribe_to_channel(self, connection_id: str, channel: str):
        if connection_id in self._connections:
            self._broadcast_channel_subscriptions[channel].add(connection_id)
            logger.debug(f"Connection {connection_id} subscribed to channel: {channel}")

    def unsubscribe_from_channel(self, connection_id: str, channel: str):
        if connection_id in self._connections:
            self._broadcast_channel_subscriptions[channel].discard(connection_id)

    def register_handler(self, message_type: str, handler: Callable):
        self._handlers[message_type] = handler
        logger.debug(f"Handler registered for message type: {message_type}")

    def handle_message(self, connection_id: str, raw_message: str):
        try:
            message = json.loads(raw_message)
            message_type = message.get("type", "unknown")
            
            if message_type in self._handlers:
                result = self._handlers[message_type](connection_id, message)
                if result:
                    self.send_message(connection_id, result)
            else:
                logger.warning(f"No handler for message type: {message_type}")
                self.send_message(connection_id, {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                })
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message from {connection_id}")
            self.send_message(connection_id, {
                "type": "error",
                "error": "Invalid JSON format"
            })
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            self.send_message(connection_id, {
                "type": "error",
                "error": str(e)
            })

    def get_connection_count(self) -> int:
        return len(self._connections)

    def get_active_channels(self) -> Dict[str, int]:
        return {channel: len(connections) for channel, connections in self._broadcast_channel_subscriptions.items() if connections}

    def shutdown(self):
        for conn_id in list(self._connections.keys()):
            self.unregister_connection(conn_id)
        self._handlers.clear()
        logger.info("WebSocketManager shutdown completed")