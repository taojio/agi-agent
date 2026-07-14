import os
import json
import time
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StoredMessage:
    message_id: str
    sender_id: str
    channel_id: str
    message_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "channel_id": self.channel_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class MessageStore:
    def __init__(self, storage_dir: str = "./data/chat"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self._cache: Dict[str, deque] = {}
        self._max_cache_per_channel = 500

    def save_message(self, message_data: Dict[str, Any]) -> bool:
        try:
            channel_id = message_data.get("channel_id", "unknown")
            channel_dir = os.path.join(self.storage_dir, channel_id)
            os.makedirs(channel_dir, exist_ok=True)

            msg = StoredMessage(
                message_id=message_data.get("message_id", ""),
                sender_id=message_data.get("sender_id", ""),
                channel_id=channel_id,
                message_type=message_data.get("message_type", "text"),
                content=message_data.get("content", ""),
                timestamp=message_data.get("timestamp", time.time()),
                metadata=message_data.get("metadata", {})
            )

            date_str = time.strftime("%Y%m%d", time.localtime(msg.timestamp))
            filename = os.path.join(channel_dir, f"messages_{date_str}.jsonl")

            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

            if channel_id not in self._cache:
                self._cache[channel_id] = deque(maxlen=self._max_cache_per_channel)
            self._cache[channel_id].append(msg)

            return True
        except Exception:
            return False

    def load_messages(self, channel_id: str, since: float = 0.0,
                       limit: int = 100, before: float = None) -> List[Dict[str, Any]]:
        messages = []

        if channel_id in self._cache:
            messages = [
                m.to_dict() for m in self._cache[channel_id]
                if m.timestamp > since and (before is None or m.timestamp < before)
            ]
            if len(messages) >= limit:
                return messages[-limit:]

        channel_dir = os.path.join(self.storage_dir, channel_id)
        if not os.path.exists(channel_dir):
            return messages[-limit:]

        files = sorted(
            [f for f in os.listdir(channel_dir) if f.startswith("messages_") and f.endswith(".jsonl")],
            reverse=True
        )

        for filename in files:
            filepath = os.path.join(channel_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg_data = json.loads(line)
                            ts = msg_data.get("timestamp", 0)
                            if ts > since and (before is None or ts < before):
                                messages.append(msg_data)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

            if len(messages) >= limit * 2:
                break

        return sorted(messages, key=lambda m: m.get("timestamp", 0))[-limit:]

    def search_messages(self, query: str, channel_id: str = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()

        channels_to_search = [channel_id] if channel_id else [
            d for d in os.listdir(self.storage_dir)
            if os.path.isdir(os.path.join(self.storage_dir, d))
        ]

        for ch_id in channels_to_search:
            channel_dir = os.path.join(self.storage_dir, ch_id)
            if not os.path.exists(channel_dir):
                continue

            for filename in os.listdir(channel_dir):
                if not filename.endswith(".jsonl"):
                    continue
                filepath = os.path.join(channel_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg_data = json.loads(line)
                                content = str(msg_data.get("content", ""))
                                if query_lower in content.lower():
                                    results.append(msg_data)
                                    if len(results) >= limit:
                                        return results
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

        return results

    def delete_channel_history(self, channel_id: str) -> bool:
        try:
            channel_dir = os.path.join(self.storage_dir, channel_id)
            if os.path.exists(channel_dir):
                import shutil
                shutil.rmtree(channel_dir)
            self._cache.pop(channel_id, None)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        total_messages = 0
        total_channels = 0
        total_size = 0

        if os.path.exists(self.storage_dir):
            for channel_dir in os.listdir(self.storage_dir):
                ch_path = os.path.join(self.storage_dir, channel_dir)
                if os.path.isdir(ch_path):
                    total_channels += 1
                    for f in os.listdir(ch_path):
                        fp = os.path.join(ch_path, f)
                        if os.path.isfile(fp):
                            total_size += os.path.getsize(fp)
                            with open(fp, "r", encoding="utf-8") as fh:
                                total_messages += sum(1 for _ in fh if _.strip())

        return {
            "storage_dir": self.storage_dir,
            "total_channels": total_channels,
            "total_messages": total_messages,
            "total_size_bytes": total_size,
            "cached_channels": len(self._cache)
        }
