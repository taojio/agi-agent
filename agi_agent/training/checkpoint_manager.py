import os
import json
import time
import hashlib
import pickle
import shutil
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class CheckpointType(Enum):
    PERIODIC = "periodic"
    MILESTONE = "milestone"
    TRIGGERED = "triggered"
    MANUAL = "manual"
    PRE_MUTATION = "pre_mutation"
    SAFETY_BACKUP = "safety_backup"


@dataclass
class CheckpointInfo:
    checkpoint_id: str
    version: str
    type: CheckpointType
    step: int
    timestamp: float = field(default_factory=time.time)
    phase: str = ""
    performance_score: float = 0.0
    file_size_bytes: int = 0
    file_hash: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True


@dataclass
class CheckpointContent:
    models: Dict[str, Any] = field(default_factory=dict)
    configs: Dict[str, Any] = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, base_dir: str = "./agent_checkpoints",
                 max_checkpoints: int = 10,
                 keep_milestones: bool = True):
        self.base_dir = os.path.abspath(base_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_milestones = keep_milestones

        self.checkpoint_history: deque = deque(maxlen=max_checkpoints * 2)
        self.checkpoint_index: Dict[str, CheckpointInfo] = {}
        self.latest_checkpoint: Optional[str] = None

        self.save_callbacks: List = []
        self.load_callbacks: List = []

        self._ensure_directories()
        self._load_checkpoint_index()

    def _ensure_directories(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "state"), exist_ok=True)

    def _load_checkpoint_index(self):
        index_file = os.path.join(self.base_dir, "checkpoint_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.latest_checkpoint = data.get("latest")
                for item in data.get("checkpoints", []):
                    info = CheckpointInfo(
                        checkpoint_id=item["checkpoint_id"],
                        version=item["version"],
                        type=CheckpointType(item["type"]),
                        step=item["step"],
                        timestamp=item.get("timestamp", 0),
                        phase=item.get("phase", ""),
                        performance_score=item.get("performance_score", 0.0),
                        file_size_bytes=item.get("file_size_bytes", 0),
                        file_hash=item.get("file_hash", ""),
                        description=item.get("description", ""),
                        metadata=item.get("metadata", {}),
                        valid=item.get("valid", True)
                    )
                    self.checkpoint_index[info.checkpoint_id] = info
                    self.checkpoint_history.append(info)
            except Exception:
                pass

    def _save_checkpoint_index(self):
        index_file = os.path.join(self.base_dir, "checkpoint_index.json")
        data = {
            "latest": self.latest_checkpoint,
            "checkpoints": [
                {
                    "checkpoint_id": info.checkpoint_id,
                    "version": info.version,
                    "type": info.type.value,
                    "step": info.step,
                    "timestamp": info.timestamp,
                    "phase": info.phase,
                    "performance_score": info.performance_score,
                    "file_size_bytes": info.file_size_bytes,
                    "file_hash": info.file_hash,
                    "description": info.description,
                    "metadata": info.metadata,
                    "valid": info.valid
                }
                for info in list(self.checkpoint_history)
            ]
        }
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    def save_checkpoint(self,
                        agent_state: Dict[str, Any],
                        step: int,
                        checkpoint_type: CheckpointType = CheckpointType.PERIODIC,
                        description: str = "",
                        phase: str = "",
                        performance_score: float = 0.0) -> Optional[str]:
        version = self._generate_version(step, checkpoint_type)
        checkpoint_id = f"cp_{version}_{int(time.time())}"

        checkpoint_dir = os.path.join(self.base_dir, checkpoint_id)
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)

            content = self._extract_checkpoint_content(agent_state)

            models_dir = os.path.join(checkpoint_dir, "models")
            configs_dir = os.path.join(checkpoint_dir, "configs")
            logs_dir = os.path.join(checkpoint_dir, "logs")
            metadata_dir = os.path.join(checkpoint_dir, "metadata")
            state_dir = os.path.join(checkpoint_dir, "state")

            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(configs_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)
            os.makedirs(state_dir, exist_ok=True)

            self._save_models(content.models, models_dir)
            self._save_configs(content.configs, configs_dir)
            self._save_logs(content.logs, logs_dir)
            self._save_state(content.state, state_dir)
            self._save_metadata(content.metadata, metadata_dir, step, phase, performance_score)

            total_size = self._calculate_dir_size(checkpoint_dir)
            file_hash = self._calculate_checkpoint_hash(checkpoint_dir)

            info = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                version=version,
                type=checkpoint_type,
                step=step,
                phase=phase,
                performance_score=performance_score,
                file_size_bytes=total_size,
                file_hash=file_hash,
                description=description,
                metadata={"checkpoint_dir": checkpoint_dir}
            )

            self.checkpoint_index[checkpoint_id] = info
            self.checkpoint_history.append(info)
            self.latest_checkpoint = checkpoint_id

            self._cleanup_old_checkpoints()
            self._save_checkpoint_index()

            for callback in self.save_callbacks:
                try:
                    callback(info)
                except Exception:
                    pass

            return checkpoint_id

        except Exception as e:
            logginglog.error(f"[CheckpointManager] Save failed: {e}")
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            return None

    def _generate_version(self, step: int, checkpoint_type: CheckpointType) -> str:
        major = step // 100000
        minor = (step % 100000) // 1000
        patch = step % 1000
        return f"v{major}.{minor}.{patch}"

    def _extract_checkpoint_content(self, agent_state: Dict[str, Any]) -> CheckpointContent:
        content = CheckpointContent()

        if "models" in agent_state:
            content.models = agent_state["models"]
        if "configs" in agent_state:
            content.configs = agent_state["configs"]
        if "logs" in agent_state:
            content.logs = agent_state["logs"]
        if "metadata" in agent_state:
            content.metadata = agent_state["metadata"]
        if "state" in agent_state:
            content.state = agent_state["state"]

        return content

    def _save_models(self, models: Dict[str, Any], directory: str):
        try:
            import torch
            for name, model in models.items():
                if hasattr(model, 'state_dict'):
                    filepath = os.path.join(directory, f"{name}.pt")
                    torch.save(model.state_dict(), filepath)
                elif isinstance(model, dict):
                    filepath = os.path.join(directory, f"{name}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(model, f, default=str)
        except ImportError:
            for name, model in models.items():
                if isinstance(model, dict):
                    filepath = os.path.join(directory, f"{name}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(model, f, default=str)

    def _save_configs(self, configs: Dict[str, Any], directory: str):
        for name, config in configs.items():
            filepath = os.path.join(directory, f"{name}.json")
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, default=str)
            except Exception:
                pass

    def _save_logs(self, logs: Dict[str, Any], directory: str):
        for name, log_data in logs.items():
            if isinstance(log_data, str):
                filepath = os.path.join(directory, f"{name}.log")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(log_data)
            elif isinstance(log_data, (list, dict)):
                filepath = os.path.join(directory, f"{name}.json")
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, default=str)
                except Exception:
                    pass

    def _save_state(self, state: Dict[str, Any], directory: str):
        for name, state_data in state.items():
            filepath = os.path.join(directory, f"{name}.pkl")
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(state_data, f)
            except Exception:
                try:
                    filepath_json = os.path.join(directory, f"{name}.json")
                    with open(filepath_json, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, default=str)
                except Exception:
                    pass

    def _save_metadata(self, metadata: Dict[str, Any], directory: str,
                       step: int, phase: str, performance_score: float):
        metadata.update({
            "step": step,
            "phase": phase,
            "performance_score": performance_score,
            "save_time": time.time(),
            "save_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        })

        try:
            import sys
            import platform
            metadata["system_info"] = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor()
            }
        except Exception:
            pass

        filepath = os.path.join(directory, "checkpoint_metadata.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_checkpoint(self, checkpoint_id: str = None) -> Optional[Dict[str, Any]]:
        if checkpoint_id is None:
            checkpoint_id = self.latest_checkpoint

        if checkpoint_id is None or checkpoint_id not in self.checkpoint_index:
            return None

        info = self.checkpoint_index[checkpoint_id]
        if not info.valid:
            return None

        checkpoint_dir = os.path.join(self.base_dir, checkpoint_id)
        if not os.path.exists(checkpoint_dir):
            return None

        try:
            content = CheckpointContent()

            models_dir = os.path.join(checkpoint_dir, "models")
            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(models_dir, filename)
                    if filename.endswith('.pt'):
                        try:
                            import torch
                            content.models[name] = torch.load(filepath, map_location='cpu')
                        except Exception:
                            pass
                    elif filename.endswith('.json'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content.models[name] = json.load(f)

            configs_dir = os.path.join(checkpoint_dir, "configs")
            if os.path.exists(configs_dir):
                for filename in os.listdir(configs_dir):
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(configs_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content.configs[name] = json.load(f)

            state_dir = os.path.join(checkpoint_dir, "state")
            if os.path.exists(state_dir):
                for filename in os.listdir(state_dir):
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(state_dir, filename)
                    if filename.endswith('.pkl'):
                        with open(filepath, 'rb') as f:
                            content.state[name] = pickle.load(f)
                    elif filename.endswith('.json'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content.state[name] = json.load(f)

            metadata_file = os.path.join(checkpoint_dir, "metadata", "checkpoint_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    content.metadata = json.load(f)

            result = {
                "models": content.models,
                "configs": content.configs,
                "logs": content.logs,
                "metadata": content.metadata,
                "state": content.state,
                "checkpoint_id": checkpoint_id,
                "info": info
            }

            for callback in self.load_callbacks:
                try:
                    callback(info, result)
                except Exception:
                    pass

            return result

        except Exception as e:
            logging.getLogger("agi_agent").error(f"[CheckpointManager] Load failed: {e}")
            return None

    def verify_checkpoint(self, checkpoint_id: str) -> bool:
        if checkpoint_id not in self.checkpoint_index:
            return False

        info = self.checkpoint_index[checkpoint_id]
        checkpoint_dir = os.path.join(self.base_dir, checkpoint_id)

        if not os.path.exists(checkpoint_dir):
            info.valid = False
            return False

        current_hash = self._calculate_checkpoint_hash(checkpoint_dir)
        if info.file_hash and current_hash != info.file_hash:
            info.valid = False
            return False

        required_subdirs = ["models", "configs", "metadata", "state"]
        for subdir in required_subdirs:
            if not os.path.exists(os.path.join(checkpoint_dir, subdir)):
                info.valid = False
                return False

        info.valid = True
        return True

    def _cleanup_old_checkpoints(self):
        if len(self.checkpoint_index) <= self.max_checkpoints:
            return

        candidates = []
        for cp_id, info in self.checkpoint_index.items():
            if info.type == CheckpointType.MILESTONE and self.keep_milestones:
                continue
            candidates.append(info)

        candidates.sort(key=lambda x: x.timestamp)

        to_remove = len(candidates) - (self.max_checkpoints // 2)
        if to_remove <= 0:
            return

        for info in candidates[:to_remove]:
            checkpoint_dir = os.path.join(self.base_dir, info.checkpoint_id)
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            if info.checkpoint_id in self.checkpoint_index:
                del self.checkpoint_index[info.checkpoint_id]

    def _calculate_dir_size(self, directory: str) -> int:
        total = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
        return total

    def _calculate_checkpoint_hash(self, directory: str) -> str:
        hasher = hashlib.md5()
        try:
            metadata_file = os.path.join(directory, "metadata", "checkpoint_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    hasher.update(f.read())
        except Exception:
            pass
        return hasher.hexdigest()

    def get_best_checkpoint(self, metric: str = "performance_score") -> Optional[CheckpointInfo]:
        best = None
        best_score = -float('inf')

        for info in self.checkpoint_index.values():
            if not info.valid:
                continue
            score = getattr(info, metric, 0)
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best = info

        return best

    def list_checkpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        checkpoints = sorted(
            self.checkpoint_index.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "version": cp.version,
                "type": cp.type.value,
                "step": cp.step,
                "phase": cp.phase,
                "performance_score": cp.performance_score,
                "timestamp": cp.timestamp,
                "file_size_mb": cp.file_size_bytes / (1024 * 1024),
                "valid": cp.valid,
                "description": cp.description
            }
            for cp in checkpoints
        ]

    def register_save_callback(self, callback):
        self.save_callbacks.append(callback)

    def register_load_callback(self, callback):
        self.load_callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        valid_count = sum(1 for cp in self.checkpoint_index.values() if cp.valid)
        total_size = sum(cp.file_size_bytes for cp in self.checkpoint_index.values())

        type_counts = {}
        for cp in self.checkpoint_index.values():
            t = cp.type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_checkpoints": len(self.checkpoint_index),
            "valid_checkpoints": valid_count,
            "latest_checkpoint": self.latest_checkpoint,
            "total_size_mb": total_size / (1024 * 1024),
            "max_checkpoints": self.max_checkpoints,
            "type_distribution": type_counts,
            "base_directory": self.base_dir
        }
