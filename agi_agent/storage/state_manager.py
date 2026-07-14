import os
import json
import torch
import pickle
import zlib
import hashlib
import shutil
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from ..config.settings import STORAGE_DIR, MODEL_DIR

logger = logging.getLogger("agi_agent.storage")


@dataclass
class SaveVersion:
    version_id: str
    timestamp: float
    train_step: int
    free_energy: float
    confidence: float
    novelty: float
    performance_score: float
    file_path: str
    file_size: int
    checksum: str
    status: str = "active"


@dataclass
class SaveConfig:
    save_interval: int = 300
    save_path: str = STORAGE_DIR
    model_path: str = MODEL_DIR
    max_versions: int = 10
    compression_level: int = 6
    retry_count: int = 3
    retry_delay: float = 1.0
    enable_auto_save: bool = True
    format: str = "json"


class AgentStateManager:
    def __init__(self, config: Optional[SaveConfig] = None):
        self.config = config or SaveConfig()
        self.versions: List[SaveVersion] = []
        self.version_index: Dict[str, SaveVersion] = {}
        self._last_save_time = 0
        self._save_lock = asyncio.Lock()
        self._init_storage()

    def _init_storage(self):
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_path, "versions"), exist_ok=True)
        self._load_version_index()

    def _load_version_index(self):
        index_path = os.path.join(self.config.save_path, "version_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    for v_data in index_data.get("versions", []):
                        version = SaveVersion(**v_data)
                        self.versions.append(version)
                        self.version_index[version.version_id] = version
                self.versions.sort(key=lambda x: x.timestamp, reverse=True)
            except Exception as e:
                logger.error(f"Failed to load version index: {e}", exc_info=True)

    def _save_version_index(self):
        index_path = os.path.join(self.config.save_path, "version_index.json")
        index_data = {"versions": [asdict(v) for v in self.versions]}
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

    def _generate_version_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"

    def _calculate_checksum(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self.config.compression_level)

    def _decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _create_backup(self, filepath: str):
        backup_path = f"{filepath}.backup"
        if os.path.exists(filepath):
            shutil.copy2(filepath, backup_path)

    def _verify_file(self, filepath: str, expected_checksum: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            actual_checksum = self._calculate_checksum(data)
            return actual_checksum == expected_checksum
        except Exception:
            return False

    def _collect_agent_state(self, agent) -> Dict[str, Any]:
        state = {
            "metadata": {
                "version": "1.0",
                "timestamp": time.time(),
                "train_step": int(agent.train_step) if hasattr(agent, 'train_step') else 0,
                "last_fe": float(agent.last_fe) if hasattr(agent, 'last_fe') else 1.0,
            },
            "metrics": {
                "metrics_history": agent.metrics_history[-1000:] if hasattr(agent, 'metrics_history') and agent.metrics_history else [],
                "execution_history": agent.execution_history[-100:] if hasattr(agent, 'execution_history') and agent.execution_history else [],
                "long_term_performance": agent.long_term_performance[-100:] if hasattr(agent, 'long_term_performance') and agent.long_term_performance else [],
            },
            "model_states": {},
            "config": {
                "input_dim": int(agent.input_dim) if hasattr(agent, 'input_dim') else 16,
                "autonomous_mode": bool(agent.autonomous_mode) if hasattr(agent, 'autonomous_mode') else True,
                "running": bool(agent.running) if hasattr(agent, 'running') else True,
            }
        }

        model_attributes = [
            'perception', 'cognitive', 'dual_cognition', 'snn_enhancer',
            'causal_reasoner', 'meta_learn', 'meta_cog', 'homeostasis',
            'evolve_engine', 'execution', 'self_model', 'decision_engine',
            'action_planner', 'execution_monitor', 'reflex_controller',
            'thinking_orchestrator', 'meta_cognitive_orchestrator', 'action_orchestrator'
        ]

        for attr_name in model_attributes:
            attr = getattr(agent, attr_name, None)
            if attr is not None and isinstance(attr, torch.nn.Module):
                try:
                    state_dict = attr.state_dict()
                    serializable_dict = {}
                    for k, v in state_dict.items():
                        try:
                            serializable_dict[k] = v.detach().cpu().numpy().tolist()
                        except Exception:
                            serializable_dict[k] = str(v)
                    state["model_states"][attr_name] = {
                        "state_dict": serializable_dict
                    }
                except Exception as e:
                    logger.warning(f"State operation failed: {e}", exc_info=True)

        if hasattr(agent, 'knowledge_graph') and agent.knowledge_graph is not None:
            try:
                kg_state = agent.knowledge_graph.get_state() if hasattr(agent.knowledge_graph, 'get_state') else {}
                state["knowledge_graph"] = kg_state
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

        if hasattr(agent, 'memory_store') and agent.memory_store is not None:
            try:
                mem_state = agent.memory_store.get_state() if hasattr(agent.memory_store, 'get_state') else {}
                state["memory_store"] = mem_state
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

        if hasattr(agent, 'reflex_controller') and agent.reflex_controller is not None:
            try:
                rc_state = agent.reflex_controller.get_state() if hasattr(agent.reflex_controller, 'get_state') else {}
                state["reflex_controller"] = rc_state
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

        return state

    def _restore_agent_state(self, agent, state: Dict[str, Any]):
        if "config" in state:
            config = state["config"]
            agent.input_dim = config.get("input_dim", agent.input_dim)
            agent.autonomous_mode = config.get("autonomous_mode", True)
            agent.running = config.get("running", True)

        if "metadata" in state:
            metadata = state["metadata"]
            agent.train_step = metadata.get("train_step", agent.train_step)
            agent.last_fe = metadata.get("last_fe", agent.last_fe)

        if "metrics" in state:
            metrics = state["metrics"]
            if hasattr(agent, 'metrics_history'):
                agent.metrics_history = metrics.get("metrics_history", [])
            if hasattr(agent, 'execution_history'):
                agent.execution_history = metrics.get("execution_history", [])
            if hasattr(agent, 'long_term_performance'):
                agent.long_term_performance = metrics.get("long_term_performance", [])

        if "model_states" in state:
            model_states = state["model_states"]
            for attr_name, model_data in model_states.items():
                attr = getattr(agent, attr_name, None)
                if attr is not None and isinstance(attr, torch.nn.Module) and "state_dict" in model_data:
                    try:
                        state_dict = {k: torch.tensor(v) for k, v in model_data["state_dict"].items()}
                        attr.load_state_dict(state_dict)
                    except Exception as e:
                        logger.warning(f"State operation failed: {e}", exc_info=True)

        if "knowledge_graph" in state and hasattr(agent, 'knowledge_graph'):
            try:
                kg_state = state["knowledge_graph"]
                if hasattr(agent.knowledge_graph, 'set_state'):
                    agent.knowledge_graph.set_state(kg_state)
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

        if "memory_store" in state and hasattr(agent, 'memory_store'):
            try:
                mem_state = state["memory_store"]
                if hasattr(agent.memory_store, 'set_state'):
                    agent.memory_store.set_state(mem_state)
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

        if "reflex_controller" in state and hasattr(agent, 'reflex_controller'):
            try:
                rc_state = state["reflex_controller"]
                if hasattr(agent.reflex_controller, 'set_state'):
                    agent.reflex_controller.set_state(rc_state)
            except Exception as e:
                logger.warning(f"State operation failed: {e}", exc_info=True)

    async def save_agent(self, agent, force: bool = False) -> Optional[SaveVersion]:
        current_time = time.time()
        
        if not force and self.config.enable_auto_save:
            if current_time - self._last_save_time < self.config.save_interval:
                return None

        async with self._save_lock:
            for attempt in range(self.config.retry_count):
                try:
                    version_id = self._generate_version_id()
                    timestamp = current_time
                    
                    agent_state = self._collect_agent_state(agent)
                    
                    if self.config.format == "pkl":
                        try:
                            raw_data = pickle.dumps(agent_state)
                        except Exception:
                            raw_data = json.dumps(agent_state, default=str).encode('utf-8')
                            file_ext = ".json"
                        else:
                            file_ext = ".pkl"
                    else:
                        try:
                            raw_data = json.dumps(agent_state, default=str).encode('utf-8')
                        except Exception:
                            return None
                        file_ext = ".json"
                    
                    compressed_data = self._compress(raw_data)
                    checksum = self._calculate_checksum(compressed_data)
                    
                    file_path = os.path.join(self.config.save_path, "versions", f"{version_id}{file_ext}.gz")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, 'wb') as f:
                        f.write(compressed_data)
                    
                    file_size = os.path.getsize(file_path)
                    
                    metrics = agent.metrics_history[-1] if hasattr(agent, 'metrics_history') and agent.metrics_history else {}
                    
                    version = SaveVersion(
                        version_id=version_id,
                        timestamp=timestamp,
                        train_step=int(agent.train_step) if hasattr(agent, 'train_step') else 0,
                        free_energy=float(metrics.get('free_energy', agent.last_fe)) if hasattr(agent, 'last_fe') else 1.0,
                        confidence=float(metrics.get('confidence', 0.0)),
                        novelty=float(metrics.get('novelty', 0.0)),
                        performance_score=float(metrics.get('performance_score', 0.5)),
                        file_path=file_path,
                        file_size=file_size,
                        checksum=checksum,
                        status="active"
                    )
                    
                    self.versions.insert(0, version)
                    self.version_index[version_id] = version
                    
                    while len(self.versions) > self.config.max_versions:
                        old_version = self.versions.pop()
                        try:
                            if os.path.exists(old_version.file_path):
                                os.remove(old_version.file_path)
                            del self.version_index[old_version.version_id]
                        except Exception as e:
                            logger.warning(f"State operation failed: {e}", exc_info=True)
                    
                    self._save_version_index()
                    self._last_save_time = current_time
                    
                    return version
                    
                except Exception as e:
                    if attempt < self.config.retry_count - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        return None

    async def load_agent(self, agent, version_id: Optional[str] = None) -> bool:
        if version_id is None:
            if not self.versions:
                return False
            version = self.versions[0]
        elif version_id in self.version_index:
            version = self.version_index[version_id]
        else:
            return False

        if not self._verify_file(version.file_path, version.checksum):
            backup_path = f"{version.file_path}.backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, version.file_path)
                if not self._verify_file(version.file_path, version.checksum):
                    return False
            else:
                return False

        try:
            with open(version.file_path, 'rb') as f:
                compressed_data = f.read()
            
            raw_data = self._decompress(compressed_data)
            
            if version.file_path.endswith('.pkl.gz'):
                agent_state = pickle.loads(raw_data)
            else:
                agent_state = json.loads(raw_data)
            
            self._restore_agent_state(agent, agent_state)
            
            version.status = "loaded"
            self._save_version_index()
            
            return True
            
        except Exception:
            return False

    def get_version_history(self) -> List[Dict[str, Any]]:
        return [asdict(v) for v in self.versions]

    def get_best_version(self) -> Optional[SaveVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda x: x.performance_score)

    def get_version_by_step(self, step: int) -> Optional[SaveVersion]:
        for version in self.versions:
            if version.train_step >= step:
                return version
        return None

    def delete_version(self, version_id: str) -> bool:
        if version_id not in self.version_index:
            return False
        
        version = self.version_index[version_id]
        
        try:
            if os.path.exists(version.file_path):
                os.remove(version.file_path)
            backup_path = f"{version.file_path}.backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            self.versions = [v for v in self.versions if v.version_id != version_id]
            del self.version_index[version_id]
            self._save_version_index()
            
            return True
        except Exception:
            return False

    async def verify_agent(self, agent) -> Dict[str, Any]:
        try:
            initial_fe = agent.last_fe if hasattr(agent, 'last_fe') else 0.0
            initial_step = agent.train_step if hasattr(agent, 'train_step') else 0
            
            test_input = torch.randn(1, agent.input_dim) if hasattr(agent, 'input_dim') else None
            
            if test_input is not None and hasattr(agent, 'perception'):
                with torch.no_grad():
                    output = agent.perception(test_input)
                    model_ok = output is not None
            else:
                model_ok = True
            
            return {
                "status": "success",
                "initial_free_energy": initial_fe,
                "train_step": initial_step,
                "model_loaded": model_ok,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.config)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._init_storage()