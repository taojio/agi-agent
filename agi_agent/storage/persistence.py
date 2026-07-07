import os
import json
import torch
import pickle
from datetime import datetime
from ..config.settings import STORAGE_DIR, MODEL_DIR


class PersistenceManager:
    def __init__(self):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        self.metadata = {}
        self.backup_count = 0

    def save_model(self, model, name: str, step: int = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{step or timestamp}.pt"
        filepath = os.path.join(MODEL_DIR, filename)
        
        torch.save(model.state_dict(), filepath)
        self.metadata[name] = {"path": filepath, "step": step, "timestamp": timestamp}
        return filepath

    def load_model(self, model, name: str, step: int = None):
        if step is not None:
            filepath = os.path.join(MODEL_DIR, f"{name}_{step}.pt")
        elif name in self.metadata:
            filepath = self.metadata[name]["path"]
        else:
            files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(name)])
            if not files:
                return False
            filepath = os.path.join(MODEL_DIR, files[-1])
        
        if os.path.exists(filepath):
            model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            return True
        return False

    def save_state(self, state: dict, filename: str = "agent_state"):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(STORAGE_DIR, f"{filename}_{timestamp}.json")
        
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                serializable_state[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, (torch.nn.Module)):
                continue
            elif isinstance(value, dict):
                serializable_state[key] = value
            else:
                serializable_state[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2)
        
        return filepath

    def load_state(self, filename: str = "agent_state"):
        files = sorted([f for f in os.listdir(STORAGE_DIR) if f.startswith(filename)])
        if not files:
            return None
        
        filepath = os.path.join(STORAGE_DIR, files[-1])
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_knowledge(self, knowledge: list, filename: str = "knowledge"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(STORAGE_DIR, f"{filename}_{timestamp}.pkl")
        
        serializable_knowledge = []
        for rule in knowledge:
            serializable_rule = {}
            for key, value in rule.items():
                if isinstance(value, torch.Tensor):
                    serializable_rule[key] = value.detach().cpu().numpy().tolist()
                else:
                    serializable_rule[key] = value
            serializable_knowledge.append(serializable_rule)
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_knowledge, f)
        
        return filepath

    def load_knowledge(self, filename: str = "knowledge"):
        files = sorted([f for f in os.listdir(STORAGE_DIR) if f.startswith(filename)])
        if not files:
            return None
        
        filepath = os.path.join(STORAGE_DIR, files[-1])
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def create_backup(self, agent, name: str = "agent_backup"):
        self.backup_count += 1
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "backup_count": self.backup_count,
            "model_states": {},
            "metadata": self.metadata
        }
        
        return self.save_state(backup_data, name)

    def get_storage_info(self):
        model_count = len([f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')])
        state_count = len([f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')])
        knowledge_count = len([f for f in os.listdir(STORAGE_DIR) if f.endswith('.pkl')])
        
        return {
            "model_count": model_count,
            "state_count": state_count,
            "knowledge_count": knowledge_count,
            "backup_count": self.backup_count
        }