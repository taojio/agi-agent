import torch
import random
from collections import deque
from ..config.settings import MEMORY_BUFFER_SIZE, FREE_ENERGY_THRESHOLD, DEVICE
from .predictive_coding import HierarchicalPredictiveCoding
from ..utils.metrics import calc_free_energy, calc_entropy


class CognitiveInferenceLayer:
    def __init__(self, feat_dim=16):
        self.pc_model = HierarchicalPredictiveCoding(feat_dim).to(DEVICE)
        self.memory_buffer = deque(maxlen=MEMORY_BUFFER_SIZE)
        self.knowledge_rules = []
        self.recent_predictions = deque(maxlen=20)
        self.thought_history = deque(maxlen=50)
        self.feat_dim = feat_dim

    def resize(self, new_dim):
        self.feat_dim = new_dim
        self.pc_model.resize(new_dim)
        self.memory_buffer.clear()
        self.recent_predictions.clear()

    def autonomous_thinking(self, current_feature):
        pred_seq = self.pc_model.predict_next(current_feature)
        self.offline_reasoning()
        self.deposit_knowledge(current_feature, pred_seq[-1])
        return pred_seq

    def offline_reasoning(self):
        if len(self.memory_buffer) < 10:
            return
        
        sample = random.sample(self.memory_buffer, 1)[0]
        feat_t, feat_t1 = sample
        self.pc_model.update(feat_t, feat_t1)

    def deposit_knowledge(self, feat, pred_feat):
        self.memory_buffer.append((feat.detach().to(DEVICE), pred_feat.detach().to(DEVICE)))
        self.recent_predictions.append((feat.detach().to(DEVICE), pred_feat.detach().to(DEVICE)))
        
        fe = calc_free_energy(pred_feat, feat)
        normalized_conf = max(0.0, min(1.0, 1.0 - fe / (FREE_ENERGY_THRESHOLD * 2)))
        
        if normalized_conf > 0.3 and len(self.knowledge_rules) < 1000:
            self.knowledge_rules.append({
                "state": feat.detach().to(DEVICE),
                "next_state": pred_feat.detach().to(DEVICE),
                "confidence": normalized_conf,
                "timestamp": len(self.memory_buffer)
            })

    def get_knowledge_summary(self):
        if not self.knowledge_rules:
            return {"count": 0, "avg_confidence": 0.0}
        
        avg_conf = sum(rule["confidence"] for rule in self.knowledge_rules) / len(self.knowledge_rules)
        return {"count": len(self.knowledge_rules), "avg_confidence": avg_conf}

    def set_lr(self, lr):
        self.pc_model.set_lr(lr)