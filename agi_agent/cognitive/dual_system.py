import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ..config.settings import DEVICE, FREE_ENERGY_THRESHOLD, MAX_INFERENCE_STEP
from ..utils.metrics import calc_free_energy, calc_entropy, calc_confidence
from .predictive_coding import HierarchicalPredictiveCoding


class System1:
    def __init__(self, feat_dim=16):
        self.feat_dim = feat_dim
        self.pattern_memory = {}
        self.production_rules = []
        self.fast_weights = nn.Linear(feat_dim, feat_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.fast_weights.parameters(), lr=1e-3)
        self.recent_patterns = deque(maxlen=100)
        self.threshold = 0.85

    def pattern_match(self, feature):
        best_match = None
        best_similarity = 0.0
        best_prediction = None

        for pattern_id, (stored_feat, stored_pred) in self.pattern_memory.items():
            stored_feat_tensor = torch.tensor(stored_feat, dtype=torch.float32).to(DEVICE)
            
            if len(feature) != len(stored_feat_tensor):
                min_len = min(len(feature), len(stored_feat_tensor))
                feat_trim = feature[:min_len]
                stored_trim = stored_feat_tensor[:min_len]
            else:
                feat_trim = feature
                stored_trim = stored_feat_tensor

            similarity = torch.nn.functional.cosine_similarity(feat_trim, stored_trim, dim=-1).item()
            
            if similarity > best_similarity and similarity > self.threshold:
                best_similarity = similarity
                best_match = pattern_id
                best_prediction = stored_pred

        return best_match, best_similarity, best_prediction

    def intuitive_inference(self, feature):
        if len(self.recent_patterns) < 5:
            return None, 0.0

        feat_tensor = feature.detach() if hasattr(feature, 'detach') else torch.tensor(feature, dtype=torch.float32).to(DEVICE)
        
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(0)
        
        fast_pred = self.fast_weights(feat_tensor)

        confidence = self._estimate_confidence(fast_pred, feat_tensor)
        return fast_pred, confidence

    def _estimate_confidence(self, pred, feat):
        fe = calc_free_energy(pred.unsqueeze(0), feat.unsqueeze(0)) if pred.dim() == 1 else calc_free_energy(pred, feat)
        return calc_confidence(fe)

    def add_pattern(self, feature, prediction):
        feat_np = feature.detach().cpu().numpy() if hasattr(feature, 'detach') else feature
        pred_np = prediction.detach().cpu().numpy() if hasattr(prediction, 'detach') else prediction
        
        pattern_id = f"pattern_{len(self.pattern_memory)}"
        self.pattern_memory[pattern_id] = (feat_np, pred_np)
        
        self.recent_patterns.append((feat_np, pred_np))
        if len(self.recent_patterns) >= 5:
            self._update_fast_weights()

    def _update_fast_weights(self):
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(DEVICE)
        
        for feat_np, pred_np in list(self.recent_patterns)[-10:]:
            feat_tensor = torch.tensor(feat_np, dtype=torch.float32).to(DEVICE)
            pred_target = torch.tensor(pred_np, dtype=torch.float32).to(DEVICE)
            
            while feat_tensor.dim() > 2:
                feat_tensor = feat_tensor.squeeze(0)
            while pred_target.dim() > 2:
                pred_target = pred_target.squeeze(0)
            
            if feat_tensor.dim() == 1:
                feat_tensor = feat_tensor.unsqueeze(0)
            if pred_target.dim() == 1:
                pred_target = pred_target.unsqueeze(0)
            
            if feat_tensor.shape[1] != self.feat_dim:
                if feat_tensor.shape[1] < self.feat_dim:
                    padding = torch.zeros(1, self.feat_dim - feat_tensor.shape[1]).to(DEVICE)
                    feat_tensor = torch.cat([feat_tensor, padding], dim=1)
                else:
                    feat_tensor = feat_tensor[:, :self.feat_dim]
            
            if pred_target.shape[1] != self.feat_dim:
                if pred_target.shape[1] < self.feat_dim:
                    padding = torch.zeros(1, self.feat_dim - pred_target.shape[1]).to(DEVICE)
                    pred_target = torch.cat([pred_target, padding], dim=1)
                else:
                    pred_target = pred_target[:, :self.feat_dim]
            
            fast_pred = self.fast_weights(feat_tensor)
            loss = torch.mean(torch.square(fast_pred - pred_target))
            total_loss += loss
        
        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()

    def add_production_rule(self, condition, action, confidence=0.9):
        self.production_rules.append({
            "condition": condition,
            "action": action,
            "confidence": confidence,
            "usage_count": 0
        })

    def match_production_rules(self, feature):
        matched_rules = []
        
        for rule in self.production_rules:
            condition_feat = torch.tensor(rule["condition"], dtype=torch.float32).to(DEVICE)
            
            if len(feature) != len(condition_feat):
                min_len = min(len(feature), len(condition_feat))
                feat_trim = feature[:min_len]
                cond_trim = condition_feat[:min_len]
            else:
                feat_trim = feature
                cond_trim = condition_feat
            
            similarity = torch.nn.functional.cosine_similarity(feat_trim, cond_trim, dim=-1).item()
            
            if similarity > 0.8:
                matched_rules.append((rule, similarity))
        
        matched_rules.sort(key=lambda x: x[1] * x[0]["confidence"], reverse=True)
        return matched_rules

    def resize(self, new_dim):
        self.feat_dim = new_dim
        self.fast_weights = nn.Linear(new_dim, new_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.fast_weights.parameters(), lr=1e-3)


class System2:
    def __init__(self, feat_dim=16):
        self.feat_dim = feat_dim
        self.pc_model = HierarchicalPredictiveCoding(feat_dim).to(DEVICE)
        self.working_memory = deque(maxlen=50)
        self.inference_stack = []
        self.is_deliberating = False
        self.max_deliberation_steps = MAX_INFERENCE_STEP

    def deliberate(self, feature, goal=None):
        self.is_deliberating = True
        results = []
        
        for step in range(self.max_deliberation_steps):
            pred_seq = self.pc_model.predict_next(feature)
            final_pred = pred_seq[-1]
            
            fe = calc_free_energy(final_pred, feature)
            confidence = calc_confidence(fe)
            
            results.append({
                "step": step,
                "prediction": final_pred,
                "confidence": confidence,
                "free_energy": fe
            })
            
            if confidence > 0.9:
                break
            
            if goal is not None:
                feature = self._refine_feature(feature, goal, final_pred)
        
        self.is_deliberating = False
        return results

    def _refine_feature(self, current, goal, prediction):
        goal_tensor = torch.tensor(goal, dtype=torch.float32).to(DEVICE) if not isinstance(goal, torch.Tensor) else goal
        
        if current.dim() == 2 and current.shape[0] == 1:
            current = current.squeeze(0)
        if prediction.dim() == 2 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if goal_tensor.dim() == 2 and goal_tensor.shape[0] == 1:
            goal_tensor = goal_tensor.squeeze(0)
        
        error = goal_tensor - prediction
        refined = current + 0.1 * error
        
        return refined.unsqueeze(0)

    def analogical_reasoning(self, source_feature, target_feature):
        source_pred = self.pc_model.predict_next(source_feature)[-1]
        
        source_pred_np = source_pred.detach().cpu().numpy()
        source_feat_np = source_feature.detach().cpu().numpy()
        
        transformation = source_pred_np - source_feat_np
        
        target_pred_np = target_feature.detach().cpu().numpy() + transformation
        target_pred = torch.tensor(target_pred_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        return target_pred

    def causal_inference(self, feature, action_hypothesis):
        predictions = []
        
        for action in action_hypothesis:
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            combined = torch.cat([feature, action_tensor], dim=-1)
            
            pred = self.pc_model.predict_next(feature)[-1]
            
            fe = calc_free_energy(pred, feature)
            predictions.append({
                "action": action,
                "prediction": pred,
                "free_energy": fe,
                "plausibility": 1.0 / (fe + 1e-8)
            })
        
        predictions.sort(key=lambda x: x["plausibility"], reverse=True)
        return predictions

    def add_to_working_memory(self, item):
        self.working_memory.append(item)

    def clear_working_memory(self):
        self.working_memory.clear()

    def resize(self, new_dim):
        self.feat_dim = new_dim
        self.pc_model.resize(new_dim)

    def set_lr(self, lr):
        self.pc_model.set_lr(lr)


class DualSystemCognition:
    def __init__(self, feat_dim=16):
        self.system1 = System1(feat_dim)
        self.system2 = System2(feat_dim)
        self.feat_dim = feat_dim
        self.mode = "auto"
        self.system1_usage_count = 0
        self.system2_usage_count = 0

    def think(self, feature, goal=None):
        fast_pred, fast_conf = self.system1.intuitive_inference(feature)
        
        if self.mode == "system1_only":
            self.system1_usage_count += 1
            return {
                "system": "system1",
                "prediction": fast_pred,
                "confidence": fast_conf,
                "deliberation_steps": 0
            }
        
        if fast_conf > 0.85 and goal is None:
            self.system1_usage_count += 1
            return {
                "system": "system1",
                "prediction": fast_pred,
                "confidence": fast_conf,
                "deliberation_steps": 0
            }
        
        self.system2_usage_count += 1
        deliberate_results = self.system2.deliberate(feature, goal)
        
        if deliberate_results:
            best_result = max(deliberate_results, key=lambda x: x["confidence"])
            return {
                "system": "system2",
                "prediction": best_result["prediction"],
                "confidence": best_result["confidence"],
                "deliberation_steps": best_result["step"] + 1,
                "free_energy": best_result["free_energy"]
            }
        
        return {
            "system": "system1",
            "prediction": fast_pred,
            "confidence": fast_conf,
            "deliberation_steps": 0
        }

    def learn_pattern(self, feature, prediction):
        self.system1.add_pattern(feature, prediction)

    def deliberate_deep(self, feature, goal, max_steps=10):
        return self.system2.deliberate(feature, goal)

    def analogical_reason(self, source, target):
        return self.system2.analogical_reasoning(source, target)

    def causal_reason(self, feature, actions):
        return self.system2.causal_inference(feature, actions)

    def resize(self, new_dim):
        self.feat_dim = new_dim
        self.system1.resize(new_dim)
        self.system2.resize(new_dim)

    def set_lr(self, lr):
        self.system2.set_lr(lr)

    def set_mode(self, mode):
        if mode in ["auto", "system1_only", "system2_only"]:
            self.mode = mode

    def get_stats(self):
        return {
            "system1_usage": self.system1_usage_count,
            "system2_usage": self.system2_usage_count,
            "mode": self.mode,
            "pattern_count": len(self.system1.pattern_memory)
        }