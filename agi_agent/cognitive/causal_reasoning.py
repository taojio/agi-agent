import numpy as np
from collections import deque
import torch
import torch.nn as nn
from ..config.settings import DEVICE
from ..learning.commonsense_knowledge import CommonsenseKnowledgeBase, create_default_commonsense_kb


class CausalGraph:
    def __init__(self, num_nodes=16):
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        self.causal_strength = np.zeros((num_nodes, num_nodes))
        self.node_names = [f"node_{i}" for i in range(num_nodes)]
        self.intervention_history = deque(maxlen=100)
        self.observation_history = deque(maxlen=200)

    def add_edge(self, from_node, to_node, strength=0.5):
        self.adjacency_matrix[from_node, to_node] = 1
        self.causal_strength[from_node, to_node] = strength

    def remove_edge(self, from_node, to_node):
        self.adjacency_matrix[from_node, to_node] = 0
        self.causal_strength[from_node, to_node] = 0

    def update_strength(self, from_node, to_node, delta):
        self.causal_strength[from_node, to_node] = max(0.0, min(1.0, 
            self.causal_strength[from_node, to_node] + delta))

    def infer_effect(self, cause_node, effect_node, intervention_value=1.0):
        if self.adjacency_matrix[cause_node, effect_node] == 0:
            return 0.0
        
        direct_effect = self.causal_strength[cause_node, effect_node] * intervention_value
        
        indirect_effect = 0.0
        for intermediate in range(self.num_nodes):
            if self.adjacency_matrix[cause_node, intermediate] and self.adjacency_matrix[intermediate, effect_node]:
                indirect_effect += (self.causal_strength[cause_node, intermediate] * 
                                  self.causal_strength[intermediate, effect_node]) * intervention_value
        
        return direct_effect + indirect_effect

    def detect_causation(self, x, y, threshold=0.3):
        correlation = np.corrcoef(x, y)[0, 1]
        
        if abs(correlation) < threshold:
            return 0.0
        
        x_shifted = np.roll(x, 1)
        x_shifted[0] = x[0]
        
        y_shifted = np.roll(y, 1)
        y_shifted[0] = y[0]
        
        x_causes_y = np.mean((x - x_shifted) * (y - y_shifted))
        y_causes_x = np.mean((y - y_shifted) * (x - x_shifted))
        
        if x_causes_y > y_causes_x and x_causes_y > 0:
            return x_causes_y
        elif y_causes_x > x_causes_y and y_causes_x > 0:
            return -y_causes_x
        return 0.0

    def learn_from_observations(self, observations):
        observations = np.array(observations)
        if observations.shape[0] < 10:
            return
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                causation = self.detect_causation(observations[:, i], observations[:, j])
                if abs(causation) > 0.1:
                    if causation > 0:
                        self.add_edge(i, j, causation)
                    else:
                        self.add_edge(j, i, abs(causation))

    def record_intervention(self, node_idx, value, outcome):
        self.intervention_history.append({
            "node": node_idx,
            "value": value,
            "outcome": outcome
        })

    def get_causal_path(self, from_node, to_node):
        path = []
        visited = set()
        stack = [(from_node, [from_node])]
        
        while stack:
            node, current_path = stack.pop()
            if node == to_node:
                path = current_path
                break
            visited.add(node)
            for neighbor in range(self.num_nodes):
                if self.adjacency_matrix[node, neighbor] and neighbor not in visited:
                    stack.append((neighbor, current_path + [neighbor]))
        
        return path


class CausalInferenceEngine:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.causal_graph = CausalGraph(num_nodes=feature_dim)
        self.propensity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.propensity_net.parameters(), lr=1e-3)
        self.counterfactual_history = deque(maxlen=100)

    def estimate_propensity(self, features, action):
        concat = torch.cat([features, action], dim=-1)
        return self.propensity_net(concat)

    def compute_cate(self, features, action, outcome):
        propensity = self.estimate_propensity(features, action)
        
        cate = (outcome - (1 - propensity) * outcome.detach()) / (propensity + 1e-8)
        return cate

    def infer_counterfactual(self, features, actual_action, actual_outcome, hypothetical_action):
        propensity_actual = self.estimate_propensity(features, actual_action)
        propensity_hypothetical = self.estimate_propensity(features, hypothetical_action)
        
        counterfactual = (actual_outcome * propensity_hypothetical / (propensity_actual + 1e-8)).detach()
        
        self.counterfactual_history.append({
            "actual_action": actual_action.detach().cpu().numpy(),
            "actual_outcome": actual_outcome.detach().cpu().numpy(),
            "hypothetical_action": hypothetical_action.detach().cpu().numpy(),
            "counterfactual": counterfactual.cpu().numpy()
        })
        
        return counterfactual

    def update_causal_model(self, features, action, outcome):
        self.optimizer.zero_grad()
        
        features_detached = features.detach().clone()
        action_detached = action.detach().clone()
        outcome_detached = outcome.detach().clone()
        
        propensity = self.estimate_propensity(features_detached, action_detached)
        loss = -torch.mean(outcome_detached * torch.log(propensity + 1e-8) + 
                          (1 - outcome_detached) * torch.log(1 - propensity + 1e-8))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def detect_spurious_correlation(self, features, threshold=0.8):
        feat_np = features.detach().cpu().numpy()
        if feat_np.ndim > 2:
            feat_np = feat_np.reshape(feat_np.shape[0], -1)
        
        if feat_np.shape[0] < 2:
            return []
        
        try:
            correlations = np.corrcoef(feat_np.T)
        except:
            return []
        
        spurious_pairs = []
        
        num_features = min(self.feature_dim, correlations.shape[0])
        for i in range(num_features):
            for j in range(i + 1, num_features):
                if abs(correlations[i, j]) > threshold:
                    has_causal_path = len(self.causal_graph.get_causal_path(i, j)) > 1
                    if not has_causal_path:
                        spurious_pairs.append((i, j, correlations[i, j]))
        
        return spurious_pairs


class AnalogicalReasoner:
    def __init__(self):
        self.situation_memory = deque(maxlen=200)
        self.analogy_threshold = 0.6

    def store_situation(self, context, action, outcome, features):
        situation = {
            "context": context,
            "action": action,
            "outcome": outcome,
            "features": np.array(features),
            "timestamp": np.random.randint(1000000)
        }
        self.situation_memory.append(situation)

    def find_analogous_situation(self, current_features, top_k=3):
        current = np.array(current_features).flatten()
        similarities = []
        
        for situation in self.situation_memory:
            stored = situation["features"].flatten()
            min_len = min(len(current), len(stored))
            if min_len == 0:
                continue
            
            sim = np.dot(current[:min_len], stored[:min_len]) / (
                np.linalg.norm(current[:min_len]) * np.linalg.norm(stored[:min_len]) + 1e-8
            )
            similarities.append((sim, situation))
        
        similarities.sort(key=lambda x: -x[0])
        return [s[1] for s in similarities[:top_k] if s[0] > self.analogy_threshold]

    def transfer_strategy(self, current_features):
        analogs = self.find_analogous_situation(current_features)
        
        if not analogs:
            return None
        
        best_analog = max(analogs, key=lambda s: s.get("outcome", 0))
        return best_analog["action"]


class CausalReasoningEngine:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.causal_graph = CausalGraph(num_nodes=feature_dim)
        self.causal_inference = CausalInferenceEngine(feature_dim=feature_dim)
        self.analogical_reasoner = AnalogicalReasoner()
        self.enabled = True
        self.commonsense_kb = create_default_commonsense_kb()
    
    def query_commonsense_rule(self, action: str, context: dict = None) -> dict:
        """查询常识规则，判断动作是否违反常识"""
        if context is None:
            context = {}
        
        return self.commonsense_kb.validate_action(action, context)
    
    def reason(self, features, action=None, outcome=None, context=None):
        if not self.enabled:
            return {
                "causal_effect": 0.0,
                "analogous_actions": [],
                "spurious_correlations": [],
                "causal_path": [],
                "commonsense_check": {"safe": True, "violations": [], "reason": "reasoning disabled"}
            }
        
        features_np = features.detach().cpu().numpy() if hasattr(features, 'detach') else np.array(features)
        
        commonsense_check = {"safe": True, "violations": [], "reason": "no action provided"}
        
        if action is not None:
            action_str = str(action)
            ctx = context.copy() if context else {}
            if hasattr(action, 'tolist'):
                action_str = str(action.tolist())
            elif isinstance(action, (list, tuple)):
                action_str = str(action)
            
            if "action" in ctx and ctx["action"] != action_str:
                ctx["action_text"] = action_str
            else:
                ctx["action"] = action_str
            
            commonsense_check = self.query_commonsense_rule(action_str, ctx)
            
            if not commonsense_check["safe"] and commonsense_check.get("highest_severity") == "high":
                return {
                    "causal_effect": 0.0,
                    "analogous_actions": [],
                    "spurious_correlations": [],
                    "num_causal_edges": np.sum(self.causal_graph.adjacency_matrix),
                    "commonsense_check": commonsense_check,
                    "warning": "Action blocked by commonsense safety check"
                }
        
        self.causal_graph.learn_from_observations([features_np])
        
        causal_effect = 0.0
        if action is not None and outcome is not None:
            action_tensor = action.detach() if hasattr(action, 'detach') else torch.tensor(action, dtype=torch.float32).to(DEVICE)
            outcome_tensor = outcome.detach() if hasattr(outcome, 'detach') else torch.tensor(outcome, dtype=torch.float32).to(DEVICE)
            
            causal_effect = self.causal_inference.compute_cate(
                torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(DEVICE),
                action_tensor.unsqueeze(0),
                outcome_tensor.unsqueeze(0)
            ).item()
            
            self.causal_inference.update_causal_model(
                torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(DEVICE),
                action_tensor.unsqueeze(0),
                outcome_tensor.unsqueeze(0)
            )
            
            self.analogical_reasoner.store_situation(
                context=features_np.tolist(),
                action=action_tensor.cpu().numpy().tolist(),
                outcome=outcome_tensor.cpu().numpy().tolist(),
                features=features_np
            )
        
        analogous_actions = []
        analogs = self.analogical_reasoner.find_analogous_situation(features_np)
        for analog in analogs:
            analogous_actions.append({
                "action": analog["action"],
                "similarity": np.dot(features_np.flatten()[:min(len(features_np.flatten()), len(analog["features"].flatten()))], 
                                    analog["features"].flatten()[:min(len(features_np.flatten()), len(analog["features"].flatten()))]) / 
                             (np.linalg.norm(features_np.flatten()[:min(len(features_np.flatten()), len(analog["features"].flatten()))]) * 
                              np.linalg.norm(analog["features"].flatten()[:min(len(features_np.flatten()), len(analog["features"].flatten()))]) + 1e-8)
            })
        
        spurious_correlations = self.causal_inference.detect_spurious_correlation(
            torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )
        
        return {
            "causal_effect": causal_effect,
            "analogous_actions": analogous_actions,
            "spurious_correlations": spurious_correlations,
            "num_causal_edges": np.sum(self.causal_graph.adjacency_matrix),
            "commonsense_check": commonsense_check
        }
    
    def infer_counterfactual_with_commonsense(self, features, actual_action, actual_outcome, 
                                              hypothetical_action, context=None):
        """结合常识规则的反事实推理"""
        if context is None:
            context = {}
        
        hypothetical_str = str(hypothetical_action)
        if hasattr(hypothetical_action, 'tolist'):
            hypothetical_str = str(hypothetical_action.tolist())
        elif isinstance(hypothetical_action, (list, tuple)):
            hypothetical_str = str(hypothetical_action)
        
        ctx = context.copy()
        ctx["action"] = hypothetical_str
        rule_check = self.query_commonsense_rule(hypothetical_str, ctx)
        
        if not rule_check["safe"]:
            return {
                "counterfactual": None,
                "reason": f"违反常识: {rule_check['suggestion']}",
                "commonsense_violation": rule_check
            }
        
        features_np = features.detach().cpu().numpy() if hasattr(features, 'detach') else np.array(features)
        
        if isinstance(actual_action, str):
            actual_action = torch.tensor([0.5], dtype=torch.float32).to(DEVICE)
        elif not hasattr(actual_action, 'detach'):
            actual_action = torch.tensor(actual_action, dtype=torch.float32).to(DEVICE)
        else:
            actual_action = actual_action.detach()
            
        if isinstance(actual_outcome, str):
            actual_outcome = torch.tensor([0.5], dtype=torch.float32).to(DEVICE)
        elif not hasattr(actual_outcome, 'detach'):
            actual_outcome = torch.tensor(actual_outcome, dtype=torch.float32).to(DEVICE)
        else:
            actual_outcome = actual_outcome.detach()
            
        if isinstance(hypothetical_action, str):
            hypothetical_action = torch.tensor([0.7], dtype=torch.float32).to(DEVICE)
        elif not hasattr(hypothetical_action, 'detach'):
            hypothetical_action = torch.tensor(hypothetical_action, dtype=torch.float32).to(DEVICE)
        else:
            hypothetical_action = hypothetical_action.detach()
        
        counterfactual = self.causal_inference.infer_counterfactual(
            torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(DEVICE),
            actual_action.unsqueeze(0),
            actual_outcome.unsqueeze(0),
            hypothetical_action.unsqueeze(0)
        )
        
        return {
            "counterfactual": counterfactual.detach().cpu().numpy().tolist() if counterfactual is not None else None,
            "reason": "valid counterfactual",
            "commonsense_violation": None
        }

    def get_causal_graph_summary(self):
        return {
            "num_nodes": self.causal_graph.num_nodes,
            "num_edges": np.sum(self.causal_graph.adjacency_matrix),
            "avg_strength": np.mean(self.causal_graph.causal_strength[self.causal_graph.causal_strength > 0])
        }

    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.causal_graph = CausalGraph(num_nodes=new_feature_dim)
        self.causal_inference = CausalInferenceEngine(feature_dim=new_feature_dim)