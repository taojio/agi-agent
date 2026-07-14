import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from ..config.settings import DEVICE


class EntityCategory(Enum):
    OBJECT = "object"
    AGENT = "agent"
    ENVIRONMENT = "environment"
    EVENT = "event"
    ACTION = "action"
    RULE = "rule"
    RELATION = "relation"
    CONCEPT = "concept"


class AbstractionLevel(Enum):
    PERCEPTUAL = "perceptual"
    OBJECT = "object"
    SCENE = "scene"
    RULE = "rule"


class ModalityType(Enum):
    VISUAL = "visual"
    TEXT = "text"
    SENSOR = "sensor"
    POINTCLOUD = "pointcloud"


class PhysicalProperty:
    def __init__(self, name: str, value: float, unit: str = "", variance: float = 0.0):
        self.name = name
        self.value = value
        self.unit = unit
        self.variance = variance
        self.history: List[Tuple[float, float]] = []

    def update(self, value: float, timestamp: float):
        self.value = value
        self.history.append((timestamp, value))
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "variance": self.variance,
            "history_length": len(self.history)
        }


class WorldEntity:
    def __init__(self, entity_id: str, category: EntityCategory,
                 features: Optional[torch.Tensor] = None,
                 position: Optional[np.ndarray] = None):
        self.entity_id = entity_id
        self.category = category
        self.features = features if features is not None else torch.tensor([])
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self.properties: Dict[str, PhysicalProperty] = {}
        self.state: Dict[str, Any] = {}
        self.confidence = 0.5
        self.last_updated = time.time()
        self.activation_level = 0.0

    def add_property(self, name: str, value: float, unit: str = "", variance: float = 0.0):
        self.properties[name] = PhysicalProperty(name, value, unit, variance)

    def get_property(self, name: str, default: float = 0.0) -> float:
        return self.properties[name].value if name in self.properties else default

    def update_state(self, key: str, value: Any):
        self.state[key] = value
        self.last_updated = time.time()

    def activate(self, strength: float = 1.0):
        self.activation_level = min(1.0, self.activation_level + strength * 0.1)
        self.confidence = min(1.0, self.confidence + 0.02)

    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "category": self.category.value,
            "position": self.position.tolist(),
            "confidence": self.confidence,
            "activation_level": self.activation_level,
            "property_count": len(self.properties),
            "state_keys": list(self.state.keys())
        }


class CausalRelation:
    def __init__(self, cause_id: str, effect_id: str,
                 strength: float = 0.5, delay: float = 0.0,
                 conditions: Optional[List[str]] = None):
        self.cause_id = cause_id
        self.effect_id = effect_id
        self.strength = strength
        self.delay = delay
        self.conditions = conditions if conditions is not None else []
        self.evidence_count = 0
        self.confidence = strength

    def update_confidence(self, evidence_strength: float):
        self.evidence_count += 1
        self.confidence = (self.confidence * (self.evidence_count - 1) + evidence_strength) / self.evidence_count

    def to_dict(self):
        return {
            "cause_id": self.cause_id,
            "effect_id": self.effect_id,
            "strength": self.strength,
            "delay": self.delay,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count
        }


class SocialRule:
    def __init__(self, rule_id: str, description: str,
                 conditions: List[str], consequences: List[str],
                 enforcement_strength: float = 0.9, domain: str = "general"):
        self.rule_id = rule_id
        self.description = description
        self.conditions = conditions
        self.consequences = consequences
        self.enforcement_strength = enforcement_strength
        self.domain = domain
        self.applied_count = 0
        self.confidence = 0.5

    def apply(self, context: Dict[str, Any]) -> bool:
        for condition in self.conditions:
            if condition not in context or not context[condition]:
                return False
        self.applied_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        return True

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "enforcement_strength": self.enforcement_strength,
            "domain": self.domain,
            "applied_count": self.applied_count,
            "confidence": self.confidence
        }


class SimulationResult:
    def __init__(self, simulation_id: str, initial_state: Dict,
                 final_state: Dict, steps: List[Dict],
                 confidence: float = 0.5):
        self.simulation_id = simulation_id
        self.initial_state = initial_state
        self.final_state = final_state
        self.steps = steps
        self.confidence = confidence
        self.valid = True

    def to_dict(self):
        return {
            "simulation_id": self.simulation_id,
            "step_count": len(self.steps),
            "confidence": self.confidence,
            "valid": self.valid,
            "final_state_keys": list(self.final_state.keys())
        }


class MultiModalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Embedding(10000, feature_dim),
            nn.LSTM(feature_dim, feature_dim, batch_first=True, bidirectional=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim)
        )

        self.pointcloud_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

        self.unified_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.modality_weights = nn.Parameter(torch.ones(4))

    def forward_visual(self, visual_input: torch.Tensor) -> torch.Tensor:
        if visual_input.dim() == 3:
            visual_input = visual_input.unsqueeze(0)
        return self.visual_encoder(visual_input)

    def forward_text(self, text_input: torch.Tensor) -> torch.Tensor:
        if text_input.dim() == 1:
            text_input = text_input.unsqueeze(0)
        embeddings, _ = self.text_encoder(text_input)
        return embeddings[:, -1, :]

    def forward_sensor(self, sensor_input: torch.Tensor) -> torch.Tensor:
        if sensor_input.dim() == 1:
            sensor_input = sensor_input.unsqueeze(0).unsqueeze(0)
        elif sensor_input.dim() == 2:
            sensor_input = sensor_input.unsqueeze(0)
        return self.sensor_encoder(sensor_input)

    def forward_pointcloud(self, pointcloud_input: torch.Tensor) -> torch.Tensor:
        features = self.pointcloud_encoder(pointcloud_input)
        return features.mean(dim=0, keepdim=True)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        modalities = []
        weights = F.softmax(self.modality_weights, dim=0)

        if "visual" in inputs:
            modalities.append(self.forward_visual(inputs["visual"]) * weights[0])
        if "text" in inputs:
            modalities.append(self.forward_text(inputs["text"]) * weights[1])
        if "sensor" in inputs:
            modalities.append(self.forward_sensor(inputs["sensor"]) * weights[2])
        if "pointcloud" in inputs:
            modalities.append(self.forward_pointcloud(inputs["pointcloud"]) * weights[3])

        if not modalities:
            return torch.randn(1, self.feature_dim).to(DEVICE)

        fused = torch.stack(modalities, dim=1).sum(dim=1)
        return self.unified_projection(fused)

    def get_modality_importance(self) -> Dict[str, float]:
        weights = F.softmax(self.modality_weights, dim=0)
        return {
            "visual": weights[0].item(),
            "text": weights[1].item(),
            "sensor": weights[2].item(),
            "pointcloud": weights[3].item()
        }


class HierarchicalRepresentation(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.perceptual_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.object_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.scene_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.rule_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.up_projection_perceptual = nn.Linear(feature_dim, feature_dim)
        self.up_projection_object = nn.Linear(feature_dim, feature_dim)
        self.up_projection_scene = nn.Linear(feature_dim, feature_dim)

        self.down_projection_scene = nn.Linear(feature_dim, feature_dim)
        self.down_projection_object = nn.Linear(feature_dim, feature_dim)
        self.down_projection_perceptual = nn.Linear(feature_dim, feature_dim)

    def forward(self, input_features: torch.Tensor,
                prev_hierarchy: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        perceptual = self.perceptual_layer(input_features)

        if prev_hierarchy is None:
            prev_hierarchy = {}

        object_context = prev_hierarchy.get("object", torch.zeros_like(perceptual))
        object_input = torch.cat([perceptual, self.up_projection_perceptual(object_context)], dim=-1)
        object_emb = self.object_layer(object_input)

        scene_context = prev_hierarchy.get("scene", torch.zeros_like(object_emb))
        scene_input = torch.cat([object_emb, self.up_projection_object(scene_context), perceptual], dim=-1)
        scene_emb = self.scene_layer(scene_input)

        rule_context = prev_hierarchy.get("rule", torch.zeros_like(scene_emb))
        rule_input = torch.cat([scene_emb, self.up_projection_scene(rule_context)], dim=-1)
        rule_emb = self.rule_layer(rule_input)

        down_scene = self.down_projection_scene(scene_emb)
        down_object = self.down_projection_object(object_emb)
        down_perceptual = self.down_projection_perceptual(perceptual)

        refined_perceptual = perceptual + down_object * 0.3 + down_scene * 0.2
        refined_object = object_emb + down_scene * 0.3 + down_perceptual * 0.2
        refined_scene = scene_emb + down_object * 0.2 + down_perceptual * 0.1

        return {
            "perceptual": refined_perceptual,
            "object": refined_object,
            "scene": refined_scene,
            "rule": rule_emb
        }

    def get_hierarchy_summary(self, hierarchy: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            "perceptual_norm": hierarchy["perceptual"].norm().item(),
            "object_norm": hierarchy["object"].norm().item(),
            "scene_norm": hierarchy["scene"].norm().item(),
            "rule_norm": hierarchy["rule"].norm().item()
        }


class DynamicsPredictor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.micro_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.meso_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.macro_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.gmm_weights = nn.Linear(feature_dim * 2, 3)
        self.gmm_means = nn.Linear(feature_dim * 2, feature_dim * 3)
        self.gmm_vars = nn.Linear(feature_dim * 2, feature_dim * 3)

    def predict_micro(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current_state, action], dim=-1)
        if combined.size(-1) > self.feature_dim * 2:
            combined = combined[:, :self.feature_dim * 2]
        return self.micro_predictor(combined)

    def predict_meso(self, current_state: torch.Tensor, action: torch.Tensor,
                     relation_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current_state, action, relation_features], dim=-1)
        if combined.size(-1) > self.feature_dim * 3:
            combined = combined[:, :self.feature_dim * 3]
        return self.meso_predictor(combined)

    def predict_macro(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current_state, action], dim=-1)
        if combined.size(-1) > self.feature_dim * 2:
            combined = combined[:, :self.feature_dim * 2]
        return self.macro_predictor(combined)

    def predict_probabilistic(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([state, action], dim=-1)
        if combined.size(-1) > self.feature_dim * 2:
            combined = combined[:, :self.feature_dim * 2]

        weights = F.softmax(self.gmm_weights(combined), dim=-1)
        means = self.gmm_means(combined).view(-1, 3, self.feature_dim)
        vars = F.softplus(self.gmm_vars(combined)).view(-1, 3, self.feature_dim)

        return {
            "weights": weights,
            "means": means,
            "vars": vars
        }

    def forward(self, hierarchy: Dict[str, torch.Tensor], action: torch.Tensor,
                relation_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if relation_features is None:
            relation_features = torch.zeros(1, self.feature_dim).to(DEVICE)

        micro_pred = self.predict_micro(hierarchy["perceptual"], action)
        meso_pred = self.predict_meso(hierarchy["object"], action, relation_features)
        macro_pred = self.predict_macro(hierarchy["scene"], action)

        probabilistic = self.predict_probabilistic(hierarchy["object"], action)

        return {
            "perceptual": micro_pred,
            "object": meso_pred,
            "scene": macro_pred,
            "probabilistic": probabilistic
        }


class CausalReasoner(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.causal_discovery = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.intervention_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.counterfactual_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.causal_graph: Dict[str, List[Dict]] = {}

    def discover_causality(self, cause_features: torch.Tensor,
                           effect_features: torch.Tensor) -> float:
        combined = torch.cat([cause_features, effect_features], dim=-1)
        if combined.size(-1) > self.feature_dim * 2:
            combined = combined[:, :self.feature_dim * 2]
        return self.causal_discovery(combined).squeeze().item()

    def predict_intervention(self, cause_features: torch.Tensor,
                            effect_features: torch.Tensor,
                            intervention: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cause_features, effect_features, intervention], dim=-1)
        if combined.size(-1) > self.feature_dim * 3:
            combined = combined[:, :self.feature_dim * 3]
        return self.intervention_predictor(combined)

    def predict_counterfactual(self, cause_features: torch.Tensor,
                               effect_features: torch.Tensor,
                               alternative_cause: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cause_features, effect_features, alternative_cause], dim=-1)
        if combined.size(-1) > self.feature_dim * 3:
            combined = combined[:, :self.feature_dim * 3]
        return self.counterfactual_predictor(combined)

    def add_causal_edge(self, cause_id: str, effect_id: str, strength: float):
        if cause_id not in self.causal_graph:
            self.causal_graph[cause_id] = []
        self.causal_graph[cause_id].append({
            "effect": effect_id,
            "strength": strength
        })

    def get_causal_graph(self) -> Dict[str, List[Dict]]:
        return dict(self.causal_graph)


class MemorySystem:
    def __init__(self, feature_dim: int = 128, working_memory_size: int = 20,
                 episodic_memory_size: int = 1000, semantic_memory_size: int = 100):
        self.feature_dim = feature_dim
        self.working_memory = deque(maxlen=working_memory_size)
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, torch.Tensor] = {}
        self.episodic_memory_size = episodic_memory_size
        self.semantic_memory_size = semantic_memory_size

    def add_working_memory(self, state: Dict[str, torch.Tensor], timestamp: float):
        self.working_memory.append({
            "state": {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in state.items()},
            "timestamp": timestamp
        })

    def get_working_memory(self) -> List[Dict[str, Any]]:
        return list(self.working_memory)

    def add_episodic_memory(self, scene_id: str, features: torch.Tensor, context: Dict[str, Any]):
        memory_item = {
            "scene_id": scene_id,
            "features": features.detach().cpu().numpy(),
            "context": context,
            "timestamp": time.time(),
            "access_count": 0
        }
        self.episodic_memory.append(memory_item)
        if len(self.episodic_memory) > self.episodic_memory_size:
            self.episodic_memory = self.episodic_memory[-self.episodic_memory_size:]

    def retrieve_episodic_memory(self, query_features: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.episodic_memory:
            return []

        query_np = query_features.detach().cpu().numpy().flatten()
        similarities = []

        for item in self.episodic_memory:
            item_features = item["features"].flatten()
            similarity = np.dot(query_np, item_features) / (np.linalg.norm(query_np) * np.linalg.norm(item_features) + 1e-8)
            similarities.append((similarity, item))

        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in similarities[:top_k]]

        for item in results:
            item["access_count"] += 1

        return results

    def add_semantic_memory(self, concept_id: str, features: torch.Tensor):
        if len(self.semantic_memory) >= self.semantic_memory_size:
            oldest_key = min(self.semantic_memory.keys(), key=lambda k: self.semantic_memory[k].get("_timestamp", 0))
            del self.semantic_memory[oldest_key]

        self.semantic_memory[concept_id] = features.detach().clone()
        self.semantic_memory[concept_id]._timestamp = time.time()

    def get_semantic_memory(self, concept_id: str) -> Optional[torch.Tensor]:
        return self.semantic_memory.get(concept_id)

    def get_semantic_keys(self) -> List[str]:
        return list(self.semantic_memory.keys())

    def distill_from_episodic(self):
        if len(self.episodic_memory) < 10:
            return

        recent_memories = self.episodic_memory[-50:]
        if not recent_memories:
            return

        avg_features = np.mean([m["features"] for m in recent_memories], axis=0)
        concept_id = f"distilled_{int(time.time())}"
        self.add_semantic_memory(concept_id, torch.tensor(avg_features, dtype=torch.float32))


class PlanningInterface:
    def __init__(self, world_model):
        self.world_model = world_model

    def imagine_trajectory(self, initial_state: Dict[str, Any],
                           actions: List[Dict], max_steps: int = 50) -> Dict[str, Any]:
        trajectory = []
        current_state = dict(initial_state)

        for step_idx in range(max_steps):
            if step_idx < len(actions):
                action = actions[step_idx]
            else:
                action = {"type": "noop"}

            next_state, confidence = self.world_model._predict_next_state(current_state, action)

            trajectory.append({
                "step": step_idx,
                "state": dict(current_state),
                "action": action,
                "next_state": next_state,
                "confidence": confidence
            })

            current_state = next_state

            if step_idx % 10 == 0:
                self.world_model._correct_state(current_state)

        return {
            "trajectory": trajectory,
            "total_steps": len(trajectory),
            "final_state": current_state
        }

    def plan_to_goal(self, goal_state: Dict[str, Any],
                     current_state: Dict[str, Any],
                     max_planning_steps: int = 20) -> List[Dict]:
        plan = []
        current = dict(current_state)
        steps_taken = 0

        while steps_taken < max_planning_steps:
            differences = self._calculate_state_difference(current, goal_state)
            if not differences:
                break

            priority_diff = max(differences.items(), key=lambda x: self._calculate_importance(x[0]))
            diff_key = priority_diff[0]

            action = self._find_action_to_resolve(diff_key, current, goal_state)
            if action:
                plan.append(action)
                current = self.world_model._simulate_action_effect(current, action)

            steps_taken += 1

        return plan

    def generate_rl_trajectories(self, num_trajectories: int = 10,
                                 max_steps: int = 50) -> List[Dict]:
        trajectories = []

        for _ in range(num_trajectories):
            initial_state = self.world_model._generate_random_state()
            actions = self.world_model._generate_random_actions(max_steps)
            trajectory = self.imagine_trajectory(initial_state, actions, max_steps)
            trajectories.append(trajectory)

        return trajectories

    def _calculate_state_difference(self, current: Dict, goal: Dict) -> Dict[str, Dict]:
        differences = {}
        for key, target_value in goal.items():
            current_value = current.get(key, {})
            if isinstance(target_value, dict) and isinstance(current_value, dict):
                for sub_key, sub_target in target_value.items():
                    if current_value.get(sub_key) != sub_target:
                        differences[f"{key}.{sub_key}"] = {
                            "current": current_value.get(sub_key),
                            "target": sub_target
                        }
            elif current_value != target_value:
                differences[key] = {"current": current_value, "target": target_value}
        return differences

    def _calculate_importance(self, key: str) -> float:
        return 1.0

    def _find_action_to_resolve(self, diff_key: str, current: Dict, goal: Dict) -> Optional[Dict]:
        for rel in self.world_model.causal_relations:
            if rel.effect_id in diff_key:
                return {
                    "action": "activate",
                    "entity_id": rel.cause_id,
                    "target": rel.effect_id,
                    "confidence": rel.confidence
                }
        return None


class WorldModelEngine:
    def __init__(self, feature_dim: int = 128, history_length: int = 100):
        self.feature_dim = feature_dim
        self.history_length = history_length

        self.multimodal_encoder = MultiModalEncoder(feature_dim).to(DEVICE)
        self.hierarchical_representation = HierarchicalRepresentation(feature_dim).to(DEVICE)
        self.dynamics_predictor = DynamicsPredictor(feature_dim).to(DEVICE)
        self.causal_reasoner = CausalReasoner(feature_dim).to(DEVICE)
        self.memory_system = MemorySystem(feature_dim)
        self.planning_interface = PlanningInterface(self)

        self.entities: Dict[str, WorldEntity] = {}
        self.causal_relations: List[CausalRelation] = []
        self.social_rules: List[SocialRule] = []
        self.state_history = deque(maxlen=history_length)
        self.simulation_history = deque(maxlen=50)
        self.hierarchy_history = deque(maxlen=history_length)

        self.closed_loop_correction_interval = 10
        self.training_mode = False
        self.optimizer = None

        self._init_default_rules()

    def _init_default_rules(self):
        self.add_social_rule(
            rule_id="rule_physical_cause",
            description="物理因果规则：动作导致状态变化",
            conditions=["action_executed"],
            consequences=["state_changed"],
            enforcement_strength=0.95,
            domain="physics"
        )

        self.add_social_rule(
            rule_id="rule_social_norms",
            description="社会规范规则：合作带来回报",
            conditions=["cooperation", "trust"],
            consequences=["positive_outcome"],
            enforcement_strength=0.8,
            domain="social"
        )

        self.add_social_rule(
            rule_id="rule_resource_limits",
            description="资源限制规则：资源有限需合理分配",
            conditions=["resource_scarce"],
            consequences=["prioritize_essential"],
            enforcement_strength=0.9,
            domain="resource"
        )

    def encode_multimodal_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        tensor_inputs = {}
        for modality, data in inputs.items():
            if isinstance(data, np.ndarray):
                tensor_inputs[modality] = torch.tensor(data, dtype=torch.float32).to(DEVICE)
            elif isinstance(data, torch.Tensor):
                tensor_inputs[modality] = data.to(DEVICE)
            else:
                continue

        return self.multimodal_encoder(tensor_inputs)

    def build_hierarchy(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        prev_hierarchy = None
        if self.hierarchy_history:
            prev_hierarchy = self.hierarchy_history[-1]

        hierarchy = self.hierarchical_representation(features, prev_hierarchy)
        detached_hierarchy = {k: v.detach() for k, v in hierarchy.items()}
        self.hierarchy_history.append(detached_hierarchy)
        return hierarchy

    def predict_dynamics(self, hierarchy: Dict[str, torch.Tensor],
                         action: torch.Tensor, step_idx: int = 0) -> Dict[str, torch.Tensor]:
        relation_features = self._get_relation_features(hierarchy)
        prediction = self.dynamics_predictor(hierarchy, action, relation_features)

        if step_idx > 0 and step_idx % self.closed_loop_correction_interval == 0:
            prediction = self._apply_closed_loop_correction(prediction)

        return prediction

    def _get_relation_features(self, hierarchy: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.causal_relations:
            return torch.zeros(1, self.feature_dim).to(DEVICE)

        relation_strengths = torch.tensor(
            [rel.strength for rel in self.causal_relations],
            dtype=torch.float32
        ).to(DEVICE)

        return relation_strengths.mean().expand(1, self.feature_dim)

    def _apply_closed_loop_correction(self, prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.state_history:
            return prediction

        last_state = self.state_history[-1]
        corrected = {}

        for key, pred in prediction.items():
            if key == "probabilistic":
                corrected[key] = pred
                continue

            if isinstance(last_state, dict) and key in last_state:
                last_val = last_state[key]
                if isinstance(last_val, torch.Tensor):
                    corrected[key] = pred * 0.8 + last_val * 0.2
                else:
                    corrected[key] = pred
            else:
                corrected[key] = pred

        return corrected

    def discover_causal_structure(self):
        if len(self.entities) < 2:
            return

        entity_ids = list(self.entities.keys())
        for i in range(len(entity_ids)):
            for j in range(len(entity_ids)):
                if i == j:
                    continue

                cause_entity = self.entities[entity_ids[i]]
                effect_entity = self.entities[entity_ids[j]]

                if len(cause_entity.features) > 0 and len(effect_entity.features) > 0:
                    cause_emb = self._ensure_feature_dim(cause_entity.features)
                    effect_emb = self._ensure_feature_dim(effect_entity.features)

                    strength = self.causal_reasoner.discover_causality(cause_emb, effect_emb)

                    if strength > 0.3:
                        self.causal_reasoner.add_causal_edge(
                            entity_ids[i], entity_ids[j], strength
                        )

                        exists = any(rel.cause_id == entity_ids[i] and rel.effect_id == entity_ids[j]
                                     for rel in self.causal_relations)
                        if not exists:
                            self.add_causal_relation(entity_ids[i], entity_ids[j], strength)

    def simulate_counterfactual(self, base_simulation: SimulationResult,
                               modified_actions: List[Dict]) -> SimulationResult:
        return self.predict_counterfactual(base_simulation.initial_state, modified_actions)

    def predict_counterfactual(self, base_state: Dict[str, Any],
                               modified_actions: List[Dict]) -> SimulationResult:
        modified_initial = dict(base_state)

        for action in modified_actions:
            entity_id = action.get("entity_id")
            if entity_id in modified_initial:
                modified_initial[entity_id] = action.get("state", {})

        scenario_id = f"counterfactual_{int(time.time())}"
        return self.simulate_scenario(scenario_id, modified_initial, modified_actions)

    def predict_intervention(self, cause_id: str, intervention_state: Dict[str, Any]) -> Dict[str, Any]:
        if cause_id not in self.entities:
            return {"error": "Cause entity not found"}

        cause_entity = self.entities[cause_id]
        cause_emb = self._ensure_feature_dim(cause_entity.features)

        effect_predictions = {}
        for rel in self.causal_relations:
            if rel.cause_id == cause_id:
                effect_entity = self.entities.get(rel.effect_id)
                if effect_entity:
                    effect_emb = self._ensure_feature_dim(effect_entity.features)
                    intervention_tensor = self._dict_to_tensor(intervention_state)

                    prediction = self.causal_reasoner.predict_intervention(
                        cause_emb, effect_emb, intervention_tensor
                    )

                    effect_predictions[rel.effect_id] = {
                        "predicted_state": prediction.detach().cpu().numpy(),
                        "confidence": rel.confidence
                    }

        return effect_predictions

    def add_entity(self, entity_id: str, category: EntityCategory,
                   features: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   position: Optional[np.ndarray] = None) -> WorldEntity:
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        elif features is None:
            features = torch.randn(self.feature_dim)

        features = self._ensure_feature_dim(features)

        entity = WorldEntity(entity_id, category, features=features, position=position)
        self.entities[entity_id] = entity
        return entity

    def add_causal_relation(self, cause_id: str, effect_id: str,
                            strength: float = 0.5, delay: float = 0.0,
                            conditions: Optional[List[str]] = None) -> bool:
        if cause_id not in self.entities or effect_id not in self.entities:
            return False

        relation = CausalRelation(cause_id, effect_id, strength, delay, conditions)
        self.causal_relations.append(relation)
        return True

    def add_social_rule(self, rule_id: str, description: str,
                        conditions: List[str], consequences: List[str],
                        enforcement_strength: float = 0.9, domain: str = "general"):
        rule = SocialRule(rule_id, description, conditions, consequences,
                          enforcement_strength, domain)
        self.social_rules.append(rule)

    def simulate_scenario(self, scenario_id: str, initial_state: Dict[str, Any],
                          actions: List[Dict], max_steps: int = 50) -> SimulationResult:
        self.discover_causal_structure()
        current_state = dict(initial_state)
        steps = []

        for step_idx in range(max_steps):
            step_result = {"step": step_idx, "state": dict(current_state)}

            applicable_rules = []
            for rule in self.social_rules:
                if rule.apply(current_state):
                    applicable_rules.append(rule.rule_id)

            step_result["applied_rules"] = applicable_rules

            if step_idx < len(actions):
                action = actions[step_idx]
                step_result["action"] = action

                action_tensor = self._dict_to_tensor(action)
                features = self._state_to_features(current_state)

                retrieved_memories = self.memory_system.retrieve_episodic_memory(features, top_k=3)
                step_result["retrieved_memories"] = len(retrieved_memories)

                hierarchy = self.build_hierarchy(features)
                dynamics = self.predict_dynamics(hierarchy, action_tensor, step_idx)

                step_result["dynamics_prediction"] = {
                    "perceptual_norm": dynamics["perceptual"].norm().item(),
                    "object_norm": dynamics["object"].norm().item(),
                    "scene_norm": dynamics["scene"].norm().item()
                }

                self.memory_system.add_working_memory(hierarchy, time.time())

                for rel in self.causal_relations:
                    if rel.cause_id == action.get("entity_id"):
                        effect_entity = self.entities.get(rel.effect_id)
                        if effect_entity:
                            effect_entity.activate(rel.strength)
                            current_state[rel.effect_id] = {
                                "activated": True,
                                "confidence": effect_entity.confidence
                            }
                            step_result["causal_effects"] = {
                                "cause": rel.cause_id,
                                "effect": rel.effect_id,
                                "strength": rel.strength
                            }

            steps.append(step_result)
            self._update_state_history(current_state)

            if step_idx % 10 == 0:
                self.memory_system.distill_from_episodic()

        self.memory_system.add_episodic_memory(scenario_id, features, {"steps": len(steps), "confidence": 0.5})

        confidence = self._calculate_simulation_confidence(steps)

        result = SimulationResult(
            simulation_id=scenario_id,
            initial_state=initial_state,
            final_state=current_state,
            steps=steps,
            confidence=confidence
        )

        self.simulation_history.append(result)
        return result

    def plan_long_term(self, goal_state: Dict[str, Any],
                       current_state: Dict[str, Any],
                       max_planning_steps: int = 20) -> List[Dict]:
        return self.planning_interface.plan_to_goal(goal_state, current_state, max_planning_steps)

    def imagine_trajectory(self, initial_state: Dict[str, Any],
                           actions: List[Dict], max_steps: int = 50) -> Dict[str, Any]:
        return self.planning_interface.imagine_trajectory(initial_state, actions, max_steps)

    def generate_rl_trajectories(self, num_trajectories: int = 10,
                                 max_steps: int = 50) -> List[Dict]:
        return self.planning_interface.generate_rl_trajectories(num_trajectories, max_steps)

    def start_training(self):
        self.training_mode = True
        self.optimizer = torch.optim.Adam([
            {"params": self.multimodal_encoder.parameters()},
            {"params": self.hierarchical_representation.parameters()},
            {"params": self.dynamics_predictor.parameters()},
            {"params": self.causal_reasoner.parameters()}
        ], lr=1e-4)

    def stop_training(self):
        self.training_mode = False
        self.optimizer = None

    def train_step(self, inputs: Dict[str, Any], actions: List[Dict],
                   target_state: Dict[str, Any]) -> float:
        if not self.training_mode or self.optimizer is None:
            return 0.0

        self.optimizer.zero_grad()

        features = self.encode_multimodal_input(inputs)
        hierarchy = self.build_hierarchy(features)

        action_tensor = self._dict_to_tensor(actions[0]) if actions else torch.zeros(1, self.feature_dim).to(DEVICE)
        prediction = self.predict_dynamics(hierarchy, action_tensor)

        target_features = self._state_to_features(target_state)
        target_hierarchy = self.build_hierarchy(target_features)

        loss = 0.0
        for level in ["perceptual", "object", "scene"]:
            loss += F.mse_loss(prediction[level], target_hierarchy[level])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _state_to_features(self, state: Dict[str, Any]) -> torch.Tensor:
        features_list = []
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                features_list.append(value.flatten())
            elif isinstance(value, np.ndarray):
                features_list.append(torch.tensor(value, dtype=torch.float32).flatten())
            elif isinstance(value, (int, float)):
                features_list.append(torch.tensor([value], dtype=torch.float32))

        if not features_list:
            return torch.randn(1, self.feature_dim).to(DEVICE)

        combined = torch.cat(features_list)
        if combined.size(0) < self.feature_dim:
            combined = torch.cat([combined, torch.zeros(self.feature_dim - combined.size(0))])
        elif combined.size(0) > self.feature_dim:
            combined = combined[:self.feature_dim]

        return combined.unsqueeze(0).to(DEVICE)

    def _dict_to_tensor(self, data: Dict[str, Any]) -> torch.Tensor:
        values = []
        for v in data.values():
            if isinstance(v, (int, float)):
                values.append(v)
            elif isinstance(v, (list, np.ndarray)):
                values.extend(list(v))

        if not values:
            return torch.zeros(1, self.feature_dim).to(DEVICE)

        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.size(0) < self.feature_dim:
            tensor = torch.cat([tensor, torch.zeros(self.feature_dim - tensor.size(0))])
        elif tensor.size(0) > self.feature_dim:
            tensor = tensor[:self.feature_dim]

        return tensor.unsqueeze(0).to(DEVICE)

    def _ensure_feature_dim(self, features: torch.Tensor) -> torch.Tensor:
        if len(features) < self.feature_dim:
            features = torch.cat([features, torch.zeros(self.feature_dim - len(features))])
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        return features

    def _update_state_history(self, state: Dict[str, Any]):
        self.state_history.append(state)

    def _calculate_simulation_confidence(self, steps: List[Dict]) -> float:
        if not steps:
            return 0.5

        confidence_sum = 0.0
        count = 0
        for step in steps:
            if "causal_effects" in step:
                confidence_sum += step["causal_effects"].get("strength", 0.5)
                count += 1
            if "dynamics_prediction" in step:
                confidence_sum += 0.5
                count += 1

        if count == 0:
            return 0.5

        return min(1.0, confidence_sum / count)

    def _predict_next_state(self, current_state: Dict[str, Any], action: Dict) -> Tuple[Dict, float]:
        features = self._state_to_features(current_state)
        hierarchy = self.build_hierarchy(features)
        action_tensor = self._dict_to_tensor(action)
        prediction = self.predict_dynamics(hierarchy, action_tensor)

        next_state = {}
        for entity_id in current_state:
            if isinstance(current_state[entity_id], dict):
                next_state[entity_id] = {**current_state[entity_id]}
            else:
                next_state[entity_id] = current_state[entity_id]

        return next_state, prediction["object"].norm().item()

    def _correct_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def _simulate_action_effect(self, current: Dict, action: Dict) -> Dict:
        new_state = dict(current)
        target_id = action.get("target")
        if target_id:
            new_state[target_id] = {"activated": True, "confidence": action.get("confidence", 0.5)}
        return new_state

    def _generate_random_state(self) -> Dict[str, Any]:
        return {
            "agent_position": [np.random.rand(), np.random.rand(), np.random.rand()],
            "goal_position": [np.random.rand(), np.random.rand(), np.random.rand()],
            "confidence": np.random.rand()
        }

    def _generate_random_actions(self, max_steps: int) -> List[Dict]:
        actions = []
        for _ in range(max_steps):
            actions.append({
                "type": np.random.choice(["move", "rotate", "wait"]),
                "direction": [np.random.randn(), np.random.randn(), np.random.randn()],
                "magnitude": np.random.rand()
            })
        return actions

    def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        if entity_id not in self.entities:
            return {"error": "Entity not found"}

        entity = self.entities[entity_id]
        return {
            "entity_id": entity.entity_id,
            "category": entity.category.value,
            "position": entity.position.tolist(),
            "properties": {k: v.to_dict() for k, v in entity.properties.items()},
            "state": entity.state,
            "confidence": entity.confidence,
            "activation_level": entity.activation_level
        }

    def get_causal_graph(self) -> Dict[str, List[Dict]]:
        graph = defaultdict(list)
        for rel in self.causal_relations:
            graph[rel.cause_id].append({
                "effect": rel.effect_id,
                "strength": rel.strength,
                "confidence": rel.confidence
            })
        return dict(graph)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self.entities),
            "causal_relation_count": len(self.causal_relations),
            "social_rule_count": len(self.social_rules),
            "state_history_length": len(self.state_history),
            "simulation_history_length": len(self.simulation_history),
            "hierarchy_history_length": len(self.hierarchy_history),
            "working_memory_size": len(self.memory_system.working_memory),
            "episodic_memory_size": len(self.memory_system.episodic_memory),
            "semantic_memory_size": len(self.memory_system.semantic_memory),
            "modality_importance": self.multimodal_encoder.get_modality_importance(),
            "causal_graph_nodes": len(self.causal_reasoner.get_causal_graph())
        }
