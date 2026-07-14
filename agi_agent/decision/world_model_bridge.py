import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..cognitive.world_model import WorldModelEngine, EntityCategory, SimulationResult, CausalRelation


class BridgeMode(Enum):
    PREDICTION = "prediction"
    SIMULATION = "simulation"
    PLANNING = "planning"
    CAUSAL_ANALYSIS = "causal_analysis"


@dataclass
class PredictionResult:
    predicted_state: Dict[str, Any]
    confidence: float
    dynamics_metrics: Dict[str, float]
    causal_effects: List[Dict[str, Any]]
    horizon: int = 1


@dataclass
class SimulationScenario:
    scenario_id: str
    initial_state: Dict[str, Any]
    actions: List[Dict[str, Any]]
    simulation_result: Optional[SimulationResult] = None
    predicted_outcomes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionSupportInfo:
    world_state: Dict[str, Any]
    predicted_next_states: List[PredictionResult]
    causal_graph: Dict[str, List[Dict]]
    recommended_actions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    confidence_bounds: Tuple[float, float]


class WorldModelDecisionBridge:
    def __init__(self, world_model: WorldModelEngine, feature_dim: int = 16):
        self.world_model = world_model
        self.feature_dim = feature_dim
        
        self._prediction_cache: deque = deque(maxlen=50)
        self._scenario_history: deque = deque(maxlen=20)
        self._causal_graph_cache: Optional[Dict] = None
        self._last_update_time = 0.0
        self._update_interval = 0.5

    def _features_to_state(self, features: np.ndarray) -> Dict[str, Any]:
        state = {}
        num_chunks = min(len(features), 8)
        chunk_size = len(features) // num_chunks if num_chunks > 0 else 1
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(features))
            chunk = features[start:end]
            state[f"feature_group_{i}"] = {
                "mean": float(np.mean(chunk)),
                "std": float(np.std(chunk)),
                "norm": float(np.linalg.norm(chunk)),
                "active": float(np.mean(chunk) > 0.1)
            }
        
        return state

    def _state_to_features(self, state: Dict[str, Any]) -> np.ndarray:
        values = []
        for key, value in state.items():
            if isinstance(value, dict):
                values.extend([v for v in value.values() if isinstance(v, (int, float))])
            elif isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return np.zeros(self.feature_dim)
        
        arr = np.array(values, dtype=np.float32)
        if len(arr) < self.feature_dim:
            arr = np.pad(arr, (0, self.feature_dim - len(arr)), mode='constant')
        elif len(arr) > self.feature_dim:
            arr = arr[:self.feature_dim]
        
        return arr

    def predict_next_state(self, current_features: np.ndarray, 
                           action_vector: Optional[np.ndarray] = None,
                           horizon: int = 1) -> PredictionResult:
        if action_vector is None:
            action_vector = np.zeros(self.feature_dim)
        
        action_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)
        feat_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0)
        
        encoded = self.world_model.encode_multimodal_input({"sensor": feat_tensor})
        hierarchy = self.world_model.build_hierarchy(encoded)
        dynamics = self.world_model.predict_dynamics(hierarchy, action_tensor)
        
        predicted_feat = dynamics["object"].detach().cpu().numpy().flatten()
        predicted_state = self._features_to_state(predicted_feat)
        
        dynamics_metrics = {
            "perceptual_norm": dynamics["perceptual"].norm().item(),
            "object_norm": dynamics["object"].norm().item(),
            "scene_norm": dynamics["scene"].norm().item(),
            "confidence": float(dynamics["object"].norm().item() / np.sqrt(self.feature_dim))
        }
        
        causal_effects = []
        for rel in self.world_model.causal_relations:
            causal_effects.append({
                "cause": rel.cause_id,
                "effect": rel.effect_id,
                "strength": rel.strength,
                "confidence": rel.confidence
            })
        
        result = PredictionResult(
            predicted_state=predicted_state,
            confidence=dynamics_metrics["confidence"],
            dynamics_metrics=dynamics_metrics,
            causal_effects=causal_effects,
            horizon=horizon
        )
        
        self._prediction_cache.append({
            "timestamp": self._last_update_time,
            "result": result
        })
        
        return result

    def simulate_scenarios(self, current_features: np.ndarray,
                           candidate_actions: List[Dict[str, Any]],
                           max_steps: int = 10) -> List[SimulationScenario]:
        scenarios = []
        entity_id = f"bridge_entity_{id(self)}_{len(self._scenario_history)}"
        
        self.world_model.add_entity(
            entity_id=entity_id,
            category=EntityCategory.AGENT,
            features=current_features
        )
        
        initial_state = {
            entity_id: {
                "features": current_features.tolist(),
                "confidence": 1.0,
                "activated": True
            }
        }
        
        for action in candidate_actions:
            scenario = SimulationScenario(
                scenario_id=f"scenario_{entity_id}_{action.get('name', 'unknown')}",
                initial_state=dict(initial_state),
                actions=[action]
            )
            
            actions_list = [{
                "entity_id": entity_id,
                "action": action.get("action", "execute"),
                "parameters": action.get("parameters", {}),
                "confidence": action.get("confidence", 0.5)
            }]
            
            simulation = self.world_model.simulate_scenario(
                scenario_id=scenario.scenario_id,
                initial_state=initial_state,
                actions=actions_list,
                max_steps=max_steps
            )
            
            scenario.simulation_result = simulation
            
            predicted_outcomes = []
            for step in simulation.steps:
                outcome = {
                    "step": step["step"],
                    "state": step.get("state", {}),
                    "action": step.get("action", {}),
                    "confidence": step.get("dynamics_prediction", {}).get("object_norm", 0.5),
                    "causal_effects": step.get("causal_effects", {})
                }
                predicted_outcomes.append(outcome)
            
            scenario.predicted_outcomes = predicted_outcomes
            scenarios.append(scenario)
        
        self._scenario_history.extend(scenarios)
        return scenarios

    def get_decision_support(self, current_features: np.ndarray,
                             available_actions: List[Dict[str, Any]],
                             goal_state: Optional[Dict[str, Any]] = None) -> DecisionSupportInfo:
        world_state = self._features_to_state(current_features)
        
        predicted_next_states = []
        for action in available_actions:
            action_vec = np.array(action.get("features", np.zeros(self.feature_dim)))
            prediction = self.predict_next_state(current_features, action_vec)
            predicted_next_states.append(prediction)
        
        causal_graph = self.world_model.get_causal_graph()
        
        recommended_actions = self._rank_actions_by_prediction(
            available_actions, predicted_next_states, goal_state
        )
        
        risk_assessment = self._assess_risk(predicted_next_states)
        
        confidences = [p.confidence for p in predicted_next_states]
        confidence_bounds = (min(confidences) if confidences else 0.0,
                             max(confidences) if confidences else 1.0)
        
        return DecisionSupportInfo(
            world_state=world_state,
            predicted_next_states=predicted_next_states,
            causal_graph=causal_graph,
            recommended_actions=recommended_actions,
            risk_assessment=risk_assessment,
            confidence_bounds=confidence_bounds
        )

    def _rank_actions_by_prediction(self, actions: List[Dict[str, Any]],
                                    predictions: List[PredictionResult],
                                    goal_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        
        for action, prediction in zip(actions, predictions):
            score = prediction.confidence
            
            if goal_state:
                score += self._goal_alignment_score(prediction.predicted_state, goal_state)
            
            causal_bonus = sum(ce["strength"] for ce in prediction.causal_effects)
            score += causal_bonus * 0.1
            
            ranked.append({
                **action,
                "predicted_confidence": prediction.confidence,
                "alignment_score": score,
                "rank": len(ranked) + 1
            })
        
        ranked.sort(key=lambda x: x["alignment_score"], reverse=True)
        for i, action in enumerate(ranked):
            action["rank"] = i + 1
        
        return ranked

    def _goal_alignment_score(self, predicted_state: Dict[str, Any],
                              goal_state: Dict[str, Any]) -> float:
        score = 0.0
        total = 0
        
        for key, target_value in goal_state.items():
            if key in predicted_state:
                predicted_value = predicted_state[key]
                if isinstance(predicted_value, dict) and isinstance(target_value, dict):
                    for sub_key, sub_target in target_value.items():
                        if sub_key in predicted_value:
                            p_val = predicted_value[sub_key]
                            t_val = sub_target
                            if isinstance(p_val, (int, float)) and isinstance(t_val, (int, float)):
                                score += 1.0 - abs(p_val - t_val)
                                total += 1
                elif isinstance(predicted_value, (int, float)) and isinstance(target_value, (int, float)):
                    score += 1.0 - abs(predicted_value - target_value)
                    total += 1
        
        return score / max(total, 1)

    def _assess_risk(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        risks = {
            "high_confidence_risk": 0.0,
            "low_confidence_risk": 0.0,
            "causal_chain_risk": 0.0,
            "overall_risk": 0.0
        }
        
        if not predictions:
            return risks
        
        confidences = [p.confidence for p in predictions]
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        risks["low_confidence_risk"] = max(0.0, 1.0 - avg_confidence)
        risks["high_confidence_risk"] = std_confidence
        
        causal_strengths = []
        for p in predictions:
            causal_strengths.extend([ce["strength"] for ce in p.causal_effects])
        
        if causal_strengths:
            avg_causal = np.mean(causal_strengths)
            risks["causal_chain_risk"] = max(0.0, 0.5 - avg_causal)
        
        risks["overall_risk"] = (
            risks["low_confidence_risk"] * 0.4 +
            risks["high_confidence_risk"] * 0.3 +
            risks["causal_chain_risk"] * 0.3
        )
        
        return risks

    def plan_to_goal(self, current_features: np.ndarray,
                     goal_state: Dict[str, Any],
                     max_planning_steps: int = 20) -> List[Dict[str, Any]]:
        current_state = self._features_to_state(current_features)
        plan = self.world_model.plan_long_term(goal_state, current_state, max_planning_steps)
        
        for step in plan:
            if "confidence" not in step:
                step["confidence"] = 0.5
        
        return plan

    def get_causal_graph_summary(self) -> Dict[str, Any]:
        if self._causal_graph_cache is None or time.time() - self._last_update_time > self._update_interval:
            self._causal_graph_cache = self.world_model.get_causal_graph()
            self._last_update_time = time.time()
        
        return {
            "nodes": list(self._causal_graph_cache.keys()),
            "edges": sum(len(edges) for edges in self._causal_graph_cache.values()),
            "graph": self._causal_graph_cache
        }

    def get_simulation_history(self) -> List[Dict[str, Any]]:
        history = []
        for scenario in self._scenario_history:
            history.append({
                "scenario_id": scenario.scenario_id,
                "action_count": len(scenario.actions),
                "predicted_outcomes": scenario.predicted_outcomes,
                "confidence": scenario.simulation_result.confidence if scenario.simulation_result else 0.5
            })
        return history

    def reset_cache(self):
        self._prediction_cache.clear()
        self._scenario_history.clear()
        self._causal_graph_cache = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "prediction_cache_size": len(self._prediction_cache),
            "scenario_history_size": len(self._scenario_history),
            "causal_graph_available": self._causal_graph_cache is not None,
            "world_model_summary": self.world_model.get_summary()
        }


import time
