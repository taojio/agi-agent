import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from ..config.settings import DEVICE
from ..core import get_adaptive_config

_adaptive_config = get_adaptive_config()


class HomeostaticNeed:
    def __init__(self, name, baseline, threshold_low, threshold_high, decay_rate=0.01, gain_rate=0.05):
        self.name = name
        self.baseline = baseline
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.decay_rate = decay_rate
        self.gain_rate = gain_rate
        self.value = baseline
        self.history = deque(maxlen=100)
        self.status = "normal"

    def update(self, delta=0.0):
        self.value = max(0.0, min(1.0, self.value + delta))
        
        if self.value < self.threshold_low:
            self.status = "deficit"
        elif self.value > self.threshold_high:
            self.status = "surplus"
        else:
            self.status = "normal"
        
        self.history.append(self.value)
        return self.status

    def decay(self):
        return self.update(-self.decay_rate)

    def gain(self):
        return self.update(self.gain_rate)

    def get_deviation(self):
        return abs(self.value - self.baseline)

    def get_priority(self):
        deviation = self.get_deviation()
        if self.status == "deficit":
            return deviation * 1.5
        elif self.status == "surplus":
            return deviation * 0.8
        return deviation * 0.3


class InternalEnergySystem:
    def __init__(self):
        self.energy = HomeostaticNeed(
            name="energy",
            baseline=_adaptive_config.get("energy_baseline", 0.7),
            threshold_low=_adaptive_config.get("energy_threshold_low", 0.3),
            threshold_high=_adaptive_config.get("energy_threshold_high", 0.9),
            decay_rate=_adaptive_config.get("energy_decay_rate", 0.005),
            gain_rate=_adaptive_config.get("energy_gain_rate", 0.08)
        )
        self.attention = HomeostaticNeed(
            name="attention",
            baseline=_adaptive_config.get("attention_baseline", 0.6),
            threshold_low=_adaptive_config.get("attention_threshold_low", 0.2),
            threshold_high=_adaptive_config.get("attention_threshold_high", 0.85),
            decay_rate=_adaptive_config.get("attention_decay_rate", 0.01),
            gain_rate=_adaptive_config.get("attention_gain_rate", 0.06)
        )
        self.security = HomeostaticNeed(
            name="security",
            baseline=_adaptive_config.get("security_baseline", 0.8),
            threshold_low=_adaptive_config.get("security_threshold_low", 0.4),
            threshold_high=_adaptive_config.get("security_threshold_high", 0.95),
            decay_rate=_adaptive_config.get("security_decay_rate", 0.003),
            gain_rate=_adaptive_config.get("security_gain_rate", 0.07)
        )
        self.curiosity = HomeostaticNeed(
            name="curiosity",
            baseline=_adaptive_config.get("curiosity_baseline", 0.5),
            threshold_low=_adaptive_config.get("curiosity_threshold_low", 0.2),
            threshold_high=_adaptive_config.get("curiosity_threshold_high", 0.8),
            decay_rate=_adaptive_config.get("curiosity_decay_rate", 0.008),
            gain_rate=_adaptive_config.get("curiosity_gain_rate", 0.04)
        )
        self.competence = HomeostaticNeed(
            name="competence",
            baseline=_adaptive_config.get("competence_baseline", 0.5),
            threshold_low=_adaptive_config.get("competence_threshold_low", 0.25),
            threshold_high=_adaptive_config.get("competence_threshold_high", 0.85),
            decay_rate=_adaptive_config.get("competence_decay_rate", 0.002),
            gain_rate=_adaptive_config.get("competence_gain_rate", 0.03)
        )

        self.needs = {
            "energy": self.energy,
            "attention": self.attention,
            "security": self.security,
            "curiosity": self.curiosity,
            "competence": self.competence
        }

    def update_all(self, activities=None):
        if activities is None:
            activities = {}

        for name, need in self.needs.items():
            if name in activities:
                need.update(activities[name])
            else:
                need.decay()

    def get_highest_priority_need(self):
        priorities = {name: need.get_priority() for name, need in self.needs.items()}
        return max(priorities, key=priorities.get), priorities[max(priorities, key=priorities.get)]

    def get_need_status(self):
        return {name: {
            "value": need.value,
            "status": need.status,
            "priority": need.get_priority(),
            "deviation": need.get_deviation()
        } for name, need in self.needs.items()}

    def consume_energy(self, amount):
        self.energy.update(-amount)

    def restore_energy(self, amount):
        self.energy.update(amount)


class GenerativeModel(nn.Module):
    def __init__(self, feature_dim=16, action_dim=8, latent_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        ).to(DEVICE)
        
        self.state_transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ).to(DEVICE)
        
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, feature_dim)
        ).to(DEVICE)
        
        self.precision_encoder = nn.Sequential(
            nn.Linear(feature_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Softplus()
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def encode(self, state):
        return self.state_encoder(state)
    
    def transition(self, latent_state, action):
        while latent_state.dim() > action.dim():
            latent_state = latent_state.squeeze(0)
        while action.dim() > latent_state.dim():
            action = action.squeeze(0)
        
        combined = torch.cat([latent_state, action], dim=-1)
        return self.state_transition(combined)
    
    def decode(self, latent_state):
        return self.state_decoder(latent_state)
    
    def predict(self, current_state, action):
        latent = self.encode(current_state)
        next_latent = self.transition(latent, action)
        return self.decode(next_latent)
    
    def forward(self, state, action):
        return self.predict(state, action)
    
    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(new_feature_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU()
        ).to(DEVICE)
        self.state_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, new_feature_dim)
        ).to(DEVICE)
        self.precision_encoder = nn.Sequential(
            nn.Linear(new_feature_dim, self.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Softplus()
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)


class ActiveInferenceEngine:
    def __init__(self, feature_dim=16, action_dim=8):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        self.generative_model = GenerativeModel(feature_dim, action_dim)
        
        history_len = _adaptive_config.get("prediction_history_len", 200)
        self.prediction_error_history = deque(maxlen=history_len)
        self.free_energy_history = deque(maxlen=history_len)
        self.variational_params = deque(maxlen=_adaptive_config.get("variational_params_len", 100))
        
        self.action_noise_scale = _adaptive_config.get("action_noise_scale", 0.1)
        self.learning_rate = _adaptive_config.get("active_inference_lr", 1e-3)
        self.epistemic_weight = _adaptive_config.get("epistemic_weight", 0.5)
        
        self.beta = _adaptive_config.get("active_inference_beta", 1.0)

    def compute_variational_free_energy(self, current_state, action, actual_next_state=None):
        state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        latent = self.generative_model.encode(state)
        next_latent = self.generative_model.transition(latent, action_tensor)
        predicted_state = self.generative_model.decode(next_latent)
        
        if actual_next_state is not None:
            actual = torch.tensor(actual_next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prediction_error = nn.MSELoss()(predicted_state, actual).item()
        else:
            prediction_error = 0.0
        #确保torch.mean和torch.sum计算结果正确转换为Python标量类型
        prediction_error = float(prediction_error)
        
        latent_entropy = -0.5 * float(torch.sum(1 + torch.log(torch.var(latent, dim=1) + 1e-10)).item())
        
        kl_div = 0.0
        if len(self.variational_params) > 0:
            prev_latent = torch.tensor(self.variational_params[-1], dtype=torch.float32).to(DEVICE)
            kl_div = float(torch.mean(torch.log((torch.var(latent, dim=1) + 1e-10) / (torch.var(prev_latent, dim=0) + 1e-10))).item())
        
        precision = self.generative_model.precision_encoder(state).item()
        
        fe = (precision * prediction_error) + kl_div - latent_entropy
        
        self.free_energy_history.append(fe)
        
        return fe, prediction_error, precision

    def compute_epistemic_value(self, current_state, candidate_actions, num_samples=5):
        epistemic_values = []
        
        for action in candidate_actions:
            state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            latent_samples = []
            for _ in range(num_samples):
                noise = torch.randn_like(state) * 0.1
                noisy_state = state + noise
                latent = self.generative_model.encode(noisy_state)
                next_latent = self.generative_model.transition(latent, action_tensor)
                latent_samples.append(next_latent.detach().cpu().numpy())
            
            latent_samples = np.array(latent_samples)
            entropy = -np.mean(np.sum(latent_samples * np.log(latent_samples + 1e-10), axis=-1))
            
            epistemic_values.append(entropy)
        
        return np.array(epistemic_values)

    def generate_action_via_gradient(self, current_state, steps=10):
        state_np = np.array(current_state)
        if state_np.ndim == 1:
            state_np = state_np.reshape(1, -1)
        elif state_np.ndim > 2:
            state_np = state_np.flatten().reshape(1, -1)
        
        state = torch.tensor(state_np, dtype=torch.float32).to(DEVICE)
        action = torch.randn(self.action_dim, requires_grad=True).to(DEVICE)
        
        optimizer = optim.Adam([action], lr=self.learning_rate)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            latent = self.generative_model.encode(state)
            next_latent = self.generative_model.transition(latent, action.unsqueeze(0))
            predicted = self.generative_model.decode(next_latent)
            prediction_loss = torch.mean(torch.square(predicted - state))
            
            latent_variance = torch.var(next_latent)
            epistemic_reward = self.epistemic_weight * latent_variance
            
            total_loss = prediction_loss - epistemic_reward
            
            total_loss.backward()
            optimizer.step()
            
            action.data = torch.clamp(action.data, -1.0, 1.0)
        
        self.variational_params.append(next_latent.detach().cpu().numpy().flatten())
        
        return action.detach().cpu().numpy()

    def predict_next_state(self, current_state, action):
        state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        return self.generative_model.predict(state, action_tensor).detach().cpu().numpy().flatten()

    def update_prediction_model(self, current_state, action, next_state):
        state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        target = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        self.generative_model.optimizer.zero_grad()
        
        predicted = self.generative_model.predict(state, action_tensor)
        
        reconstruction_loss = nn.MSELoss()(predicted, target)
        latent = self.generative_model.encode(state)
        regularization = 1e-4 * torch.mean(torch.square(latent))
        
        loss = reconstruction_loss + regularization
        loss.backward()
        self.generative_model.optimizer.step()
        
        error = float(reconstruction_loss.item())
        self.prediction_error_history.append(error)

    def compute_prediction_error(self, predicted, actual):
        return np.mean(np.square(predicted - actual))

    def compute_free_energy(self, prediction_error, entropy=0.0):
        precision = 1.0
        fe = precision * prediction_error + entropy
        self.free_energy_history.append(fe)
        return fe

    def generate_action_to_minimize_fe(self, current_state):
        return self.generate_action_via_gradient(current_state)

    def get_prediction_accuracy(self):
        if len(self.prediction_error_history) < 10:
            return 0.5
        recent_errors = list(self.prediction_error_history)[-10:]
        avg_error = np.mean(recent_errors)
        return max(0.0, min(1.0, 1.0 - avg_error))

    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.generative_model.resize(new_feature_dim)


class ValueSystem:
    def __init__(self):
        self.hierarchical_values = {
            'survival': {
                'energy_conservation': 2.0,
                'security': 1.8,
                'attention': 1.2
            },
            'growth': {
                'competence_gain': 1.5,
                'learning': 1.3,
                'exploration': 1.0
            },
            'curiosity': {
                'novelty': 0.8,
                'information_gain': 0.7
            }
        }
        
        self.survival_veto_threshold = 0.3
        
        self.discount_factor = 0.95
        self.time_horizon = 10
        
        self.value_history = deque(maxlen=200)
        self.experience_buffer = deque(maxlen=500)
        
        self.need_sensitivity = {
            'energy': 0.8,
            'attention': 0.6,
            'security': 1.0,
            'curiosity': 0.4,
            'competence': 0.5
        }

    def compute_intrinsic_value(self, fe, novelty, energy_level, competence_gain, 
                                exploration_bonus=0.0, needs_status=None, time_step=0):
        survival_value = 0.0
        growth_value = 0.0
        curiosity_value = 0.0
        
        if needs_status:
            survival_value = (
                self.hierarchical_values['survival']['energy_conservation'] * needs_status['energy']['value'] +
                self.hierarchical_values['survival']['security'] * needs_status['security']['value'] +
                self.hierarchical_values['survival']['attention'] * needs_status['attention']['value']
            )
            
            growth_value = (
                self.hierarchical_values['growth']['competence_gain'] * competence_gain +
                self.hierarchical_values['growth']['exploration'] * exploration_bonus
            )
            
            curiosity_value = (
                self.hierarchical_values['curiosity']['novelty'] * novelty
            )
        else:
            survival_value = self.hierarchical_values['survival']['energy_conservation'] * energy_level
            growth_value = self.hierarchical_values['growth']['competence_gain'] * competence_gain
            curiosity_value = self.hierarchical_values['curiosity']['novelty'] * novelty
        
        temporal_discount = self.discount_factor ** time_step
        
        survival_ok = energy_level > self.survival_veto_threshold if needs_status is None else \
                      needs_status['energy']['value'] > self.survival_veto_threshold
        
        if not survival_ok:
            total_value = survival_value * temporal_discount
        else:
            total_value = (survival_value * 0.3 + growth_value * 0.5 + curiosity_value * 0.2) * temporal_discount
        
        total_value += -1.0 * fe
        
        self.value_history.append(total_value)
        return total_value

    def learn_value_weights(self, outcome, expected_value, actual_reward, needs_status=None):
        error = actual_reward - expected_value
        
        learning_rate = 0.01
        
        if needs_status:
            for need_name, sensitivity in self.need_sensitivity.items():
                need_deviation = needs_status[need_name].get('deviation', 0.0)
                adaptive_lr = learning_rate * (1 + need_deviation)
                
                if need_name == 'energy' or need_name == 'security':
                    self.hierarchical_values['survival'][need_name] = \
                        self.hierarchical_values['survival'].get(need_name, 1.0) + adaptive_lr * error * sensitivity
                elif need_name == 'competence':
                    self.hierarchical_values['growth']['competence_gain'] += adaptive_lr * error * sensitivity
                elif need_name == 'curiosity':
                    self.hierarchical_values['curiosity']['novelty'] += adaptive_lr * error * sensitivity
        else:
            for level, weights in self.hierarchical_values.items():
                for key in weights:
                    weights[key] += learning_rate * error * 0.1

        for level in self.hierarchical_values:
            self.hierarchical_values[level] = {
                k: max(-3.0, min(3.0, v)) 
                for k, v in self.hierarchical_values[level].items()
            }

    def record_experience(self, context, action, outcome, value):
        experience = {
            "context": context,
            "action": action,
            "outcome": outcome,
            "value": value,
            "timestamp": np.random.randint(1000000)
        }
        self.experience_buffer.append(experience)

    def get_value_preferences(self):
        flat_weights = {}
        for level, weights in self.hierarchical_values.items():
            for key, value in weights.items():
                flat_weights[f"{level}_{key}"] = value
        return flat_weights

    def get_hierarchical_preferences(self):
        return dict(self.hierarchical_values)


class AutonomousGoalGenerator:
    def __init__(self, feature_dim=16):
        self.internal_energy = InternalEnergySystem()
        self.active_inference = ActiveInferenceEngine(feature_dim=feature_dim)
        self.value_system = ValueSystem()
        
        self.active_goals = []
        self.goal_history = deque(maxlen=200)
        self.goal_priority_threshold = 0.3
        self.max_active_goals = 5
        
        self.self_model = None
        
        self.goal_hierarchy = {
            'survival': ['maintain_energy', 'ensure_security'],
            'growth': ['improve_competence', 'learn_new_skill'],
            'exploration': ['explore_unknown', 'acquire_information'],
            'cognitive': ['reduce_free_energy', 'improve_prediction']
        }

    def set_self_model(self, self_model):
        self.self_model = self_model

    def _extract_internal_state_vector(self):
        needs = self.internal_energy.get_need_status()
        return np.array([
            needs['energy']['value'],
            needs['attention']['value'],
            needs['security']['value'],
            needs['curiosity']['value'],
            needs['competence']['value']
        ])

    def _detect_emergent_goals_from_trajectory(self, trajectory, free_energy, novelty, confidence):
        emergent_goals = []
        
        if self.self_model is not None:
            problems = self.self_model.detect_future_problems(trajectory)
            
            for problem in problems:
                goal = self._derive_goal_from_problem(problem, free_energy, novelty)
                if goal:
                    emergent_goals.append(goal)
        
        if free_energy > 0.4:
            fe_goal = {
                "id": f"goal_fe_{len(self.goal_history) + 1}",
                "type": "reduce_free_energy",
                "priority": free_energy * 0.9,
                "context": {
                    "free_energy": free_energy,
                    "source": "emergent",
                    "severity": "high" if free_energy > 0.7 else "medium"
                },
                "target": "minimize_prediction_error",
                "status": "active",
                "urgency": min(1.0, free_energy),
                "hierarchy": "cognitive"
            }
            emergent_goals.append(fe_goal)
        
        if novelty > 0.5 and confidence < 0.7:
            exp_goal = {
                "id": f"goal_explore_{len(self.goal_history) + 1}",
                "type": "explore",
                "priority": novelty * 0.8,
                "context": {
                    "novelty": novelty,
                    "confidence": confidence,
                    "source": "emergent",
                    "reason": "high_novelty_low_confidence"
                },
                "target": "acquire_new_information",
                "status": "active",
                "urgency": novelty * (1.0 - confidence),
                "hierarchy": "exploration"
            }
            emergent_goals.append(exp_goal)
        
        if confidence < 0.4:
            learn_goal = {
                "id": f"goal_learn_{len(self.goal_history) + 1}",
                "type": "learn",
                "priority": (1.0 - confidence) * 0.85,
                "context": {
                    "confidence": confidence,
                    "source": "emergent",
                    "reason": "low_confidence"
                },
                "target": "improve_prediction_accuracy",
                "status": "active",
                "urgency": 1.0 - confidence,
                "hierarchy": "growth"
            }
            emergent_goals.append(learn_goal)
        
        return emergent_goals

    def _derive_goal_from_problem(self, problem, free_energy, novelty):
        if isinstance(problem, str):
            return {
                "id": f"goal_str_{len(self.goal_history) + 1}",
                "type": "address_issue",
                "priority": 0.5,
                "context": {"problem": problem, "source": "emergent"},
                "target": "monitor_system",
                "status": "active",
                "urgency": 0.5,
                "hierarchy": "cognitive"
            }
        need_name = problem.get('need', 'unknown')
        severity = problem.get('severity', 0.5)
        urgency = problem.get('urgency', 0.5)
        
        goal_type_map = {
            'energy': 'maintain_energy',
            'attention': 'allocate_attention',
            'security': 'ensure_security',
            'curiosity': 'satisfy_curiosity',
            'competence': 'improve_competence'
        }
        
        target_map = {
            'energy': 'conserve_energy' if severity < 0.5 else 'find_energy_source',
            'attention': 'rest_attention' if severity < 0.5 else 'focus_on_task',
            'security': 'maintain_safety' if severity < 0.5 else 'avoid_danger',
            'curiosity': 'deepen_understanding' if severity < 0.5 else 'explore_unknown',
            'competence': 'practice_existing' if severity < 0.5 else 'learn_new_skill'
        }
        
        hierarchy_map = {
            'energy': 'survival',
            'attention': 'survival',
            'security': 'survival',
            'curiosity': 'exploration',
            'competence': 'growth'
        }
        
        goal_type = goal_type_map.get(need_name, 'maintain_energy')
        target = target_map.get(need_name, 'maintain_state')
        hierarchy = hierarchy_map.get(need_name, 'survival')
        
        priority = severity * urgency * (1.5 if hierarchy == 'survival' else 1.0)
        
        return {
            "id": f"goal_{need_name}_{len(self.goal_history) + 1}",
            "type": goal_type,
            "priority": priority,
            "context": {
                "need": need_name,
                "severity": severity,
                "urgency": urgency,
                "source": "self_model_prediction",
                "timestep": problem.get('timestep', 0)
            },
            "target": target,
            "status": "active",
            "urgency": urgency,
            "hierarchy": hierarchy,
            "derived_from": "self_model_trajectory"
        }

    def _resolve_goal_conflicts(self, goals):
        survival_goals = [g for g in goals if g.get('hierarchy') == 'survival']
        other_goals = [g for g in goals if g.get('hierarchy') != 'survival']
        
        if survival_goals:
            survival_goals.sort(key=lambda x: x['priority'], reverse=True)
            selected_survival = survival_goals[:2]
            
            needs_status = self.internal_energy.get_need_status()
            energy_ok = needs_status['energy']['value'] > 0.3
            security_ok = needs_status['security']['value'] > 0.3
            
            if energy_ok and security_ok:
                other_goals.sort(key=lambda x: x['priority'], reverse=True)
                selected_other = other_goals[:3]
                return selected_survival + selected_other
            else:
                return selected_survival
        else:
            goals.sort(key=lambda x: x['priority'], reverse=True)
            return goals[:self.max_active_goals]

    def _create_hierarchical_subgoals(self, goal):
        subgoals = []
        
        capability_assessment = None
        if self.self_model is not None:
            if hasattr(self.self_model, 'assess_capability_for_goal'):
                capability_assessment = self.self_model.assess_capability_for_goal(goal['type'])
            elif hasattr(self.self_model, 'assess_capability'):
                raw = self.self_model.assess_capability(goal['type'])
                capability_assessment = {
                    'feasibility': raw.get('success_rate', 0.5),
                    'capability_gaps': {'practice': 0.3} if raw.get('is_weakness') else {}
                }
        
        if capability_assessment and capability_assessment.get('feasibility', 1.0) < 0.5:
            for cap_name, gap in capability_assessment['capability_gaps'].items():
                if gap > 0.2:
                    subgoal = {
                        "id": f"subgoal_{cap_name}_{len(self.goal_history) + 1}",
                        "type": f"acquire_{cap_name}",
                        "priority": goal['priority'] * 0.7,
                        "context": {
                            "parent_goal": goal['id'],
                            "capability_gap": gap,
                            "reason": "capability_deficit"
                        },
                        "target": f"improve_{cap_name}_capability",
                        "status": "active",
                        "urgency": goal['urgency'] * 0.8,
                        "hierarchy": goal.get('hierarchy', 'growth'),
                        "is_subgoal": True,
                        "parent_goal_id": goal['id']
                    }
                    subgoals.append(subgoal)
        
        return subgoals

    def generate_goals(self, current_state, free_energy, novelty, confidence):
        internal_state_vec = self._extract_internal_state_vector()
        
        trajectory = None
        if self.self_model is not None:
            trajectory = self.self_model.predict_self_trajectory(internal_state_vec)
        
        emergent_goals = self._detect_emergent_goals_from_trajectory(
            trajectory, free_energy, novelty, confidence
        )
        
        resolved_goals = self._resolve_goal_conflicts(emergent_goals)
        
        all_goals = []
        for goal in resolved_goals:
            all_goals.append(goal)
            subgoals = self._create_hierarchical_subgoals(goal)
            all_goals.extend(subgoals)
        
        for goal in all_goals:
            if goal["id"] not in [g["id"] for g in self.active_goals]:
                self.active_goals.append(goal)
                self.goal_history.append(goal)
        
        self._prune_low_priority_goals()
        
        return all_goals

    def _prune_low_priority_goals(self):
        self.active_goals.sort(key=lambda x: x["priority"], reverse=True)
        
        self.active_goals = self.active_goals[:self.max_active_goals]

    def update_goal_status(self, goal_id, status, result=None):
        for goal in self.active_goals:
            if goal["id"] == goal_id:
                goal["status"] = status
                if result is not None:
                    goal["result"] = result
                break

    def complete_goal(self, goal_id, result):
        self.update_goal_status(goal_id, "completed", result)
        
        for goal in self.active_goals:
            if goal.get('parent_goal_id') == goal_id:
                goal['status'] = 'completed'

    def fail_goal(self, goal_id, reason):
        self.update_goal_status(goal_id, "failed", reason)

    def get_highest_priority_goal(self):
        if not self.active_goals:
            return None
        return max(self.active_goals, key=lambda x: x["priority"])

    def update_internal_state(self, activities):
        self.internal_energy.update_all(activities)

    def get_homeostatic_state(self):
        return {
            "needs": self.internal_energy.get_need_status(),
            "active_goals": self.active_goals,
            "value_weights": self.value_system.get_value_preferences(),
            "prediction_accuracy": self.active_inference.get_prediction_accuracy()
        }


class HomeostasisEngine:
    def __init__(self, feature_dim=16):
        self.goal_generator = AutonomousGoalGenerator(feature_dim=feature_dim)
        self.feature_dim = feature_dim
        
        self.self_model = None

    def set_self_model(self, self_model):
        self.self_model = self_model
        self.goal_generator.set_self_model(self_model)

    def step(self, current_state, free_energy, novelty, confidence, action_result=None):
        needs_status = self.goal_generator.internal_energy.get_need_status()
        
        activity_updates = {}
        if free_energy < 0.3:
            activity_updates["competence"] = 0.02
        if novelty > 0.5:
            activity_updates["curiosity"] = 0.03
        if confidence > 0.7:
            activity_updates["competence"] = 0.01
        
        self.goal_generator.update_internal_state(activity_updates)

        if action_result is not None:
            predicted = self.goal_generator.active_inference.predict_next_state(
                current_state, action_result
            )
            prediction_error = self.goal_generator.active_inference.compute_prediction_error(
                predicted, current_state
            )
            self.goal_generator.active_inference.prediction_error_history.append(prediction_error)
            self.goal_generator.active_inference.update_prediction_model(
                current_state, action_result, current_state
            )

            value = self.goal_generator.value_system.compute_intrinsic_value(
                free_energy, novelty, needs_status["energy"]["value"], 
                confidence - 0.5, needs_status=needs_status
            )
            self.goal_generator.value_system.record_experience(
                context={"fe": free_energy, "novelty": novelty},
                action=action_result,
                outcome=current_state,
                value=value
            )

        new_goals = self.goal_generator.generate_goals(current_state, free_energy, novelty, confidence)
        
        highest_goal = self.goal_generator.get_highest_priority_goal()
        
        self_reflection = None
        if self.self_model is not None:
            internal_state_vec = self.goal_generator._extract_internal_state_vector()
            trajectory = self.self_model.predict_self_trajectory(internal_state_vec)
            self_reflection = self.self_model.generate_self_reflection(
                internal_state_vec, trajectory, action_result
            )
        
        return {
            "new_goals": new_goals,
            "highest_priority_goal": highest_goal,
            "homeostatic_state": self.goal_generator.get_homeostatic_state(),
            "predicted_action": self.goal_generator.active_inference.generate_action_to_minimize_fe(current_state),
            "self_reflection": self_reflection,
            "self_model_state": self.self_model.get_self_model_state() if self.self_model else None
        }

    def get_needs_status(self):
        return self.goal_generator.internal_energy.get_need_status()

    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.goal_generator.active_inference.resize(new_feature_dim)
        if self.self_model is not None:
            self.self_model.resize(new_feature_dim)