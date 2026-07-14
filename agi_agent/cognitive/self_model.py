import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from ..config.settings import DEVICE


class InternalStatePredictor(nn.Module):
    def __init__(self, state_dim=5, horizon=10):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        
        self.recurrent_layer = nn.LSTM(
            input_size=state_dim + 8,
            hidden_size=state_dim * 2,
            num_layers=2,
            batch_first=True
        ).to(DEVICE)
        
        self.output_layer = nn.Linear(state_dim * 2, state_dim).to(DEVICE)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        self.training_history = deque(maxlen=500)
        self.prediction_errors = deque(maxlen=200)
    
    def forward(self, current_state, actions=None, steps=1):
        state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        predictions = []
        
        if actions is None:
            actions = torch.zeros(steps, 8).to(DEVICE)
        
        hidden = None
        for t in range(steps):
            action_t = actions[t].unsqueeze(0) if steps > 1 else actions.unsqueeze(0)
            input_t = torch.cat([state, action_t], dim=1)
            input_t = input_t.unsqueeze(0)
            
            output, hidden = self.recurrent_layer(input_t, hidden)
            pred = self.output_layer(output[:, -1, :])
            state = pred
            predictions.append(pred.detach().cpu().numpy().flatten())
        
        return np.array(predictions)
    
    def train_step(self, state_sequence, action_sequence, target_sequence):
        self.optimizer.zero_grad()
        
        states = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        actions = torch.tensor(action_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        targets = torch.tensor(target_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        inputs = torch.cat([states[:, :-1], actions], dim=2)
        outputs, _ = self.recurrent_layer(inputs)
        preds = self.output_layer(outputs)
        
        loss = nn.MSELoss()(preds, targets[:, 1:])
        loss.backward()
        self.optimizer.step()
        
        error = loss.item()
        self.prediction_errors.append(error)
        self.training_history.append(error)
        
        return error
    
    def get_prediction_accuracy(self):
        if len(self.prediction_errors) < 10:
            return 0.5
        recent_errors = list(self.prediction_errors)[-10:]
        avg_error = np.mean(recent_errors)
        return max(0.0, min(1.0, 1.0 - avg_error * 10))


class CompetenceAssessor:
    def __init__(self, num_capabilities=5):
        self.num_capabilities = num_capabilities
        self.capability_names = ["perception", "cognition", "action", "learning", "adaptation"]
        
        self.success_counts = {name: deque(maxlen=100) for name in self.capability_names}
        self.attempt_counts = {name: deque(maxlen=100) for name in self.capability_names}
        
        self.capability_estimates = {name: 0.5 for name in self.capability_names}
        self.capability_variances = {name: 0.25 for name in self.capability_names}
        
        self.last_update_step = 0
    
    def record_attempt(self, capability_name, success, confidence=1.0):
        if capability_name in self.success_counts:
            self.success_counts[capability_name].append(1 if success else 0)
            self.attempt_counts[capability_name].append(1)
    
    def update_estimates(self):
        for name in self.capability_names:
            successes = list(self.success_counts[name])
            attempts = list(self.attempt_counts[name])
            
            if len(attempts) > 0:
                avg_success = np.mean(successes) if successes else 0.5
                variance = np.var(successes) if len(successes) > 1 else 0.25
                
                self.capability_estimates[name] = 0.7 * self.capability_estimates[name] + 0.3 * avg_success
                self.capability_variances[name] = 0.7 * self.capability_variances[name] + 0.3 * variance
    
    def get_capability_score(self, capability_name):
        return self.capability_estimates.get(capability_name, 0.5)
    
    def get_capability_confidence(self, capability_name):
        variance = self.capability_variances.get(capability_name, 0.25)
        return max(0.1, min(1.0, 1.0 - variance))
    
    def evaluate_task_feasibility(self, required_capabilities):
        feasibility = 1.0
        confidence_product = 1.0
        
        for cap_name, required_level in required_capabilities.items():
            current_level = self.get_capability_score(cap_name)
            cap_confidence = self.get_capability_confidence(cap_name)
            
            if current_level < required_level:
                feasibility *= (current_level / required_level) * cap_confidence
            else:
                feasibility *= cap_confidence
            
            confidence_product *= cap_confidence
        
        return {
            'feasibility': max(0.0, min(1.0, feasibility)),
            'confidence': confidence_product,
            'capability_gaps': {
                cap: max(0, req - self.get_capability_score(cap))
                for cap, req in required_capabilities.items()
            }
        }
    
    def get_overall_competence(self):
        return np.mean(list(self.capability_estimates.values()))


class SelfBoundaryDetector:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        
        self.self_state_buffer = deque(maxlen=200)
        self.env_state_buffer = deque(maxlen=200)
        
        self.self_mean = np.zeros(feature_dim)
        self.self_covariance = np.eye(feature_dim) * 0.1
        
        self.env_mean = np.zeros(feature_dim)
        self.env_covariance = np.eye(feature_dim) * 0.1
        
        self.update_count = 0
    
    def record_self_state(self, state_vector):
        state_np = np.array(state_vector)
        if state_np.ndim > 1:
            state_np = state_np.flatten()[:self.feature_dim]
        elif state_np.ndim == 1 and len(state_np) > self.feature_dim:
            state_np = state_np[:self.feature_dim]
        self.self_state_buffer.append(state_np)
        self._update_self_model()
    
    def record_env_state(self, state_vector):
        state_np = np.array(state_vector)
        if state_np.ndim > 1:
            state_np = state_np.flatten()[:self.feature_dim]
        elif state_np.ndim == 1 and len(state_np) > self.feature_dim:
            state_np = state_np[:self.feature_dim]
        self.env_state_buffer.append(state_np)
        self._update_env_model()
    
    def _update_self_model(self):
        if len(self.self_state_buffer) > 10:
            states = np.array(list(self.self_state_buffer))
            self.self_mean = np.mean(states, axis=0)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            elif states.ndim > 2:
                states = states.reshape(len(states), -1)
            self.self_covariance = np.cov(states, rowvar=False) + np.eye(self.feature_dim) * 1e-6
            self.update_count += 1
    
    def _update_env_model(self):
        if len(self.env_state_buffer) > 10:
            states = np.array(list(self.env_state_buffer))
            self.env_mean = np.mean(states, axis=0)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            elif states.ndim > 2:
                states = states.reshape(len(states), -1)
            self.env_covariance = np.cov(states, rowvar=False) + np.eye(self.feature_dim) * 1e-6
    
    def is_self_state(self, state_vector, threshold=0.5):
        mahalanobis_self = self._mahalanobis_distance(state_vector, self.self_mean, self.self_covariance)
        mahalanobis_env = self._mahalanobis_distance(state_vector, self.env_mean, self.env_covariance)
        
        self_prob = 1.0 / (1.0 + np.exp(mahalanobis_self - mahalanobis_env))
        
        return self_prob > threshold, self_prob
    
    def _mahalanobis_distance(self, x, mean, cov):
        diff = x - mean
        try:
            inv_cov = np.linalg.inv(cov)
            return np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
        except np.linalg.LinAlgError:
            return np.sqrt(np.sum(diff ** 2))
    
    def get_self_environment_boundary(self):
        return {
            'self_mean': self.self_mean,
            'self_covariance': self.self_covariance,
            'env_mean': self.env_mean,
            'env_covariance': self.env_covariance,
            'separation_score': np.linalg.norm(self.self_mean - self.env_mean)
        }


class SelfModel:
    def __init__(self, state_dim=5, feature_dim=16, horizon=10):
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.horizon = horizon
        
        self.internal_predictor = InternalStatePredictor(state_dim=state_dim, horizon=horizon)
        self.competence_assessor = CompetenceAssessor()
        self.boundary_detector = SelfBoundaryDetector(feature_dim=feature_dim)
        
        self.trajectory_history = deque(maxlen=100)
        self.self_awareness_score = 0.5
        
        self.homeostatic_baselines = {
            'energy': 0.7,
            'attention': 0.6,
            'security': 0.8,
            'curiosity': 0.5,
            'competence': 0.5
        }
    
    def predict_self_trajectory(self, current_internal_state, planned_actions=None):
        pred_trajectory = self.internal_predictor.forward(current_internal_state, planned_actions, steps=self.horizon)
        
        self.trajectory_history.append({
            'start_state': current_internal_state,
            'predicted': pred_trajectory,
            'timestamp': np.random.randint(1000000)
        })
        
        return pred_trajectory
    
    def evaluate_trajectory_deviation(self, trajectory):
        deviations = []
        critical_points = []
        
        for t, state in enumerate(trajectory):
            need_names = list(self.homeostatic_baselines.keys())
            for i, need_name in enumerate(need_names):
                baseline = self.homeostatic_baselines[need_name]
                value = state[i] if i < len(state) else baseline
                
                deviation = abs(value - baseline)
                deviations.append({
                    'timestep': t,
                    'need': need_name,
                    'value': value,
                    'baseline': baseline,
                    'deviation': deviation,
                    'critical': deviation > 0.4
                })
                
                if deviation > 0.4:
                    critical_points.append({
                        'timestep': t,
                        'need': need_name,
                        'severity': deviation,
                        'type': 'deficit' if value < baseline else 'surplus'
                    })
        
        return {
            'deviations': deviations,
            'critical_points': critical_points,
            'max_deviation': max(d['deviation'] for d in deviations) if deviations else 0.0,
            'critical_count': len(critical_points)
        }
    
    def detect_future_problems(self, trajectory):
        eval_result = self.evaluate_trajectory_deviation(trajectory)
        critical_points = eval_result['critical_points']
        
        problems = []
        for cp in critical_points:
            if cp['type'] == 'deficit':
                problems.append({
                    'type': 'need_deficit',
                    'need': cp['need'],
                    'timestep': cp['timestep'],
                    'severity': cp['severity'],
                    'urgency': 1.0 - (cp['timestep'] / self.horizon)
                })
        
        problems.sort(key=lambda x: x['urgency'] * x['severity'], reverse=True)
        
        return problems
    
    def generate_self_reflection(self, current_state, trajectory, action_result=None):
        reflection = {
            'current_state': current_state,
            'predicted_trajectory': trajectory,
            'competence_estimates': self.competence_assessor.capability_estimates,
            'self_awareness': self.self_awareness_score,
            'detected_problems': self.detect_future_problems(trajectory)
        }
        
        if action_result is not None:
            reflection['prediction_accuracy'] = self.internal_predictor.get_prediction_accuracy()
            reflection['learning_signal'] = reflection['prediction_accuracy']
            
            for cap_name in self.competence_assessor.capability_names:
                success = reflection['prediction_accuracy'] > 0.7
                self.competence_assessor.record_attempt(cap_name, success)
        
        self.competence_assessor.update_estimates()
        
        return reflection
    
    def assess_capability_for_goal(self, goal_type):
        capability_requirements = {
            'maintain_energy': {'action': 0.6, 'learning': 0.4},
            'explore_unknown': {'perception': 0.5, 'action': 0.5},
            'learn_new_skill': {'learning': 0.7, 'cognition': 0.6},
            'improve_competence': {'learning': 0.6, 'adaptation': 0.5},
            'minimize_prediction_error': {'cognition': 0.7, 'perception': 0.6}
        }
        
        if goal_type in capability_requirements:
            return self.competence_assessor.evaluate_task_feasibility(
                capability_requirements[goal_type]
            )
        
        return {'feasibility': 0.5, 'confidence': 0.5, 'capability_gaps': {}}
    
    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.boundary_detector = SelfBoundaryDetector(feature_dim=new_feature_dim)
    
    def get_self_model_state(self):
        return {
            'self_awareness': self.self_awareness_score,
            'competence': self.competence_assessor.get_overall_competence(),
            'prediction_accuracy': self.internal_predictor.get_prediction_accuracy(),
            'boundary_separation': self.boundary_detector.get_self_environment_boundary()['separation_score'],
            'capabilities': self.competence_assessor.capability_estimates
        }