import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ..config.settings import DEVICE


class Option:
    def __init__(self, name, policy_net, termination_net, initiation_net, action_dim=8, feature_dim=16):
        self.name = name
        self.policy_net = policy_net
        self.termination_net = termination_net
        self.initiation_net = initiation_net
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.usage_count = 0
        self.success_count = 0
        self.current_steps = 0
        self.max_steps = 50

    def get_action(self, feature, pred_feature):
        concat_feat = torch.cat([feature, pred_feature], dim=-1)
        action = self.policy_net(concat_feat)
        return action

    def should_terminate(self, feature):
        if self.current_steps >= self.max_steps:
            return True
        prob = self.termination_net(feature).sigmoid().item()
        self.current_steps += 1
        return np.random.random() < prob

    def is_initiated(self, feature):
        prob = self.initiation_net(feature).sigmoid().item()
        return prob > 0.5

    def reset(self):
        self.current_steps = 0

    def update_performance(self, success):
        self.usage_count += 1
        if success:
            self.success_count += 1

    def get_success_rate(self):
        return self.success_count / max(self.usage_count, 1)


class HighLevelPolicy:
    def __init__(self, feature_dim=16, num_options=5):
        self.feature_dim = feature_dim
        self.num_options = num_options
        self.options = []
        self.option_selection_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_options),
            nn.Softmax(dim=-1)
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.option_selection_net.parameters(), lr=1e-3)
        self.option_history = deque(maxlen=100)

    def add_option(self, option):
        if len(self.options) < self.num_options:
            self.options.append(option)
            return True
        return False

    def select_option(self, feature):
        if not self.options:
            return None
        
        feature_tensor = feature.detach() if hasattr(feature, 'detach') else torch.tensor(feature, dtype=torch.float32).to(DEVICE)
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        
        selection_probs = self.option_selection_net(feature_tensor)
        option_idx = torch.multinomial(selection_probs, 1).item()
        
        option = self.options[option_idx]
        self.option_history.append({"option": option.name, "prob": selection_probs[0, option_idx].item()})
        
        return option

    def update_selection(self, feature, option_idx, advantage):
        self.optimizer.zero_grad()
        
        feature_tensor = feature.detach() if hasattr(feature, 'detach') else torch.tensor(feature, dtype=torch.float32).to(DEVICE)
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        
        selection_probs = self.option_selection_net(feature_tensor)
        log_prob = torch.log(selection_probs[0, option_idx] + 1e-8)
        loss = -log_prob * advantage
        
        loss.backward()
        self.optimizer.step()
        return loss.item()


class HierarchicalValueFunction:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        
        self.high_level_value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(DEVICE)
        
        self.low_level_value_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(DEVICE)
        
        self.high_optimizer = torch.optim.Adam(self.high_level_value_net.parameters(), lr=1e-3)
        self.low_optimizer = torch.optim.Adam(self.low_level_value_net.parameters(), lr=1e-3)

    def compute_high_level_value(self, feature):
        feature_tensor = feature.detach() if hasattr(feature, 'detach') else torch.tensor(feature, dtype=torch.float32).to(DEVICE)
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        return self.high_level_value_net(feature_tensor)

    def compute_low_level_value(self, feature, pred_feature):
        concat_feat = torch.cat([feature, pred_feature], dim=-1)
        return self.low_level_value_net(concat_feat)

    def update_high_value(self, feature, target_value):
        self.high_optimizer.zero_grad()
        value = self.compute_high_level_value(feature)
        loss = torch.mean(torch.square(value - target_value))
        loss.backward()
        self.high_optimizer.step()
        return loss.item()

    def update_low_value(self, feature, pred_feature, target_value):
        self.low_optimizer.zero_grad()
        value = self.compute_low_level_value(feature, pred_feature)
        loss = torch.mean(torch.square(value - target_value))
        loss.backward()
        self.low_optimizer.step()
        return loss.item()


class ActionExecutionLayer:
    def __init__(self, action_dim=8, feature_dim=16):
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        self.base_action_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        ).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.base_action_net.parameters(), lr=1e-3)
        self.action_history = []
        self.exploration_noise = 0.1
        
        self.high_level_policy = HighLevelPolicy(feature_dim=feature_dim, num_options=3)
        self.value_function = HierarchicalValueFunction(feature_dim=feature_dim)
        
        self._init_default_options()
        
        self.current_option = None
        self.option_rewards = deque(maxlen=50)

    def _init_default_options(self):
        for i in range(3):
            policy_net = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim),
                nn.Tanh()
            ).to(DEVICE)
            
            termination_net = nn.Sequential(
                nn.Linear(self.feature_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ).to(DEVICE)
            
            initiation_net = nn.Sequential(
                nn.Linear(self.feature_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ).to(DEVICE)
            
            option = Option(
                name=f"option_{i}",
                policy_net=policy_net,
                termination_net=termination_net,
                initiation_net=initiation_net,
                action_dim=self.action_dim,
                feature_dim=self.feature_dim
            )
            self.high_level_policy.add_option(option)

    def autonomous_action(self, feature, pred_feature):
        if self.current_option is None:
            self.current_option = self.high_level_policy.select_option(feature)
        
        if self.current_option:
            action = self.current_option.get_action(feature, pred_feature)
            
            if self.current_option.should_terminate(feature):
                self.current_option.reset()
                self.current_option = None
        else:
            concat_feat = torch.cat([feature, pred_feature], dim=-1)
            action = self.base_action_net(concat_feat)
        
        if self.exploration_noise > 0:
            noise = torch.randn_like(action) * self.exploration_noise
            action = action + noise
        
        action = torch.clamp(action, -1, 1)
        
        self.action_history.append(action.detach().cpu().numpy())
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        
        return action.detach().cpu().numpy()

    def update_action_net(self, feature, pred_feature, reward):
        self.optimizer.zero_grad()
        
        feat_detached = feature.detach().clone()
        pred_detached = pred_feature.detach().clone()
        
        concat_feat = torch.cat([feat_detached, pred_detached], dim=-1)
        action = self.base_action_net(concat_feat)
        
        high_value = self.value_function.compute_high_level_value(feature)
        low_value = self.value_function.compute_low_level_value(feat_detached, pred_detached)
        
        combined_value = 0.6 * high_value + 0.4 * low_value
        advantage = reward - combined_value.detach()
        
        loss = -torch.mean(action) * advantage
        loss.backward()
        self.optimizer.step()
        
        self.value_function.update_high_value(feature, reward)
        self.value_function.update_low_value(feat_detached, pred_detached, reward)
        
        self.option_rewards.append(reward)
        if self.current_option:
            self.current_option.update_performance(reward > 0)
        
        return loss.item()

    def hardware_adapt(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.base_action_net[0] = nn.Linear(new_feature_dim * 2, 32).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.base_action_net.parameters(), lr=1e-3)
        
        self.high_level_policy = HighLevelPolicy(feature_dim=new_feature_dim, num_options=3)
        self.value_function = HierarchicalValueFunction(feature_dim=new_feature_dim)
        self._init_default_options()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.high_level_policy.optimizer.param_groups[0]['lr'] = lr
        self.value_function.high_optimizer.param_groups[0]['lr'] = lr
        self.value_function.low_optimizer.param_groups[0]['lr'] = lr

    def set_causal_bias(self, bias):
        self.causal_bias = max(-1.0, min(1.0, bias))

    def set_exploration_noise(self, noise):
        self.exploration_noise = max(0.0, min(0.5, noise))

    def get_action_stats(self):
        if not self.action_history:
            return {"count": 0, "mean_magnitude": 0.0}
        
        magnitudes = [np.mean(np.abs(a)) for a in self.action_history]
        option_success_rates = {opt.name: opt.get_success_rate() for opt in self.high_level_policy.options}
        
        return {
            "count": len(self.action_history),
            "mean_magnitude": np.mean(magnitudes),
            "exploration_noise": self.exploration_noise,
            "current_option": self.current_option.name if self.current_option else None,
            "option_success_rates": option_success_rates,
            "avg_option_reward": np.mean(self.option_rewards) if self.option_rewards else 0.0
        }