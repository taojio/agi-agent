import numpy as np
from ..config.settings import LEARNING_RATE_POOL, INITIAL_LEARNING_RATE


class MetaLearningLayer:
    def __init__(self):
        self.lr_pool = LEARNING_RATE_POOL
        self.best_lr = INITIAL_LEARNING_RATE
        self.arm_reward = np.zeros(len(self.lr_pool))
        self.arm_count = np.zeros(len(self.lr_pool))
        self.exploration_rate = 0.2
        self.convergence_history = []

    def adaptive_hyper_update(self, free_energy, convergence_speed):
        best_idx = np.argmax(self.arm_reward / (self.arm_count + 1e-8))
        
        if np.random.random() < self.exploration_rate:
            best_idx = np.random.randint(len(self.lr_pool))
        
        self.best_lr = self.lr_pool[best_idx]
        
        reward = convergence_speed * (1 / (free_energy + 1e-8))
        self.arm_reward[best_idx] += reward
        self.arm_count[best_idx] += 1
        
        self.convergence_history.append((free_energy, convergence_speed))
        if len(self.convergence_history) > 100:
            self.convergence_history.pop(0)
        
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        return self.best_lr

    def freeze_base_finetune_top(self, encoder_model):
        for idx, param in enumerate(encoder_model.encoder.parameters()):
            if idx < 2:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_meta_stats(self):
        avg_reward = np.mean(self.arm_reward / (self.arm_count + 1e-8))
        return {
            "best_lr": self.best_lr,
            "avg_reward": avg_reward,
            "exploration_rate": self.exploration_rate,
            "arm_counts": dict(zip(self.lr_pool, self.arm_count.tolist()))
        }