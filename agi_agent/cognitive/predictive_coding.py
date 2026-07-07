import torch
import torch.nn as nn
import torch.optim as optim
from ..config.settings import DEVICE, MAX_INFERENCE_STEP, FREE_ENERGY_THRESHOLD
from ..utils.metrics import calc_free_energy


class HierarchicalPredictiveCoding(nn.Module):
    def __init__(self, feat_dim=16, num_layers=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.lr = 8e-4
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.ReLU(),
                nn.Linear(feat_dim * 2, feat_dim)
            ).to(DEVICE))
        
        self.optimizers = [optim.Adam(layer.parameters(), lr=self.lr) for layer in self.layers]
    
    def resize(self, new_dim):
        self.feat_dim = new_dim
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(new_dim, new_dim * 2),
                nn.ReLU(),
                nn.Linear(new_dim * 2, new_dim)
            ).to(DEVICE))
        self.optimizers = [optim.Adam(layer.parameters(), lr=self.lr) for layer in self.layers]

    def predict_next(self, current_feat, steps=MAX_INFERENCE_STEP):
        pred_list = []
        feat = current_feat
        
        for _ in range(steps):
            for layer in self.layers:
                feat = layer(feat)
            pred_list.append(feat)
        
        return pred_list

    def update(self, current_feat, real_next_feat):
        total_fe = 0.0
        feat = current_feat
        
        for layer_idx, layer in enumerate(self.layers):
            self.optimizers[layer_idx].zero_grad()
            pred = layer(feat)
            fe = calc_free_energy(pred, real_next_feat)
            loss = torch.tensor(fe, requires_grad=True)
            loss.backward(retain_graph=True)
            self.optimizers[layer_idx].step()
            total_fe += fe
            feat = pred
        
        return total_fe / self.num_layers, feat

    def set_lr(self, lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr