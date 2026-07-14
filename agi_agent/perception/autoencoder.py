import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from ..config.settings import DEVICE, FREE_ENERGY_THRESHOLD, MAX_HIDDEN_DIM, MIN_HIDDEN_DIM, GROWTH_STEP, PRUNE_STEP
from ..utils.metrics import calc_free_energy


class GrowingAutoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_fe = FREE_ENERGY_THRESHOLD
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU()
        ).to(DEVICE)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        ).to(DEVICE)
        
        self.act_stats = deque(maxlen=50)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.reconstruction_errors = deque(maxlen=100)
        self.structure_stable_steps = 0

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def adaptive_grow_prune(self, free_energy):
        self.structure_stable_steps += 1
        
        if self.structure_stable_steps < 50:
            return False
        
        avg_recent_fe = float(np.mean(list(self.reconstruction_errors)[-20:])) if len(self.reconstruction_errors) >= 20 else float(free_energy)
        
        if avg_recent_fe > FREE_ENERGY_THRESHOLD * 1.5 and self.hidden_dim < MAX_HIDDEN_DIM:
            new_hidden = min(self.hidden_dim + GROWTH_STEP, MAX_HIDDEN_DIM)
            self.hidden_dim = new_hidden
            self._resize_layers()
            self.structure_stable_steps = 0
            return True
        elif avg_recent_fe < FREE_ENERGY_THRESHOLD * 0.3 and self.hidden_dim > MIN_HIDDEN_DIM:
            self.hidden_dim = max(self.hidden_dim - PRUNE_STEP, MIN_HIDDEN_DIM)
            self._resize_layers()
            self.structure_stable_steps = 0
            return True
        return False

    def _resize_layers(self):
        self.encoder[0] = nn.Linear(self.input_dim, self.hidden_dim).to(DEVICE)
        self.encoder[2] = nn.Linear(self.hidden_dim, self.hidden_dim // 2).to(DEVICE)
        self.decoder[0] = nn.Linear(self.hidden_dim // 2, self.hidden_dim).to(DEVICE)
        self.decoder[2] = nn.Linear(self.hidden_dim, self.input_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=self.optimizer.param_groups[0]['lr'], weight_decay=1e-4)

    def update(self, x):
        z, recon = self(x)
        fe = calc_free_energy(recon, x)
        
        for _ in range(3):
            self.optimizer.zero_grad()
            z, recon = self(x)
            mse_loss = torch.mean(torch.square(recon - x))
            mse_loss.backward()
            self.optimizer.step()
        
        z, recon = self(x)
        fe = calc_free_energy(recon, x)
        mse_loss = torch.mean(torch.square(recon - x))
        self.act_stats.append(float(torch.mean(torch.abs(z)).item()))
        self.reconstruction_errors.append(float(fe))
        
        structure_changed = self.adaptive_grow_prune(fe)
        if structure_changed:
            z, _ = self(x)
        return z, fe, structure_changed

    def get_feature_dim(self):
        return self.hidden_dim // 2