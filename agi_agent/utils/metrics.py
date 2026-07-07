import torch
import numpy as np
from scipy.stats import entropy


def calc_free_energy(pred: torch.Tensor, obs: torch.Tensor) -> float:
    mse = torch.mean(torch.square(pred - obs)).item()
    return mse


def calc_entropy(x: torch.Tensor) -> float:
    x_np = torch.softmax(x.detach().cpu(), dim=-1).numpy()
    if x_np.ndim > 1:
        x_np = x_np.flatten()
    return float(entropy(x_np + 1e-8))


def calc_kl_divergence(old_dist: torch.Tensor, new_dist: torch.Tensor) -> float:
    old_flat = old_dist.detach().cpu().flatten()
    new_flat = new_dist.detach().cpu().flatten()
    
    max_len = max(len(old_flat), len(new_flat))
    
    if len(old_flat) < max_len:
        padding = torch.zeros(max_len - len(old_flat))
        old_flat = torch.cat([old_flat, padding])
    if len(new_flat) < max_len:
        padding = torch.zeros(max_len - len(new_flat))
        new_flat = torch.cat([new_flat, padding])
    
    old_sm = torch.softmax(old_flat + 1e-8, dim=-1)
    new_sm = torch.softmax(new_flat + 1e-8, dim=-1)
    kl = torch.sum(old_sm * torch.log(old_sm / new_sm)).item()
    return max(0.0, kl)


def calc_confidence(fe: float, threshold: float = 0.3) -> float:
    base_confidence = max(0.0, min(1.0, 1.0 - fe / (threshold * 2)))
    return base_confidence


def calc_novelty(kl_shift: float, threshold: float = 0.5) -> float:
    return min(1.0, kl_shift / threshold)


def calc_convergence_speed(old_fe: float, new_fe: float) -> float:
    if old_fe < 1e-8:
        return 0.0
    return (old_fe - new_fe) / old_fe


def moving_average(data: list, window_size: int = 10) -> float:
    if len(data) < window_size:
        return float(np.mean(data)) if data else 0.0
    return float(np.mean(data[-window_size:]))


def normalize_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)