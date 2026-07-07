import torch
import torch.nn as nn
from ..config.settings import DEVICE


class MultimodalFusion(nn.Module):
    def __init__(self, modalities: dict, output_dim: int = 16):
        super().__init__()
        self.modalities = modalities
        self.output_dim = output_dim
        
        self.modal_encoders = {}
        self.fusion_weights = nn.ParameterDict()
        
        for name, dim in modalities.items():
            self.modal_encoders[name] = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, output_dim)
            ).to(DEVICE)
            self.fusion_weights[name] = nn.Parameter(torch.tensor(1.0 / len(modalities)))
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * len(modalities), output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        ).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, inputs: dict):
        encoded_features = []
        weights = []
        
        for name in self.modalities.keys():
            if name in inputs:
                x = torch.tensor(inputs[name], dtype=torch.float32).to(DEVICE)
                
                expected_dim = self.modalities[name]
                if x.dim() == 2 and x.shape[1] != expected_dim:
                    if x.shape[1] < expected_dim:
                        padding = torch.zeros(x.shape[0], expected_dim - x.shape[1]).to(DEVICE)
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :expected_dim]
                
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                feat = self.modal_encoders[name](x)
                encoded_features.append(feat)
                weights.append(self.fusion_weights[name])
        
        if not encoded_features:
            return torch.zeros(1, self.output_dim).to(DEVICE)
        
        weights = torch.softmax(torch.stack(weights), dim=0)
        weighted_features = []
        
        for i, feat in enumerate(encoded_features):
            weighted_features.append(feat * weights[i])
        
        concatenated = torch.cat(weighted_features, dim=-1)
        fused = self.fusion_layer(concatenated)
        return fused

    def update(self, inputs: dict, target: torch.Tensor):
        self.optimizer.zero_grad()
        output = self(inputs)
        loss = torch.mean(torch.square(output - target))
        loss.backward()
        self.optimizer.step()
        return loss 

    def add_modality(self, name: str, dim: int):
        self.modalities[name] = dim
        self.modal_encoders[name] = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, self.output_dim)
        ).to(DEVICE)
        self.fusion_weights[name] = nn.Parameter(torch.tensor(1.0 / len(self.modalities)))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def remove_modality(self, name: str):
        if name in self.modalities:
            del self.modalities[name]
            del self.modal_encoders[name]
            del self.fusion_weights[name]
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)