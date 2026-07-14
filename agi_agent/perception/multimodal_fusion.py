import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import numpy as np
from ..config.settings import DEVICE


class ModalityAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim).to(DEVICE)
        self.key_proj = nn.Linear(input_dim, hidden_dim).to(DEVICE)
        self.value_proj = nn.Linear(input_dim, input_dim).to(DEVICE)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32)).to(DEVICE)

    def forward(self, features: torch.Tensor):
        queries = self.query_proj(features)
        keys = self.key_proj(features)
        values = self.value_proj(features)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, values)

        return attended, attention_weights


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim: int, num_modalities: int, hidden_dim: int = 32):
        super().__init__()
        self.self_attentions = nn.ModuleList([
            ModalityAttention(input_dim, hidden_dim) for _ in range(num_modalities)
        ])
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=4, batch_first=True
        ).to(DEVICE)

    def forward(self, modality_features: List[torch.Tensor]):
        attended_features = []
        attention_weights = []

        for i, feat in enumerate(modality_features):
            attended, weights = self.self_attentions[i](feat)
            attended_features.append(attended)
            attention_weights.append(weights)

        stacked = torch.stack(attended_features, dim=1)
        batch_size, num_modalities, dim = stacked.shape
        reshaped = stacked.reshape(batch_size, num_modalities, dim)

        cross_attended, cross_weights = self.cross_attention(reshaped, reshaped, reshaped)

        return cross_attended, cross_weights


class AdaptiveModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, modality_name: str):
        super().__init__()
        self.modality_name = modality_name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim * 2, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(output_dim * 2, output_dim)
        ).to(DEVICE)

        self.importance_scorer = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(DEVICE)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        importance = self.importance_scorer(encoded).mean()
        return encoded, importance


class DynamicFusionLayer(nn.Module):
    def __init__(self, input_dim: int, num_modalities: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        self.output_dim = output_dim

        self.attention = CrossModalAttention(input_dim, num_modalities)

        self.fusion_strategy = nn.Sequential(
            nn.Linear(input_dim * num_modalities, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh()
        ).to(DEVICE)

        self.residual = nn.Linear(input_dim, output_dim).to(DEVICE) if input_dim != output_dim else nn.Identity()

    def forward(self, modality_features: List[torch.Tensor], importance_scores: torch.Tensor):
        batch_size = modality_features[0].shape[0] if modality_features else 1

        if len(modality_features) == 1:
            return modality_features[0]

        cross_attended, _ = self.attention(modality_features)

        weighted_features = []
        for i, feat in enumerate(modality_features):
            importance = importance_scores[i] if i < len(importance_scores) else 1.0
            weighted_features.append(feat * importance)

        concatenated = torch.cat(weighted_features, dim=-1)
        fused = self.fusion_strategy(concatenated)

        first_feat = modality_features[0]
        residual = self.residual(first_feat)
        fused = fused + residual

        return fused


class MultimodalFusion(nn.Module):
    def __init__(self, modalities: dict, output_dim: int = 16):
        super().__init__()
        self.modalities = modalities
        self.output_dim = output_dim

        self.modal_encoders = nn.ModuleDict()
        for name, dim in modalities.items():
            self.modal_encoders[name] = AdaptiveModalityEncoder(dim, output_dim, name)

        self.dynamic_fusion = DynamicFusionLayer(
            output_dim, len(modalities), output_dim
        )

        self.adaptive_weights = nn.ParameterDict()
        for name in modalities.keys():
            self.adaptive_weights[name] = nn.Parameter(torch.tensor(1.0 / len(modalities)))

        self.context_encoder = nn.Sequential(
            nn.Linear(output_dim * 3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, len(modalities)),
            nn.Softmax(dim=-1)
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.modality_history = []
        self.max_history = 10

    def forward(self, inputs: dict, context: Optional[torch.Tensor] = None):
        encoded_features = []
        importance_scores = []
        modality_names = []

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

                feat, importance = self.modal_encoders[name](x)
                encoded_features.append(feat)
                importance_scores.append(importance)
                modality_names.append(name)

        if not encoded_features:
            return torch.zeros(1, self.output_dim).to(DEVICE)

        importance_tensor = torch.stack(importance_scores)

        if context is not None and len(encoded_features) > 1:
            context_weights = self._compute_contextual_weights(encoded_features, context)
            importance_tensor = importance_tensor * context_weights

        self._update_modality_history(modality_names, importance_scores)

        fused = self.dynamic_fusion(encoded_features, importance_tensor)

        return fused

    def _compute_contextual_weights(self, features: List[torch.Tensor], context: torch.Tensor):
        context_expanded = context.unsqueeze(0).repeat(len(features), 1, 1)

        context_input = []
        for i, feat in enumerate(features):
            context_input.append(torch.cat([feat, context_expanded[i]], dim=-1))

        context_input_tensor = torch.stack(context_input)
        batch_size, num_modalities, dim = context_input_tensor.shape
        context_input_reshaped = context_input_tensor.reshape(batch_size * num_modalities, dim)

        weights = self.context_encoder(context_input_reshaped)
        weights = weights.reshape(batch_size, num_modalities, len(self.modalities))
        weights = weights.mean(dim=0)

        return weights

    def _update_modality_history(self, names: List[str], scores: List[float]):
        self.modality_history.append({
            "modalities": names,
            "scores": scores,
            "timestamp": torch.tensor([[np.random.randint(1000000)]])
        })

        if len(self.modality_history) > self.max_history:
            self.modality_history.pop(0)

    def get_modality_importance(self) -> Dict[str, float]:
        importance = {}
        for name, encoder in self.modal_encoders.items():
            dummy_input = torch.randn(1, self.modalities[name]).to(DEVICE)
            _, score = encoder(dummy_input)
            importance[name] = score.item()
        return importance

    def update(self, inputs: dict, target: torch.Tensor, context: Optional[torch.Tensor] = None):
        self.optimizer.zero_grad()
        output = self(inputs, context)
        loss = torch.mean(torch.square(output - target))
        loss.backward()
        self.optimizer.step()
        return loss

    def add_modality(self, name: str, dim: int):
        self.modalities[name] = dim
        self.modal_encoders[name] = AdaptiveModalityEncoder(dim, self.output_dim, name)
        self.adaptive_weights[name] = nn.Parameter(torch.tensor(1.0 / len(self.modalities)))

        self.dynamic_fusion = DynamicFusionLayer(
            self.output_dim, len(self.modalities), self.output_dim
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(self.output_dim * 3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, len(self.modalities)),
            nn.Softmax(dim=-1)
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def remove_modality(self, name: str):
        if name in self.modalities:
            del self.modalities[name]
            del self.modal_encoders[name]
            del self.adaptive_weights[name]

            self.dynamic_fusion = DynamicFusionLayer(
                self.output_dim, len(self.modalities), self.output_dim
            )

            self.context_encoder = nn.Sequential(
                nn.Linear(self.output_dim * 3, 64),
                nn.LeakyReLU(),
                nn.Linear(64, len(self.modalities)),
                nn.Softmax(dim=-1)
            ).to(DEVICE)

            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_fusion_summary(self) -> Dict[str, Any]:
        importance = self.get_modality_importance()
        return {
            "modalities": list(self.modalities.keys()),
            "modality_dims": dict(self.modalities),
            "output_dim": self.output_dim,
            "modality_importance": importance,
            "history_length": len(self.modality_history),
            "total_parameters": sum(p.numel() for p in self.parameters())
        }

    def resize(self, new_dim):
        self.output_dim = new_dim

        new_encoders = nn.ModuleDict()
        for name, dim in self.modalities.items():
            new_encoders[name] = AdaptiveModalityEncoder(dim, new_dim, name)
        self.modal_encoders = new_encoders

        self.dynamic_fusion = DynamicFusionLayer(new_dim, len(self.modalities), new_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)