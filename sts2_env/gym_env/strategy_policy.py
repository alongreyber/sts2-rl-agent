"""Small attention encoder for masked strategy PPO (no teacher data required)."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StrategyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, width=32):
        n, f = observation_space['candidates'].shape
        super().__init__(observation_space, features_dim=(n + 1) * width)
        self.project = nn.Linear(f, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, obs):
        tokens = self.project(torch.cat([obs['context'].unsqueeze(1), obs['candidates']], dim=1))
        padding = torch.cat([torch.zeros_like(obs['valid'][:, :1]), 1 - obs['valid']], dim=1).bool()
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=padding, need_weights=False)
        result = self.norm(tokens + attended)
        return result.masked_fill(padding.unsqueeze(-1), 0).flatten(1)
