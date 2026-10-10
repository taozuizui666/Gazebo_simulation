"""Policy preprocessing uses only the canonical lidar vector."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarFeatures(BaseFeaturesExtractor):
    """Keep all bins and their zero mask; spread short ranges for the small MLP."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, observations):
        return torch.sqrt(torch.clamp(observations, min=0.0, max=1.0))
