import torch.nn as nn
import torch


class PoolingBottleneck(nn.Module):
    def __init__(self):
        super(PoolingBottleneck, self).__init__()

    def forward(self, features: torch.Tensor):
        features, _ = torch.max(features, dim=1)
        return features
