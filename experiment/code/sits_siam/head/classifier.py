import torch.nn as nn
import torch


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 128, num_classes: int = 10):
        super(ClassifierHead, self).__init__()
        layers = []
        decoder = [d_model, 64, 32, num_classes]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([nn.BatchNorm1d(decoder[i + 1]), nn.ReLU()])
        self.decoder = nn.Sequential(*layers)

        self.example_input = torch.randn(2, d_model)
        self.example_output = torch.randn(2, num_classes)

    def forward(self, features: torch.Tensor):
        return self.decoder(features)
