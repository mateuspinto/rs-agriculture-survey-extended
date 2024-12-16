import torch.nn as nn
import torch


class BertHead(nn.Module):
    def __init__(self, hidden, num_features):
        super().__init__()
        self.linear = nn.Linear(hidden, num_features)

        self.input_sample = torch.rand((2, 70, hidden), dtype=torch.float32)
        self.output_sample = torch.rand((2, 70, num_features), dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # Testing the model output
    model = BertHead(hidden=128, num_features=10)
    with torch.inference_mode():
        output = model(model.input_sample)
        print(output.shape)
        print(model.output_sample.shape)
