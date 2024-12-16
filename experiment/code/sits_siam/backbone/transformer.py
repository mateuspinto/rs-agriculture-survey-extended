import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from print_color import print

if (
    "TRANSFORMER_FROM_SCRATCH" in os.environ
    and os.environ["TRANSFORMER_FROM_SCRATCH"] == "True"
):
    print(
        "Using Transformer from scratch. Performance could get worse...",
        tag="WARNING",
        tag_color="yellow",
        color="magenta",
    )
    from .scratch import TransformerEncoder, TransformerEncoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 128,
        n_head: int = 16,
        n_layers: int = 1,
        d_inner: int = 128,
        activation: str = "relu",
        dropout: int = 0.2,
        max_len: int = 366,
        max_seq_len: int = 70,
        T: int = 1000,
        max_temporal_shift: int = 30,
    ):
        super(TransformerBackbone, self).__init__()
        self.modelname = self._get_name()
        self.max_seq_len = max_seq_len

        self.mlp_dim = [input_dim, 32, 64, d_model]
        layers = []
        for i in range(len(self.mlp_dim) - 1):
            layers.append(LinLayer(self.mlp_dim[i], self.mlp_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(
            d_model, max_len=max_len + 2 * max_temporal_shift, T=T
        )

        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, d_inner, dropout, activation, batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformerencoder = TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )

        # Input and output sample for batch size 2
        self.input_sample = (
            torch.rand((2, self.max_seq_len, input_dim), dtype=torch.float32),
            torch.randint(1, max_len, (2, self.max_seq_len), dtype=torch.int64),
            torch.zeros((2, self.max_seq_len), dtype=torch.bool),
        )
        self.output_sample = torch.rand((2, max_seq_len, d_model), dtype=torch.float32)

    def forward(self, x: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor):
        x = self.mlp1(x)
        x = self.inlayernorm(x)
        x = self.dropout(x + self.position_enc(doy))
        x = self.transformerencoder(x, src_key_padding_mask=mask)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, T: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, doy):
        """
        Args:
            doy: Tensor, shape [batch_size, seq_len]
        """
        return self.pe[doy]


class LinLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinLayer, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = self.ln(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    # Testing the model output
    model = TransformerBackbone(d_inner=168)
    with torch.no_grad():
        output = model(*model.input_sample)
        print(output.shape)
        print(model.output_sample.shape)
