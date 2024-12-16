import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

from typing import Optional


class Attention(nn.Module):
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention weights and apply them to the input.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, num_heads, seq_length, d_k).
        mask : torch.Tensor
            Mask tensor of shape (batch_size, num_heads, seq_length, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        d_k = X.size(-1)
        scores = torch.matmul(X, X.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=X.dtype)
        )
        scores = scores.masked_fill(mask, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, X)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        """
        Initialize the MultiHeadAttention module.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input (input dimensionality).
        num_heads : int
            The number of heads in the multi-head attention models.
        dropout : float, optional
            The dropout probability, by default 0.1.
        batch_first : bool, optional
            If True, the input and output tensors are provided as (batch, seq, feature), by default True.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(
        self, X: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multi-head attention to the input.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, seq_length, d_model).
        src_key_padding_mask : torch.Tensor
            Mask tensor of shape (batch_size, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, seq_len, _ = X.size()
        X = self.linear(X)
        X = X.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        mask = (
            src_key_padding_mask.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.num_heads, -1, -1)
        )
        X = self.attention(X, mask)
        X = (
            X.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.d_k)
        )
        X = self.output_linear(X)

        return X


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
    ):
        """
        Initialize the TransformerEncoderLayer module.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        num_heads : int
            The number of heads in the multi-head attention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network model, by default 2048.
        dropout : float, optional
            The dropout value, by default 0.1.
        activation : str, optional
            The activation function of intermediate layers, by default "relu".
        batch_first : bool, optional
            If True, the input and output tensors are provided as (batch, seq, feature), by default True.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(
        self,
        X: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, seq_length, d_model).
        src_mask : torch.Tensor
            Mask tensor (not used in this implementation).
        src_key_padding_mask : torch.Tensor
            Mask tensor of shape (batch_size, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_length, d_model).
        """
        X2 = self.self_attn(X, src_key_padding_mask=src_key_padding_mask)
        X = X + self.dropout(X2)
        X = self.norm1(X)
        X2 = self.linear2(self.dropout(self.activation(self.linear1(X))))
        X = X + self.dropout(X2)
        X = self.norm2(X)
        return X


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module):
        """
        Initialize the TransformerEncoder module.

        Parameters
        ----------
        encoder_layer : nn.Module
            An instance of the TransformerEncoderLayer class.
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        norm : nn.Module
            The layer normalization component.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass the input through the transformer encoder layers.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, seq_length, d_model).
        mask : torch.Tensor, optional
            Mask tensor (not used in this implementation). The default is None.
        src_key_padding_mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, seq_length). The default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_length, d_model).
        """
        for layer in self.layers:
            X = layer(X, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        X = self.norm(X)
        return X
