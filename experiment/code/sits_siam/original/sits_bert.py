import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))  # noqa: E741
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()  # [d_model/2,]

        pe[1:, 0::2] = torch.sin(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]

        self.register_buffer("pe", pe)

    def forward(self, doy):
        return self.pe[doy, :]


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=366)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence, doy_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)  # noqa: F841
        obs_embed = self.input(
            input_sequence
        )  # [batch_size, seq_length, embedding_dim]
        x = obs_embed.repeat(1, 1, 2)  # [batch_size, seq_length, embedding_dim*2]
        for i in range(batch_size):
            x[i, :, self.embed_size :] = self.position(
                doy_sequence[i, :]
            )  # [seq_length, embedding_dim]

        return self.dropout(x)


class SBERT(nn.Module):
    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-BERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(num_features, int(hidden / 2))

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, doy, mask):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class MulticlassClassification(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool1d(64)
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        x = self.linear(x)
        return x


class OriginalSITSBert(nn.Module):
    def __init__(self, num_classes=13, pretrained=True):
        super(OriginalSITSBert, self).__init__()
        self.sbert = SBERT(num_features=10, hidden=256, n_layers=3, attn_heads=8)
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

        if pretrained:
            self.sbert.load_state_dict(
                torch.load(
                    "weights/original_sits_bert/checkpoint.bert.pth",
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )

    def forward(self, x, doy, mask):
        # print("x shape and dtype", x.shape, x.dtype)
        # print("doy shape and dtype", doy.shape, doy.dtype)
        # print("mask shape and dtype", mask.shape, mask.dtype)

        # exit(0)
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)


class OriginalSITSBertMissingMaskFix:
    def __call__(self, sample):
        # Invert sample mask and cast to int8
        sample["mask"] = (~sample["mask"]).astype(np.int8)

        return sample


class OriginalSITSBertDataset(Dataset):
    def __init__(self, file_path, feature_num, seq_len):
        """
        :param file_path: fine-tuning file path
        :param feature_num: number of input features
        :param seq_len: padded sequence length
        """

        self.seq_len = seq_len
        self.dimension = feature_num

        with open(file_path, "r") as ifile:
            self.Data = ifile.readlines()
            self.TS_num = len(self.Data)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        line = self.Data[item]

        # line[-1] == '\n' should be discarded
        line_data = line[:-1].split(",")
        line_data = list(map(float, line_data))
        line_data = np.array(line_data, dtype=float)

        class_label = np.array([line_data[-1]], dtype=int)

        ts = np.reshape(line_data[:-1], (self.dimension + 1, -1)).T
        ts_length = ts.shape[0]

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        ts_origin[:ts_length, :] = ts[:, :-1] / 10000.0

        # day of year
        doy = np.zeros((self.seq_len,), dtype=int)
        doy[:ts_length] = np.squeeze(ts[:, -1])

        output = {
            "x": ts_origin,
            "mask": bert_mask,
            "y": class_label,
            "doy": doy,
        }

        for key, value in output.items():
            if (
                (value.dtype == np.half)
                or (value.dtype == np.float32)
                or (value.dtype == np.float64)
            ):
                output[key] = torch.from_numpy(value.astype(np.float32))
            elif value.dtype == np.bool:
                output[key] = torch.from_numpy(value.astype(np.bool))
            else:
                output[key] = torch.from_numpy(value.astype(np.int64))

        return output


# Testing if the model is working if script is run
if __name__ == "__main__":
    model = OriginalSITSBert()
    x = torch.randn(2, 64, 10, dtype=torch.float32)
    doy = torch.randint(0, 366, (2, 64), dtype=torch.int64)
    mask = torch.randint(0, 2, (2, 64), dtype=torch.int64)
    out = model(x, doy, mask)
    print(out.shape)
