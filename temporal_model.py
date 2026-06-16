"""
Temporal Models: LSTM and Transformer for scoring frame importance over time.
Input : (T, D) frame embeddings
Output: (T,)  importance scores in [0, 1]
"""
import torch
import torch.nn as nn
import math


class LSTMTemporalModel(nn.Module):
    """Bidirectional LSTM → importance scorer."""
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T, D)  →  out: (1, T, 2*H)
        out, _ = self.lstm(x)
        scores = self.scorer(out).squeeze(-1).squeeze(0)  # (T,)
        return scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerTemporalModel(nn.Module):
    """
    Transformer encoder for temporal self-attention over frame embeddings.
    Learns which frames are globally important via attention.
    """
    def __init__(self, input_dim: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.scorer  = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T, D)
        h = self.pos_enc(self.input_proj(x))    # (1, T, d_model)
        h = self.encoder(h)                      # (1, T, d_model)
        scores = self.scorer(h).squeeze(-1).squeeze(0)  # (T,)
        return scores


def build_temporal_model(kind: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory: 'lstm' or 'transformer'."""
    if kind == "lstm":
        return LSTMTemporalModel(input_dim, **kwargs)
    elif kind == "transformer":
        return TransformerTemporalModel(input_dim, **kwargs)
    raise ValueError(f"Unknown temporal model: {kind}")
