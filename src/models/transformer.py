from __future__ import annotations
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE for sequences.
    Input:  x [B, T, D]
    Output: x + pe[:, :T]
    """
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]
    
def masked_mean_pool(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    lengths: [B]
    returns: [B, D]
    """
    B, T, D = x.shape
    device = x.device
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
    mask_f = mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
    x_sum = (x * mask_f).sum(dim=1)  # [B, D]
    denom = mask_f.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return x_sum / denom

class TransformerSER(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 80,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2000,
        use_subsample: bool = True,
    ):
        super().__init__()
        self.use_subsample = use_subsample
        if use_subsample:
            # light temporal subsampling to reduce T (stride=2)
            self.subsample = nn.Sequential(
                nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
            proj_out = d_model
        else:
            self.subsample = None
            self.in_proj = nn.Linear(n_mels, d_model)
            proj_out = d_model

        self.pos = SinusoidalPositionalEncoding(proj_out, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_out,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(proj_out, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, 1, 80, T]
        x = x.squeeze(1)  # [B, 80, T]

        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(-1), device=x.device, dtype=torch.long)
        
        if self.use_subsample:
            # Conv1d expects [B, C, T], already [B, 80, T]
            h = self.subsample(x)  # [B, D, T']
            # update lengths for stride=2 (ceil division)
            lengths = (lengths + 1) // 2
            h = h.transpose(1, 2)  # [B, T', D]
        else:
            # tokens per frame
            h = x.transpose(1, 2)  # [B, T, 80]
            h = self.in_proj(h)    # [B, T, D]
        
        h = self.pos(h)          # [B, T, D]

        # Padding mask: True = pad positions
        T = h.size(1)
        mask = torch.arange(T, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T]

        h = self.encoder(h, src_key_padding_mask=mask)  # [B, T, D]
        h = masked_mean_pool(h, lengths)  # [B, D]
        h = self.drop(h)
        return self.fc(h)
