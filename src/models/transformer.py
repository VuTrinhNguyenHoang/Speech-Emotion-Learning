from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

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

class LinearMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        kernel: Literal["elu", "relu"] = "elu",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.kernel = kernel
        self.eps = eps

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.record_attn: bool = False
        self.attn_map: Optional[torch.Tensor] = None

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "elu":
            return F.elu(x, alpha=1.0) + 1.0
        elif self.kernel == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported kernel: {self.kernel}")

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, C]
        src_key_padding_mask: [B, N] with True=PAD (same convention as PyTorch)
        """
        B, N, C = x.shape
        H, D = self.h, self.d

        qkv = self.qkv(x).view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,D]

        qf = self._phi(q)
        kf = self._phi(k)

        # Mask padded tokens out of K/V (important for speech batching)
        if src_key_padding_mask is not None:
            # mask: True=PAD -> valid = ~mask
            valid = (~src_key_padding_mask).to(kf.dtype)  # [B, N]
            valid = valid[:, None, :, None]               # [B, 1, N, 1]
            kf = kf * valid
            v = v * valid

        kf_drop = self.attn_drop(kf)

        if not self.record_attn:
            kv = torch.matmul(kf_drop.transpose(-2, -1), v)        # [B, H, D, D]
            z = kf_drop.sum(dim=2)                                 # [B, H, D]
            y_num = torch.matmul(qf, kv)                           # [B, H, N, D]
            y_den = torch.einsum("bhnd,bhd->bhn", qf, z)           # [B, H, N]
            y_den = y_den.unsqueeze(-1).clamp_min(self.eps)        # [B, H, N, 1]
            y = (y_num / y_den).transpose(1, 2).reshape(B, N, C)   # [B, N, C]
            y = self.proj_drop(self.out_proj(y))
            self.attn_map = None
            return y

        # Debug path (O(N^2))
        sim = torch.einsum("bhid,bhjd->bhij", qf, kf_drop).clamp_min(0.0)
        if src_key_padding_mask is not None:
            # zero-out attention to PAD keys
            key_valid = (~src_key_padding_mask).to(sim.dtype)      # [B, N]
            sim = sim * key_valid[:, None, None, :]                # [B,H,N,N]

        attn = sim / (sim.sum(dim=-1, keepdim=True) + self.eps)     # [B, H, N, N]
        self.attn_map = attn
        if attn.requires_grad:
            self.attn_map.retain_grad()

        y = torch.einsum("bhij,bhjd->bhid", attn, v)                # [B, H, N, D]
        y = y.transpose(1, 2).reshape(B, N, C)                      # [B, N, C]
        y = self.proj_drop(self.out_proj(y))
        return y

class LinearTransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer that uses LinearMultiheadAttention.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        kernel: Literal["elu", "relu"] = "elu",
    ):
        super().__init__()
        self.self_attn = LinearMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            kernel=kernel,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.norm_first = norm_first

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_first:
            attn_out = self.self_attn(self.norm1(src), src_key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)

            ff = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(ff)
        else:
            attn_out = self.self_attn(src, src_key_padding_mask=src_key_padding_mask)
            src = self.norm1(src + self.dropout1(attn_out))

            ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ff))
        return src

class _LinearEncoder(nn.Module):
    def __init__(self, layers: list[LinearTransformerEncoderLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

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
        attn_type: Literal["sdpa", "linear"] = "sdpa",
        kernel: Literal["elu", "relu"] = "elu",
    ):
        super().__init__()
        self.use_subsample = use_subsample
        self.attn_type = attn_type
        self.kernel = kernel

        if use_subsample:
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

        if attn_type == "linear":
            layers = [
                LinearTransformerEncoderLayer(
                    d_model=proj_out,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    norm_first=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qkv_bias=True,
                    kernel=kernel,
                )
                for _ in range(num_layers)
            ]
            self.encoder = _LinearEncoder(layers)
        else:
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
        else:
            lengths = lengths.to(x.device)

        if self.use_subsample:
            h = self.subsample(x)          # [B, D, T']
            lengths = (lengths + 1) // 2   # stride=2
            h = h.transpose(1, 2)          # [B, T', D]
        else:
            h = x.transpose(1, 2)          # [B, T, 80]
            h = self.in_proj(h)            # [B, T, D]

        h = self.pos(h)                    # [B, T, D]

        T = h.size(1)
        pad_mask = torch.arange(T, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T], True=PAD

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        h = masked_mean_pool(h, lengths)
        h = self.drop(h)
        return self.fc(h)
