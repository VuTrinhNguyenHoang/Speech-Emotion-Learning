from __future__ import annotations
import torch
import torchaudio

def load_mono_resample(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # [C, N]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr  # [1, N], sr

def pad_or_crop(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    # wav: [1, N]
    n = wav.size(1)
    if n == target_len:
        return wav
    if n > target_len:
        return wav[:, :target_len]
    # pad
    pad_len = target_len - n
    return torch.nn.functional.pad(wav, (0, pad_len))

def peak_normalize(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = wav.abs().max()
    return wav / (peak + eps)
