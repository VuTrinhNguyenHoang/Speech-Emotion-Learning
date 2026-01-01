from __future__ import annotations
import torch
import torchaudio
from ..config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, FMIN, FMAX, EPS

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=2.0,  # power spectrogram
)

def log_mel(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [1, N]
    returns: [N_MELS, T]
    """
    mel = _mel(wav)  # [1, n_mels, T]
    mel = mel.squeeze(0)
    return torch.log(mel + EPS)

def to_ml_vector(logmel: torch.Tensor) -> torch.Tensor:
    """
    logmel: [N_MELS, T]
    returns: [2*N_MELS] = concat(mean, std) over time
    """
    mean = logmel.mean(dim=1)
    std = logmel.std(dim=1)
    return torch.cat([mean, std], dim=0)
