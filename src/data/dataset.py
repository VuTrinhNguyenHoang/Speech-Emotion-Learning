from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class LogMelCacheDataset(Dataset):
    """
    Reads cached .pt files with structure:
      {"x": [80, T], "y": int, "speaker": str, "path": str}
    Returns:
      x: FloatTensor [1, 80, T]  (channel-first for CNN)
      y: LongTensor []
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.files: List[Path] = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt cache files in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = torch.load(self.files[idx], map_location="cpu")
        x = obj["x"]  # [80, T]
        y = obj["y"]
        x = x.unsqueeze(0)  # [1, 80, T]
        return x, torch.tensor(y, dtype=torch.long)

def pad_collate(batch):
    """
    Pad variable T to max T in batch along time axis.
    batch: list of (x [1,80,T], y)
    returns:
      X: [B,1,80,Tmax]
      y: [B]
      lengths: [B] original T
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([t.size(-1) for t in xs], dtype=torch.long)
    Tmax = int(lengths.max().item())

    X = torch.zeros((len(xs), 1, xs[0].size(1), Tmax), dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.size(-1)
        X[i, :, :, :T] = x
    y = torch.stack(list(ys), dim=0)
    return X, y, lengths