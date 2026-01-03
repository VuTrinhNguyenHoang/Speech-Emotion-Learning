from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
import torch
from tqdm import tqdm

from ..config import SAMPLE_RATE, DURATION_SEC
from ..paths import PROCESSED_DIR
from .audio import load_mono_resample, pad_or_crop, peak_normalize
from .features import log_mel
from .labels import EMO2ID

def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()

def cache_split(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    target_len = int(SAMPLE_RATE * DURATION_SEC)
    n_ok, n_skip = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cache {csv_path.name}"):
        wav_path = str(row["path"])
        emo = row["emotion"]
        spk = str(row["speaker"])

        if emo not in EMO2ID:
            raise ValueError(f"Unknown emotion '{emo}' in {csv_path}")

        # Stable filename
        uid = _hash_path(wav_path)
        out_file = out_dir / f"{uid}.pt"
        if out_file.exists():
            n_skip += 1
            continue

        wav, _ = load_mono_resample(wav_path, SAMPLE_RATE)
        wav = peak_normalize(wav)
        wav = pad_or_crop(wav, target_len)

        lm = log_mel(wav)  # [80, T]
        sample = {
            "x": lm.to(torch.float32),      # [80, T]
            "y": int(EMO2ID[emo]),          # int
            "speaker": spk,                 # str
            "path": wav_path,               # for debugging
        }
        torch.save(sample, out_file)
        n_ok += 1

    print(f"[OK] wrote={n_ok}, skipped(existing)={n_skip}, out_dir={out_dir}")

def cache_split_fsc(csv_path: Path, out_dir: Path, intent2id: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    target_len = int(SAMPLE_RATE * DURATION_SEC)
    n_ok, n_skip = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cache {csv_path.name}"):
        wav_path = str(row["path"])
        intent = str(row["intent"])
        spk = str(row["speaker"])
        y = int(intent2id[intent])

        # Stable filename
        uid = _hash_path(wav_path)
        out_file = out_dir / f"{uid}.pt"
        if out_file.exists():
            n_skip += 1
            continue

        wav, _ = load_mono_resample(p, SAMPLE_RATE)
        wav = peak_normalize(wav)
        wav = pad_or_crop(wav, target_len)
        x = log_mel(wav).to(torch.float32)  # [80, T]
        sample = {
            "x": x,              # [80, T]
            "y": y,              # int
            "speaker": spk,     # str
            "intent": intent,  # str
            "path": wav_path,   # for debugging
        }
        torch.save(sample, out_file)
        n_ok += 1

    print(f"[OK] wrote={n_ok}, skipped(existing)={n_skip}, out_dir={out_dir}")

def cache_all_splits(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    cache_root: Path,
) -> None:
    cache_split(train_csv, cache_root / "train")
    cache_split(val_csv, cache_root / "val")
    cache_split(test_csv, cache_root / "test")

def cache_all_splits_fsc(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    cache_root: Path,
    intent2id: dict,
) -> None:
    cache_split_fsc(train_csv, cache_root / "train", intent2id)
    cache_split_fsc(val_csv, cache_root / "val", intent2id)
    cache_split_fsc(test_csv, cache_root / "test", intent2id)