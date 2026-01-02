from __future__ import annotations
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import torch

from .paths import SPLITS_DIR, PROCESSED_DIR
from .config import SAMPLE_RATE, DURATION_SEC
from .data.labels import EMO2ID
from .data.features import to_ml_vector

def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()

def load_cached_features(csv_path: str, cache_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load pre-cached log-mel features from .pt files."""
    df = pd.read_csv(csv_path)
    X_list = []
    y_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(csv_path).name}"):
        wav_path = str(row["path"])
        uid = _hash_path(wav_path)
        cache_file = cache_root / f"{uid}.pt"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cached file not found: {cache_file}")
        
        sample = torch.load(cache_file, map_location="cpu")
        lm = sample["x"]  # [80, T]
        vec = to_ml_vector(lm).numpy()  # [FEATURE_DIM]
        X_list.append(vec)
        y_list.append(row["emotion"])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    labels = sorted(df["emotion"].unique().tolist())
    return X, y, labels

def main():
    train_csv = str(SPLITS_DIR / "crema_train.csv")
    val_csv   = str(SPLITS_DIR / "crema_val.csv")
    test_csv  = str(SPLITS_DIR / "crema_test.csv")
    
    cache_root = PROCESSED_DIR / "logmel_3s_16k_nmel80"

    X_tr, y_tr, _ = load_cached_features(train_csv, cache_root / "train")
    X_va, y_va, _ = load_cached_features(val_csv, cache_root / "val")
    X_te, y_te, _ = load_cached_features(test_csv, cache_root / "test")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC()),
    ])

    clf.fit(X_tr, y_tr)
    print("\n[VAL]\n", classification_report(y_va, clf.predict(X_va), digits=4))
    print("\n[TEST]\n", classification_report(y_te, clf.predict(X_te), digits=4))

if __name__ == "__main__":
    main()