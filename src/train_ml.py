from __future__ import annotations
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import torch

from .paths import SPLITS_DIR
from .config import SAMPLE_RATE, DURATION_SEC
from .data.audio import load_mono_resample, pad_or_crop, peak_normalize
from .data.features import log_mel, to_ml_vector

def featurize_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    X_list = []
    y_list = []
    target_len = int(SAMPLE_RATE * DURATION_SEC)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Featurize {csv_path}"):
        wav, _ = load_mono_resample(row["path"], SAMPLE_RATE)
        wav = peak_normalize(wav)
        wav = pad_or_crop(wav, target_len)
        lm = log_mel(wav)                    # [N_MELS, T]
        vec = to_ml_vector(lm).numpy()       # [HOP_LENGTH]
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

    X_tr, y_tr, _ = featurize_csv(train_csv)
    X_va, y_va, _ = featurize_csv(val_csv)
    X_te, y_te, _ = featurize_csv(test_csv)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC()),
    ])

    clf.fit(X_tr, y_tr)
    print("\n[VAL]\n", classification_report(y_va, clf.predict(X_va), digits=4))
    print("\n[TEST]\n", classification_report(y_te, clf.predict(X_te), digits=4))

if __name__ == "__main__":
    main()