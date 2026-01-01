from __future__ import annotations
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable
import pandas as pd

from ..config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
from .crema import CremaItem

def make_speaker_split(items: list[CremaItem]) -> dict[str, list[CremaItem]]:
    speakers = sorted({it.speaker for it in items})
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train:n_train + n_val])
    test_spk = set(speakers[n_train + n_val:])

    split = {"train": [], "val": [], "test": []}
    for it in items:
        if it.speaker in train_spk:
            split["train"].append(it)
        elif it.speaker in val_spk:
            split["val"].append(it)
        else:
            split["test"].append(it)
    return split

def split_to_df(split_items: list[CremaItem]) -> pd.DataFrame:
    return pd.DataFrame([asdict(it) for it in split_items])
