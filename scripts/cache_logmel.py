from pathlib import Path
from src.paths import SPLITS_DIR, PROCESSED_DIR
from src.data.cache import cache_all_splits

def main():
    train_csv = SPLITS_DIR / "crema_train.csv"
    val_csv   = SPLITS_DIR / "crema_val.csv"
    test_csv  = SPLITS_DIR / "crema_test.csv"

    cache_root = PROCESSED_DIR / "logmel_3s_16k_nmel80"
    cache_all_splits(train_csv, val_csv, test_csv, cache_root)

if __name__ == "__main__":
    main()