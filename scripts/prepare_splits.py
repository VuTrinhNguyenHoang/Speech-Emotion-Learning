from pathlib import Path
from src.paths import CREMA_WAV_DIR, SPLITS_DIR
from src.data.crema import scan_crema
from src.data.split import make_speaker_split, split_to_df

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    items = scan_crema(CREMA_WAV_DIR)
    split = make_speaker_split(items)

    for name, its in split.items():
        df = split_to_df(its)
        out = SPLITS_DIR / f"crema_{name}.csv"
        df.to_csv(out, index=False)
        print(f"[OK] {name}: {len(df)} -> {out}")

    # sanity: number of unique speakers per split
    for name in ["train", "val", "test"]:
        df = split_to_df(split[name])
        print(name, "unique speakers:", df["speaker"].nunique())

if __name__ == "__main__":
    main()
