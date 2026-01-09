from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from .paths import PROCESSED_DIR, SPLITS_DIR
from .data.dataset import LogMelCacheDataset, pad_collate
from .models.cnn import SimpleCNN, ResNet18
from .models.transformer import TransformerSER
from .train_dl import set_seed, build_model

def load_label_map(label_map_path: Path) -> tuple[dict[str, int], list[str]]:
    """
    Returns:
      intent2id: dict intent -> id
      id2intent_list: list where index=id, value=intent
    """
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Missing label map: {label_map_path}. "
            f"Expected file created by your FSC prep notebook (fsc_label_map.json)."
        )
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    intent2id = data["intent2id"]
    id2intent = data.get("id2intent", None)

    if isinstance(id2intent, dict):
        # keys might be strings in JSON
        max_id = max(int(k) for k in id2intent.keys())
        id2intent_list = [""] * (max_id + 1)
        for k, v in id2intent.items():
            id2intent_list[int(k)] = v
    else:
        # fallback: derive from intent2id
        max_id = max(intent2id.values())
        id2intent_list = [""] * (max_id + 1)
        for intent, idx in intent2id.items():
            id2intent_list[int(idx)] = intent

    # sanity: no empty holes
    if any(x == "" for x in id2intent_list):
        raise ValueError("Label map has missing ids (holes). Please regenerate fsc_label_map.json.")

    return intent2id, id2intent_list

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, float, str]:
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    crit = nn.CrossEntropyLoss()

    for X, y, lengths in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X, lengths)
        loss = crit(logits, y)
        total_loss += float(loss.item()) * y.size(0)

        pred = logits.argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    avg_loss = total_loss / max(1, len(ys))
    macro_f1 = float(f1_score(ys, ps, average="macro"))
    report = classification_report(ys, ps, target_names=target_names, digits=4, zero_division=0)
    return avg_loss, macro_f1, report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="Transformer", choices=["SimpleCNN", "ResNet18", "Transformer"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_subsample", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--attn_type", type=str, default="sdpa", choices=["sdpa", "linear"])
    ap.add_argument("--kernel", type=str, default="elu", choices=["elu", "relu"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label map
    intent2id, id2intent = load_label_map(SPLITS_DIR / "fsc_label_map.json")
    num_classes = len(id2intent)

    # Load cached datasets (your cache is here: data/processed/fsc_logmel_3s_16k_nmel80/train)
    cache_root = PROCESSED_DIR / "fsc_logmel_3s_16k_nmel80"
    train_dir = cache_root / "train"
    val_dir = cache_root / "val"
    test_dir = cache_root / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():

            raise FileNotFoundError(f"Missing cache split dir: {d}")

    train_ds = LogMelCacheDataset(train_dir)
    val_ds = LogMelCacheDataset(val_dir)
    test_ds = LogMelCacheDataset(test_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    model = build_model(args, num_classes=num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_path = Path(f"best_slu_{args.arch.lower()}_{args.attn_type}.pt")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X, lengths)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * y.size(0)
            n_seen += y.size(0)

        train_loss = running / max(1, n_seen)
        val_loss, val_f1, _ = evaluate(model, val_loader, device, target_names=id2intent)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": num_classes,
                    "label_map": str(SPLITS_DIR / "fsc_label_map.json"),
                },
                best_path,
            )
            print(f"  [OK] saved best -> {best_path} (val_macroF1={best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("  [STOP] early stopping")
                break

    # Load best and evaluate on test set
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_loss, test_f1, test_report = evaluate(model, test_loader, device, target_names=id2intent)
    print(f"\n[TEST] loss={test_loss:.4f} macroF1={test_f1:.4f}\n")
    print(test_report)


if __name__ == "__main__":
    main()