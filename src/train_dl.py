from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from .paths import PROCESSED_DIR
from .data.dataset import LogMelCacheDataset, pad_collate
from .data.labels import EMO_MAP
from .models.cnn import SimpleCNN, ResNet18
from .models.transformer import TransformerSER

EMOTIONS = list(EMO_MAP.values())

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(args, num_classes: int) -> nn.Module:
    arch = args.arch.lower()
    if arch == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif arch == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif arch == "transformer":
        return TransformerSER(
            num_classes=num_classes,
            n_mels=80,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.layers,
            dim_feedforward=args.ffn,
            dropout=args.dropout,
            use_subsample=not args.no_subsample,
        )
    raise ValueError(f"Unknown arch: {arch}")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, str]:
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

    avg_loss = total_loss / len(ys)
    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average="macro"))
    report = classification_report(ys, ps, target_names=EMOTIONS, digits=4)
    return avg_loss, macro_f1, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="SimpleCNN", choices=["SimpleCNN", "ResNet18", "Transformer"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_subsample", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = PROCESSED_DIR / "logmel_3s_16k_nmel80"
    train_ds = LogMelCacheDataset(cache_root / "train")
    val_ds = LogMelCacheDataset(cache_root / "val")
    test_ds = LogMelCacheDataset(cache_root / "test")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=True
    )

    num_classes = len(EMOTIONS)
    model = build_model(args, num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_path = Path(f"best_{args.arch.lower()}.pt")
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
        val_loss, val_f1, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({"model": model.state_dict(), "arch": args.arch}, best_path)
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

    test_loss, test_f1, test_report = evaluate(model, test_loader, device)
    print(f"\n[TEST] loss={test_loss:.4f} macroF1={test_f1:.4f}\n")
    print(test_report)

if __name__ == "__main__":
    main()