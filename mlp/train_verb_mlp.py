"""Train CartoonAliveMLP on a verb-generated NPZ dataset.

Usage:
    uv run python -m mlp.train_verb_mlp \\
        --data mlp/data/live_portrait/datasets/dev_500.npz \\
        --out mlp/checkpoints/humanoid-anime-dev \\
        --epochs 200
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp.model import CartoonAliveMLP

logger = logging.getLogger(__name__)


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
    early_stop_patience: int = 20,
    seed: int = 42,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", data_path)
    d = np.load(data_path, allow_pickle=True)
    features = d["features"].astype(np.float32)   # (N, 1014)
    labels = d["labels"].astype(np.float32)       # (N, P)
    param_names = [str(s) for s in d["param_names"]]

    N, input_dim = features.shape
    _, n_params = labels.shape
    logger.info("N=%d  input_dim=%d  n_params=%d (%s)", N, input_dim, n_params, param_names)

    # Train/val split BEFORE computing norm stats (avoid leakage)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = features[train_idx]
    Y_train = labels[train_idx]
    X_val = features[val_idx]
    Y_val = labels[val_idx]

    # Normalization stats from training data
    lm_mean = X_train.mean(axis=0)
    lm_std = X_train.std(axis=0)
    p_mean = Y_train.mean(axis=0)
    p_std = Y_train.std(axis=0)
    # Guard against zero-std params (constant targets → skip normalization)
    p_std_safe = np.where(p_std < 1e-6, 1.0, p_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    model = CartoonAliveMLP(n_params=n_params, input_dim=input_dim).to(device)
    model.set_norm_stats(lm_mean, lm_std, p_mean, p_std_safe)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 0
    history: list[tuple[int, float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        scheduler.step()

        # Validation
        model.eval()
        val_preds: list[np.ndarray] = []
        val_trues: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(yb.numpy())
        preds = np.concatenate(val_preds)
        trues = np.concatenate(val_trues)
        val_mse = float(((preds - trues) ** 2).mean())
        history.append((epoch, train_loss, val_mse))

        if val_mse < best_val:
            best_val = val_mse
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "n_params": n_params,
                    "param_names": param_names,
                    "epoch": epoch,
                    "val_mse": val_mse,
                },
                output_dir / "model.pt",
            )
        else:
            patience += 1
            if patience >= early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info("epoch %3d  train=%.5f  val=%.5f  best=%.5f", epoch, train_loss, val_mse, best_val)

    # Final evaluation on best model
    logger.info("Best val MSE: %.5f", best_val)
    best = torch.load(output_dir / "model.pt", weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val).to(device)
        preds = model(xb).cpu().numpy()

    # Per-param R² and RMSE
    ss_res = ((preds - Y_val) ** 2).sum(axis=0)
    ss_tot = ((Y_val - Y_val.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot_safe = np.where(ss_tot < 1e-9, 1.0, ss_tot)
    r2 = 1.0 - (ss_res / ss_tot_safe)
    rmse = np.sqrt(((preds - Y_val) ** 2).mean(axis=0))

    logger.info("Per-param validation metrics:")
    logger.info("  %-12s  %7s  %7s  %7s", "param", "R²", "RMSE", "label_std")
    for name, r, e, ls in zip(param_names, r2, rmse, Y_val.std(axis=0)):
        logger.info("  %-12s  %+7.3f  %7.4f  %7.4f", name, r, e, ls)

    # Save history + metrics
    (output_dir / "history.csv").write_text(
        "epoch,train_loss,val_mse\n" + "\n".join(f"{e},{t:.6f},{v:.6f}" for e, t, v in history) + "\n"
    )
    (output_dir / "metrics.csv").write_text(
        "param,r2,rmse,label_std\n"
        + "\n".join(f"{n},{r:.4f},{e:.4f},{ls:.4f}" for n, r, e, ls in zip(param_names, r2, rmse, Y_val.std(axis=0)))
        + "\n"
    )
    logger.info("Saved model + history + metrics to %s", output_dir)
    return {"best_val_mse": best_val, "r2": r2, "rmse": rmse}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )
