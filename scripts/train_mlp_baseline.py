#!/usr/bin/env python3
"""Train the MLP baseline using the `cdtl` package (publication version).

Numerically identical to scripts/mlp_baseline_standalone.py, but imports the
shared data/loss/model modules instead of duplicating them. Use this version
inside the repository; use the standalone script if you want a single file with
no package dependency.

Example
-------
    python scripts/train_mlp_baseline.py \
        --data_dir /path/to/Solovev \
        --geometries NSTX ITER D3D0 D3D1 ITER_REACTOR \
        --epochs 150 --out results/mlp_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

# Allow running as `python scripts/train_mlp_baseline.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdtl.data import load_geometry_data
from cdtl.losses import LpLoss
from cdtl.baselines.mlp import MLPBaseline, count_parameters
from cdtl.geometries import SOURCE_GEOMETRIES


def train_one(name, data_dir, args, device):
    data = load_geometry_data(
        name, data_dir,
        n_train=args.n_train, n_test=args.n_test,
        subsample=args.subsample, grid_size=args.grid_size,
        random_state=args.data_seed,
        mat_file=args.mat_file.get(name) if args.mat_file else None,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.train_input, data.train_output),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.test_input, data.test_output),
        batch_size=args.batch_size)

    model = MLPBaseline(in_channels=6, width=args.width, depth=args.depth).to(device)
    loss_fn = LpLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_steps, gamma=args.decay_rate)

    best, patience = float("inf"), 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                total += loss_fn(model(xb.to(device)), yb.to(device)).item()
        test_loss = total / len(test_loader)
        if test_loss < best:
            best, patience = test_loss, 0
        else:
            patience += 1
        if patience > args.patience:
            break

    return best, count_parameters(model)


def main():
    p = argparse.ArgumentParser(description="MLP baseline (package version)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--geometries", nargs="+", default=SOURCE_GEOMETRIES)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--decay_steps", type=int, default=10)
    p.add_argument("--decay_rate", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--subsample", type=int, default=2)
    p.add_argument("--grid_size", type=int, default=64)
    p.add_argument("--data_seed", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/mlp_baseline.csv")
    args = p.parse_args()
    args.mat_file = {}  # optional per-geometry filename overrides

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    for name in args.geometries:
        err, n_params = train_one(name, args.data_dir, args, device)
        pct = 100.0 * err
        rows.append((name, pct, n_params))
        print(f"{name:>14s}: mean rel L2 = {pct:6.3f}%  (params={n_params})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "rel_L2_error_pct", "n_params"])
        for name, pct, n in rows:
            w.writerow([name, f"{pct:.3f}", n])
    print(f"\nWrote {args.out}")

    order = ["NSTX", "ITER", "D3D0", "D3D1", "ITER_REACTOR"]
    lut = {n: pct for n, pct, _ in rows}
    cells = " & ".join(f"{lut[g]:.2f}" if g in lut else "--" for g in order)
    print("\nLaTeX row for Table 1:")
    print(f"\\textbf{{MLP}} & Individual & {cells} \\\\")


if __name__ == "__main__":
    sys.exit(main())
