#!/usr/bin/env python3
"""Generate Solov'ev equilibrium datasets for one or more geometries.

Produces ``.mat`` files compatible with stone.data.load_geometry_data and the
per-architecture training scripts. For each geometry it samples the profile
constant A uniformly on [-0.2, 1.0] and evaluates the analytic Cerfon--Freidberg
Solov'ev flux on an ``s x s`` grid, storing:

    coeff : (N, s)      particular-solution profile psi_p(R)
    sol   : (N, s, s)   homogeneous-solution grid psi_H(R, Z)

The full flux used in training is psi = psi_H + broadcast(psi_p).

Requires JAX and SymPy (see requirements-data.txt). Generation is CPU-bound and
embarrassingly parallel across samples; use --workers to parallelize.

Example
-------
    python scripts/generate_data.py \
        --geometries NSTX ITER D3D0 D3D1 ITER_REACTOR SPHER KSTAR D3D_ITER \
        --n_samples 1200 --grid_size 64 --out_dir data/Solovev --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdtl.geometries import GEOMETRIES, get_geometry
from cdtl.solovev import Device


def generate_one_sample(A: float, geom_name: str, grid_size: int):
    """Compute (psi_p_row, psi_H_grid) for a single profile constant A."""
    geom = get_geometry(geom_name)
    device = Device(
        pressure=A,
        epsilon=geom.epsilon,
        kappa=geom.kappa,
        delta=geom.delta,
        device_name=geom.name,
    )
    r_min, r_max = 1.0 - geom.epsilon, 1.0 + geom.epsilon
    z_max = geom.kappa * geom.epsilon
    r_arr = np.linspace(r_min, r_max, grid_size)
    z_arr = np.linspace(-z_max, z_max, grid_size)

    # Particular-solution profile along R (homogeneous part handled separately).
    psi_p_row = np.array([device.psiP(r, 0.0) for r in r_arr], dtype=np.float64)

    # Homogeneous solution on the full grid.
    psi_H = np.zeros((grid_size, grid_size), dtype=np.float64)
    for iz, z in enumerate(z_arr):
        for ir, r in enumerate(r_arr):
            psi_H[iz, ir] = device.psiH(r, z)
    return psi_p_row, psi_H


def generate_geometry(geom_name, n_samples, grid_size, a_low, a_high, workers, out_dir, seed):
    rng = np.random.default_rng(seed)
    A_values = rng.uniform(a_low, a_high, size=n_samples)

    worker = partial(generate_one_sample, geom_name=geom_name, grid_size=grid_size)
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(worker, A_values.tolist())
    else:
        results = [worker(A) for A in A_values.tolist()]

    coeff = np.stack([r[0] for r in results], axis=0)   # (N, s)
    sol = np.stack([r[1] for r in results], axis=0)     # (N, s, s)

    os.makedirs(out_dir, exist_ok=True)
    fname = GEOMETRIES[geom_name.upper()].mat_file
    path = os.path.join(out_dir, fname)
    savemat(path, {"coeff": coeff, "sol": sol, "A": A_values})
    print(f"{geom_name}: wrote {path}  coeff{coeff.shape} sol{sol.shape}")


def main():
    p = argparse.ArgumentParser(description="Generate Solov'ev datasets")
    p.add_argument("--geometries", nargs="+", default=list(GEOMETRIES))
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--grid_size", type=int, default=64)
    p.add_argument("--a_low", type=float, default=-0.2)
    p.add_argument("--a_high", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out_dir", default="data/Solovev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for name in args.geometries:
        generate_geometry(
            name, args.n_samples, args.grid_size,
            args.a_low, args.a_high, args.workers, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
