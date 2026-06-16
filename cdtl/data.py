"""Dataset loading and feature construction for Solov'ev equilibria.

This module reproduces, in a single reusable place, the data-handling logic
that was previously duplicated across the per-architecture notebooks. A sample
maps the conditioning vector and grid coordinates

    (A, epsilon, kappa, delta, R, Z)   ->   psi(R, Z)

evaluated on a uniform ``s x s`` grid, where ``A`` is the Solov'ev profile
constant and ``(epsilon, kappa, delta)`` are the geometry parameters.

The raw ``.mat`` files contain two arrays:
    ``coeff`` : (N, s) particular-solution profile psi_p evaluated along R
    ``sol``   : (N, s, s) homogeneous-solution grid psi_H

and the full flux is reconstructed as psi = psi_H + broadcast(psi_p), matching
the original training pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from .geometries import Geometry, get_geometry


@dataclass
class GeometryData:
    """Train/test tensors for one geometry in grid (image) layout.

    Tensors use the channel-last convention expected by the grid-based
    operators and the MLP baseline:
        inputs  : (N, s, s, 6)  channels = [A, epsilon, kappa, delta, R, Z]
        outputs : (N, s, s, 1)
    """

    name: str
    train_input: torch.Tensor
    train_output: torch.Tensor
    test_input: torch.Tensor
    test_output: torch.Tensor
    grid: torch.Tensor  # (s, s, 2) the (R, Z) coordinates for this geometry

    @property
    def s(self) -> int:
        return self.train_input.shape[1]


def _build_grid(geom: Geometry, s: int) -> np.ndarray:
    """Construct the (s, s, 2) uniform (R, Z) grid for a geometry."""
    r_min, r_max = 1.0 - geom.epsilon, 1.0 + geom.epsilon
    z_max = geom.kappa * geom.epsilon
    r_array = np.linspace(r_min, r_max, s).astype(np.float32)
    z_array = np.linspace(-z_max, z_max, s).astype(np.float32)
    grid = np.vstack(
        [xx.ravel() for xx in np.meshgrid(r_array, z_array)]
    ).T  # (s*s, 2)
    return grid.reshape(s, s, 2), r_array


def _recover_A(coeff_rows: np.ndarray, r_array: np.ndarray) -> np.ndarray:
    """Recover the scalar profile constant A from the particular-solution row.

    The particular solution is psi_p(R) = A * R^4 / 8 + (1 - A) * (R^2 / 2) ln R,
    rearranged here to solve for A and averaged over R for numerical stability,
    exactly as done in the original notebooks.
    """
    n = coeff_rows.shape[0]
    a_vals = np.zeros(n, dtype=np.float32)
    denom = 0.5 * np.log(r_array) * (r_array ** 2) - 0.125 * r_array ** 4
    for i, psi_p in enumerate(coeff_rows):
        a_vals[i] = np.mean((psi_p - 0.125 * r_array ** 4) / denom)
    return a_vals


def load_geometry_data(
    geom_name: str,
    data_dir: str,
    n_train: int = 1000,
    n_test: int = 200,
    subsample: int = 2,
    grid_size: int = 64,
    random_state: int = 100,
    mat_file: str | None = None,
) -> GeometryData:
    """Load and featurize one geometry's Solov'ev dataset.

    Parameters
    ----------
    geom_name : str
        Geometry key (e.g. "NSTX", "SPHER").
    data_dir : str
        Directory containing the ``.mat`` files.
    n_train, n_test : int
        Number of training / test samples.
    subsample : int
        Stride applied to the high-fidelity grid (2 maps 64x64 -> 32x32),
        reproducing the resolution-independent evaluation in the paper.
    grid_size : int
        Native resolution of the stored data before subsampling.
    random_state : int
        Seed for the train/test split (fixed at 100 in the paper).
    mat_file : str, optional
        Override the default ``.mat`` filename for this geometry.
    """
    geom = get_geometry(geom_name)
    fname = mat_file or geom.mat_file
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset for {geom_name} not found at '{path}'. "
            f"Generate it with scripts/generate_data.py or download the release."
        )

    r = subsample
    s = grid_size // r
    raw = loadmat(path)
    coeff = raw["coeff"][:, ::r]          # (N, s)
    sol = raw["sol"][:, ::r, ::r]         # (N, s, s)

    grid_np, r_array = _build_grid(geom, s)
    grid = torch.tensor(grid_np)          # (s, s, 2)

    coeff_tr, coeff_te, sol_tr, sol_te = train_test_split(
        coeff, sol, test_size=n_test, random_state=random_state, shuffle=True
    )
    coeff_tr, sol_tr = coeff_tr[:n_train], sol_tr[:n_train]

    a_tr = _recover_A(coeff_tr, r_array)
    a_te = _recover_A(coeff_te, r_array)

    def _assemble(a_vals, coeff_rows, sol_grid, n):
        a_ch = torch.tensor(a_vals).view(n, 1, 1, 1).repeat(1, s, s, 1)
        geom_ch = (
            torch.tensor(geom.params, dtype=torch.float32)
            .view(1, 1, 1, 3)
            .repeat(n, s, s, 1)
        )
        grid_ch = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        inp = torch.cat([a_ch, geom_ch, grid_ch], dim=-1).float()  # (n, s, s, 6)

        psi_p = torch.tensor(coeff_rows).view(n, 1, s, 1).repeat(1, s, 1, 1)
        out = (torch.tensor(sol_grid).view(n, s, s, 1) + psi_p).float()
        return inp, out

    train_input, train_output = _assemble(a_tr, coeff_tr, sol_tr, n_train)
    test_input, test_output = _assemble(a_te, coeff_te, sol_te, n_test)

    return GeometryData(
        name=geom_name,
        train_input=train_input,
        train_output=train_output,
        test_input=test_input,
        test_output=test_output,
        grid=grid,
    )
