"""Tokamak geometry definitions for the Solov'ev transfer-learning benchmark.

Each geometry is described by three boundary-shape parameters used in the
Cerfon--Freidberg analytic Solov'ev solution (Cerfon & Freidberg, 2010):

    epsilon : inverse aspect ratio (a / R0)
    kappa   : elongation
    delta   : triangularity

The eight configurations are split into five ``source`` geometries used for
pre-training and three ``target`` geometries held out for transfer evaluation.
These values match Supplementary Table S1 of the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Geometry:
    """Boundary-shape parameters for a single tokamak-like configuration."""

    name: str
    epsilon: float  # inverse aspect ratio a / R0
    kappa: float    # elongation
    delta: float    # triangularity
    role: str       # "source" or "target"
    mat_file: str = field(default="")  # filename of the .mat dataset

    @property
    def params(self) -> List[float]:
        """Return [epsilon, kappa, delta] in the channel order used by the models."""
        return [self.epsilon, self.kappa, self.delta]


# ---------------------------------------------------------------------------
# Canonical geometry table (Supplementary Table S1).
# `mat_file` names follow the dataset release; override via the data config if
# your local filenames differ.
# ---------------------------------------------------------------------------
GEOMETRIES: Dict[str, Geometry] = {
    "NSTX": Geometry(
        "NSTX", epsilon=0.78, kappa=2.00, delta=0.35, role="source",
        mat_file="solovev_data_NSTX.mat",
    ),
    "ITER": Geometry(
        "ITER", epsilon=0.32, kappa=1.70, delta=0.33, role="source",
        mat_file="solovev_data_ITER.mat",
    ),
    "D3D0": Geometry(
        "D3D0", epsilon=0.37, kappa=1.55, delta=0.43, role="source",
        mat_file="solovev_data_DIII_D_0.mat",
    ),
    "D3D1": Geometry(
        "D3D1", epsilon=0.37, kappa=1.60, delta=0.36, role="source",
        mat_file="solovev_data_DIII_D_1.mat",
    ),
    "ITER_REACTOR": Geometry(
        "ITER_REACTOR", epsilon=0.43, kappa=1.98, delta=0.22, role="source",
        mat_file="solovev_data_ITER_reactor.mat",
    ),
    "SPHER": Geometry(
        "SPHER", epsilon=0.95, kappa=1.00, delta=0.20, role="target",
        mat_file="solovev_data_Spheromak.mat",
    ),
    "KSTAR": Geometry(
        "KSTAR", epsilon=0.25, kappa=1.81, delta=0.76, role="target",
        mat_file="solovev_data_KSTAR.mat",
    ),
    "D3D_ITER": Geometry(
        "D3D_ITER", epsilon=0.35, kappa=1.17, delta=0.08, role="target",
        mat_file="solovev_data_DIII_D_ITER_Like.mat",
    ),
}

SOURCE_GEOMETRIES: List[str] = [k for k, g in GEOMETRIES.items() if g.role == "source"]
TARGET_GEOMETRIES: List[str] = [k for k, g in GEOMETRIES.items() if g.role == "target"]


def get_geometry(name: str) -> Geometry:
    """Look up a geometry by name (case-insensitive)."""
    key = name.upper()
    if key not in GEOMETRIES:
        raise KeyError(
            f"Unknown geometry '{name}'. Available: {sorted(GEOMETRIES)}"
        )
    return GEOMETRIES[key]
