# Cross-Device Transfer Learning for Grad–Shafranov Equilibria

Reference implementation for **"Towards Data-Efficient Cross-Device
Generalization of Grad–Shafranov Equilibria via Transfer Learning Neural
Operators."**

This repository benchmarks five neural operator architectures — Fourier (FNO),
Wavelet (WNO), MIONet, NOMAD, and a Spatio-Spectral Graph operator (Sp²GNO) —
on cross-geometry transfer of Solov'ev Grad–Shafranov equilibria, together with
a simple pointwise **MLP baseline**. Models are pre-trained on five source
tokamak geometries and adapted to three held-out target geometries under a
freeze-then-fine-tune protocol.

## Repository layout

```
cdtl/                     core library (importable package)
├── geometries.py          the 8 tokamak configurations (source/target split)
├── data.py                .mat loading + feature construction
├── losses.py              relative L2 loss / metric (LpLoss)
├── solovev.py             analytic Cerfon–Freidberg equilibrium solver
├── utilities.py           spectral/wavelet layers, MLP head, graph utilities
├── lipschitz.py           Lipschitz node embeddings for Sp²GNO
├── baselines/
│   └── mlp.py             pointwise MLP baseline model
└── models/                operator backbones + transfer-learning heads
    ├── fno.py  wno.py  mionet.py  nomad.py  sp2gno.py

scripts/
├── generate_data.py            (re)generate the Solov'ev .mat datasets
└── train_mlp_baseline.py       MLP baseline (uses the package)

configs/                    example configuration files
docs/                       model card and reproducibility notes
data/                       .mat datasets, not tracked in git
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For dataset generation only (JAX + SymPy):
pip install -r requirements-data.txt
# For the graph operator (Sp²GNO):
pip install -r requirements-graph.txt
```

Tested with Python 3.10, PyTorch 2.x, CUDA 12. Operator training and inference
were run on NVIDIA A100 / GH200 GPUs; the MLP baseline runs comfortably on CPU.

## Data

The datasets are analytically generated and fully reproducible. Either:

1. **Generate locally** (recommended for exact reproduction):

   ```bash
   python scripts/generate_data.py \
       --geometries NSTX ITER D3D0 D3D1 ITER_REACTOR SPHER KSTAR D3D_ITER \
       --n_samples 1200 --grid_size 64 --out_dir data/Solovev --workers 8
   ```

2. **Download** the released dataset archive (see *Data Availability* in the
   paper) and unpack it into `data/Solovev/`.

Each geometry yields 1,200 samples (1,000 train / 200 test). The 64×64 grids are
subsampled to 32×32 at load time for the resolution-independent evaluation used
in the paper.

## Quick start: MLP baseline

The MLP baseline is the simple/trivial reference model required by the Nature
Communications ML reporting checklist (Item 4D). It regresses each grid point's
`(A, ε, κ, δ, R, Z)` features to the flux ψ independently, with no operator
structure.

```bash
# Package version (run from repo root):
python scripts/train_mlp_baseline.py \
    --data_dir data/Solovev \
    --geometries NSTX ITER D3D0 D3D1 ITER_REACTOR \
    --epochs 150 --out results/mlp_baseline.csv
```

The script prints a per-geometry mean relative L2 error and a ready-to-paste
LaTeX row for Table 1 of the manuscript.

## Reproducing the operator benchmark

The five operator architectures and the four transfer strategies
(individual, multi-head, single-head base, single-head full) are defined in
`cdtl/models/`. Training and transfer driver scripts mirror the methodology in
the paper; see `docs/REPRODUCIBILITY.md` for the exact hyperparameters
(150 epochs full-training, 100 epochs transfer, Adam, step decay) and the
source→target split.

## Citation

```bibtex
@article{yoo2026cdtl,
  title   = {Towards Data-Efficient Cross-Device Generalization of
             Grad--Shafranov Equilibria via Transfer Learning Neural Operators},
  author  = {Yoo, Jay Phil and Howes, William and Ghai, Yashika and
             Kobayashi, Kazuma and Chakraborty, Souvik and Alam, Syed Bahauddin},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This work used Delta and DeltaAI at NCSA (NSF awards OAC-2005572 and
OAC-2320345).