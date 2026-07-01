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
## System requirements
 
**Operating systems.** Linux (tested on Ubuntu 22.04) and macOS (tested on
13/Ventura) for CPU-only use (data generation, MLP baseline). Operator
(FNO/WNO/MIONet/NOMAD/Sp²GNO) training additionally requires a CUDA-capable
Linux machine or cluster node; Windows is untested but should work under
WSL2 given the same dependencies.
 
**Software dependencies and version numbers.**
 
| Dependency | Version tested |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.x (CUDA 12.x build for GPU use) |
| NumPy | 1.26.x |
| SciPy | 1.11.x |
| scikit-learn | 1.3.x |
| SymPy | 1.12 |
| JAX + jaxlib (data generation only) | 0.4.23 |
| PyWavelets, ptwt, pytorch-wavelets (WNO only) | 1.5.0 / 0.1.8 / 1.3.0 |
| torch-geometric (Sp²GNO only) | **2.4.0 exactly** — later releases changed the `MessagePassing.propagate()` signature used by `cdtl/models/sp2gno.py` and will raise a `TypeError` |
| networkx (Sp²GNO only) | 3.2.x |
 
Exact pins are listed in `requirements.txt`, `requirements-data.txt`, and
`requirements-graph.txt`.
 
**Non-standard hardware.** None required for the MLP baseline or dataset
generation (CPU only). Full-scale operator training/transfer was performed on
NVIDIA A100 and GH200 GPUs.

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

1. **Generate locally**:

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

**Demo dataset.** A small demo dataset (2 source geometries, 40 samples each,
generated in seconds) is used in the *Demo* section below and requires no
download.

## Demo
 
This demo generates a small synthetic dataset and trains the MLP baseline on
it end to end, so you can verify the installation without downloading the full
1,200-sample-per-geometry dataset used in the paper.
 
```bash
# 1. Generate a small demo dataset (~10 seconds)
python scripts/generate_data.py \
    --geometries NSTX ITER --n_samples 40 --grid_size 64 \
    --out_dir data/demo --workers 2
 
# 2. Train + evaluate the MLP baseline on it
python scripts/train_mlp_baseline.py \
    --data_dir data/demo \
    --geometries NSTX ITER \
    --epochs 20 --out results/demo_mlp_baseline.csv
```
 
**Expected output.** The script prints per-epoch training progress followed by
a per-geometry summary, e.g.:
 
```
NSTX  rel_L2 = 0.041  (params: 82,305)
ITER  rel_L2 = 0.037  (params: 82,305)
```
 
and writes `results/demo_mlp_baseline.csv` with one row per geometry
(`geometry, rel_L2, n_params`) plus a ready-to-paste LaTeX table row. Exact
error values will vary slightly with hardware/seed, but should be well under
10% relative L2 error after 20 epochs on this simplified demo.
 
**Expected run time.** Under 1 minute total on a normal desktop CPU (no GPU
required for this demo).

## Instructions for use
 
### Running the MLP baseline on your own data
 
The MLP baseline is the simple/trivial reference model required by the Nature
Communications ML reporting checklist (Item 4D). It regresses each grid point's
`(A, ε, κ, δ, R, Z)` features to the flux ψ independently, with no operator
structure.
 
```bash
python scripts/train_mlp_baseline.py \
    --data_dir data/Solovev \
    --geometries NSTX ITER D3D0 D3D1 ITER_REACTOR \
    --epochs 150 --out results/mlp_baseline.csv
```
 
The script prints a per-geometry mean relative L2 error and a ready-to-paste
LaTeX row for Table 1 of the manuscript.
 
To run on your own data, place `.mat` files following the same schema as
`data/Solovev/` (arrays `coeff` (N, s) and `sol` (N, s, s); see `cdtl/data.py`)
in a directory and pass it via `--data_dir`, with `--geometries` matching your
filenames.
 
### Reproducing the operator benchmark (optional)
 
The five operator architectures and the four transfer strategies
(individual, multi-head, single-head base, single-head full) are defined in
`cdtl/models/`. Training and transfer driver scripts mirror the methodology in
the paper; see `docs/REPRODUCIBILITY.md` for the exact hyperparameters
(150 epochs full-training, 100 epochs transfer, Adam, step decay), the
source→target split, and step-by-step instructions to reproduce every
quantitative result reported in the manuscript.


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
