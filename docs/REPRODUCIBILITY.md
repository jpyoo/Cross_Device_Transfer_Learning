# Reproducibility Notes

This document records the exact settings needed to reproduce the results in the
paper.

## Hardware

- Dataset generation and the numerical baseline: NVIDIA GH200 Superchip,
  72-core ARM Grace CPU, `multiprocessing` parallelization (no GPU).
- Operator training/inference: NVIDIA A100 (Delta cluster, NCSA) and GH200.
- The MLP baseline runs on CPU or a single GPU.

## Source / target split

| Role   | Geometries                                  |
|--------|---------------------------------------------|
| Source | NSTX, ITER, D3D0, D3D1, ITER-Reactor        |
| Target | SPHER, KSTAR, D3D-ITER                       |

Geometry parameters `(ε, κ, δ)` are listed in `cdtl/geometries.py` and match
Supplementary Table S1. No target-geometry sample is used during pre-training.

## Data

- `A ~ Uniform[-0.2, 1.0]`, 1,200 samples per geometry (1,000 train / 200 test).
- Native 64×64 grid, subsampled by stride 2 to 32×32 (1,024 points) at load time.
- Train/test split seed: `random_state = 100` (fixed; see *Cross-validation*).

## Hyperparameters

| Parameter        | Full-training | Transfer |
|------------------|---------------|----------|
| Batch size       | 32            | 32       |
| Max epochs       | 150           | 100      |
| Learning rate    | 1e-3          | 1e-2     |
| LR decay steps   | 10            | 10       |
| LR decay rate    | 0.5           | 0.5      |
| Optimizer        | Adam          | Adam     |
| Weight decay     | 1e-5          | 1e-6     |
| Early-stop patience | 50         | 50       |

These match Supplementary Table S12 and are the defaults in
`scripts/train_mlp_baseline.py`.

## Transfer strategies

1. **Individual** — pre-train per source geometry, transfer directly.
2. **Multi-head** — shared backbone, one head per source task.
3. **Single-head base** — shared backbone + single head; fine-tune backbone.
4. **Single-head full** — shared backbone + single head; fine-tune head only.

Transfer freezes the operator backbone and updates only the projection head
(`Q` for FNO/WNO/Sp²GNO; final combined layers for NOMAD; final branch/trunk
layers for MIONet).

## Cross-validation

A single fixed random split is used rather than k-fold cross-validation,
consistent with standard practice in neural-operator benchmarking: training five
architectures across four transfer strategies and four fine-tuning budgets makes
repeated splitting computationally prohibitive. The 200-sample test set per
geometry is drawn independently of `N_target` and never used for training or
model selection. To assess split sensitivity, vary `--data_seed` / `--seed`.

## Determinism

Set `--seed` (model init / shuffling) and `--data_seed` (train/test split).
Exact bitwise reproducibility additionally requires fixed cuDNN flags; small
run-to-run variation on GPU is expected and does not affect the reported
conclusions.

## Note on A recovery

When loading data, the profile constant `A` is recovered from the stored
particular-solution row by averaging over R. Because the 32×32 evaluation grid
is rebuilt as `linspace(r_min, r_max, 32)` rather than the stride-2 subsample of
the 64-point grid, this recovery is approximate (a known property of the original
pipeline, preserved here for exact reproducibility). `scripts/generate_data.py`
also stores the exact `A` array, which can be used directly if preferred.
