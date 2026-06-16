# Model Card — Neural Operators for Solov'ev Equilibria

## Model details

- **Developed by:** Yoo, Howes, Ghai, Kobayashi, Chakraborty, Alam (2026).
- **Model type:** Neural operators learning the geometry-conditioned solution
  operator of the Solov'ev family of the Grad–Shafranov equation.
- **Architectures:** FNO, WNO, MIONet, NOMAD, Sp²GNO, plus a pointwise MLP
  baseline.
- **Mapping learned:** `(A, ε, κ, δ) → ψ(R, Z)`, where `A` is the Solov'ev
  profile constant and `(ε, κ, δ)` are inverse aspect ratio, elongation, and
  triangularity. Output is the poloidal flux on a 32×32 grid.
- **Parameters:** FNO 2.18M, WNO 1.79M, Sp²GNO 1.50M, NOMAD 0.36M,
  MIONet 0.25M, MLP baseline ≈ 0.05M.

## Intended use

- **Primary:** Real-time, amortized MHD equilibrium inference across tokamak
  geometries within the analytic Solov'ev family; benchmarking transfer-learning
  strategies for operator surrogates.
- **Out of scope:** Free-boundary equilibria, nonlinear pressure/current
  profiles, 3D configurations, and experimental reconstruction with diagnostic
  constraints. Predictions outside the trained geometry range require fine-tuning
  and are not validated.

## Training data

- Analytically generated Solov'ev equilibria (Cerfon–Freidberg, 2010).
- **Source geometries (pre-training):** NSTX, ITER, D3D0, D3D1, ITER-Reactor.
- 1,000 training + 200 test samples per geometry; `A ~ Uniform[-0.2, 1.0]`.
- 64×64 grids subsampled to 32×32 at load time.
- Fixed train/test split (`random_state = 100`).

## Evaluation data

- **Target geometries (held out):** SPHER, KSTAR, D3D-ITER.
- No target-geometry samples appear in pre-training (leakage-free split).
- Fine-tuning budgets: `N_target ∈ {0, 10, 100, 1000}`; 200 test samples each.

## Metrics

- Mean relative L2 error (`‖ψ̂ − ψ‖₂ / ‖ψ‖₂`, batch-averaged).
- Physics consistency: mean absolute `∇·B` (finite differences on predicted ψ).

## Quantitative results (summary)

- WNO: mean rel. L2 < 4% with 100 fine-tuning samples; < 2% with 1,000.
- All operators implicitly satisfy `∇·B ≈ 0` to ~1e-8 (single-precision FD).
- MLP baseline: see `results/mlp_baseline.csv` after running the baseline script.
- Inference: four of five operators achieve sub-millisecond to millisecond
  latency (15×–~1124× over the parallelized analytic solver).

## Limitations and biases

- The Solov'ev restriction introduces a systematic bias toward smooth,
  linearized-profile equilibria; sharper experimental gradients are unseen.
- Graph-based Sp²GNO does not transfer reliably across geometries under shallow
  fine-tuning in the current implementation.
- No probabilistic uncertainty quantification is provided.
- Fixed-grid masking creates a domain mismatch contributing to boundary-localized
  error in the few-shot regime.

## How to use

See `README.md`. Datasets are reproducible via `scripts/generate_data.py`;
pre-trained weights are released alongside the paper (see *Data/Code
Availability*).
