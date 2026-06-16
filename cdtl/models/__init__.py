"""Neural operator architectures benchmarked in the paper.

Each module exposes a ``Base*`` backbone plus the transfer-learning head
wrappers (single-task, multi-head, single-head, frozen-base) used in the
pre-train / fine-tune experiments.

    fno     : Fourier Neural Operator
    wno     : Wavelet Neural Operator
    mionet  : Multiple-Input Operator Network (branch--trunk)
    nomad   : Nonlinear Manifold Decoder operator
    sp2gno  : Spatio-Spectral Graph Neural Operator
"""
