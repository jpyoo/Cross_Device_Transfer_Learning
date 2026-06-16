"""Loss functions and evaluation metrics.

The primary metric throughout the paper is the relative L2 error,

    e_rel = || psi_hat - psi ||_2 / || psi ||_2,

averaged over examples. ``LpLoss.rel`` implements this and doubles as the
training objective, matching the original benchmark.
"""

from __future__ import annotations

import torch


class LpLoss:
    """Relative (or absolute) Lp loss over batched fields.

    By default returns the batch-mean relative L2 error, which is both the
    training objective and the reported evaluation metric.
    """

    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True,
                 reduction: bool = True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        diff = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, 1)
        ynorm = torch.norm(y.reshape(n, -1), self.p, 1)
        ratio = diff / ynorm
        if self.reduction:
            return torch.mean(ratio) if self.size_average else torch.sum(ratio)
        return ratio

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)
        norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape(n, -1) - y.reshape(n, -1), self.p, 1
        )
        if self.reduction:
            return torch.mean(norms) if self.size_average else torch.sum(norms)
        return norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)
