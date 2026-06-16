"""Pointwise MLP baseline for Solov'ev equilibrium regression.

This is the simple/trivial baseline requested by the Nature Communications
Machine Learning reporting checklist (Item 4D). It treats equilibrium
prediction as pointwise regression: each grid point's 6-dimensional feature
vector (A, epsilon, kappa, delta, R, Z) is mapped independently to the scalar
flux psi at that point by a shared multilayer perceptron. The MLP therefore
has no operator structure and no spatial coupling -- it is the natural
lower-bound reference against which the neural operators are measured.

The parameter count is kept comparable to the operator projection heads
(width 128) so the comparison isolates the value of operator structure rather
than raw capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Shared pointwise MLP applied independently at every grid location.

    Input  : (B, s, s, in_channels) with channels [A, epsilon, kappa, delta, R, Z]
    Output : (B, s, s, 1)
    """

    def __init__(self, in_channels: int = 6, width: int = 128,
                 depth: int = 4, out_channels: int = 1):
        super().__init__()
        assert depth >= 2, "depth must include at least input and output layers"
        layers = [nn.Linear(in_channels, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The MLP is applied to the last (channel) dimension; spatial dims pass
        # through unchanged, so each point is regressed independently.
        return self.net(x.to(torch.float32))


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
