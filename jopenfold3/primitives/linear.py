"""Linear layer translation."""

from __future__ import annotations

import einops
import jax.numpy as jnp
import torch.nn as nn
from jaxtyping import Array, Float

from jopenfold3.backend import AbstractFromTorch, from_torch


class Linear(AbstractFromTorch):
    """Linear layer matching PyTorch nn.Linear semantics.

    Weight shape: (out_features, in_features)
    """

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None = None

    def __call__(self, x: Array) -> Array:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, o.shape)
        return o


# Register for both the custom Linear and nn.Linear
import jopenfold3._vendor.openfold3.core.model.primitives.linear as pt_linear

from_torch.register(pt_linear.Linear, Linear.from_torch)
from_torch.register(nn.Linear, Linear.from_torch)
