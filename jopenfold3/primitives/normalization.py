"""Normalization layer translations (LayerNorm, AdaLN)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.primitives.normalization as pt_norm
from jaxtyping import Array, Float

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives.activations import Sigmoid
from jopenfold3.primitives.linear import Linear


class LayerNorm(AbstractFromTorch):
    """LayerNorm matching PyTorch semantics (last-dim normalization)."""

    weight: Float[Array, "D"] | None = None
    bias: Float[Array, "D"] | None = None
    eps: float = 1e-5

    @classmethod
    def from_torch(cls, model) -> "LayerNorm":
        weight = from_torch(model.weight) if model.weight is not None else None
        bias = from_torch(model.bias) if model.bias is not None else None
        return cls(weight=weight, bias=bias, eps=model.eps)

    def __call__(self, x: Array) -> Array:
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class AdaLN(AbstractFromTorch):
    """Adaptive LayerNorm (AF3 Algorithm 26)."""

    layer_norm_a: LayerNorm
    layer_norm_s: LayerNorm
    sigmoid: Sigmoid
    linear_g: Linear
    linear_s: Linear

    def __call__(self, a: Array, s: Array) -> Array:
        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s))
        a = g * a + self.linear_s(s)
        return a


# Register converters
from_torch.register(pt_norm.LayerNorm, LayerNorm.from_torch)
from_torch.register(pt_norm.AdaLN, AdaLN.from_torch)
