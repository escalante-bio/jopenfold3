"""Transition layer translations (SwiGLU, ReLU, Conditioned)."""

from __future__ import annotations

import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.transition as pt_transition
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import (
    AdaLN,
    LayerNorm,
    Linear,
    Sigmoid,
    SwiGLU,
)

# ---------------------------------------------------------------------------
# SwiGLUTransition  (AF3 Algorithm 11)
# ---------------------------------------------------------------------------

class SwiGLUTransition(AbstractFromTorch):
    """SwiGLU feed-forward transition block."""

    layer_norm: LayerNorm
    swiglu: SwiGLU
    linear_out: Linear

    def __call__(self, x: Array, mask: Array | None = None, **kwargs) -> Array:
        if mask is None:
            mask = jnp.ones(x.shape[:-1])
        mask = mask[..., None]

        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask
        return x


# ---------------------------------------------------------------------------
# ReLUTransitionLayer
# ---------------------------------------------------------------------------

class ReLUTransitionLayer(AbstractFromTorch):
    """MLP with one or more Linear+ReLU layers followed by a linear out."""

    layers: list  # list of Sequential(Linear, ReLU)
    linear_out: Linear

    @classmethod
    def from_torch(cls, model) -> "ReLUTransitionLayer":
        layers = [from_torch(layer) for layer in model.layers]
        linear_out = from_torch(model.linear_out)
        return cls(layers=layers, linear_out=linear_out)

    def __call__(self, x: Array, mask: Array | None = None, **kwargs) -> Array:
        if mask is None:
            mask = jnp.ones(x.shape[:-1])[..., None]

        for layer in self.layers:
            x = layer(x)

        x = self.linear_out(x) * mask
        return x


# ---------------------------------------------------------------------------
# ReLUTransition  (AF2 Algorithm 9 / 15)
# ---------------------------------------------------------------------------

class ReLUTransition(AbstractFromTorch):
    """LayerNorm + ReLU MLP transition."""

    layer_norm: LayerNorm
    transition_mlp: ReLUTransitionLayer

    def __call__(self, x: Array, mask: Array | None = None, **kwargs) -> Array:
        if mask is None:
            mask = jnp.ones(x.shape[:-1])
        mask = mask[..., None]

        x = self.layer_norm(x)
        x = self.transition_mlp(x, mask=mask)
        return x


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock  (AF3 Algorithm 25)
# ---------------------------------------------------------------------------

class ConditionedTransitionBlock(AbstractFromTorch):
    """SwiGLU transition with adaptive layer-norm conditioning."""

    layer_norm: AdaLN
    swiglu: SwiGLU
    sigmoid: Sigmoid
    linear_g: Linear
    linear_out: Linear

    def __call__(
        self,
        a: Array,
        s: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        if mask is None:
            mask = jnp.ones(a.shape[:-1])
        mask = mask[..., None]

        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        a = a * mask
        return a


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(pt_transition.SwiGLUTransition, SwiGLUTransition.from_torch)
from_torch.register(pt_transition.ReLUTransitionLayer, ReLUTransitionLayer.from_torch)
from_torch.register(pt_transition.ReLUTransition, ReLUTransition.from_torch)
from_torch.register(
    pt_transition.ConditionedTransitionBlock, ConditionedTransitionBlock.from_torch
)
