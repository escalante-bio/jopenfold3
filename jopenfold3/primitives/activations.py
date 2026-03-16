"""Activation function translations."""

from __future__ import annotations

import equinox as eqx
import jax
import jopenfold3._vendor.openfold3.core.model.primitives.activations as pt_act
import torch.nn as nn
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives.linear import Linear


class ReLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class SiLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.silu(x)


class Sigmoid(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.sigmoid(x)


class SwiGLU(AbstractFromTorch):
    """SwiGLU activation: silu(linear_a(x)) * linear_b(x)."""

    linear_a: Linear
    linear_b: Linear
    swish: SiLU

    def __call__(self, x: Array) -> Array:
        return self.swish(self.linear_a(x)) * self.linear_b(x)


# Register converters
from_torch.register(nn.ReLU, lambda _: ReLU())
from_torch.register(nn.SiLU, lambda _: SiLU())
from_torch.register(nn.Sigmoid, lambda _: Sigmoid())
from_torch.register(pt_act.SwiGLU, SwiGLU.from_torch)
