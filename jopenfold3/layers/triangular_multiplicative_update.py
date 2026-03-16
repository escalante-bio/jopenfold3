"""Triangle multiplicative update layer translations."""

from __future__ import annotations

import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.triangular_multiplicative_update as pt_tri_mul
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import LayerNorm, Linear, Sigmoid

# ---------------------------------------------------------------------------
# TriangleMultiplicativeUpdate  (AF2 Alg 11/12, AF3 Alg 12/13)
# ---------------------------------------------------------------------------

class TriangleMultiplicativeUpdate(AbstractFromTorch):
    """Non-fused triangle multiplicative update.

    ``_outgoing`` controls the einsum contraction order:
      - outgoing (True):  a=[...,C,I,J], b=[...,C,J,I]  -> p=[...,I,J,C]
      - incoming (False): a=[...,C,J,I], b=[...,C,I,J]  -> p=[...,I,J,C]
    """

    linear_g: Linear
    linear_z: Linear
    layer_norm_in: LayerNorm
    layer_norm_out: LayerNorm
    sigmoid: Sigmoid
    linear_a_p: Linear
    linear_a_g: Linear
    linear_b_p: Linear
    linear_b_g: Linear
    _outgoing: bool = True

    @classmethod
    def from_torch(cls, model) -> "TriangleMultiplicativeUpdate":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["_outgoing"] = bool(model._outgoing)
        return cls(**kwargs)

    def __call__(
        self,
        z: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        if mask is None:
            mask = jnp.ones(z.shape[:-1])
        mask = mask[..., None]

        z_normed = self.layer_norm_in(z)

        a = mask * self.sigmoid(self.linear_a_g(z_normed)) * self.linear_a_p(z_normed)
        b = mask * self.sigmoid(self.linear_b_g(z_normed)) * self.linear_b_p(z_normed)

        # Combine projections
        if self._outgoing:
            a = rearrange(a, '... I J C -> ... C I J')
            b = rearrange(b, '... I J C -> ... C J I')
        else:
            a = rearrange(a, '... I J C -> ... C J I')
            b = rearrange(b, '... I J C -> ... C I J')

        x = jnp.einsum("...ij,...jk->...ik", a, b)
        x = rearrange(x, '... C I J -> ... I J C')

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z_normed))
        x = x * g

        return x


# ---------------------------------------------------------------------------
# FusedTriangleMultiplicativeUpdate  (AF2-Multimer)
# ---------------------------------------------------------------------------

class FusedTriangleMultiplicativeUpdate(AbstractFromTorch):
    """Fused triangle multiplicative update (AF2-Multimer variant).

    Uses a single pair of fused projections (``linear_ab_p``, ``linear_ab_g``)
    instead of separate a/b projections.
    """

    linear_g: Linear
    linear_z: Linear
    layer_norm_in: LayerNorm
    layer_norm_out: LayerNorm
    sigmoid: Sigmoid
    linear_ab_p: Linear
    linear_ab_g: Linear
    _outgoing: bool = True
    c_hidden: int = 0

    @classmethod
    def from_torch(cls, model) -> "FusedTriangleMultiplicativeUpdate":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["_outgoing"] = bool(model._outgoing)
        kwargs["c_hidden"] = model.c_hidden
        return cls(**kwargs)

    def __call__(
        self,
        z: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        if mask is None:
            mask = jnp.ones(z.shape[:-1])
        mask = mask[..., None]

        z_normed = self.layer_norm_in(z)

        ab = mask * self.sigmoid(self.linear_ab_g(z_normed)) * self.linear_ab_p(z_normed)
        a = ab[..., : self.c_hidden]
        b = ab[..., self.c_hidden :]

        # Combine projections (same logic as non-fused)
        if self._outgoing:
            a = rearrange(a, '... I J C -> ... C I J')
            b = rearrange(b, '... I J C -> ... C J I')
        else:
            a = rearrange(a, '... I J C -> ... C J I')
            b = rearrange(b, '... I J C -> ... C I J')

        x = jnp.einsum("...ij,...jk->...ik", a, b)
        x = rearrange(x, '... C I J -> ... I J C')

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z_normed))
        x = x * g

        return x


# ---------------------------------------------------------------------------
# Convenience aliases matching the PyTorch subclasses
# ---------------------------------------------------------------------------

TriangleMultiplicationOutgoing = TriangleMultiplicativeUpdate
TriangleMultiplicationIncoming = TriangleMultiplicativeUpdate
FusedTriangleMultiplicationOutgoing = FusedTriangleMultiplicativeUpdate
FusedTriangleMultiplicationIncoming = FusedTriangleMultiplicativeUpdate


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

# Non-fused variants
from_torch.register(
    pt_tri_mul.TriangleMultiplicativeUpdate,
    TriangleMultiplicativeUpdate.from_torch,
)
from_torch.register(
    pt_tri_mul.TriangleMultiplicationOutgoing,
    TriangleMultiplicativeUpdate.from_torch,
)
from_torch.register(
    pt_tri_mul.TriangleMultiplicationIncoming,
    TriangleMultiplicativeUpdate.from_torch,
)

# Fused variants
from_torch.register(
    pt_tri_mul.FusedTriangleMultiplicativeUpdate,
    FusedTriangleMultiplicativeUpdate.from_torch,
)
from_torch.register(
    pt_tri_mul.FusedTriangleMultiplicationOutgoing,
    FusedTriangleMultiplicativeUpdate.from_torch,
)
from_torch.register(
    pt_tri_mul.FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicativeUpdate.from_torch,
)
