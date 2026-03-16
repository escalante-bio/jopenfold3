"""Outer product mean layer translation."""

from __future__ import annotations

import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.outer_product_mean as pt_opm
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import LayerNorm, Linear


class OuterProductMean(AbstractFromTorch):
    """Outer product mean (AF2 Algorithm 10 / AF3 Algorithm 9)."""

    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: float = 1e-3

    @classmethod
    def from_torch(cls, model) -> "OuterProductMean":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["eps"] = model.eps
        return cls(**kwargs)

    def __call__(self, m: Array, mask: Array | None = None, **kwargs) -> Array:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = jnp.ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask[..., None]
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        a = rearrange(a, '... S N C -> ... N S C')
        b = rearrange(b, '... S N C -> ... N S C')

        # [*, N_res, N_res, C, C]
        outer = jnp.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = rearrange(outer, '... I J Ca Cb -> ... I J (Ca Cb)')

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        norm = jnp.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm
        return outer


# Register converter
from_torch.register(pt_opm.OuterProductMean, OuterProductMean.from_torch)
