"""Triangle attention layer translation."""

from __future__ import annotations

import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.triangular_attention as pt_tri_att
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import Attention, LayerNorm, Linear

# ---------------------------------------------------------------------------
# TriangleAttention  (AF2 Alg 13/14, AF3 Alg 14/15)
# ---------------------------------------------------------------------------

class TriangleAttention(AbstractFromTorch):
    """Triangle attention over pair representations.

    When ``starting=True`` attention is applied along the "starting node"
    (row) dimension; when ``starting=False`` the input is transposed so
    attention runs along columns instead.
    """

    layer_norm: LayerNorm
    linear_z: Linear
    mha: Attention
    starting: bool = True
    inf: float = 1e9

    @classmethod
    def from_torch(cls, model) -> "TriangleAttention":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["starting"] = bool(model.starting)
        kwargs["inf"] = float(model.inf)
        return cls(**kwargs)

    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        if mask is None:
            mask = jnp.ones(x.shape[:-1])

        if not self.starting:
            x = jnp.swapaxes(x, -2, -3)
            mask = jnp.swapaxes(mask, -1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # mask_bias: [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # triangle_bias: [*, I, J, H] -> [*, 1, H, I, J]
        triangle_bias = rearrange(self.linear_z(x), '... I J H -> ... H I J')
        triangle_bias = triangle_bias[..., None, :, :, :]  # unsqueeze -4

        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = jnp.swapaxes(x, -2, -3)

        return x


# Convenience aliases
TriangleAttentionStartingNode = TriangleAttention
TriangleAttentionEndingNode = TriangleAttention


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(pt_tri_att.TriangleAttention, TriangleAttention.from_torch)

# TriangleAttentionStartingNode is just an alias in PyTorch
# TriangleAttentionEndingNode is a subclass with starting=False
from_torch.register(
    pt_tri_att.TriangleAttentionEndingNode, TriangleAttention.from_torch
)
