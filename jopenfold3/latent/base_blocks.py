"""Base blocks for MSA and Pair transformer stacks (JAX translation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.latent.base_blocks as pt_blocks

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.layers.transition import SwiGLUTransition
from jopenfold3.layers.triangular_attention import TriangleAttention
from jopenfold3.layers.triangular_multiplicative_update import TriangleMultiplicativeUpdate
from jopenfold3.primitives import DropoutRowwise


class PairBlock(AbstractFromTorch):
    """Pair block: tri_mul_out/in -> tri_att_start/end -> pair_transition.

    Implements AF3 Algorithm 15.
    """

    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    pair_transition: SwiGLUTransition
    ps_dropout_row_layer: DropoutRowwise

    def __call__(self, z, pair_mask=None, *, key, deterministic=True, **kwargs):
        if pair_mask is None:
            pair_mask = jnp.ones(z.shape[:-1])

        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Triangle multiplicative updates
        z = z + self.ps_dropout_row_layer(
            self.tri_mul_out(z, mask=pair_mask), key=k1, deterministic=deterministic
        )
        z = z + self.ps_dropout_row_layer(
            self.tri_mul_in(z, mask=pair_mask), key=k2, deterministic=deterministic
        )

        # Triangle attention (starting node)
        z = z + self.ps_dropout_row_layer(
            self.tri_att_start(z, mask=pair_mask), key=k3, deterministic=deterministic
        )

        # Triangle attention (ending node) — matches PyTorch tri_att_start_end:
        # external transpose so dropout_row acts on the correct axis,
        # tri_att_end (starting=False) internally transposes again.
        z = jnp.swapaxes(z, -2, -3)
        z = z + self.ps_dropout_row_layer(
            self.tri_att_end(z, mask=jnp.swapaxes(pair_mask, -1, -2)),
            key=k4, deterministic=deterministic,
        )
        z = jnp.swapaxes(z, -2, -3)

        # Pair transition
        z = z + self.pair_transition(z, mask=pair_mask)

        return z


from_torch.register(pt_blocks.PairBlock, PairBlock.from_torch)
