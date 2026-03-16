"""PairFormer block and stack (JAX translation)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.latent.pairformer as pt_pairformer

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.latent.base_blocks import PairBlock
from jopenfold3.layers.attention_pair_bias import AttentionPairBias
from jopenfold3.layers.transition import SwiGLUTransition


class PairFormerBlock(AbstractFromTorch):
    """Implements block of AF3 Algorithm 17."""

    pair_stack: PairBlock
    attn_pair_bias: AttentionPairBias
    single_transition: SwiGLUTransition

    def __call__(self, s, z, single_mask=None, pair_mask=None, *,
                 key, deterministic=True, **kwargs):
        if single_mask is None:
            single_mask = jnp.ones(s.shape[:-1])
        if pair_mask is None:
            pair_mask = jnp.ones(z.shape[:-1])

        z = self.pair_stack(z=z, pair_mask=pair_mask,
                           key=key, deterministic=deterministic)
        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)
        s = s + self.single_transition(s, mask=single_mask)
        return s, z


class PairFormerStack(eqx.Module):
    """Stack of PairFormerBlocks using scan for JIT/memory efficiency."""

    stacked_params: PairFormerBlock  # arrays stacked along dim 0
    static: PairFormerBlock          # non-array structure

    @classmethod
    def from_torch(cls, model) -> "PairFormerStack":
        layers = [from_torch(b) for b in model.blocks]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return cls(stacked_params=stacked, static=static)

    def __call__(self, s, z, single_mask=None, pair_mask=None, *,
                 key, deterministic=True, **kwargs):
        @jax.checkpoint
        def body_fn(carry, params):
            s, z, key = carry
            block = eqx.combine(params, self.static)
            s, z = block(s=s, z=z, single_mask=single_mask, pair_mask=pair_mask,
                        key=key, deterministic=deterministic)
            return (s, z, jax.random.fold_in(key, 1)), None

        (s, z, _), _ = jax.lax.scan(body_fn, (s, z, key), self.stacked_params)
        return s, z


from_torch.register(pt_pairformer.PairFormerBlock, PairFormerBlock.from_torch)
from_torch.register(pt_pairformer.PairFormerStack, PairFormerStack.from_torch)
