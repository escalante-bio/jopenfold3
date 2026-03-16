"""Diffusion transformer block and stack translation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.diffusion_transformer as pt_dt
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch

# Ensure layer converters are registered before use
from jopenfold3.layers.attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from jopenfold3.layers.transition import ConditionedTransitionBlock  # noqa: F401
from jopenfold3.primitives import LayerNorm


class DiffusionTransformerBlock(AbstractFromTorch):
    """Diffusion transformer block (AF3 Algorithm 23)."""

    attention_pair_bias: AttentionPairBias | CrossAttentionPairBias
    conditioned_transition: AbstractFromTorch  # ConditionedTransitionBlock

    def __call__(self, a: Array, s: Array, z: Array, mask: Array | None = None, **kwargs) -> Array:
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)
        a = a + self.conditioned_transition(a=a, s=s, mask=mask)
        return a


class DiffusionTransformer(eqx.Module):
    """Diffusion transformer stack using scan (AF3 Algorithm 23)."""

    stacked_params: DiffusionTransformerBlock
    static: DiffusionTransformerBlock
    layer_norm_z: LayerNorm | None = None
    use_cross_attention: bool = False

    @classmethod
    def from_torch(cls, model) -> "DiffusionTransformer":
        layers = [from_torch(child) for child in model.blocks]

        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )

        layer_norm_z = None
        if model.use_cross_attention and hasattr(model, "layer_norm_z"):
            layer_norm_z = from_torch(model.layer_norm_z)

        return cls(
            stacked_params=stacked,
            static=static,
            layer_norm_z=layer_norm_z,
            use_cross_attention=model.use_cross_attention,
        )

    def __call__(self, a: Array, s: Array, z: Array, mask: Array | None = None,
                 *, key, deterministic=True, **kwargs) -> Array:
        if self.use_cross_attention and self.layer_norm_z is not None:
            z = self.layer_norm_z(z)

        @jax.checkpoint
        def body_fn(carry, params):
            a, key = carry
            block = eqx.combine(params, self.static)
            a = block(a=a, s=s, z=z, mask=mask)
            return (a, jax.random.fold_in(key, 1)), None

        (a, _), _ = jax.lax.scan(body_fn, (a, key), self.stacked_params)
        return a


# Register converters
from_torch.register(pt_dt.DiffusionTransformerBlock, DiffusionTransformerBlock.from_torch)
from_torch.register(pt_dt.DiffusionTransformer, DiffusionTransformer.from_torch)
