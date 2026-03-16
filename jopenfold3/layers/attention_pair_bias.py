"""Attention layer with pair bias translation."""

from __future__ import annotations

import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.attention_pair_bias as pt_apb
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import AdaLN, Attention, LayerNorm, Linear, Sigmoid
from jopenfold3.utils import convert_single_rep_to_blocks


class AttentionPairBias(AbstractFromTorch):
    """Attention layer with pair bias (AF3 Algorithm 24)."""

    layer_norm_a: LayerNorm | AdaLN
    layer_norm_z: LayerNorm
    linear_z: Linear
    mha: Attention
    sigmoid: Sigmoid
    linear_ada_out: Linear | None = None
    use_ada_layer_norm: bool = False
    inf: float = 1e9

    @classmethod
    def from_torch(cls, model) -> "AttentionPairBias":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["use_ada_layer_norm"] = model.use_ada_layer_norm
        kwargs["inf"] = model.inf
        return cls(**kwargs)

    def __call__(
        self,
        a: Array,
        z: Array,
        s: Array | None = None,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            a: [*, N, C_q] Token or atom-level embedding
            z: [*, N, N, C_z] Pair embedding
            s: [*, N, C_s] Single embedding (used with AdaLN)
            mask: [*, N] Mask

        Returns:
            [*, N, C_q] attention updated embedding
        """
        if self.use_ada_layer_norm:
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        # Prepare biases
        if mask is None:
            mask = jnp.ones(a.shape[:-1])

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]

        # [*, N, N, C_z]
        z_normed = self.layer_norm_z(z)

        # [*, N, N, no_heads]
        z_bias = self.linear_z(z_normed)

        # [*, no_heads, N, N]
        z_bias = rearrange(z_bias, '... I J H -> ... H I J')

        biases = [mask_bias, z_bias]

        a = self.mha(q_x=a, kv_x=a, biases=biases)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class CrossAttentionPairBias(AbstractFromTorch):
    """Attention layer with pair bias and sequence-local blocking (AF3 Algorithm 24)."""

    layer_norm_a_q: LayerNorm | AdaLN
    layer_norm_a_k: LayerNorm | AdaLN
    linear_z: Linear
    mha: Attention
    sigmoid: Sigmoid
    linear_ada_out: Linear | None = None
    use_ada_layer_norm: bool = False
    n_query: int | None = None
    n_key: int | None = None
    inf: float = 1e9

    @classmethod
    def from_torch(cls, model) -> "CrossAttentionPairBias":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["use_ada_layer_norm"] = model.use_ada_layer_norm
        kwargs["n_query"] = model.n_query
        kwargs["n_key"] = model.n_key
        kwargs["inf"] = model.inf
        return cls(**kwargs)

    def _prep_block_inputs(
        self,
        a: Array,
        z: Array,
        mask: Array,
    ) -> tuple[Array, Array, list[Array]]:
        """Convert inputs to q/k blocks and compute biases."""
        a_query, a_key, block_mask = convert_single_rep_to_blocks(
            a, self.n_query, self.n_key, mask
        )

        # [*, N_blocks, 1, N_query, N_key]
        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = rearrange(z, '... I J H -> ... H I J')

        biases.append(z)

        return a_query, a_key, biases

    def __call__(
        self,
        a: Array,
        z: Array,
        s: Array | None = None,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            a: [*, N, C_q] Token or atom-level embedding
            z: [*, N, N, C_z] Pair embedding
            s: [*, N, C_s] Single embedding (used with AdaLN)
            mask: [*, N] Mask

        Returns:
            [*, N, C_q] attention updated embedding
        """
        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = jnp.ones(a.shape[:-1])

        a_q, a_k, biases = self._prep_block_inputs(a, z, mask)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = convert_single_rep_to_blocks(
                s, self.n_query, self.n_key, mask
            )
            a_q = self.layer_norm_a_q(a_q, s_q)
            a_k = self.layer_norm_a_k(a_k, s_k)
        else:
            a_q = self.layer_norm_a_q(a_q)
            a_k = self.layer_norm_a_k(a_k)

        a = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        # Unblock: [*, N_blocks, N_query, c] -> [*, N_atom, c]
        a = a.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


# Register converters
from_torch.register(pt_apb.AttentionPairBias, AttentionPairBias.from_torch)
from_torch.register(pt_apb.CrossAttentionPairBias, CrossAttentionPairBias.from_torch)
