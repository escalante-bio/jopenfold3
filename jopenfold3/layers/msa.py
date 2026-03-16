"""MSA attention layer translations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.msa as pt_msa
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives import Attention, GlobalAttention, LayerNorm, Linear, Sigmoid


class MSAAttention(AbstractFromTorch):
    """MSA row attention, optionally with pair bias."""

    layer_norm_m: LayerNorm
    mha: Attention
    layer_norm_z: LayerNorm | None = None
    linear_z: Linear | None = None
    pair_bias: bool = False
    inf: float = 1e9

    @classmethod
    def from_torch(cls, model) -> "MSAAttention":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["pair_bias"] = model.pair_bias
        kwargs["inf"] = model.inf
        return cls(**kwargs)

    def __call__(
        self,
        m: Array,
        z: Array | None = None,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            z: [*, N_res, N_res, C_z] Pair embedding (required when pair_bias=True)
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            mask = jnp.ones(m.shape[:-1])

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        biases = [mask_bias]

        if (
            self.pair_bias
            and z is not None
            and self.layer_norm_z is not None
            and self.linear_z is not None
        ):
            # [*, N_res, N_res, C_z]
            z_normed = self.layer_norm_z(z)

            # [*, N_res, N_res, no_heads]
            z_proj = self.linear_z(z_normed)

            # [*, 1, no_heads, N_res, N_res]
            z_proj = rearrange(z_proj, '... I J H -> ... H I J')[..., None, :, :, :]
            biases.append(z_proj)

        m = self.layer_norm_m(m)
        m = self.mha(q_x=m, kv_x=m, biases=biases)
        return m


class MSAColumnAttention(AbstractFromTorch):
    """MSA column attention (AF2 Algorithm 8). Transposes and delegates to MSAAttention."""

    _msa_att: MSAAttention

    def __call__(
        self,
        m: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        # [*, N_res, N_seq, C_in]
        m = jnp.swapaxes(m, -2, -3)
        if mask is not None:
            mask = jnp.swapaxes(mask, -1, -2)

        m = self._msa_att(m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = jnp.swapaxes(m, -2, -3)
        return m


class MSAColumnGlobalAttention(AbstractFromTorch):
    """MSA column global attention. Transposes, applies layer norm + global attention."""

    layer_norm_m: LayerNorm
    global_attention: GlobalAttention

    def __call__(
        self,
        m: Array,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if mask is None:
            mask = jnp.ones(m.shape[:-1])

        # [*, N_res, N_seq, C_in]
        m = jnp.swapaxes(m, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

        m = self.layer_norm_m(m)
        m = self.global_attention(m=m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = jnp.swapaxes(m, -2, -3)
        return m


class MSAPairWeightedAveraging(AbstractFromTorch):
    """MSA pair weighted averaging (AF3 Algorithm 10)."""

    layer_norm_m: LayerNorm
    layer_norm_z: LayerNorm
    linear_z: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear
    sigmoid: Sigmoid
    no_heads: int = 0
    inf: float = 1e9

    @classmethod
    def from_torch(cls, model) -> "MSAPairWeightedAveraging":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["no_heads"] = model.no_heads
        kwargs["inf"] = model.inf
        return cls(**kwargs)

    def __call__(
        self,
        m: Array,
        z: Array | None = None,
        mask: Array | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            m: [*, N_seq, N_token, C_m] MSA embedding
            z: [*, N_token, N_token, C_z] Pair embedding
            mask: [*, N_token, N_token] Pair mask

        Returns:
            [*, N_seq, N_token, C_m] updated MSA embedding
        """
        # Prepare pair weights
        if mask is None:
            mask = jnp.ones(z.shape[:-1]) if z is not None else jnp.ones(m.shape[:-1])

        # [*, 1, 1, N_token, N_token]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]

        # [*, N_token, N_token, C_z]
        z = self.layer_norm_z(z)

        # [*, N_token, N_token, no_heads]
        z = self.linear_z(z)

        # [*, 1, no_heads, N_token, N_token]
        z = rearrange(z, '... I J H -> ... H I J')[..., None, :, :, :]
        z = z + mask_bias

        # Apply to MSA
        # [*, N_seq, N_token, C_m]
        m = self.layer_norm_m(m)

        # [*, N_seq, N_token, H * C_hidden]
        v = self.linear_v(m)

        # [*, N_seq, H, N_token, C_hidden]
        v = rearrange(v, '... S N (H C) -> ... S H N C', H=self.no_heads)

        # Softmax over key dimension
        weights = jax.nn.softmax(z, axis=-1)

        # [*, N_seq, Q, H, C_hidden]  (contract over K)
        o = jnp.einsum("...hqk,...hkc->...qhc", weights, v)

        # Gate
        # [*, N_seq, N_token, H, C_hidden]
        g = rearrange(self.sigmoid(self.linear_g(m)), '... N (H C) -> ... N H C', H=self.no_heads)

        o = o * g

        # [*, N_seq, N_token, H * C_hidden]
        o = rearrange(o, '... N H C -> ... N (H C)')

        # [*, N_seq, N_token, C_m]
        o = self.linear_o(o)
        return o


# Register converters
from_torch.register(pt_msa.MSAAttention, MSAAttention.from_torch)
from_torch.register(pt_msa.MSARowAttentionWithPairBias, MSAAttention.from_torch)
from_torch.register(pt_msa.MSAColumnAttention, MSAColumnAttention.from_torch)
from_torch.register(pt_msa.MSAColumnGlobalAttention, MSAColumnGlobalAttention.from_torch)
from_torch.register(pt_msa.MSAPairWeightedAveraging, MSAPairWeightedAveraging.from_torch)
