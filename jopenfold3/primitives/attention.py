"""Attention layer translations (Attention, GlobalAttention)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.primitives.attention as pt_attn
from einops import rearrange
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.primitives.activations import Sigmoid
from jopenfold3.primitives.linear import Linear


def _attention_einsum(
    query: Array,
    key: Array,
    value: Array,
    biases: list[Array],
) -> Array:
    """Naive attention via explicit score materialisation.

    Used only by :class:`GlobalAttention` whose Q/K/V shapes are
    non-standard (no separate head dimension on K/V).

    Args:
        query: [*, H, Q, C_hidden] (already scaled)
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = jnp.einsum("...qc,...kc->...qk", query, key)
    for b in biases:
        scores = scores + b
    scores = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("...qk,...kc->...qc", scores, value)


def _attention(
    query: Array,
    key: Array,
    value: Array,
    biases: list[Array],
) -> Array:
    """Flash-compatible attention via ``jax.nn.dot_product_attention``.

    On GPU this dispatches to cuDNN flash attention (O(N) memory instead
    of O(N²) for the attention matrix).  On CPU/TPU it falls back to the
    XLA default implementation.

    Args:
        query: [*, H, Q, C_hidden] (already scaled by 1/√d)
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    H, Q, C = query.shape[-3:]
    K = key.shape[-2]
    batch_shape = query.shape[:-3]

    # Merge biases into a single additive bias [*, H, Q, K]
    bias = None
    if biases:
        bias = biases[0]
        for b in biases[1:]:
            bias = bias + b
        # Broadcast + flatten to 4-D so dot_product_attention sees (B, N, T, S)
        bias = jnp.broadcast_to(bias, batch_shape + (H, Q, K)).reshape(-1, H, Q, K)

    # Flatten leading dims, then transpose [B, H, Q, C] -> [B, Q, H, C]
    q = jnp.swapaxes(query.reshape(-1, H, Q, C), -3, -2)
    k = jnp.swapaxes(key.reshape(-1, H, K, C), -3, -2)
    v = jnp.swapaxes(value.reshape(-1, H, K, C), -3, -2)

    # scale=1.0 because _prep_qkv already scales q by 1/sqrt(c_hidden)
    o = jax.nn.dot_product_attention(q, k, v, bias=bias, scale=1.0)

    # Transpose back [B, Q, H, C] -> [B, H, Q, C], then unflatten
    o = jnp.swapaxes(o, -3, -2)
    return o.reshape(batch_shape + (H, Q, C))


class Attention(AbstractFromTorch):
    """Standard multi-head attention with optional gating."""

    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None = None
    sigmoid: Sigmoid | None = None
    c_hidden: int = 0
    no_heads: int = 0

    @classmethod
    def from_torch(cls, model) -> "Attention":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["c_hidden"] = model.c_hidden
        kwargs["no_heads"] = model.no_heads
        return cls(**kwargs)

    def _prep_qkv(self, q_x: Array, kv_x: Array) -> tuple[Array, Array, Array]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H*C] -> [*, H, Q/K, C]
        q = rearrange(q, '... Q (H C) -> ... H Q C', H=self.no_heads)
        k = rearrange(k, '... K (H C) -> ... H K C', H=self.no_heads)
        v = rearrange(v, '... V (H C) -> ... H V C', H=self.no_heads)

        q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: Array, q_x: Array) -> Array:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            # [*, Q, H, C_hidden]
            g = rearrange(g, '... Q (H C) -> ... Q H C', H=self.no_heads)
            o = o * g

        # [*, Q, H * C_hidden]
        o = rearrange(o, '... Q H C -> ... Q (H C)')

        # [*, Q, C_q]
        o = self.linear_o(o)
        return o

    def __call__(
        self,
        q_x: Array,
        kv_x: Array,
        biases: list[Array] | None = None,
        **kwargs,
    ) -> Array:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        # [*, H, Q, C] -> [*, Q, H, C]
        o = rearrange(o, '... H Q C -> ... Q H C')
        o = self._wrap_up(o, q_x)
        return o


class GlobalAttention(AbstractFromTorch):
    """Global attention variant."""

    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_g: Linear
    linear_o: Linear
    sigmoid: Sigmoid
    c_hidden: int = 0
    no_heads: int = 0
    inf: float = 1e9
    eps: float = 1e-8

    @classmethod
    def from_torch(cls, model) -> "GlobalAttention":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["c_hidden"] = model.c_hidden
        kwargs["no_heads"] = model.no_heads
        kwargs["inf"] = model.inf
        kwargs["eps"] = model.eps
        return cls(**kwargs)

    def __call__(self, m: Array, mask: Array, **kwargs) -> Array:
        # [*, N_res, C_in]  (mean over seq dim, weighted by mask)
        q = jnp.sum(m * mask[..., None], axis=-2) / (
            jnp.sum(mask, axis=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q = q * (self.c_hidden ** -0.5)

        # [*, N_res, H, C_hidden]
        q = rearrange(q, '... N (H C) -> ... N H C', H=self.no_heads)

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]

        o = _attention_einsum(q, k, v, [bias])

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))
        # [*, N_res, N_seq, H, C_hidden]
        g = rearrange(g, '... S (H C) -> ... S H C', H=self.no_heads)

        # [*, N_res, N_seq, H, C_hidden]
        o = o[..., None, :, :] * g

        # [*, N_res, N_seq, H * C_hidden]
        o = rearrange(o, '... S H C -> ... S (H C)')

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)
        return m


# Register converters
from_torch.register(pt_attn.Attention, Attention.from_torch)
from_torch.register(pt_attn.GlobalAttention, GlobalAttention.from_torch)
