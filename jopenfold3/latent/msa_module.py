"""MSA module block and stack (JAX translation)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.latent.msa_module as pt_msa

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.latent.base_blocks import PairBlock
from jopenfold3.layers.msa import MSAAttention
from jopenfold3.layers.outer_product_mean import OuterProductMean
from jopenfold3.layers.transition import SwiGLUTransition
from jopenfold3.primitives import DropoutRowwise


class MSAModuleBlock(AbstractFromTorch):
    """Implements block of AF3 Algorithm 8.

    MSAModuleBlock may or may not have msa_att_row/msa_dropout_layer/msa_transition
    (skipped on last block when opm_first=True).
    """

    msa_att_row: MSAAttention | None = None
    msa_dropout_layer: DropoutRowwise | None = None
    msa_transition: SwiGLUTransition | None = None
    outer_product_mean: OuterProductMean | None = None
    pair_stack: PairBlock | None = None
    opm_first: bool = True
    skip_msa_update: bool = False

    @classmethod
    def from_torch(cls, model) -> "MSAModuleBlock":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["opm_first"] = model.opm_first
        kwargs["skip_msa_update"] = model.skip_msa_update
        return cls(**kwargs)

    def __call__(self, m, z, msa_mask=None, pair_mask=None, *,
                 key, deterministic=True, **kwargs):
        if msa_mask is None:
            msa_mask = jnp.ones(m.shape[:-1]) if m is not None else None
        if pair_mask is None:
            pair_mask = jnp.ones(z.shape[:-1])

        k_msa, k_pair = jax.random.split(key)

        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            msa_out = self.msa_att_row(m, z=z, mask=pair_mask)
            if self.msa_dropout_layer is not None:
                msa_out = self.msa_dropout_layer(
                    msa_out, key=k_msa, deterministic=deterministic)
            m = m + msa_out
            m = m + self.msa_transition(m, mask=msa_mask)

            if not self.opm_first:
                z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask,
                           key=k_pair, deterministic=deterministic)

        return m, z


class MSAModuleStack(eqx.Module):
    """Stack of MSAModuleBlocks.

    First N-1 homogeneous blocks are executed via ``scan`` (single trace);
    the last (heterogeneous, ``skip_msa_update=True``) block runs explicitly.
    """

    stacked_params: MSAModuleBlock   # first N-1 blocks, arrays stacked dim-0
    static: MSAModuleBlock           # shared non-array structure
    final_block: MSAModuleBlock      # last block (different structure)

    @classmethod
    def from_torch(cls, model) -> "MSAModuleStack":
        blocks = [from_torch(b) for b in model.blocks]
        scan_blocks = blocks[:-1]
        final_block = blocks[-1]

        _, static = eqx.partition(scan_blocks[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(b, eqx.is_inexact_array) for b in scan_blocks],
        )
        return cls(stacked_params=stacked, static=static, final_block=final_block)

    def __call__(self, m, z, msa_mask=None, pair_mask=None, *,
                 key, deterministic=True, **kwargs):
        @jax.checkpoint
        def body_fn(carry, params):
            m, z, key = carry
            block = eqx.combine(params, self.static)
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask,
                        key=key, deterministic=deterministic)
            return (m, z, jax.random.fold_in(key, 1)), None

        (m, z, key), _ = jax.lax.scan(body_fn, (m, z, key), self.stacked_params)

        # Final heterogeneous block
        m, z = eqx.filter_checkpoint(self.final_block)(
            m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask,
            key=key, deterministic=deterministic)
        return z


from_torch.register(pt_msa.MSAModuleBlock, MSAModuleBlock.from_torch)
from_torch.register(pt_msa.MSAModuleStack, MSAModuleStack.from_torch)
