"""Diffusion conditioning module translation (AF3 Algorithm 21)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.diffusion_conditioning as pt_dc
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.feature_embedders.input_embedders import FourierEmbedding
from jopenfold3.layers.transition import SwiGLUTransition
from jopenfold3.primitives import LayerNorm, Linear
from jopenfold3.utils import relpos_complex

# ---------------------------------------------------------------------------
# DiffusionConditioning
# ---------------------------------------------------------------------------

class DiffusionConditioning(eqx.Module):
    """Implements AF3 Algorithm 21."""

    layer_norm_z: LayerNorm
    linear_z: Linear
    transition_z: list[SwiGLUTransition]
    layer_norm_s: LayerNorm
    linear_s: Linear
    fourier_emb: FourierEmbedding
    layer_norm_n: LayerNorm
    linear_n: Linear
    transition_s: list[SwiGLUTransition]
    sigma_data: float
    max_relative_idx: int
    max_relative_chain: int

    @classmethod
    def from_torch(cls, model) -> "DiffusionConditioning":
        return cls(
            layer_norm_z=from_torch(model.layer_norm_z),
            linear_z=from_torch(model.linear_z),
            transition_z=[from_torch(l) for l in model.transition_z],
            layer_norm_s=from_torch(model.layer_norm_s),
            linear_s=from_torch(model.linear_s),
            fourier_emb=from_torch(model.fourier_emb),
            layer_norm_n=from_torch(model.layer_norm_n),
            linear_n=from_torch(model.linear_n),
            transition_s=[from_torch(l) for l in model.transition_s],
            sigma_data=model.sigma_data,
            max_relative_idx=model.max_relative_idx,
            max_relative_chain=model.max_relative_chain,
        )

    def __call__(
        self,
        batch: Batch,
        t: Array,
        si_input: Array,
        si_trunk: Array,
        zij_trunk: Array,
        use_conditioning: bool,
        **kwargs,
    ) -> tuple[Array, Array]:
        """
        Args:
            batch: Feature dictionary
            t: [*] Noise level at a diffusion timestep
            si_input: [*, N_token, c_s_input] Input embedding
            si_trunk: [*, N_token, c_s] Single representation
            zij_trunk: [*, N_token, N_token, c_z] Pair representation
            use_conditioning: Whether to condition with trunk representations

        Returns:
            si: [*, N_token, c_s] Conditioned single representation
            zij: [*, N_token, N_token, c_z] Conditioned pair representation
        """
        token_mask = batch.token_mask

        if not use_conditioning:
            si_trunk = si_trunk * 0
            zij_trunk = zij_trunk * 0

        # Pair conditioning
        relpos_zij = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).astype(zij_trunk.dtype)

        zij = jnp.concatenate([zij_trunk, relpos_zij], axis=-1)
        zij = self.linear_z(self.layer_norm_z(zij))

        # Single conditioning
        si = jnp.concatenate([si_trunk, si_input], axis=-1)
        si = self.linear_s(self.layer_norm_s(si))

        n = 0.25 * jnp.log(t / self.sigma_data)
        n = self.fourier_emb(n[..., None])

        si = si + self.linear_n(self.layer_norm_n(n))[..., None, :]

        # Transitions
        pair_token_mask = token_mask[..., None] * token_mask[..., None, :]

        for l in self.transition_z:
            zij = zij + l(zij, mask=pair_token_mask)

        for l in self.transition_s:
            si = si + l(si, mask=token_mask)

        return si, zij


# ---------------------------------------------------------------------------
# Register converter
# ---------------------------------------------------------------------------

from_torch.register(pt_dc.DiffusionConditioning, DiffusionConditioning.from_torch)
