"""Prediction head translations for AF3 model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.heads.prediction_heads as pt_heads
from jaxtyping import Array

from jopenfold3.backend import AbstractFromTorch, from_torch
from jopenfold3.latent.pairformer import PairFormerStack
from jopenfold3.primitives import LayerNorm, Linear
from jopenfold3.utils import max_atom_per_token_masked_select

# ---------------------------------------------------------------------------
# PairformerEmbedding  (AF3 Algorithm 31, lines 1-6)
# ---------------------------------------------------------------------------

class PairformerEmbedding(eqx.Module):
    """Pairformer embedding for confidence heads."""

    linear_i: Linear
    linear_j: Linear
    linear_distance: Linear
    pairformer_stack: PairFormerStack
    min_bin: float
    max_bin: float
    no_bin: int
    inf: float

    @classmethod
    def from_torch(cls, model) -> "PairformerEmbedding":
        return cls(
            linear_i=from_torch(model.linear_i),
            linear_j=from_torch(model.linear_j),
            linear_distance=from_torch(model.linear_distance),
            pairformer_stack=from_torch(model.pairformer_stack),
            min_bin=model.min_bin,
            max_bin=model.max_bin,
            no_bin=model.no_bin,
            inf=model.inf,
        )

    def embed_zij(
        self,
        si_input: Array,
        zij: Array,
        x_pred: Array,
    ) -> Array:
        """Embed pair representation with distance features."""
        zij = (
            zij
            + self.linear_i(si_input[..., None, :])
            + self.linear_j(si_input[..., None, :, :])
        )

        # Embed pair distances of representative atoms
        bins = jnp.linspace(self.min_bin, self.max_bin, self.no_bin)
        squared_bins = bins**2
        upper = jnp.concatenate(
            [squared_bins[1:], jnp.array([self.inf])], axis=-1
        )
        dij = jnp.sum(
            (x_pred[..., None, :] - x_pred[..., None, :, :]) ** 2,
            axis=-1,
            keepdims=True,
        )
        dij = ((dij > squared_bins) * (dij < upper)).astype(x_pred.dtype)
        zij = zij + self.linear_distance(dij)

        return zij

    def __call__(
        self,
        si_input: Array,
        si: Array,
        zij: Array,
        x_pred: Array,
        single_mask: Array,
        pair_mask: Array,
        *,
        key,
        **kwargs,
    ) -> tuple[Array, Array]:
        """
        Args:
            si_input: [*, N_token, C_s] Output of InputFeatureEmbedder
            si: [*, N_token, C_s] Single embedding
            zij: [*, N_token, N_token, C_z] Pairwise embedding
            x_pred: [*, N_token, 3] Representative atom predicted coordinates
            single_mask: [*, N_token] Single mask
            pair_mask: [*, N_token, N_token] Pair mask
            key: PRNG key.

        Returns:
            si: [*, N_token, C_s] Updated single representation
            zij: [*, N_token, N_token, C_z] Updated pair representation
        """
        zij = self.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)

        si, zij = self.pairformer_stack(
            s=si,
            z=zij,
            single_mask=single_mask,
            pair_mask=pair_mask,
            key=key,
        )

        return si, zij


# ---------------------------------------------------------------------------
# PredictedAlignedErrorHead  (AF3 Algorithm 31, Line 5)
# ---------------------------------------------------------------------------

class PredictedAlignedErrorHead(AbstractFromTorch):
    """PAE head."""

    layer_norm: LayerNorm
    linear: Linear

    def __call__(self, zij: Array, **kwargs) -> Array:
        """
        Args:
            zij: [*, N, N, C_z] Pair embedding

        Returns:
            logits: [*, N, N, C_out] Logits
        """
        return self.linear(self.layer_norm(zij))


# ---------------------------------------------------------------------------
# PredictedDistanceErrorHead  (AF3 Algorithm 31, Line 6)
# ---------------------------------------------------------------------------

class PredictedDistanceErrorHead(AbstractFromTorch):
    """PDE head."""

    layer_norm: LayerNorm
    linear: Linear

    def __call__(self, zij: Array, **kwargs) -> Array:
        """
        Args:
            zij: [*, N, N, C_z] Pair embedding

        Returns:
            logits: [*, N, N, C_out] Logits
        """
        logits = self.linear(self.layer_norm(zij))
        logits = logits + jnp.swapaxes(logits, -2, -3)
        return logits


# ---------------------------------------------------------------------------
# PerResidueLDDTAllAtom  (AF3 Algorithm 31, Line 7)
# ---------------------------------------------------------------------------

class PerResidueLDDTAllAtom(eqx.Module):
    """pLDDT head."""

    layer_norm: LayerNorm
    linear: Linear
    max_atoms_per_token: int
    c_out: int

    @classmethod
    def from_torch(cls, model) -> "PerResidueLDDTAllAtom":
        return cls(
            layer_norm=from_torch(model.layer_norm),
            linear=from_torch(model.linear),
            max_atoms_per_token=model.max_atoms_per_token,
            c_out=model.c_out,
        )

    def __call__(
        self,
        s: Array,
        max_atom_per_token_mask: Array,
        n_atom: int | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            s: [*, N_token, C_s] Single embedding
            max_atom_per_token_mask: [*, N_token * max_atoms_per_token] Mask
            n_atom: Output atom dimension size.

        Returns:
            logits: [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(
            *batch_dims, n_token * self.max_atoms_per_token, self.c_out
        )

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
            n_atom=n_atom,
        )

        return logits


# ---------------------------------------------------------------------------
# ExperimentallyResolvedHeadAllAtom
# ---------------------------------------------------------------------------

class ExperimentallyResolvedHeadAllAtom(eqx.Module):
    """Experimentally resolved head."""

    layer_norm: LayerNorm
    linear: Linear
    max_atoms_per_token: int
    c_out: int

    @classmethod
    def from_torch(cls, model) -> "ExperimentallyResolvedHeadAllAtom":
        return cls(
            layer_norm=from_torch(model.layer_norm),
            linear=from_torch(model.linear),
            max_atoms_per_token=model.max_atoms_per_token,
            c_out=model.c_out,
        )

    def __call__(
        self,
        s: Array,
        max_atom_per_token_mask: Array,
        n_atom: int | None = None,
        **kwargs,
    ) -> Array:
        """
        Args:
            s: [*, N_token, C_s] Single embedding
            max_atom_per_token_mask: [*, N_token * max_atoms_per_token] Mask
            n_atom: Output atom dimension size.

        Returns:
            logits: [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        logits = self.linear(self.layer_norm(s))

        logits = logits.reshape(
            *batch_dims, n_token * self.max_atoms_per_token, self.c_out
        )

        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
            n_atom=n_atom,
        )

        return logits


# ---------------------------------------------------------------------------
# DistogramHead
# ---------------------------------------------------------------------------

class DistogramHead(AbstractFromTorch):
    """Distogram head."""

    linear: Linear

    def __call__(self, z: Array, **kwargs) -> Array:
        """
        Args:
            z: [*, N, N, C_z] Pair embedding

        Returns:
            logits: [*, N, N, C_out] Distogram logits
        """
        logits = self.linear(z)
        logits = logits + jnp.swapaxes(logits, -2, -3)
        return logits


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(pt_heads.PairformerEmbedding, PairformerEmbedding.from_torch)
from_torch.register(pt_heads.PredictedAlignedErrorHead, PredictedAlignedErrorHead.from_torch)
from_torch.register(pt_heads.PredictedDistanceErrorHead, PredictedDistanceErrorHead.from_torch)
from_torch.register(pt_heads.PerResidueLDDTAllAtom, PerResidueLDDTAllAtom.from_torch)
from_torch.register(pt_heads.ExperimentallyResolvedHeadAllAtom, ExperimentallyResolvedHeadAllAtom.from_torch)
from_torch.register(pt_heads.DistogramHead, DistogramHead.from_torch)
