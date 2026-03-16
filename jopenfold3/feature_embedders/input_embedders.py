"""Input embedder translations (FourierEmbedding, InputEmbedderAllAtom, MSAModuleEmbedder)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.feature_embedders.input_embedders as pt_ie
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.layers.sequence_local_atom_attention import AtomAttentionEncoder
from jopenfold3.primitives import Linear
from jopenfold3.utils import relpos_complex

# ---------------------------------------------------------------------------
# FourierEmbedding  (AF3 Algorithm 22)
# ---------------------------------------------------------------------------

class FourierEmbedding(eqx.Module):
    """Fourier embedding for noise-level conditioning."""

    w: Array
    b: Array

    @classmethod
    def from_torch(cls, model) -> "FourierEmbedding":
        return cls(
            w=from_torch(model.w),
            b=from_torch(model.b),
        )

    def __call__(self, x: Array, **kwargs) -> Array:
        """
        Args:
            x: [*, 1] Input tensor
        Returns:
            [*, c] Embedding
        """
        x = x * self.w + self.b
        return jnp.cos(2 * jnp.pi * x)


# ---------------------------------------------------------------------------
# InputEmbedderAllAtom  (AF3 Algorithm 1 lines 1-5)
# ---------------------------------------------------------------------------

class InputEmbedderAllAtom(eqx.Module):
    """Embeds input features for the all-atom model."""

    atom_attn_enc: AtomAttentionEncoder
    linear_s: Linear
    linear_z_i: Linear
    linear_z_j: Linear
    linear_relpos: Linear
    linear_token_bonds: Linear
    max_relative_idx: int
    max_relative_chain: int

    @classmethod
    def from_torch(cls, model) -> "InputEmbedderAllAtom":
        return cls(
            atom_attn_enc=from_torch(model.atom_attn_enc),
            linear_s=from_torch(model.linear_s),
            linear_z_i=from_torch(model.linear_z_i),
            linear_z_j=from_torch(model.linear_z_j),
            linear_relpos=from_torch(model.linear_relpos),
            linear_token_bonds=from_torch(model.linear_token_bonds),
            max_relative_idx=model.max_relative_idx,
            max_relative_chain=model.max_relative_chain,
        )

    def __call__(self, batch: Batch, *, key, **kwargs) -> tuple[Array, Array, Array]:
        """
        Args:
            batch: Input feature dictionary
            key: PRNG key.

        Returns:
            s_input: [*, N_token, C_s_input] Single (input) representation
            s: [*, N_token, C_s] Single representation
            z: [*, N_token, N_token, C_z] Pair representation
        """
        a, _, _, _ = self.atom_attn_enc(batch=batch, key=key)

        # [*, N_token, C_s_input]
        s_input = jnp.concatenate(
            [
                a,
                batch.restype,
                batch.profile,
                batch.deletion_mean[..., None],
            ],
            axis=-1,
        )

        # [*, N_token, C_s]
        s = self.linear_s(s_input)

        s_input_emb_i = self.linear_z_i(s_input)
        s_input_emb_j = self.linear_z_j(s_input)
        token_bonds_emb = self.linear_token_bonds(
            batch.token_bonds[..., None].astype(s.dtype)
        )

        # [*, N_token, N_token, C_z]
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]

        relpos_feats = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).astype(z.dtype)
        relpos_emb = self.linear_relpos(relpos_feats)
        z = z + relpos_emb

        z = z + token_bonds_emb

        return s_input, s, z


# ---------------------------------------------------------------------------
# MSAModuleEmbedder  (AF3 Algorithm 8 lines 1-4)
# ---------------------------------------------------------------------------

class MSAModuleEmbedder(eqx.Module):
    """Sample MSA features and embed them."""

    linear_m: Linear
    linear_s_input: Linear

    @classmethod
    def from_torch(cls, model) -> "MSAModuleEmbedder":
        return cls(
            linear_m=from_torch(model.linear_m),
            linear_s_input=from_torch(model.linear_s_input),
        )

    def __call__(self, batch: Batch, s_input: Array, **kwargs) -> tuple[Array, Array]:
        """
        Args:
            batch: Input feature dictionary
            s_input: [*, N_token, C_s_input] single embedding

        Returns:
            m: [*, N_seq, N_token, C_m] MSA embedding
            msa_mask: [*, N_seq, N_token] MSA mask
        """
        # [*, N_msa, N_token, 34]
        msa_feat = jnp.concatenate(
            [
                batch.msa,
                batch.has_deletion[..., None],
                batch.deletion_value[..., None],
            ],
            axis=-1,
        )
        msa_mask = batch.msa_mask

        # [*, N_seq, N_token, C_m]
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input)[..., None, :, :]

        return m, msa_mask


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(pt_ie.FourierEmbedding, FourierEmbedding.from_torch)
from_torch.register(pt_ie.InputEmbedderAllAtom, InputEmbedderAllAtom.from_torch)
from_torch.register(pt_ie.MSAModuleEmbedder, MSAModuleEmbedder.from_torch)
