"""Sequence-local atom attention module translations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.layers.sequence_local_atom_attention as pt_slaa
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.layers.diffusion_transformer import DiffusionTransformer
from jopenfold3.primitives import LayerNorm, Linear, ReLU, Sequential
from jopenfold3.utils import (
    aggregate_atom_feat_to_tokens,
    broadcast_token_feat_to_atoms,
    convert_pair_rep_to_blocks,
    convert_single_rep_to_blocks,
)

# ---------------------------------------------------------------------------
# RefAtomFeatureEmbedder  (AF3 Algorithm 5, lines 1-6)
# ---------------------------------------------------------------------------

class RefAtomFeatureEmbedder(eqx.Module):
    """Reference atom feature embedder."""

    linear_ref_pos: Linear
    linear_ref_charge: Linear
    linear_ref_mask: Linear
    linear_ref_element: Linear
    linear_ref_atom_chars: Linear
    linear_ref_offset: Linear
    linear_inv_sq_dists: Linear
    linear_valid_mask: Linear

    @classmethod
    def from_torch(cls, model) -> "RefAtomFeatureEmbedder":
        return cls(
            linear_ref_pos=from_torch(model.linear_ref_pos),
            linear_ref_charge=from_torch(model.linear_ref_charge),
            linear_ref_mask=from_torch(model.linear_ref_mask),
            linear_ref_element=from_torch(model.linear_ref_element),
            linear_ref_atom_chars=from_torch(model.linear_ref_atom_chars),
            linear_ref_offset=from_torch(model.linear_ref_offset),
            linear_inv_sq_dists=from_torch(model.linear_inv_sq_dists),
            linear_valid_mask=from_torch(model.linear_valid_mask),
        )

    def __call__(
        self,
        batch: Batch,
        n_query: int,
        n_key: int,
        **kwargs,
    ) -> tuple[Array, Array]:
        """
        Args:
            batch: Input feature dictionary
            n_query: Block height
            n_key: Block width

        Returns:
            cl: [*, N_atom, c_atom] Atom single conditioning
            plm: [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
        """
        dtype = batch.ref_pos.dtype

        # Embed atom features
        # [*, N_atom, c_atom]
        cl = self.linear_ref_pos(batch.ref_pos)
        cl = cl + self.linear_ref_charge(
            jnp.arcsinh(batch.ref_charge[..., None])
        )
        cl = cl + self.linear_ref_mask(
            batch.ref_mask[..., None].astype(dtype)
        )
        cl = cl + self.linear_ref_element(
            batch.ref_element.astype(dtype)
        )
        cl = cl + self.linear_ref_atom_chars(
            batch.ref_atom_name_chars.reshape(
                *batch.ref_atom_name_chars.shape[:-2], -1
            ).astype(dtype)
        )

        # Embed offsets
        d_l, d_m, atom_mask = convert_single_rep_to_blocks(
            ql=batch.ref_pos,
            n_query=n_query,
            n_key=n_key,
            atom_mask=batch.atom_mask,
        )
        v_l, v_m, _ = convert_single_rep_to_blocks(
            ql=batch.ref_space_uid[..., None],
            n_query=n_query,
            n_key=n_key,
            atom_mask=batch.atom_mask,
        )

        # dlm: [*, N_blocks, N_query, N_key, 3]
        # vlm: [*, N_blocks, N_query, N_key, 1]
        dlm = (d_l[..., :, None, :] - d_m[..., None, :, :]) * atom_mask[..., None]
        vlm = (v_l[..., :, None, :] == v_m[..., None, :, :]).astype(
            dlm.dtype
        ) * atom_mask[..., None]

        plm = self.linear_ref_offset(dlm) * vlm

        # Embed pairwise inverse squared distances
        inv_sq_dists = 1.0 / (1 + jnp.sum(dlm**2, axis=-1, keepdims=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


# ---------------------------------------------------------------------------
# NoisyPositionEmbedder  (AF3 Algorithm 5, lines 8-12)
# ---------------------------------------------------------------------------

class NoisyPositionEmbedder(eqx.Module):
    """Noisy position embedder for diffusion conditioning."""

    layer_norm_s: LayerNorm
    linear_s: Linear
    layer_norm_z: LayerNorm
    linear_z: Linear
    linear_r: Linear

    @classmethod
    def from_torch(cls, model) -> "NoisyPositionEmbedder":
        return cls(
            layer_norm_s=from_torch(model.layer_norm_s),
            linear_s=from_torch(model.linear_s),
            layer_norm_z=from_torch(model.layer_norm_z),
            linear_z=from_torch(model.linear_z),
            linear_r=from_torch(model.linear_r),
        )

    def __call__(
        self,
        batch: Batch,
        cl: Array,
        plm: Array,
        si_trunk: Array,
        zij_trunk: Array,
        rl: Array,
        n_query: int,
        n_key: int,
        **kwargs,
    ) -> tuple[Array, Array, Array]:
        """
        Args:
            batch: Input feature dictionary
            cl: [*, N_atom, c_atom] Atom single conditioning
            plm: [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
            si_trunk: [*, N_token, c_s] Trunk single representation
            zij_trunk: [*, N_token, N_token, c_z] Trunk pair representation
            rl: [*, N_atom, 3] Noisy atom positions
            n_query: Block height
            n_key: Block width

        Returns:
            cl: [*, N_atom, c_atom] Updated atom single conditioning
            plm: [*, N_blocks, N_query, N_key, c_atom_pair] Updated atom pair conditioning
            ql: [*, N_atom, c_atom] Atom single representation
        """
        # Broadcast trunk single representation into atom single conditioning
        si_trunk = self.linear_s(self.layer_norm_s(si_trunk))
        si_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch.token_mask,
            num_atoms_per_token=batch.num_atoms_per_token,
            token_feat=si_trunk,
            token_dim=-2,
            atom_to_token_index=batch.atom_to_token_index,
            atom_mask=batch.atom_mask,
        )
        cl = cl + si_trunk

        # Broadcast trunk pair representation into atom pair conditioning
        zij_trunk = self.linear_z(self.layer_norm_z(zij_trunk))
        zij_trunk = convert_pair_rep_to_blocks(
            batch=batch, zij_trunk=zij_trunk, n_query=n_query, n_key=n_key
        )
        plm = plm + zij_trunk

        # Add noisy coordinate projection
        ql = cl + self.linear_r(rl)

        return cl, plm, ql


# ---------------------------------------------------------------------------
# AtomAttentionEncoder  (AF3 Algorithm 5)
# ---------------------------------------------------------------------------

class AtomAttentionEncoder(eqx.Module):
    """Atom attention encoder."""

    ref_atom_feature_embedder: RefAtomFeatureEmbedder
    noisy_position_embedder: NoisyPositionEmbedder | None
    relu: ReLU
    linear_l: Linear
    linear_m: Linear
    pair_mlp: Sequential
    atom_transformer: DiffusionTransformer
    linear_q: Sequential
    n_query: int
    n_key: int

    @classmethod
    def from_torch(cls, model) -> "AtomAttentionEncoder":
        noisy_pos = None
        if hasattr(model, "noisy_position_embedder"):
            noisy_pos = from_torch(model.noisy_position_embedder)

        return cls(
            ref_atom_feature_embedder=from_torch(model.ref_atom_feature_embedder),
            noisy_position_embedder=noisy_pos,
            relu=from_torch(model.relu),
            linear_l=from_torch(model.linear_l),
            linear_m=from_torch(model.linear_m),
            pair_mlp=from_torch(model.pair_mlp),
            atom_transformer=from_torch(model.atom_transformer),
            linear_q=from_torch(model.linear_q),
            n_query=model.n_query,
            n_key=model.n_key,
        )

    def get_atom_reps(
        self,
        batch: Batch,
        rl: Array | None = None,
        si_trunk: Array | None = None,
        zij_trunk: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Compute atom representations."""
        # Embed reference atom features
        cl, plm = self.ref_atom_feature_embedder(
            batch=batch, n_query=self.n_query, n_key=self.n_key
        )

        # Embed noisy atom positions and trunk embeddings
        if rl is not None and self.noisy_position_embedder is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch,
                cl=cl,
                plm=plm,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                rl=rl,
                n_query=self.n_query,
                n_key=self.n_key,
            )
        else:
            ql = cl.copy()

        # Add the combined single conditioning to the pair rep (line 13-14)
        cl_l, cl_m, atom_mask = convert_single_rep_to_blocks(
            ql=cl,
            n_query=self.n_query,
            n_key=self.n_key,
            atom_mask=batch.atom_mask,
        )

        cl_lm = (
            self.linear_l(self.relu(cl_l[..., :, None, :]))
            + self.linear_m(self.relu(cl_m[..., None, :, :]))
        ) * atom_mask[..., None]

        plm = plm + cl_lm
        plm = plm + self.pair_mlp(plm)
        plm = plm * atom_mask[..., None]

        return ql, cl, plm

    def __call__(
        self,
        batch: Batch,
        rl: Array | None = None,
        si_trunk: Array | None = None,
        zij_trunk: Array | None = None,
        *,
        key,
        **kwargs,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Args:
            batch: Input feature dictionary
            rl: [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk: [*, N_token, c_s] Trunk single representation (optional)
            zij_trunk: [*, N_token, N_token, c_z] Trunk pair representation (optional)
            key: PRNG key.

        Returns:
            ai: [*, N_token, c_token] Token representation
            ql: [*, N_atom, c_atom] Atom single representation
            cl: [*, N_atom, c_atom] Atom single conditioning
            plm: [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
        """
        atom_mask = batch.atom_mask

        ql, cl, plm = self.get_atom_reps(
            batch=batch, rl=rl, si_trunk=si_trunk, zij_trunk=zij_trunk
        )

        # Cross attention transformer (line 15)
        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=atom_mask, key=key, deterministic=True
        )

        ql = ql * atom_mask[..., None]

        # Aggregate atom features to tokens
        ai = aggregate_atom_feat_to_tokens(
            token_mask=batch.token_mask,
            atom_to_token_index=batch.atom_to_token_index,
            atom_mask=atom_mask,
            atom_feat=self.linear_q(ql),
            atom_dim=-2,
            aggregate_fn="mean",
        )

        return ai, ql, cl, plm


# ---------------------------------------------------------------------------
# AtomAttentionDecoder  (AF3 Algorithm 6)
# ---------------------------------------------------------------------------

class AtomAttentionDecoder(eqx.Module):
    """Atom attention decoder."""

    linear_q_in: Linear
    atom_transformer: DiffusionTransformer
    layer_norm: LayerNorm
    linear_q_out: Linear

    @classmethod
    def from_torch(cls, model) -> "AtomAttentionDecoder":
        return cls(
            linear_q_in=from_torch(model.linear_q_in),
            atom_transformer=from_torch(model.atom_transformer),
            layer_norm=from_torch(model.layer_norm),
            linear_q_out=from_torch(model.linear_q_out),
        )

    def __call__(
        self,
        batch: Batch,
        ai: Array,
        ql: Array,
        cl: Array,
        plm: Array,
        *,
        key,
        **kwargs,
    ) -> Array:
        """
        Args:
            batch: Input feature dictionary
            ai: [*, N_token, c_token] Token representation
            ql: [*, N_atom, c_atom] Atom single representation
            cl: [*, N_atom, c_atom] Atom single conditioning
            plm: [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
            key: PRNG key.

        Returns:
            rl_update: [*, N_atom, 3] Atom position updates
        """
        # Broadcast per-token activations to atoms
        ql = ql + broadcast_token_feat_to_atoms(
            token_mask=batch.token_mask,
            num_atoms_per_token=batch.num_atoms_per_token,
            token_feat=self.linear_q_in(ai),
            token_dim=-2,
            atom_to_token_index=batch.atom_to_token_index,
            atom_mask=batch.atom_mask,
        )

        # Atom transformer
        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=batch.atom_mask, key=key, deterministic=True
        )

        # Compute updates for atom positions
        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(pt_slaa.RefAtomFeatureEmbedder, RefAtomFeatureEmbedder.from_torch)
from_torch.register(pt_slaa.NoisyPositionEmbedder, NoisyPositionEmbedder.from_torch)
from_torch.register(pt_slaa.AtomAttentionEncoder, AtomAttentionEncoder.from_torch)
from_torch.register(pt_slaa.AtomAttentionDecoder, AtomAttentionDecoder.from_torch)
