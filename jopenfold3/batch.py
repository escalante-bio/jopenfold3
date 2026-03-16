"""Typed batch container for OpenFold3 JAX model inputs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float, Int


class Batch(eqx.Module):
    """Typed, JIT-friendly batch of model inputs.

    All fields are JAX arrays. Non-array metadata (``atom_array``,
    ``query_id``, etc.) lives outside this container and is handled
    by the prediction pipeline separately.

    Dimension key::

        B     = batch
        Nt    = number of tokens (residues / ligand groups)
        Na    = number of atoms
        Nm    = number of MSA sequences
        Ntp   = number of templates
    """

    # ------------------------------------------------------------------
    # Token-level features  (B, Nt, ...)
    # ------------------------------------------------------------------
    token_mask: Float[Array, "B Nt"]
    token_index: Int[Array, "B Nt"]
    token_bonds: Int[Array, "B Nt Nt"]
    restype: Float[Array, "B Nt 32"]
    profile: Float[Array, "B Nt 32"]
    deletion_mean: Float[Array, "B Nt"]
    residue_index: Int[Array, "B Nt"]
    asym_id: Int[Array, "B Nt"]
    entity_id: Int[Array, "B Nt"]
    sym_id: Int[Array, "B Nt"]
    is_protein: Int[Array, "B Nt"]
    is_dna: Int[Array, "B Nt"]
    is_rna: Int[Array, "B Nt"]
    is_atomized: Int[Array, "B Nt"]
    start_atom_index: Int[Array, "B Nt"]
    num_atoms_per_token: Int[Array, "B Nt"]

    # ------------------------------------------------------------------
    # Atom-level features  (B, Na, ...)
    # ------------------------------------------------------------------
    atom_mask: Float[Array, "B Na"]
    atom_to_token_index: Int[Array, "B Na"]
    ref_pos: Float[Array, "B Na 3"]
    ref_charge: Float[Array, "B Na"]
    ref_mask: Int[Array, "B Na"]
    ref_element: Int[Array, "B Na 119"]
    ref_atom_name_chars: Int[Array, "B Na 4 64"]
    ref_space_uid: Int[Array, "B Na"]

    # ------------------------------------------------------------------
    # MSA features  (B, Nm, Nt, ...)
    # ------------------------------------------------------------------
    msa: Float[Array, "B Nm Nt 32"]
    msa_mask: Float[Array, "B Nm Nt"]
    has_deletion: Float[Array, "B Nm Nt"]
    deletion_value: Float[Array, "B Nm Nt"]

    # ------------------------------------------------------------------
    # Template features  (B, Ntp, ...)
    # ------------------------------------------------------------------
    template_distogram: Float[Array, "B Ntp Nt Nt 39"]
    template_unit_vector: Float[Array, "B Ntp Nt Nt 3"]
    template_restype: Int[Array, "B Ntp Nt 32"]
    template_backbone_frame_mask: Float[Array, "B Ntp Nt"]
    template_pseudo_beta_mask: Float[Array, "B Ntp Nt"]

    # ------------------------------------------------------------------
    # Precomputed representative atom indices  (B, Nt)
    # ------------------------------------------------------------------
    representative_atom_index: Int[Array, "B Nt"]
    representative_atom_mask: Float[Array, "B Nt"]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_representative_atom_indices(
        restype, is_protein, is_dna, is_rna, is_atomized, start_atom_index,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-token representative atom index and mask.

        Proteins: CB (CA for glycine). DNA purines: C4. DNA pyrimidines: C2.
        Must be called with one-hot restype (before soft sequence injection).
        """
        from jopenfold3._vendor.openfold3.core.data.resources.residues import (
            STANDARD_PROTEIN_RESIDUES_ORDER,
            STANDARD_RESIDUES_WITH_GAP_3,
        )
        from jopenfold3._vendor.openfold3.core.data.resources.token_atom_constants import (
            atom_name_to_index_by_restype,
        )

        def _offset_and_mask(atom_name):
            index_arr = jnp.array(
                atom_name_to_index_by_restype[atom_name]["index"], dtype=jnp.float32,
            )
            mask_arr = jnp.array(
                atom_name_to_index_by_restype[atom_name]["mask"], dtype=jnp.float32,
            )
            offset = jnp.einsum("...k,k->...", restype.astype(jnp.float32), index_arr).astype(jnp.int32)
            mask = jnp.einsum("...k,k->...", restype.astype(jnp.float32), mask_arr).astype(jnp.int32)
            return offset, mask

        is_standard_protein = is_protein * (1 - is_atomized)
        is_standard_glycine = is_standard_protein * restype[..., STANDARD_PROTEIN_RESIDUES_ORDER["G"]]
        is_standard_dna = is_dna * (1 - is_atomized)
        is_standard_rna = is_rna * (1 - is_atomized)

        is_standard_purine = is_standard_dna * (
            restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("DA")]
            + restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("DG")]
        ) + is_standard_rna * (
            restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("A")]
            + restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("G")]
        )
        is_standard_pyrimidine = is_standard_dna * (
            restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("DC")]
            + restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("DT")]
        ) + is_standard_rna * (
            restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("C")]
            + restype[..., STANDARD_RESIDUES_WITH_GAP_3.index("U")]
        )

        start = start_atom_index.astype(jnp.int32)
        cb_offset, cb_mask = _offset_and_mask("CB")
        ca_offset, ca_mask = _offset_and_mask("CA")
        c4_offset, c4_mask = _offset_and_mask("C4")
        c2_offset, c2_mask = _offset_and_mask("C2")

        rep_index = (
            (start + cb_offset) * is_standard_protein * (1 - is_standard_glycine)
            + (start + ca_offset) * is_standard_glycine
            + (start + c4_offset) * is_standard_purine
            + (start + c2_offset) * is_standard_pyrimidine
            + start * is_atomized
        ).astype(jnp.int32)

        rep_mask = (
            cb_mask * is_standard_protein * (1 - is_standard_glycine)
            + ca_mask * is_standard_glycine
            + c4_mask * is_standard_purine
            + c2_mask * is_standard_pyrimidine
            + is_atomized
        ).astype(jnp.float32)

        return rep_index, rep_mask

    @classmethod
    def from_dict(cls, d: dict) -> "Batch":
        """Build a Batch from a dict of JAX arrays, ignoring non-model keys."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: d[k] for k in fields if k in d})

    @classmethod
    def from_torch_dict(cls, d: dict) -> "Batch":
        """Build a Batch from a dict of PyTorch/numpy tensors.

        Converts tensors to JAX arrays and adds a leading batch dimension.
        Non-model keys (``atom_array``, ``query_id``, etc.) are ignored.
        """
        def _to_jax(v):
            match v:
                case torch.Tensor():
                    return jnp.array(v.detach().cpu().numpy())[None]
                case np.ndarray():
                    return jnp.array(v)[None]
                case _:
                    return jnp.asarray(v)[None]

        fields = {f.name for f in cls.__dataclass_fields__.values()}
        jax_dict = {k: _to_jax(d[k]) for k in fields if k in d}

        # Compute representative atom indices from one-hot restype before casting
        rep_index, rep_mask = cls._compute_representative_atom_indices(
            jax_dict["restype"], jax_dict["is_protein"], jax_dict["is_dna"],
            jax_dict["is_rna"], jax_dict["is_atomized"], jax_dict["start_atom_index"],
        )
        jax_dict["representative_atom_index"] = rep_index
        jax_dict["representative_atom_mask"] = rep_mask

        jax_dict["restype"] = jax_dict["restype"].astype(jnp.float32)
        jax_dict["msa"] = jax_dict["msa"].astype(jnp.float32)
        return cls(**jax_dict)

    def expand_sample_dim(self) -> "Batch":
        """Insert a sample dimension at axis 1: ``(B, ...) → (B, 1, ...)``."""
        return jax.tree.map(lambda x: x[:, None, ...], self)
