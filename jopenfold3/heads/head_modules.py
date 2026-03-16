"""Auxiliary head module translations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.heads.head_modules as pt_hm
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.heads.prediction_heads import (
    DistogramHead,
    ExperimentallyResolvedHeadAllAtom,
    PairformerEmbedding,
    PerResidueLDDTAllAtom,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
)
from jopenfold3.utils import (
    get_token_representative_atoms,
)


class ConfidenceOutput(eqx.Module):
    """Outputs from the auxiliary confidence heads."""
    distogram_logits: Array
    plddt_logits: Array
    pde_logits: Array
    experimentally_resolved_logits: Array
    pae_logits: Array | None


# ---------------------------------------------------------------------------
# AuxiliaryHeadsAllAtom  (AF3 Algorithm 31)
# ---------------------------------------------------------------------------

class AuxiliaryHeadsAllAtom(eqx.Module):
    """Auxiliary head container for OF3."""

    pairformer_embedding: PairformerEmbedding
    pde: PredictedDistanceErrorHead
    plddt: PerResidueLDDTAllAtom
    distogram: DistogramHead
    experimentally_resolved: ExperimentallyResolvedHeadAllAtom
    pae: PredictedAlignedErrorHead | None
    max_atoms_per_token: int
    pae_enabled: bool

    @classmethod
    def from_torch(cls, model) -> "AuxiliaryHeadsAllAtom":
        pae = None
        pae_enabled = model.config.pae.enabled
        if hasattr(model, "pae"):
            pae = from_torch(model.pae)

        return cls(
            pairformer_embedding=from_torch(model.pairformer_embedding),
            pde=from_torch(model.pde),
            plddt=from_torch(model.plddt),
            distogram=from_torch(model.distogram),
            experimentally_resolved=from_torch(model.experimentally_resolved),
            pae=pae,
            max_atoms_per_token=model.max_atoms_per_token,
            pae_enabled=pae_enabled,
        )

    def __call__(
        self,
        batch: Batch,
        si_input: Array,
        output: dict,
        use_zij_trunk_embedding: bool,
        *,
        key,
        **kwargs,
    ) -> ConfidenceOutput:
        """
        Args:
            batch: Input feature dictionary
            si_input: [*, N_token, C_s_input] Single (input) representation
            output: Dict containing:
                "si_trunk": [*, N_token, C_s] Single representation
                "zij_trunk": [*, N_token, N_token, C_z] Pair representation
                "atom_positions_predicted": [*, N_atom, 3] Predicted positions
            use_zij_trunk_embedding: Whether to use zij trunk embedding

        Returns:
            ConfidenceOutput with all head logits.
        """
        si = output["si_trunk"]
        zij = output["zij_trunk"]
        atom_positions_predicted = output["atom_positions_predicted"].astype(si.dtype)

        # Distogram head
        distogram_logits = self.distogram(z=zij)

        token_mask = batch.token_mask
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch, x=atom_positions_predicted, atom_mask=batch.atom_mask
        )

        if not use_zij_trunk_embedding:
            zij = zij * 0

        # Embed trunk outputs
        si, zij = self.pairformer_embedding(
            si_input=si_input,
            si=si,
            zij=zij,
            x_pred=repr_x_pred,
            single_mask=repr_x_mask,
            pair_mask=pair_mask,
            key=key,
        )

        # Build atom mask padded to MAX_ATOMS_PER_TOKEN per token.
        # For each token, create max_atoms_per_token slots; valid ones are
        # where slot_index < num_atoms_per_token[token].
        n_token = token_mask.shape[-1]
        slot_indices = jnp.arange(self.max_atoms_per_token)[None, :]  # [1, M]
        per_token_counts = batch.num_atoms_per_token * token_mask.astype(
            batch.num_atoms_per_token.dtype
        )  # [*, N_token]
        # [*, N_token, M] -> [*, N_token * M]
        max_atom_per_token_mask = (
            slot_indices < per_token_counts[..., None]
        ).astype(token_mask.dtype).reshape(*token_mask.shape[:-1], n_token * self.max_atoms_per_token)
        # Expand to match sample dimension
        max_atom_per_token_mask = jnp.broadcast_to(
            max_atom_per_token_mask,
            (*atom_positions_predicted.shape[:-2], max_atom_per_token_mask.shape[-1]),
        )

        n_atom = batch.atom_mask.shape[-1]

        plddt_logits = self.plddt(
            s=si, max_atom_per_token_mask=max_atom_per_token_mask, n_atom=n_atom,
        )

        experimentally_resolved_logits = self.experimentally_resolved(
            si, max_atom_per_token_mask, n_atom=n_atom,
        )

        pde_logits = self.pde(zij)

        pae_logits = None
        if self.pae_enabled and self.pae is not None:
            pae_logits = self.pae(zij)

        return ConfidenceOutput(
            distogram_logits=distogram_logits,
            plddt_logits=plddt_logits,
            pde_logits=pde_logits,
            experimentally_resolved_logits=experimentally_resolved_logits,
            pae_logits=pae_logits,
        )


# ---------------------------------------------------------------------------
# Register converter
# ---------------------------------------------------------------------------

from_torch.register(pt_hm.AuxiliaryHeadsAllAtom, AuxiliaryHeadsAllAtom.from_torch)
