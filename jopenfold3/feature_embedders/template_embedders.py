"""Template embedder translations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.feature_embedders.template_embedders as pt_te
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.primitives import LayerNorm, Linear

# ---------------------------------------------------------------------------
# TemplatePairEmbedderAllAtom  (AF3 Algorithm 16 lines 1-5, 8)
# ---------------------------------------------------------------------------

class TemplatePairEmbedderAllAtom(eqx.Module):
    """Embeds template pair features for the all-atom model."""

    dgram_linear: Linear
    aatype_linear_1: Linear
    aatype_linear_2: Linear
    pseudo_beta_mask_linear: Linear
    x_linear: Linear
    y_linear: Linear
    z_linear: Linear
    backbone_mask_linear: Linear
    layer_norm_z: LayerNorm
    linear_z: Linear

    @classmethod
    def from_torch(cls, model) -> "TemplatePairEmbedderAllAtom":
        return cls(
            dgram_linear=from_torch(model.dgram_linear),
            aatype_linear_1=from_torch(model.aatype_linear_1),
            aatype_linear_2=from_torch(model.aatype_linear_2),
            pseudo_beta_mask_linear=from_torch(model.pseudo_beta_mask_linear),
            x_linear=from_torch(model.x_linear),
            y_linear=from_torch(model.y_linear),
            z_linear=from_torch(model.z_linear),
            backbone_mask_linear=from_torch(model.backbone_mask_linear),
            layer_norm_z=from_torch(model.layer_norm_z),
            linear_z=from_torch(model.linear_z),
        )

    def _embed_feats(self, batch: Batch) -> Array:
        dtype = batch.template_unit_vector.dtype

        # [*, N_token, N_token]
        multichain_pair_mask = (
            batch.asym_id[..., None] == batch.asym_id[..., None, :]
        )
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        # [*, N_templ, N_token, N_token]
        pseudo_beta_pair_mask = (
            batch.template_pseudo_beta_mask[..., None]
            * batch.template_pseudo_beta_mask[..., None, :]
        )[..., None] * multichain_pair_mask

        template_distogram = batch.template_distogram

        backbone_frame_pair_mask = (
            batch.template_backbone_frame_mask[..., None]
            * batch.template_backbone_frame_mask[..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch.template_unit_vector
        x = template_unit_vector[..., 0]
        y = template_unit_vector[..., 1]
        z = template_unit_vector[..., 2]

        # [*, N_templ, N_token, N_token, 32]
        template_restype = batch.template_restype
        n_token = template_restype.shape[-2]
        template_restype_ti = jnp.broadcast_to(
            template_restype[..., None, :],
            (*template_restype.shape[:-2], template_restype.shape[-2], n_token, template_restype.shape[-1]),
        )
        template_restype_tj = jnp.broadcast_to(
            template_restype[..., None, :, :],
            (*template_restype.shape[:-2], n_token, template_restype.shape[-2], template_restype.shape[-1]),
        )

        a = self.dgram_linear(template_distogram)
        a = a + self.pseudo_beta_mask_linear(pseudo_beta_pair_mask)
        a = a + self.aatype_linear_1(template_restype_ti.astype(dtype))
        a = a + self.aatype_linear_2(template_restype_tj.astype(dtype))
        a = a + self.x_linear(x[..., None])
        a = a + self.y_linear(y[..., None])
        a = a + self.z_linear(z[..., None])
        a = a + self.backbone_mask_linear(backbone_frame_pair_mask)

        return a

    def __call__(self, batch: Batch, z: Array, **kwargs) -> Array:
        """
        Args:
            batch: Input template feature dictionary
            z: Pair embedding

        Returns:
            [*, N_templ, N_token, N_token, C_out] Template pair feature embedding
        """
        a = self._embed_feats(batch=batch)

        # [*, N_templ, N_token, N_token, C_out]
        z = self.linear_z(self.layer_norm_z(z))
        z = z[..., None, :, :, :] + a

        return z


# ---------------------------------------------------------------------------
# Register converters
# ---------------------------------------------------------------------------

from_torch.register(
    pt_te.TemplatePairEmbedderAllAtom,
    TemplatePairEmbedderAllAtom.from_torch,
)
