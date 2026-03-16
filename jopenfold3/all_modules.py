"""Import all modules to register all from_torch converters.

Import this module before calling from_torch on the OpenFold3 model.
Each import triggers from_torch.register() calls at module scope.
"""

import jopenfold3.feature_embedders.input_embedders  # noqa: F401
import jopenfold3.feature_embedders.template_embedders  # noqa: F401
import jopenfold3.heads.head_modules  # noqa: F401
import jopenfold3.heads.prediction_heads  # noqa: F401
import jopenfold3.latent.base_blocks  # noqa: F401
import jopenfold3.latent.msa_module  # noqa: F401
import jopenfold3.latent.pairformer  # noqa: F401
import jopenfold3.latent.template_module  # noqa: F401
import jopenfold3.layers.attention_pair_bias  # noqa: F401
import jopenfold3.layers.diffusion_conditioning  # noqa: F401
import jopenfold3.layers.diffusion_transformer  # noqa: F401
import jopenfold3.layers.msa  # noqa: F401
import jopenfold3.layers.outer_product_mean  # noqa: F401
import jopenfold3.layers.sequence_local_atom_attention  # noqa: F401
import jopenfold3.layers.transition  # noqa: F401
import jopenfold3.layers.triangular_attention  # noqa: F401
import jopenfold3.layers.triangular_multiplicative_update  # noqa: F401
import jopenfold3.model  # noqa: F401
import jopenfold3.primitives  # noqa: F401
import jopenfold3.structure.diffusion_module  # noqa: F401
