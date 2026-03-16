"""Top-level OpenFold3 model (JAX translation)."""

from __future__ import annotations

import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.projects.of3_all_atom.model as pt_model
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.batch import Batch
from jopenfold3.feature_embedders.input_embedders import InputEmbedderAllAtom, MSAModuleEmbedder
from jopenfold3.heads.head_modules import AuxiliaryHeadsAllAtom, ConfidenceOutput
from jopenfold3.latent.msa_module import MSAModuleStack
from jopenfold3.latent.pairformer import PairFormerStack
from jopenfold3.latent.template_module import TemplateEmbedderAllAtom
from jopenfold3.primitives import LayerNorm, Linear
from jopenfold3.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    create_noise_schedule,
)

# ---------------------------------------------------------------------------
# Typed state containers
# ---------------------------------------------------------------------------

class InitialEmbedding(eqx.Module):
    """Output of input embedding: s_input, s_init, z_init."""
    s_input: Array
    s_init: Array
    z_init: Array


class TrunkEmbedding(eqx.Module):
    """Output of trunk: single and pair representations."""
    s: Array
    z: Array


class ModelOutput(eqx.Module):
    """Full model output."""
    coordinates: Array
    confidence: ConfidenceOutput
    si_trunk: Array
    zij_trunk: Array


class OpenFold3(eqx.Module):
    """OpenFold3 model. Implements AF3 Algorithm 1."""

    input_embedder: InputEmbedderAllAtom
    layer_norm_z: LayerNorm
    linear_z: Linear
    template_embedder: TemplateEmbedderAllAtom
    msa_module_embedder: MSAModuleEmbedder
    msa_module: MSAModuleStack
    pairformer_stack: PairFormerStack
    diffusion_module: DiffusionModule
    sample_diffusion: SampleDiffusion
    aux_heads: AuxiliaryHeadsAllAtom
    layer_norm_s: LayerNorm | None = None
    linear_s: Linear | None = None
    num_recycles: int = 3

    # Noise schedule / sampling config
    default_num_sampling_steps: int = 200
    default_num_samples: int = 5
    noise_sigma_data: float = 16.0
    noise_s_max: float = 160.0
    noise_s_min: float = 4e-4
    noise_p: int = 7

    @classmethod
    def from_torch(cls, model) -> "OpenFold3":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["num_recycles"] = model.shared.num_recycles

        # Extract noise schedule and sampling config
        shared_diff = model.shared.diffusion
        kwargs["default_num_sampling_steps"] = shared_diff.no_full_rollout_steps
        kwargs["default_num_samples"] = shared_diff.no_full_rollout_samples
        ns = model.config.architecture.noise_schedule
        kwargs["noise_sigma_data"] = float(ns.sigma_data)
        kwargs["noise_s_max"] = float(ns.s_max)
        kwargs["noise_s_min"] = float(ns.s_min)
        kwargs["noise_p"] = int(ns.p)

        return cls(**kwargs)

    def _trunk_iteration(self, batch, s_input, s_init, z_init, token_mask, pair_mask,
                         s, z, *, key, deterministic=True):
        """Single trunk iteration (template + MSA + pairformer)."""
        k_tmpl, k_msa, k_pf = jax.random.split(key, 3)

        z = z_init + self.linear_z(self.layer_norm_z(z))

        z = z + self.template_embedder(
            batch=batch, z=z, pair_mask=pair_mask,
            key=k_tmpl, deterministic=deterministic,
        )

        m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)

        z = self.msa_module(
            m, z,
            msa_mask=msa_mask.astype(m.dtype),
            pair_mask=pair_mask.astype(z.dtype),
            key=k_msa, deterministic=deterministic,
        )

        s = s_init + self.linear_s(self.layer_norm_s(s))
        s, z = self.pairformer_stack(
            s=s, z=z,
            single_mask=token_mask.astype(z.dtype),
            pair_mask=pair_mask.astype(s.dtype),
            key=k_pf, deterministic=deterministic,
        )

        return s, z

    def _run_trunk_loop(self, batch, s_input, s_init, z_init, num_cycles,
                        *, key, deterministic=True):
        """Unified trunk loop: all iterations in a single fori_loop.

        Uses ``jax.lax.cond`` to apply ``stop_gradient`` on all but the
        last iteration, so XLA only traces ``_trunk_iteration`` once.
        """
        token_mask = batch.token_mask
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        s = jnp.zeros_like(s_init)
        z = jnp.zeros_like(z_init)

        def body(i, carry):
            s, z = carry
            # stop_gradient on all iterations except the last
            s, z = jax.lax.cond(
                i < num_cycles - 1,
                jax.lax.stop_gradient,
                lambda sz: sz,
                (s, z),
            )
            iter_key = jax.random.fold_in(key, i)
            return self._trunk_iteration(
                batch, s_input, s_init, z_init, token_mask, pair_mask, s, z,
                key=iter_key, deterministic=deterministic,
            )

        s, z = jax.lax.fori_loop(0, num_cycles, body, (s, z))
        return s, z

    def run_trunk(self, batch, num_cycles, *, key,
                  deterministic=True) -> tuple[InitialEmbedding, TrunkEmbedding]:
        """Implements Algorithm 1 lines 1-14."""
        k_embed, key = jax.random.split(key)
        s_input, s_init, z_init = self.input_embedder(batch=batch, key=k_embed)
        s, z = self._run_trunk_loop(
            batch, s_input, s_init, z_init, num_cycles,
            key=key, deterministic=deterministic,
        )
        init_emb = InitialEmbedding(s_input=s_input, s_init=s_init, z_init=z_init)
        trunk_emb = TrunkEmbedding(s=s, z=z)
        return init_emb, trunk_emb

    # ------------------------------------------------------------------
    # Split JIT boundaries — each stage can be compiled independently
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def embed_inputs(self, batch: Batch, *, key) -> InitialEmbedding:
        """Stage 1: Compute input embeddings (no recycling)."""
        s_input, s_init, z_init = self.input_embedder(batch=batch, key=key)
        return InitialEmbedding(s_input=s_input, s_init=s_init, z_init=z_init)

    @eqx.filter_jit
    def recycle(self, init_emb: InitialEmbedding, batch: Batch,
                num_cycles: int, *, key, deterministic=True) -> TrunkEmbedding:
        """Stage 2: Run recycling trunk iterations."""
        s, z = self._run_trunk_loop(
            batch, init_emb.s_input, init_emb.s_init, init_emb.z_init,
            num_cycles, key=key, deterministic=deterministic,
        )
        return TrunkEmbedding(s=s, z=z)

    @eqx.filter_jit
    def sample_structures(
        self, init_emb: InitialEmbedding, trunk_emb: TrunkEmbedding,
        batch: Batch, num_sampling_steps: int, num_samples: int,
        *, key, deterministic=True,
    ) -> Array:
        """Stage 3: Run diffusion sampling to produce atom coordinates."""
        si_input = init_emb.s_input[:, None, ...]
        si_trunk = trunk_emb.s[:, None, ...]
        zij_trunk = trunk_emb.z[:, None, ...]
        batch = batch.expand_sample_dim()

        noise_schedule = create_noise_schedule(
            num_sampling_steps=num_sampling_steps,
            sigma_data=self.noise_sigma_data,
            s_max=self.noise_s_max,
            s_min=self.noise_s_min,
            p=self.noise_p,
            dtype=si_trunk.dtype,
        )

        return self.sample_diffusion(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            noise_schedule=noise_schedule,
            num_samples=num_samples,
            use_conditioning=True,
            key=key,
            deterministic=deterministic,
        )

    @eqx.filter_jit
    def confidence_metrics(
        self, init_emb: InitialEmbedding, trunk_emb: TrunkEmbedding,
        batch: Batch, coords: Array, *, key,
    ) -> ConfidenceOutput:
        """Stage 4: Compute confidence / auxiliary head outputs."""
        si_input = init_emb.s_input[:, None, ...]
        si_trunk = trunk_emb.s[:, None, ...]
        zij_trunk = trunk_emb.z[:, None, ...]
        batch = batch.expand_sample_dim()

        output_dict = {
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "atom_positions_predicted": coords,
        }
        return self.aux_heads(
            batch=batch,
            si_input=si_input,
            output=output_dict,
            use_zij_trunk_embedding=True,
            key=key,
        )

    def __call__(
        self,
        batch: Batch,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        num_samples: int | None = None,
        *,
        key,
        deterministic=True,
    ) -> ModelOutput:
        """Run the full inference forward pass.

        Composes the 4 split-JIT stages (``embed_inputs``, ``recycle``,
        ``sample_structures``, ``confidence_metrics``).  When called
        directly (not inside an outer ``jit``), each stage is compiled
        and cached independently, giving much faster first-call
        compilation than a single monolithic trace.

        Args:
            batch: Typed batch of model inputs.
            num_recycles: Override number of recycling iterations.
            num_sampling_steps: Override number of diffusion sampling steps.
            num_samples: Override number of diffusion samples.
            key: PRNG key for sampling (required).
            deterministic: If True, dropout is identity.

        Returns:
            ModelOutput with coordinates, confidence metrics, and trunk embeddings.
        """
        if num_recycles is None:
            num_recycles = self.num_recycles
        if num_sampling_steps is None:
            num_sampling_steps = self.default_num_sampling_steps
        if num_samples is None:
            num_samples = self.default_num_samples

        num_cycles = num_recycles + 1
        k_embed, k_trunk, k_diff, k_conf = jax.random.split(key, 4)

        # Stage 1+2: embed + recycle (each @eqx.filter_jit)
        init_emb = self.embed_inputs(batch, key=k_embed)
        trunk_emb = self.recycle(
            init_emb, batch, num_cycles,
            key=k_trunk, deterministic=deterministic,
        )

        # Stage 3: diffusion sampling (@eqx.filter_jit)
        coords = self.sample_structures(
            init_emb, trunk_emb, batch,
            num_sampling_steps, num_samples,
            key=k_diff, deterministic=deterministic,
        )

        # Stage 4: confidence heads (@eqx.filter_jit)
        confidence = self.confidence_metrics(
            init_emb, trunk_emb, batch, coords, key=k_conf,
        )

        return ModelOutput(
            coordinates=coords,
            confidence=confidence,
            si_trunk=trunk_emb.s[:, None, ...],
            zij_trunk=trunk_emb.z[:, None, ...],
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save model weights and skeleton (torch-free loading).

        Writes ``{path}.eqx`` (weights) and ``{path}.skeleton.pkl``
        (pytree structure with ShapeDtypeStruct leaves).
        Reload with ``OpenFold3.load(path)`` — no template model needed.
        """
        path = Path(path).with_suffix("")
        eqx.tree_serialise_leaves(f"{path}.eqx", self)
        skeleton = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x,
            self, is_leaf=eqx.is_array,
        )
        with open(f"{path}.skeleton.pkl", "wb") as f:
            pickle.dump(skeleton, f, protocol=pickle.HIGHEST_PROTOCOL)
        return Path(f"{path}.eqx")

    @classmethod
    def load(cls, path: str | Path = "~/.openfold3/jax/of3") -> "OpenFold3":
        """Load model from saved JAX weights.

        If no saved model exists, automatically downloads the default
        OpenFold3 PyTorch checkpoint and converts it to JAX.
        """
        path = Path(path).expanduser().with_suffix("")
        skeleton_path = Path(f"{path}.skeleton.pkl")
        eqx_path = Path(f"{path}.eqx")

        if not skeleton_path.exists():
            cls._download_and_convert(path)

        with open(skeleton_path, "rb") as f:
            skeleton = pickle.load(f)
        return eqx.tree_deserialise_leaves(str(eqx_path), skeleton)

    @staticmethod
    def _download_and_convert(save_path: Path) -> None:
        """Download the default OpenFold3 checkpoint and convert to JAX."""
        import logging
        import torch

        logger = logging.getLogger(__name__)

        from jopenfold3._vendor.openfold3.entry_points.parameters import (
            DEFAULT_CHECKPOINT_NAME,
            OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
            download_model_parameters,
            get_default_checkpoint_dir,
        )
        from jopenfold3._vendor.openfold3.core.utils.checkpoint_loading_utils import (
            load_checkpoint,
            get_state_dict_from_checkpoint,
        )
        from jopenfold3._vendor.openfold3.projects.of3_all_atom.project_entry import (
            OF3ProjectEntry,
            ModelUpdate,
        )
        from jopenfold3._vendor.openfold3.projects.of3_all_atom.runner import (
            OpenFold3AllAtom,
        )

        # 1. Download checkpoint if needed
        param_dir = get_default_checkpoint_dir()
        param_dir.mkdir(parents=True, exist_ok=True)
        entry = OPENFOLD_MODEL_CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT_NAME]
        ckpt_path = param_dir / entry.file_name

        if not ckpt_path.exists():
            logger.info("Downloading OpenFold3 checkpoint to %s ...", ckpt_path)
            download_model_parameters(
                param_dir, DEFAULT_CHECKPOINT_NAME,
                skip_confirmation=True,
            )

        # 2. Load PyTorch model
        logger.info("Loading PyTorch checkpoint ...")
        pe = OF3ProjectEntry()
        config = pe.get_model_config_with_update(
            ModelUpdate(presets=["predict"], custom={})
        )
        config.settings.memory.eval.use_deepspeed_evo_attention = False
        runner = OpenFold3AllAtom(config)
        ckpt = load_checkpoint(ckpt_path)
        state_dict, _ = get_state_dict_from_checkpoint(
            ckpt, init_from_ema_weights=True,
        )
        state_dict["model.version_tensor"] = torch.tensor([0, 4, 0])
        runner.load_state_dict(state_dict, strict=True)
        runner.model.eval()

        # 3. Convert to JAX
        import jopenfold3.all_modules  # noqa: F401 — register converters

        logger.info("Converting to JAX ...")
        jax_model = from_torch(runner.model)

        # 4. Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        jax_model.save(save_path)
        logger.info("Saved JAX model to %s", save_path)

        del runner, ckpt, state_dict


from_torch.register(pt_model.OpenFold3, OpenFold3.from_torch)
