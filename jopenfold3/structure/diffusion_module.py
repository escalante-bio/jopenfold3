"""Diffusion module and sampling (JAX translation)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.structure.diffusion_module as pt_diff
from jaxtyping import Array

from jopenfold3.backend import from_torch
from jopenfold3.layers.diffusion_conditioning import DiffusionConditioning
from jopenfold3.layers.diffusion_transformer import DiffusionTransformer
from jopenfold3.layers.sequence_local_atom_attention import AtomAttentionDecoder, AtomAttentionEncoder
from jopenfold3.primitives import LayerNorm, Linear


def create_noise_schedule(
    num_sampling_steps: int,
    sigma_data: float,
    s_max: float = 160.0,
    s_min: float = 4e-4,
    p: int = 7,
    dtype=jnp.float32,
) -> Array:
    """Implements AF3 noise schedule (Page 24).

    Args:
        num_sampling_steps: Number of diffusion sampling steps.
        sigma_data: Constant determined by data variance.
        s_max: Maximum standard deviation of noise.
        s_min: Minimum standard deviation of noise.
        p: Constant controlling step distribution.
        dtype: Output dtype.

    Returns:
        Noise schedule array of shape ``[num_sampling_steps + 1]``.
    """
    t = jnp.arange(0, 1 + num_sampling_steps, dtype=dtype) / num_sampling_steps
    return (
        sigma_data
        * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
    )


class DiffusionModule(eqx.Module):
    """Implements AF3 Algorithm 20."""

    diffusion_conditioning: DiffusionConditioning
    atom_attn_enc: AtomAttentionEncoder
    layer_norm_s: LayerNorm
    linear_s: Linear
    diffusion_transformer: DiffusionTransformer
    layer_norm_a: LayerNorm
    atom_attn_dec: AtomAttentionDecoder
    sigma_data: float

    @classmethod
    def from_torch(cls, model) -> "DiffusionModule":
        return cls(
            diffusion_conditioning=from_torch(model.diffusion_conditioning),
            atom_attn_enc=from_torch(model.atom_attn_enc),
            layer_norm_s=from_torch(model.layer_norm_s),
            linear_s=from_torch(model.linear_s),
            diffusion_transformer=from_torch(model.diffusion_transformer),
            layer_norm_a=from_torch(model.layer_norm_a),
            atom_attn_dec=from_torch(model.atom_attn_dec),
            sigma_data=model.sigma_data,
        )

    def __call__(
        self,
        batch,
        xl_noisy,
        token_mask,
        atom_mask,
        t,
        si_input,
        si_trunk,
        zij_trunk,
        use_conditioning,
        *,
        key,
        deterministic=True,
        **kwargs,
    ):
        si, zij = self.diffusion_conditioning(
            batch=batch, t=t, si_input=si_input,
            si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=use_conditioning,
        )

        xl_noisy = xl_noisy * atom_mask[..., None]
        rl_noisy = xl_noisy / jnp.sqrt(t[..., None, None] ** 2 + self.sigma_data**2)

        k_enc, k_dt, k_dec = jax.random.split(key, 3)

        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch, rl=rl_noisy, si_trunk=si_trunk, zij_trunk=zij,
            key=k_enc,
        )

        ai = ai + self.linear_s(self.layer_norm_s(si))

        ai = self.diffusion_transformer(a=ai, s=si, z=zij, mask=token_mask,
                                        key=k_dt, deterministic=deterministic)

        ai = self.layer_norm_a(ai)

        rl_update = self.atom_attn_dec(
            batch=batch, ai=ai, ql=ql, cl=cl, plm=plm,
            key=k_dec,
        )

        xl_out = (
            self.sigma_data**2 / (self.sigma_data**2 + t[..., None, None] ** 2) * xl_noisy
            + self.sigma_data * t[..., None, None]
            / jnp.sqrt(self.sigma_data**2 + t[..., None, None] ** 2) * rl_update
        )

        return xl_out * atom_mask[..., None]


class SampleDiffusion(eqx.Module):
    """Implements AF3 Algorithm 18."""

    diffusion_module: DiffusionModule
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float

    @classmethod
    def from_torch(cls, model) -> "SampleDiffusion":
        return cls(
            diffusion_module=from_torch(model.diffusion_module),
            gamma_0=model.gamma_0,
            gamma_min=model.gamma_min,
            noise_scale=model.noise_scale,
            step_scale=model.step_scale,
        )

    def __call__(
        self,
        batch,
        si_input,
        si_trunk,
        zij_trunk,
        noise_schedule,
        num_samples,
        use_conditioning=True,
        *,
        key,
        deterministic=True,
        **kwargs,
    ):
        atom_mask = batch.atom_mask
        batch_dim = atom_mask.shape[0]
        num_atoms = atom_mask.shape[-1]

        key, subkey = jax.random.split(key)
        xl = noise_schedule[0] * jax.random.normal(
            subkey, (batch_dim, num_samples, num_atoms, 3),
            dtype=atom_mask.dtype,
        )

        # Pair consecutive noise levels: (sigma[tau], sigma[tau+1])
        sigmas_cur = noise_schedule[:-1]   # [T]
        sigmas_next = noise_schedule[1:]   # [T]

        @jax.checkpoint
        def step_fn(carry, step_inputs):
            xl, key = carry
            sigma_cur, sigma_next = step_inputs

            gamma = jnp.where(sigma_next > self.gamma_min, self.gamma_0, 0.0)
            t_val = sigma_cur * (gamma + 1)

            key, k1, k_diff = jax.random.split(key, 3)
            noise = (
                self.noise_scale
                * jnp.sqrt(t_val**2 - sigma_cur**2)
                * jax.random.normal(k1, xl.shape, dtype=xl.dtype)
            )
            xl_noisy = xl + noise

            xl_denoised = self.diffusion_module(
                batch=batch, xl_noisy=xl_noisy,
                token_mask=batch.token_mask, atom_mask=atom_mask,
                t=t_val, si_input=si_input,
                si_trunk=si_trunk, zij_trunk=zij_trunk,
                use_conditioning=use_conditioning,
                key=k_diff, deterministic=deterministic,
            )

            delta = (xl_noisy - xl_denoised) / t_val
            dt = sigma_next - t_val
            xl = xl_noisy + self.step_scale * dt * delta

            return (xl, key), None

        (xl, _), _ = jax.lax.scan(step_fn, (xl, key), (sigmas_cur, sigmas_next))
        return xl


from_torch.register(pt_diff.DiffusionModule, DiffusionModule.from_torch)
from_torch.register(pt_diff.SampleDiffusion, SampleDiffusion.from_torch)
