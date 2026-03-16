"""Template embedding modules (JAX translation)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jopenfold3._vendor.openfold3.core.model.latent.template_module as pt_tmpl

from jopenfold3.backend import from_torch
from jopenfold3.feature_embedders.template_embedders import TemplatePairEmbedderAllAtom
from jopenfold3.latent.base_blocks import PairBlock
from jopenfold3.primitives import LayerNorm, Linear


class TemplatePairBlock(PairBlock):
    """Implements one block of AF2 Algorithm 16.

    Processes each template independently via vmap over the template dim.
    """

    tri_mul_first: bool = True

    @classmethod
    def from_torch(cls, model) -> "TemplatePairBlock":
        kwargs = {}
        for name, child in model.named_children():
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        kwargs["tri_mul_first"] = model.tri_mul_first
        return cls(**kwargs)

    def _forward_single(self, t, mask, *, key, deterministic=True):
        """Process a single template: t [*, N, N, C], mask [*, N, N]."""
        k1, k2, k3, k4 = jax.random.split(key, 4)

        if self.tri_mul_first:
            t = t + self.ps_dropout_row_layer(
                self.tri_mul_out(t, mask=mask), key=k1, deterministic=deterministic)
            t = t + self.ps_dropout_row_layer(
                self.tri_mul_in(t, mask=mask), key=k2, deterministic=deterministic)
            t = t + self.ps_dropout_row_layer(
                self.tri_att_start(t, mask=mask), key=k3, deterministic=deterministic)
            t = jnp.swapaxes(t, -2, -3)
            t = t + self.ps_dropout_row_layer(
                self.tri_att_end(t, mask=jnp.swapaxes(mask, -1, -2)),
                key=k4, deterministic=deterministic)
            t = jnp.swapaxes(t, -2, -3)
        else:
            t = t + self.ps_dropout_row_layer(
                self.tri_att_start(t, mask=mask), key=k1, deterministic=deterministic)
            t = jnp.swapaxes(t, -2, -3)
            t = t + self.ps_dropout_row_layer(
                self.tri_att_end(t, mask=jnp.swapaxes(mask, -1, -2)),
                key=k2, deterministic=deterministic)
            t = jnp.swapaxes(t, -2, -3)
            t = t + self.ps_dropout_row_layer(
                self.tri_mul_out(t, mask=mask), key=k3, deterministic=deterministic)
            t = t + self.ps_dropout_row_layer(
                self.tri_mul_in(t, mask=mask), key=k4, deterministic=deterministic)

        t = t + self.pair_transition(t, mask=mask)
        return t

    def __call__(self, t, mask=None, *, key, deterministic=True, **kwargs):
        if mask is None:
            mask = jnp.ones(t.shape[:-1])

        # vmap over the template dimension (-4)
        # t: [*, N_templ, N, N, C], mask: [*, N_templ, N, N]
        n_templ = t.shape[-4]
        keys = jax.random.split(key, n_templ)
        def fn(ti, mi, ki):
            return self._forward_single(ti, mi, key=ki, deterministic=deterministic)

        return jax.vmap(fn, in_axes=(-4, -3, 0), out_axes=-4)(t, mask, keys)


class TemplatePairStack(eqx.Module):
    """Stack of TemplatePairBlocks using scan."""

    stacked_params: TemplatePairBlock
    static: TemplatePairBlock
    layer_norm: LayerNorm

    @classmethod
    def from_torch(cls, model) -> "TemplatePairStack":
        layers = [from_torch(b) for b in model.blocks]
        layer_norm = from_torch(model.layer_norm)

        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return cls(stacked_params=stacked, static=static, layer_norm=layer_norm)

    def __call__(self, t, mask=None, *, key, deterministic=True, **kwargs):
        if mask is not None and mask.shape[-3] == 1:
            mask = jnp.broadcast_to(
                mask, mask.shape[:-3] + (t.shape[-4],) + mask.shape[-2:]
            )

        @jax.checkpoint
        def body_fn(carry, params):
            t, key = carry
            block = eqx.combine(params, self.static)
            t = block(t, mask=mask, key=key, deterministic=deterministic)
            return (t, jax.random.fold_in(key, 1)), None

        (t, _), _ = jax.lax.scan(body_fn, (t, key), self.stacked_params)
        t = self.layer_norm(t)
        return t


class TemplateEmbedderAllAtom(eqx.Module):
    """Template embedder. Implements AF3 Algorithm 16."""

    template_pair_embedder: TemplatePairEmbedderAllAtom
    template_pair_stack: TemplatePairStack
    linear_t: Linear

    @classmethod
    def from_torch(cls, model) -> "TemplateEmbedderAllAtom":
        return cls(
            template_pair_embedder=from_torch(model.template_pair_embedder),
            template_pair_stack=from_torch(model.template_pair_stack),
            linear_t=from_torch(model.linear_t),
        )

    def __call__(self, batch, z, pair_mask=None, *, key, deterministic=True, **kwargs):
        template_embeds = self.template_pair_embedder(batch, z)
        n_templ = template_embeds.shape[-4]

        if pair_mask is not None:
            pair_mask = pair_mask[..., None, :, :].astype(z.dtype)

        t = self.template_pair_stack(
            template_embeds, pair_mask, key=key, deterministic=deterministic)

        # Average over templates
        t = jnp.sum(t, axis=-4) / n_templ
        t = jax.nn.relu(t)
        t = self.linear_t(t)
        return t


from_torch.register(pt_tmpl.TemplatePairBlock, TemplatePairBlock.from_torch)
from_torch.register(pt_tmpl.TemplatePairStack, TemplatePairStack.from_torch)
from_torch.register(pt_tmpl.TemplateEmbedderAllAtom, TemplateEmbedderAllAtom.from_torch)
