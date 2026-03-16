"""Dropout layer translations."""

from __future__ import annotations

import equinox as eqx
import jax
import jopenfold3._vendor.openfold3.core.model.primitives.dropout as pt_dropout
from jaxtyping import Array

from jopenfold3.backend import from_torch


class Dropout(eqx.Module):
    """Dropout with shared mask along batch_dim(s).

    In inference mode (deterministic=True), this is identity.
    """

    r: float
    batch_dim: list[int]

    def __call__(
        self, x: Array, *, key: jax.Array, deterministic: bool = True
    ) -> Array:
        if deterministic or self.r == 0.0:
            return x
        shape = list(x.shape)
        for bd in self.batch_dim:
            shape[bd] = 1
        mask = jax.random.bernoulli(key, p=1 - self.r, shape=shape)
        return x * mask / (1 - self.r)


class DropoutRowwise(Dropout):
    pass


class DropoutColumnwise(Dropout):
    pass


def _dropout_from_torch(model) -> Dropout:
    batch_dim = model.batch_dim if isinstance(model.batch_dim, list) else [model.batch_dim]
    return Dropout(r=model.r, batch_dim=batch_dim)


def _dropout_rowwise_from_torch(model) -> DropoutRowwise:
    batch_dim = model.batch_dim if isinstance(model.batch_dim, list) else [model.batch_dim]
    return DropoutRowwise(r=model.r, batch_dim=batch_dim)


from_torch.register(pt_dropout.Dropout, _dropout_from_torch)
from_torch.register(pt_dropout.DropoutRowwise, _dropout_rowwise_from_torch)
from_torch.register(pt_dropout.DropoutColumnwise,
    lambda m: DropoutColumnwise(
        r=m.r,
        batch_dim=m.batch_dim if isinstance(m.batch_dim, list) else [m.batch_dim],
    )
)
