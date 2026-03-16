"""Sequential container translation."""

from __future__ import annotations

import equinox as eqx
import torch.nn as nn

from jopenfold3.backend import from_torch


class Sequential(eqx.Module):
    """Sequential container matching PyTorch nn.Sequential."""

    _modules: dict[str, eqx.Module]

    def __call__(self, x, **kwargs):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x


def _sequential_from_torch(model: nn.Sequential) -> Sequential:
    modules = {}
    for name, child in model.named_children():
        modules[name] = from_torch(child)
    return Sequential(_modules=modules)


from_torch.register(nn.Sequential, _sequential_from_torch)
