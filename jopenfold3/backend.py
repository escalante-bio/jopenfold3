"""Backend utilities for PyTorch-to-Equinox conversion.

Provides:
- `from_torch`: singledispatch converter from PyTorch modules/tensors to Equinox/JAX
- `AbstractFromTorch`: base class for Equinox modules with automatic from_torch
- `TestModule`: wrapper for side-by-side numerical comparison
"""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

# ---------------------------------------------------------------------------
# singledispatch converter
# ---------------------------------------------------------------------------

@functools.singledispatch
def from_torch(obj):
    """Convert a PyTorch object to its JAX/Equinox equivalent.

    Dispatches on type. Register new converters with:
        from_torch.register(TorchClass, converter_fn)
    """
    raise TypeError(f"No from_torch converter registered for {type(obj)}")


@from_torch.register(torch.Tensor)
def _tensor(t: torch.Tensor):
    return jnp.array(t.detach().cpu().numpy())


@from_torch.register(np.ndarray)
def _ndarray(a: np.ndarray):
    return jnp.array(a)


@from_torch.register(int)
def _int(x: int):
    return x


@from_torch.register(float)
def _float(x: float):
    return x


@from_torch.register(bool)
def _bool(x: bool):
    return x


@from_torch.register(type(None))
def _none(x):
    return None


@from_torch.register(tuple)
def _tuple(t: tuple):
    return tuple(from_torch(v) for v in t)


@from_torch.register(list)
def _list(lst: list):
    return [from_torch(v) for v in lst]


@from_torch.register(dict)
def _dict(d: dict):
    return {k: from_torch(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# AbstractFromTorch base class
# ---------------------------------------------------------------------------

class AbstractFromTorch(eqx.Module):
    """Base class for Equinox modules that can be auto-converted from PyTorch.

    Subclasses declare fields whose names match the PyTorch module's
    named_children() and named_parameters(recurse=False). The classmethod
    `from_torch` iterates over those and recursively converts each one.
    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module) -> "AbstractFromTorch":
        """Auto-convert a PyTorch module by matching field names."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}

        # Convert named children (submodules)
        for name, child in model.named_children():
            if name not in field_names:
                raise ValueError(
                    f"{cls.__name__}.from_torch: PyTorch module has child '{name}' "
                    f"but no matching field. Fields: {field_names}"
                )
            kwargs[name] = from_torch(child)

        # Convert direct parameters (not in submodules)
        for name, param in model.named_parameters(recurse=False):
            if name not in field_names:
                raise ValueError(
                    f"{cls.__name__}.from_torch: PyTorch module has parameter '{name}' "
                    f"but no matching field. Fields: {field_names}"
                )
            kwargs[name] = from_torch(param)

        # Convert direct buffers (not in submodules)
        for name, buf in model.named_buffers(recurse=False):
            if name not in field_names:
                # Buffers like version_tensor may not need translation
                continue
            kwargs[name] = from_torch(buf)

        # Set missing optional fields (ones that accept None) to None
        for f in dataclasses.fields(cls):
            if f.name not in kwargs:
                # Check if the field has a default or can accept None
                if f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    kwargs[f.name] = f.default_factory()
                # Otherwise leave it out — will error if truly required

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# TestModule: side-by-side comparison
# ---------------------------------------------------------------------------

class TestModule:
    """Wraps a PyTorch module for side-by-side comparison with its JAX translation."""

    def __init__(self, pt_module: torch.nn.Module):
        self.pt_module = pt_module
        self.jax_module = from_torch(pt_module)

    def __call__(self, *pt_args, **pt_kwargs):
        """Run both PyTorch and JAX, print max abs error and timing."""
        # PyTorch forward
        self.pt_module.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            pt_out = self.pt_module(*pt_args, **pt_kwargs)
            pt_time = time.perf_counter() - t0

        # Convert inputs to JAX
        jax_args = from_torch(tuple(
            a.detach().cpu() if isinstance(a, torch.Tensor) else a
            for a in pt_args
        ))
        jax_kwargs = {
            k: from_torch(v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in pt_kwargs.items()
        }

        # JAX forward
        with jax.default_matmul_precision("float32"):
            t0 = time.perf_counter()
            jax_out = self.jax_module(*jax_args, **jax_kwargs)
            jax_time = time.perf_counter() - t0

        # Compare
        pt_np = pt_out.detach().cpu().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
        jax_np = np.array(jax_out) if not isinstance(jax_out, np.ndarray) else jax_out

        max_err = np.abs(pt_np - jax_np).max()
        mean_err = np.abs(pt_np - jax_np).mean()
        print(f"max abs error: {max_err:.2e}  mean abs error: {mean_err:.2e}")
        print(f"PyTorch: {pt_time*1000:.1f}ms  JAX: {jax_time*1000:.1f}ms")

        return jax_out
