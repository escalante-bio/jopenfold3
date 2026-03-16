"""Confidence metrics for OpenFold3 JAX model outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def compute_plddt(logits: Float[Array, "... 50"]) -> np.ndarray:
    """Compute per-atom pLDDT from logits (50 bins, range 0–1).

    Args:
        logits: Raw pLDDT logits from the model, last dim = 50 bins.

    Returns:
        pLDDT scores in [0, 100] as a numpy array.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    width = 1.0 / 50
    bin_centers = jnp.linspace(width / 2, 1.0 - width / 2, 50)
    return np.array(jnp.sum(probs * bin_centers, axis=-1)) * 100.0
