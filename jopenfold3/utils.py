"""JAX equivalents of PyTorch utility functions used by the OpenFold3 model."""

from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

# ---------------------------------------------------------------------------
# Simple tensor manipulation helpers
# ---------------------------------------------------------------------------


def permute_final_dims(tensor: Array, inds: tuple[int, ...] | list[int]) -> Array:
    """Permute only the last ``len(inds)`` dimensions of *tensor*.

    ``inds`` are zero-indexed relative to the first permuted dimension.
    E.g. ``permute_final_dims(t, (2, 0, 1))`` on a 5-D tensor permutes
    axes ``(2, 3, 4)`` to ``(4, 2, 3)``.
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return jnp.transpose(tensor, first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: Array, no_dims: int) -> Array:
    """Flatten the last *no_dims* dimensions into one."""
    return t.reshape(t.shape[:-no_dims] + (-1,))


# ---------------------------------------------------------------------------
# Binned one-hot encoding
# ---------------------------------------------------------------------------


def binned_one_hot(x: Array, v_bins: Array) -> Array:
    """Bin values in *x* to the closest entry in *v_bins* and return one-hot.

    Args:
        x: Arbitrary-shape array of values.
        v_bins: 1-D array of bin centres.

    Returns:
        One-hot array with an extra trailing dimension of size ``len(v_bins)``.
    """
    n_bins = v_bins.shape[0]
    reshaped_bins = v_bins.reshape(((1,) * len(x.shape)) + (n_bins,))
    diffs = x[..., None] - reshaped_bins
    am = jnp.argmin(jnp.abs(diffs), axis=-1)
    return jax.nn.one_hot(am, num_classes=n_bins)


# ---------------------------------------------------------------------------
# Atom-attention block utilities (JAX port of atom_attention_block_utils)
# ---------------------------------------------------------------------------


def _get_query_block_padding(n_atom: int, n_query: int) -> int:
    """Padding so that *n_atom* is evenly divisible by *n_query*."""
    return (-n_atom) % n_query


def _get_block_indices(
    atom_mask: Array,
    n_query: int,
    n_key: int,
) -> tuple[Array, Array]:
    """Calculate key-block gather indices and an invalid-position mask.

    For each query block the function computes a window of *n_key* indices
    centred on the block midpoint, then shifts the window so that it does not
    underflow (index < 0) or overflow (index >= n_atom).

    Args:
        atom_mask: ``[*, N_atom]`` boolean/float mask.
        n_query: Block height (number of queries per block).
        n_key: Block width (number of keys per block).

    Returns:
        safe_indices: ``[*, N_blocks, N_key]`` clamped gather indices.
        invalid_mask: ``[*, N_blocks, N_key]`` True where index is out of range.
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom_padded = atom_mask.shape[-1]
    offset = n_query // 2
    num_blocks = math.ceil(n_atom_padded / n_query)

    # Block centres: [N_blocks]
    subset_centers = offset + jnp.arange(num_blocks) * n_query
    # Broadcast to [*, N_blocks]
    subset_centers = subset_centers.reshape(*(1,) * len(batch_dims), num_blocks)
    subset_centers = jnp.broadcast_to(subset_centers, (*batch_dims, num_blocks))

    # True atom count per sample: [*, 1] -> [*, N_blocks]
    n_atom = jnp.sum(atom_mask, axis=-1, keepdims=True).astype(jnp.int32)
    n_atom = jnp.broadcast_to(n_atom, (*batch_dims, num_blocks))

    # Initial gather window: [*, N_blocks, N_key]
    key_offsets = jnp.arange(-n_key // 2, n_key // 2)  # [N_key]
    initial_gathers = subset_centers[..., None] + key_offsets[None, :]
    initial_gathers = initial_gathers.astype(jnp.int32)

    # Correct underflow / overflow
    underflow = jax.nn.relu(-initial_gathers[..., 0])
    overflow = jax.nn.relu(initial_gathers[..., -1] - (n_atom - 1))
    total_shift = jnp.where(underflow > 0, underflow, -overflow)

    final_gathers = initial_gathers + total_shift[..., None]

    # Invalid mask
    n_atom_exp = n_atom[..., None]
    invalid_mask = (final_gathers < 0) | (final_gathers >= n_atom_exp)

    # Clamp to valid range
    safe_indices = jnp.clip(final_gathers, 0, jnp.maximum(n_atom_exp - 1, 0))

    return safe_indices, invalid_mask


def _get_pair_atom_block_mask(
    atom_mask: Array,
    num_blocks: int,
    n_query: int,
    n_key: int,
    pad_len_right_q: int,
    key_block_idxs: Array,
    invalid_mask: Array,
) -> Array:
    """Build the ``[*, N_blocks, N_query, N_key]`` pair mask from an atom mask.

    Internal helper shared by :func:`convert_single_rep_to_blocks` and
    :func:`convert_pair_rep_to_blocks`.
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    flat_batch_size = int(math.prod(batch_dims)) if batch_dims else 1

    # Query mask: [*, N_blocks, N_query]
    atom_mask_q = jnp.pad(
        atom_mask, [(0, 0)] * len(batch_dims) + [(0, pad_len_right_q)]
    )
    atom_mask_q = atom_mask_q.reshape((*batch_dims, num_blocks, n_query))

    # Key mask via gather
    atom_mask_flat = atom_mask.reshape(flat_batch_size, n_atom)
    key_block_idxs_flat = key_block_idxs.reshape(
        flat_batch_size, num_blocks * n_key
    ).astype(jnp.int32)
    invalid_mask_flat = invalid_mask.reshape(flat_batch_size, num_blocks * n_key)

    # Batched gather along axis 1
    batch_idx = jnp.arange(flat_batch_size)[:, None]
    atom_mask_k_flat = atom_mask_flat[batch_idx, key_block_idxs_flat]
    atom_mask_k_flat = jnp.where(invalid_mask_flat, 0.0, atom_mask_k_flat)

    atom_mask_k = atom_mask_k_flat.reshape((*batch_dims, num_blocks, n_key))

    # [*, N_blocks, N_query, N_key]
    atom_pair_mask = atom_mask_q[..., None] * atom_mask_k[..., None, :]
    return atom_pair_mask


def convert_single_rep_to_blocks(
    ql: Array,
    n_query: int,
    n_key: int,
    atom_mask: Array,
) -> tuple[Array, Array, Array]:
    """Convert single atom representation to query/key blocks for attention.

    Args:
        ql: ``[*, N_atom, c_atom]`` atom single representation.
        n_query: Block height (queries per block).
        n_key: Block width (keys per block).
        atom_mask: ``[*, N_atom]`` atom-level mask.

    Returns:
        ql_query: ``[*, N_blocks, N_query, c_atom]``
        ql_key:   ``[*, N_blocks, N_key, c_atom]``
        atom_pair_mask: ``[*, N_blocks, N_query, N_key]``
    """
    batch_dims = ql.shape[:-2]
    n_atom, n_dim = ql.shape[-2:]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q = _get_query_block_padding(n_atom, n_query)

    # Pad and reshape into query blocks
    ql_query = jnp.pad(
        ql, [(0, 0)] * len(batch_dims) + [(0, pad_len_right_q), (0, 0)]
    )
    ql_query = ql_query.reshape((*batch_dims, num_blocks, n_query, n_dim))

    # Ensure atom_mask is broadcast to batch_dims
    atom_mask = jnp.broadcast_to(atom_mask, (*batch_dims, n_atom))

    # Key gather indices
    key_block_idxs, invalid_mask = _get_block_indices(
        atom_mask=atom_mask, n_query=n_query, n_key=n_key
    )

    # Flatten for gather
    flat_batch_size = int(math.prod(batch_dims)) if batch_dims else 1
    ql_flat = ql.reshape(flat_batch_size, n_atom, n_dim)

    index_flat = key_block_idxs.reshape(
        flat_batch_size, num_blocks * n_key
    ).astype(jnp.int32)
    mask_flat = invalid_mask.reshape(flat_batch_size, num_blocks * n_key)

    # Batched gather
    batch_idx = jnp.arange(flat_batch_size)[:, None]
    ql_key_flat = ql_flat[batch_idx, index_flat]  # [flat, num_blocks*n_key, n_dim]
    ql_key_flat = jnp.where(mask_flat[..., None], 0.0, ql_key_flat)

    ql_key = ql_key_flat.reshape((*batch_dims, num_blocks, n_key, n_dim))

    atom_pair_mask = _get_pair_atom_block_mask(
        atom_mask=atom_mask,
        num_blocks=num_blocks,
        n_query=n_query,
        n_key=n_key,
        pad_len_right_q=pad_len_right_q,
        key_block_idxs=key_block_idxs,
        invalid_mask=invalid_mask,
    )

    return ql_query, ql_key, atom_pair_mask


# ---------------------------------------------------------------------------
# Pair representation -> blocks (for sequence-local atom attention)
# ---------------------------------------------------------------------------


def convert_pair_rep_to_blocks(
    batch: Batch,
    zij_trunk: Array,
    n_query: int,
    n_key: int,
) -> Array:
    """Convert pair atom representation to blocks for attention.

    Args:
        batch: Feature dictionary (must contain ``atom_to_token_index``,
            ``atom_mask``).
        zij_trunk: ``[*, N_token, N_token, c_atom_pair]`` pair trunk embedding.
        n_query: Block height (queries per block).
        n_key: Block width (keys per block).

    Returns:
        plm: ``[*, N_blocks, N_query, N_key, c_atom_pair]`` blocked pair
            conditioning.
    """
    atom_to_token_index = batch.atom_to_token_index
    atom_mask = batch.atom_mask

    batch_dims = zij_trunk.shape[:-3]
    n_atom = atom_to_token_index.shape[-1]
    flat_batch_size = int(math.prod(batch_dims)) if batch_dims else 1

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q = _get_query_block_padding(n_atom, n_query)

    # --- Q token indices ---
    atom_to_token_index_q = jnp.pad(
        atom_to_token_index,
        [(0, 0)] * (len(atom_to_token_index.shape) - 1) + [(0, pad_len_right_q)],
    )
    # [flat_batch_size, N_blocks, N_query]
    q_indices = atom_to_token_index_q.reshape(
        flat_batch_size, num_blocks, n_query
    ).astype(jnp.int32)

    # Ensure atom_mask is broadcast to batch_dims
    atom_mask = jnp.broadcast_to(atom_mask, (*batch_dims, n_atom))

    # --- K token indices ---
    key_block_idxs, invalid_mask = _get_block_indices(
        atom_mask=atom_mask, n_query=n_query, n_key=n_key
    )

    atom_to_token_index_flat = atom_to_token_index.reshape(flat_batch_size, n_atom)
    key_block_idxs_flat = key_block_idxs.reshape(
        flat_batch_size, num_blocks * n_key
    ).astype(jnp.int32)

    # Batched gather of token indices for keys
    batch_idx_1d = jnp.arange(flat_batch_size)[:, None]
    k_indices_flat = atom_to_token_index_flat[batch_idx_1d, key_block_idxs_flat]

    # [flat_batch_size, N_blocks, N_key]
    k_indices = k_indices_flat.reshape(
        flat_batch_size, num_blocks, n_key
    ).astype(jnp.int32)

    # [flat_batch_size, N_blocks, N_key]
    invalid_mask_reshaped = invalid_mask.reshape(flat_batch_size, num_blocks, n_key)

    # --- Index into zij_trunk ---
    zij_trunk = zij_trunk.reshape(flat_batch_size, *zij_trunk.shape[-3:])

    # batch_index: [flat_batch_size, 1, 1, 1]
    batch_index = jnp.arange(
        flat_batch_size, dtype=jnp.int32
    ).reshape(-1, 1, 1, 1)

    # Advanced indexing: [flat_batch_size, N_blocks, N_query, N_key, C]
    plm = zij_trunk[
        batch_index,
        q_indices[:, :, :, None],   # [fb, N_blocks, N_query, 1]
        k_indices[:, :, None, :],   # [fb, N_blocks, 1, N_key]
    ]

    # Mask out invalid key positions
    plm = jnp.where(
        invalid_mask_reshaped[:, :, None, :, None],  # [fb, N_blocks, 1, N_key, 1]
        0.0,
        plm,
    )

    # --- Pair atom mask ---
    atom_pair_mask = _get_pair_atom_block_mask(
        atom_mask=atom_mask,
        num_blocks=num_blocks,
        n_query=n_query,
        n_key=n_key,
        pad_len_right_q=pad_len_right_q,
        key_block_idxs=key_block_idxs.reshape(*batch_dims, num_blocks, n_key),
        invalid_mask=invalid_mask.reshape(*batch_dims, num_blocks, n_key),
    )

    # Reshape back and mask padding
    plm = plm.reshape((*batch_dims, num_blocks, n_query, n_key, plm.shape[-1]))
    plm = plm * atom_pair_mask[..., None]

    return plm


# ---------------------------------------------------------------------------
# Token <-> atom broadcasting / aggregation
# ---------------------------------------------------------------------------


def broadcast_token_feat_to_atoms(
    token_mask: Array,
    num_atoms_per_token: Array,
    token_feat: Array,
    token_dim: int = -2,
    atom_to_token_index: Array | None = None,
    atom_mask: Array | None = None,
    **kwargs,
) -> Array:
    """Broadcast token-level features to atom-level features via gather.

    Simply indexes ``token_feat[atom_to_token_index]`` — no repeat_interleave,
    no data-dependent shapes. Fully JIT-compatible.

    Args:
        token_mask: ``[*, N_token]`` token mask.
        num_atoms_per_token: ``[*, N_token]`` (unused, kept for API compat).
        token_feat: ``[*, N_token, ...]`` token-level feature.
        token_dim: Which axis is the token dimension (default ``-2``).
        atom_to_token_index: ``[*, N_atom]`` mapping from each atom to its
            token index. Required.
        atom_mask: ``[*, N_atom]`` atom mask. Required.

    Returns:
        atom_feat: ``[*, N_atom, ...]`` broadcasted atom-level feature.
    """
    if atom_to_token_index is None or atom_mask is None:
        raise ValueError(
            "atom_to_token_index and atom_mask are required. "
            "Pass batch['atom_to_token_index'] and batch['atom_mask']."
        )

    batch_dims = token_mask.shape[:-1]
    n_token = token_mask.shape[-1]
    n_atom = atom_to_token_index.shape[-1]

    # Resolve token_dim
    ndim_feat = len(token_feat.shape)
    td = token_dim if token_dim >= 0 else ndim_feat + token_dim
    feat_dims = token_feat.shape[td + 1:]

    # Mask token features
    mask_shape = (*batch_dims, n_token) + ((1,) * len(feat_dims))
    token_feat = token_feat * token_mask.reshape(mask_shape)

    # Clamp invalid atom indices to 0 (zeroed out by atom_mask below)
    safe_index = jnp.where(
        atom_mask.astype(jnp.bool_), atom_to_token_index, 0
    ).astype(jnp.int32)

    # Gather: token_feat[safe_index]
    fb_size = int(math.prod(batch_dims)) if batch_dims else 1
    flat_feat = token_feat.reshape(fb_size, n_token, *feat_dims)
    flat_index = safe_index.reshape(fb_size, n_atom)

    batch_idx = jnp.arange(fb_size)[:, None]
    atom_feat = flat_feat[batch_idx, flat_index]

    # Zero out masked atoms
    atom_feat = atom_feat * atom_mask.reshape(fb_size, n_atom)[..., None] if feat_dims else atom_feat * atom_mask.reshape(fb_size, n_atom)

    return atom_feat.reshape(*batch_dims, n_atom, *feat_dims)


def aggregate_atom_feat_to_tokens(
    token_mask: Array,
    atom_to_token_index: Array,
    atom_mask: Array,
    atom_feat: Array,
    atom_dim: int = -1,
    aggregate_fn: Literal["mean", "sum"] = "mean",
    eps: float = 1e-9,
) -> Array:
    """Aggregate atom-level features to token-level features.

    Uses a scatter-add approach via ``jnp.zeros(...).at[...].add()`` to
    accumulate atom features into their respective token bins.

    Args:
        token_mask: ``[*, N_token]`` token mask.
        atom_to_token_index: ``[*, N_atom]`` mapping from each atom to its
            token index.
        atom_mask: ``[*, N_atom]`` atom mask.
        atom_feat: ``[*, N_atom, *feat_dims]`` atom-level features.
        atom_dim: Which axis is the atom dimension (default ``-1``).
        aggregate_fn: ``"mean"`` or ``"sum"``.
        eps: Small constant for numerical stability in mean mode.

    Returns:
        token_feat: ``[*, N_token, *feat_dims]`` aggregated token features.
    """
    if aggregate_fn not in ("mean", "sum"):
        raise ValueError(f"Invalid aggregation function: {aggregate_fn}")

    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]

    # Resolve atom_dim to a positive index
    ndim = len(atom_feat.shape)
    ad = atom_dim if atom_dim >= 0 else ndim + atom_dim
    feat_dims = atom_feat.shape[ad + 1:]
    n_atom = atom_feat.shape[ad]

    # Mask atom features
    mask_shape = atom_mask.shape + (1,) * len(feat_dims)
    atom_feat = atom_feat * atom_mask.reshape(mask_shape)

    # Map invalid atoms to an overflow bin (index = n_token) that we discard
    atom_to_token_index = (
        atom_to_token_index * atom_mask.astype(jnp.int32)
        + n_token * (1 - atom_mask.astype(jnp.int32))
    )

    # Flatten batch dims for vmap scatter
    fb_size = int(math.prod(batch_dims)) if batch_dims else 1
    idx_1d = atom_to_token_index.reshape(fb_size, n_atom).astype(jnp.int32)

    if feat_dims:
        flat_feat_size = int(math.prod(feat_dims))
        atom_feat_flat = atom_feat.reshape(fb_size, n_atom, flat_feat_size)

        def _scatter_row(feat_row, idx_row):
            return jnp.zeros(
                (n_token + 1, flat_feat_size), dtype=feat_row.dtype
            ).at[idx_row].add(feat_row)

        out = jax.vmap(_scatter_row)(atom_feat_flat, idx_1d)
        token_feat = out[:, :n_token, :]
        token_feat = token_feat.reshape((*batch_dims, n_token, *feat_dims))
    else:
        atom_feat_flat = atom_feat.reshape(fb_size, n_atom)

        def _scatter_row_1d(feat_row, idx_row):
            return jnp.zeros(
                (n_token + 1,), dtype=feat_row.dtype
            ).at[idx_row].add(feat_row)

        out = jax.vmap(_scatter_row_1d)(atom_feat_flat, idx_1d)
        token_feat = out[:, :n_token]
        token_feat = token_feat.reshape((*batch_dims, n_token))

    # Mean aggregation: divide by per-token atom count
    if aggregate_fn == "mean":
        atom_mask_flat = atom_mask.reshape(fb_size, n_atom).astype(atom_feat.dtype)

        def _count_row(mask_row, idx_row):
            return jnp.zeros(
                (n_token + 1,), dtype=mask_row.dtype
            ).at[idx_row].add(mask_row)

        counts = jax.vmap(_count_row)(atom_mask_flat, idx_1d)
        counts = counts[:, :n_token]
        counts = counts.reshape((*batch_dims, n_token))

        counts_shape = counts.shape + (1,) * len(feat_dims)
        token_feat = token_feat / (counts.reshape(counts_shape) + eps)

    return token_feat


# ---------------------------------------------------------------------------
# Max-atom-per-token masked select
# ---------------------------------------------------------------------------


def max_atom_per_token_masked_select(
    atom_feat: Array,
    max_atom_per_token_mask: Array,
    n_atom: int | None = None,
) -> Array:
    """Select atoms from features padded to max atoms per token.

    Uses a cumsum-based gather to compact valid atoms into a dense
    ``[*, N_atom, c_out]`` array. Fully JIT-compatible.

    Args:
        atom_feat: ``[*, N_token * max_atoms_per_token, c_out]`` atom features
            padded to max atoms per token.
        max_atom_per_token_mask: ``[*, N_token * max_atoms_per_token]`` mask
            denoting valid atoms.
        n_atom: Output atom dimension size. Required (use
            ``batch["atom_mask"].shape[-1]``).

    Returns:
        atom_feat: ``[*, N_atom, c_out]`` selected valid atom features.
    """
    if n_atom is None:
        raise ValueError("n_atom is required (use batch['atom_mask'].shape[-1]).")

    batch_dims = atom_feat.shape[:-2]
    c_out = atom_feat.shape[-1]
    n_padded = atom_feat.shape[-2]
    fb_size = int(math.prod(batch_dims)) if batch_dims else 1

    feat_flat = atom_feat.reshape(fb_size, n_padded, c_out)
    mask_flat = max_atom_per_token_mask.reshape(fb_size, n_padded).astype(jnp.bool_)

    def _select_row(feat_row, mask_row):
        # Compute destination indices via exclusive cumsum of mask
        dest = jnp.cumsum(mask_row.astype(jnp.int32), axis=-1) - 1  # [n_padded]
        # Scatter valid entries into output
        out = jnp.zeros((n_atom, c_out), dtype=feat_row.dtype)
        out = out.at[dest].add(feat_row * mask_row[:, None])
        return out

    result = jax.vmap(_select_row)(feat_flat, mask_flat)
    return result.reshape(*batch_dims, n_atom, c_out)


# ---------------------------------------------------------------------------
# Relative position embedding
# ---------------------------------------------------------------------------


def relpos_complex(
    batch: Batch, max_relative_idx: int, max_relative_chain: int
) -> Array:
    """Compute relative position features for the pair representation.

    Implements Algorithm 3 from AF3: relative position encoding with
    residue-level, token-level, and chain-level components.

    Args:
        batch: Feature dictionary containing ``residue_index``, ``asym_id``,
            ``entity_id``, ``token_index``, ``sym_id``.
        max_relative_idx: Maximum relative residue/token index (clipped).
        max_relative_chain: Maximum relative chain index (clipped).

    Returns:
        rel_feat: ``[*, N_token, N_token, C_z]`` relative position features,
            where ``C_z = 2*(2*max_relative_idx+2) + 1 + (2*max_relative_chain+2)``.
    """
    res_idx = batch.residue_index
    asym_id = batch.asym_id
    entity_id = batch.entity_id
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(pos, condition, rel_clip_idx):
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = jnp.clip(offset + rel_clip_idx, 0, 2 * rel_clip_idx)
        final_offset = jnp.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * jnp.ones_like(clipped_offset),
        )
        boundaries = jnp.arange(0, 2 * rel_clip_idx + 2, dtype=jnp.float32)
        return binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(res_idx, same_chain, max_relative_idx)
    rel_token = _relpos(
        batch.token_index,
        same_chain & same_res,
        max_relative_idx,
    )
    rel_chain = _relpos(batch.sym_id, same_entity, max_relative_chain)

    same_entity_feat = same_entity[..., None].astype(rel_pos.dtype)

    rel_feat = jnp.concatenate(
        [rel_pos, rel_token, same_entity_feat, rel_chain], axis=-1
    )

    return rel_feat


# ---------------------------------------------------------------------------
# Token representative atom extraction
# ---------------------------------------------------------------------------


def _get_token_atom_index_offset(atom_name: str, restype: Array) -> tuple[Array, Array]:
    """Get index of a given atom (within its residue) in each residue type.

    Uses the PyTorch data constants.

    Args:
        atom_name: Atom name to get indices for.
        restype: ``[*, N_token, 32]`` one-hot residue types.

    Returns:
        token_atom_index_offset: ``[*, N_token]`` atom indices.
        token_atom_mask: ``[*, N_token]`` mask for valid atoms.
    """
    from jopenfold3._vendor.openfold3.core.data.resources.token_atom_constants import (
        atom_name_to_index_by_restype,
    )

    index_arr = jnp.array(
        atom_name_to_index_by_restype[atom_name]["index"], dtype=jnp.float32
    )
    mask_arr = jnp.array(
        atom_name_to_index_by_restype[atom_name]["mask"], dtype=jnp.float32
    )
    token_atom_index_offset = jnp.einsum(
        "...k,k->...", restype.astype(jnp.float32), index_arr
    ).astype(jnp.int32)
    token_atom_mask = jnp.einsum(
        "...k,k->...", restype.astype(jnp.float32), mask_arr
    ).astype(jnp.int32)
    return token_atom_index_offset, token_atom_mask


def get_token_representative_atoms(
    batch: Batch,
    x: Array,
    atom_mask: Array,
) -> tuple[Array, Array]:
    """Extract representative atoms per token.

    Uses precomputed ``batch.representative_atom_index`` and
    ``batch.representative_atom_mask`` (computed at Batch construction
    from one-hot restype, before any soft sequence injection).

    Args:
        batch: Batch with precomputed representative atom indices.
        x: ``[*, N_atom, 3]`` atom positions.
        atom_mask: ``[*, N_atom]`` atom mask.

    Returns:
        rep_x: ``[*, N_token, 3]`` representative atom positions.
        rep_atom_mask: ``[*, N_token]`` representative atom mask.
    """
    rep_index = batch.representative_atom_index
    n_atom = x.shape[-2]
    n_token = rep_index.shape[-1]

    batch_shape = x.shape[:-2]
    fb_size = int(math.prod(batch_shape)) if batch_shape else 1

    safe_index = jnp.clip(rep_index, 0, n_atom - 1).astype(jnp.int32)

    x_flat = x.reshape(fb_size, n_atom, 3)
    idx_flat = safe_index.reshape(fb_size, n_token)
    mask_flat = atom_mask.reshape(fb_size, n_atom)

    batch_idx = jnp.arange(fb_size)[:, None]
    rep_x = x_flat[batch_idx, idx_flat]
    rep_atom_mask = mask_flat[batch_idx, idx_flat]

    rep_x = rep_x.reshape(*batch_shape, n_token, 3)
    rep_atom_mask = rep_atom_mask.reshape(*batch_shape, n_token)

    rep_atom_mask = rep_atom_mask * batch.token_mask * batch.representative_atom_mask

    return rep_x, rep_atom_mask
