"""
GPU-accelerated operations for sequential generation.

This module provides torch-based implementations of operations that were
previously done in pandas on CPU, enabling GPU acceleration.
"""

import torch

from mostlyai.engine._common import (
    RIDX_SUB_COLUMN_PREFIX,
    SIDX_RIDX_DIGIT_ENCODING_THRESHOLD,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
)


def decode_positional_column_torch(
    encoded_tensor: torch.Tensor,
    max_seq_len: int,
    prefix: str = "",
    sub_column_indices: dict[str, int] | None = None,
) -> torch.Tensor:
    """
    GPU-accelerated version of decode_positional_column.

    Decodes positional columns (SIDX, SLEN, RIDX) from their encoded representation
    back to integer values. Operates on tensors instead of DataFrames.

    Args:
        encoded_tensor: Tensor of shape (batch, n_sub_cols) containing encoded values
        max_seq_len: Maximum sequence length (determines encoding type)
        prefix: Prefix for sub-columns (e.g., SIDX_SUB_COLUMN_PREFIX)
        sub_column_indices: Dict mapping sub-column names to tensor column indices.
                           Required for digit encoding. For cat encoding, pass None.

    Returns:
        Tensor of shape (batch,) with decoded integer values
    """
    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # Categorical encoding - single column
        # For single column, encoded_tensor should be 1D or we take the first column
        if encoded_tensor.dim() == 1:
            return encoded_tensor
        else:
            return encoded_tensor[:, 0]
    else:
        # Digit encoding - reconstruct from digit columns
        if sub_column_indices is None:
            raise ValueError("sub_column_indices required for digit encoding")

        n_digits = len(str(max_seq_len))
        result = torch.zeros(encoded_tensor.size(0), dtype=torch.long, device=encoded_tensor.device)

        for d in range(n_digits):
            col_name = f"{prefix}E{d}"
            if col_name in sub_column_indices:
                col_idx = sub_column_indices[col_name]
                digit_value = encoded_tensor[:, col_idx]
                result += digit_value * (10**d)

        return result


def compute_sequence_continue_mask_torch(
    sidx: torch.Tensor,
    slen: torch.Tensor,
    ridx: torch.Tensor | None,
    n_seed_steps: int,
) -> torch.Tensor:
    """
    GPU-accelerated version of _compute_sequence_continue_mask.

    Computes a boolean mask indicating which sequences should continue to the next step.

    Args:
        sidx: Tensor of current sequence indices (batch,)
        slen: Tensor of sequence lengths (batch,)
        ridx: Tensor of remaining steps (batch,), or None if not using RIDX
        n_seed_steps: Number of seed steps (always included in mask)

    Returns:
        Boolean tensor of shape (batch,) where True means sequence continues
    """
    device = sidx.device

    if ridx is not None:
        # RIDX > 0 means more steps remaining
        include_mask = ridx > 0
    else:
        # Fallback: SIDX < SLEN means we haven't reached the end
        include_mask = sidx < slen

    # Always include seeded steps
    seed_mask = sidx < n_seed_steps
    include_mask = include_mask | seed_mask

    return include_mask


def filter_tensors_by_mask(
    mask: torch.Tensor,
    *tensors: torch.Tensor | None,
) -> list[torch.Tensor | None]:
    """
    Filter multiple tensors using a boolean mask.

    Args:
        mask: Boolean tensor of shape (batch,)
        *tensors: Variable number of tensors to filter. None values are preserved.

    Returns:
        List of filtered tensors (or None for None inputs)
    """
    result = []
    for tensor in tensors:
        if tensor is None:
            result.append(None)
        else:
            # Handle tensors of different shapes
            if tensor.dim() == 1:
                result.append(tensor[mask])
            else:
                # Multi-dimensional: filter along first dimension
                result.append(tensor[mask.cpu().numpy()])  # Indexing needs CPU boolean array
    return result


def decode_and_filter_step(
    out_tensor: torch.Tensor,
    sub_column_names: list[str],
    max_seq_len: int,
    n_seed_steps: int,
    has_slen: bool,
    has_ridx: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Decode positional columns and compute filtering mask for a generation step.

    This combines the decoding and masking operations to minimize CPU-GPU transfers.

    Args:
        out_tensor: Output tensor from model, shape (batch, n_sub_cols)
        sub_column_names: List of sub-column names corresponding to tensor columns
        max_seq_len: Maximum sequence length
        n_seed_steps: Number of seed steps
        has_slen: Whether SLEN columns are present
        has_ridx: Whether RIDX columns are present

    Returns:
        Tuple of (filtered_tensor, include_mask, next_batch_size)
        - filtered_tensor: Tensor with continuing sequences only
        - include_mask: Boolean mask for filtering
        - next_batch_size: Number of continuing sequences
    """
    device = out_tensor.device

    # Build sub-column index mapping
    sub_col_idx = {name: idx for idx, name in enumerate(sub_column_names)}

    # Decode SIDX
    sidx_idx = None
    for name, idx in sub_col_idx.items():
        if name.startswith(SIDX_SUB_COLUMN_PREFIX):
            if sidx_idx is None or name.endswith("cat"):
                sidx_idx = idx
                break

    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        sidx = out_tensor[:, sidx_idx]
    else:
        sidx = decode_positional_column_torch(
            out_tensor,
            max_seq_len,
            SIDX_SUB_COLUMN_PREFIX,
            sub_col_idx,
        )

    # Decode SLEN if present
    slen = None
    if has_slen:
        if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
            slen_idx = None
            for name, idx in sub_col_idx.items():
                if name.startswith(SLEN_SUB_COLUMN_PREFIX):
                    if slen_idx is None or name.endswith("cat"):
                        slen_idx = idx
                        break
            if slen_idx is not None:
                slen = out_tensor[:, slen_idx]
        else:
            slen = decode_positional_column_torch(
                out_tensor,
                max_seq_len,
                SLEN_SUB_COLUMN_PREFIX,
                sub_col_idx,
            )

    # Decode RIDX if present
    ridx = None
    if has_ridx:
        if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
            ridx_idx = None
            for name, idx in sub_col_idx.items():
                if name.startswith(RIDX_SUB_COLUMN_PREFIX):
                    if ridx_idx is None or name.endswith("cat"):
                        ridx_idx = idx
                        break
            if ridx_idx is not None:
                ridx = out_tensor[:, ridx_idx]
        else:
            ridx = decode_positional_column_torch(
                out_tensor,
                max_seq_len,
                RIDX_SUB_COLUMN_PREFIX,
                sub_col_idx,
            )

    # Compute continue mask
    if slen is None:
        # Create dummy slen based on sidx (won't be used with ridx)
        slen = sidx + 1

    include_mask = compute_sequence_continue_mask_torch(sidx, slen, ridx, n_seed_steps)

    # Filter tensor
    next_batch_size = include_mask.sum().item()
    if next_batch_size < out_tensor.size(0):
        filtered_tensor = out_tensor[include_mask]
    else:
        filtered_tensor = out_tensor

    return filtered_tensor, include_mask, next_batch_size
