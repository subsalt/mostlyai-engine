"""
High-performance in-memory training functions.

This module provides training functions optimized for efficient GPU utilization:
1. Stats computed once upfront (single pass over data)
2. Full dataset encoded to CPU tensors once (no per-batch encoding overhead)
3. Batches transferred to GPU on-demand (VRAM-safe for large datasets)
4. Non-blocking transfers for overlapped computation

The encode-once-on-CPU, transfer-per-batch design allows training on datasets
larger than GPU memory while avoiding repeated encoding overhead.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mostlyai.engine._artifact import ModelArtifact, minimize_stats
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length
from mostlyai.engine._stats import compute_stats
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.training import ModelConfig
from mostlyai.engine.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)


def train_flat(
    tgt_data: pd.DataFrame,
    tgt_encoding_types: dict[str, ModelEncodingType | str],
    *,
    val_split: float = 0.1,
    model_size: str = "M",
    max_epochs: int = 100,
    max_training_time: float = 14400.0,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    enable_flexible_generation: bool = False,
    on_epoch: "OnEpochCallback | None" = None,
) -> ModelArtifact:
    """
    Train a flat (non-sequential) model entirely in memory.

    This function is optimized for the case where the full dataset fits in memory:
    - Stats computed once upfront
    - Full dataset encoded to GPU tensors once
    - Efficient batching from pre-loaded tensors

    Args:
        tgt_data: Target DataFrame for training
        tgt_encoding_types: Dict mapping column names to encoding types (must not be empty)
        val_split: Fraction of data to use for validation (default: 0.1)
        model_size: Model size ("S", "M", "L")
        max_epochs: Maximum training epochs
        max_training_time: Maximum training time in seconds
        batch_size: Batch size (auto-determined if None)
        device: Device for training (auto-detected if None)
        enable_flexible_generation: Enable flexible column order generation
        on_epoch: Callback invoked after each epoch with EpochInfo dict

    Returns:
        ModelArtifact containing trained weights and stats

    Raises:
        ValueError: If tgt_encoding_types is empty (no columns to train on)

    Example:
        >>> from mostlyai.engine import train_flat, generate_flat
        >>>
        >>> artifact = train_flat(
        ...     tgt_data=df,
        ...     tgt_encoding_types={"col1": "tabular_categorical", "col2": "tabular_numeric_auto"},
        ...     model_size="M",
        ...     max_epochs=50,
        ... )
        >>> synthetic = generate_flat(artifact, sample_size=1000)
    """
    from mostlyai.engine._tabular.training import train as train_internal, OnEpochCallback

    # Validate that we have columns to train on
    if not tgt_encoding_types:
        raise ValueError(
            "tgt_encoding_types cannot be empty. At least one column must be specified for training. "
            "If you have no columns to train on (e.g., only key columns), consider skipping this model."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    _LOG.info(f"Training flat model on {device} with {len(tgt_data)} samples")

    # Step 1: Compute stats (single pass over data)
    _LOG.info("Computing statistics...")
    tgt_stats, _ = compute_stats(
        tgt_data=tgt_data,
        tgt_encoding_types=tgt_encoding_types,
    )

    # Step 2: Split data into train/val
    n_samples = len(tgt_data)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_df = tgt_data.iloc[train_indices].reset_index(drop=True)
    val_df = tgt_data.iloc[val_indices].reset_index(drop=True)

    _LOG.info(f"Split: {n_train} training, {n_val} validation samples")

    # Step 3: Encode full datasets to tensors on CPU (single encoding pass)
    # Tensors are kept on CPU to avoid VRAM exhaustion; batches are moved to GPU on demand
    _LOG.info("Encoding to tensors...")
    train_tensors = _encode_df_to_tensors(train_df, tgt_stats, torch.device("cpu"))
    val_tensors = _encode_df_to_tensors(val_df, tgt_stats, torch.device("cpu"))

    # Step 4: Build model config
    tgt_cardinalities = get_cardinalities(tgt_stats)
    model_config: ModelConfig = {
        "tgt_cardinalities": tgt_cardinalities,
        "ctx_cardinalities": {},
        "is_sequential": False,
        "trn_cnt": n_train,
        "val_cnt": n_val,
        "tgt_seq_len_median": 1,
        "tgt_seq_len_max": 1,
        "ctx_seq_len_median": {},
        "empirical_probs": None,
    }

    # Step 5: Create efficient batch iterators
    if batch_size is None:
        batch_size = min(n_train, 2048)

    def make_batch_iter(
        tensor_dict: dict[str, torch.Tensor], batch_sz: int, target_device: torch.device, shuffle: bool = False
    ):
        """Create iterator over batches from pre-loaded CPU tensors.

        Tensors are stored on CPU and moved to GPU per-batch to avoid VRAM exhaustion.
        This allows training on datasets larger than GPU memory while still benefiting
        from single-pass encoding.
        """
        n = next(iter(tensor_dict.values())).shape[0]
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            batch_indices = indices[start:end]
            # Move only this batch to target device (GPU)
            yield {k: v[batch_indices].to(target_device, non_blocking=True) for k, v in tensor_dict.items()}

    # Step 6: Train using temporary workspace for model outputs
    with tempfile.TemporaryDirectory(prefix="train_flat_") as workspace_dir:
        workspace_path = Path(workspace_dir)

        # Write stats for training (required by train_internal)
        model_store = workspace_path / "ModelStore"
        tgt_stats_dir = model_store / "tgt-stats"
        tgt_stats_dir.mkdir(parents=True)
        with open(tgt_stats_dir / "stats.json", "w") as f:
            json.dump(tgt_stats, f)

        _LOG.info(f"Training for up to {max_epochs} epochs...")

        train_internal(
            train_tensors=make_batch_iter(train_tensors, batch_size, device, shuffle=True),
            val_tensors=make_batch_iter(val_tensors, batch_size, device, shuffle=False),
            model_config=model_config,
            workspace_dir=workspace_dir,
            model=f"MOSTLY_AI/{_model_size_name(model_size)}",
            max_training_time=max_training_time,
            max_epochs=max_epochs,
            batch_size=batch_size,
            enable_flexible_generation=enable_flexible_generation,
            device=device,
            on_epoch=on_epoch,
        )

        # Step 7: Load trained weights and create artifact
        weights_path = model_store / "model-data" / "model-weights.pt"
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        artifact = ModelArtifact.from_state_dict(
            state_dict=state_dict,
            is_sequential=False,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            tgt_stats=minimize_stats(tgt_stats),
            enable_flexible_generation=enable_flexible_generation,
        )

    _LOG.info(f"Training complete. Artifact size: {len(artifact.to_bytes()):,} bytes")
    return artifact


def train_sequential(
    tgt_data: pd.DataFrame,
    tgt_encoding_types: dict[str, ModelEncodingType | str],
    tgt_context_key: str,
    *,
    ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    ctx_encoding_types: dict[str, ModelEncodingType | str] | None = None,
    val_split: float = 0.1,
    model_size: str = "M",
    max_epochs: int = 100,
    max_training_time: float = 14400.0,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    enable_flexible_generation: bool = False,
    on_epoch: "OnEpochCallback | None" = None,
) -> ModelArtifact:
    """
    Train a sequential (longitudinal) model entirely in memory.

    This function trains models for data with multiple rows per context
    (e.g., transactions per user, events per session).

    Args:
        tgt_data: Target DataFrame containing sequences
        tgt_encoding_types: Dict mapping column names to encoding types (must not be empty)
        tgt_context_key: Column linking target rows to contexts
        ctx_data: Optional context DataFrame (one row per context)
        ctx_primary_key: Primary key column in context
        ctx_encoding_types: Dict mapping context column names to encoding types (can be None or empty)
        val_split: Fraction of contexts to use for validation
        model_size: Model size ("S", "M", "L")
        max_epochs: Maximum training epochs
        max_training_time: Maximum training time in seconds
        batch_size: Batch size (auto-determined if None)
        device: Device for training
        enable_flexible_generation: Enable flexible column order generation
        on_epoch: Callback invoked after each epoch with EpochInfo dict

    Returns:
        ModelArtifact containing trained weights and stats

    Raises:
        ValueError: If tgt_encoding_types is empty (no columns to train on)

    Example:
        >>> artifact = train_sequential(
        ...     tgt_data=transactions_df,
        ...     tgt_encoding_types={"amount": "tabular_numeric_auto"},
        ...     tgt_context_key="user_id",
        ...     ctx_data=users_df,
        ...     ctx_primary_key="user_id",
        ...     ctx_encoding_types={"age": "tabular_numeric_auto"},
        ... )
    """
    from mostlyai.engine._tabular.training import train as train_internal, OnEpochCallback

    # Validate that we have target columns to train on
    if not tgt_encoding_types:
        raise ValueError(
            "tgt_encoding_types cannot be empty. At least one target column must be specified for training."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Context is optional - treat empty dict same as None
    has_context = ctx_data is not None and ctx_encoding_types

    # Count contexts
    context_ids = tgt_data[tgt_context_key].unique()
    n_contexts = len(context_ids)

    _LOG.info(f"Training sequential model on {device} with {n_contexts} contexts, {len(tgt_data)} rows")

    # Step 1: Compute stats
    _LOG.info("Computing statistics...")
    tgt_stats, ctx_stats = compute_stats(
        tgt_data=tgt_data,
        tgt_encoding_types=tgt_encoding_types,
        tgt_context_key=tgt_context_key,
        ctx_data=ctx_data,
        ctx_primary_key=ctx_primary_key,
        ctx_encoding_types=ctx_encoding_types,
    )

    # Step 2: Split by context (not by row)
    n_val = int(n_contexts * val_split)
    n_train = n_contexts - n_val

    shuffled_ids = np.random.permutation(context_ids)
    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:])

    train_tgt = tgt_data[tgt_data[tgt_context_key].isin(train_ids)].reset_index(drop=True)
    val_tgt = tgt_data[tgt_data[tgt_context_key].isin(val_ids)].reset_index(drop=True)

    if has_context:
        train_ctx = ctx_data[ctx_data[ctx_primary_key].isin(train_ids)].reset_index(drop=True)
        val_ctx = ctx_data[ctx_data[ctx_primary_key].isin(val_ids)].reset_index(drop=True)
    else:
        train_ctx = val_ctx = None

    _LOG.info(f"Split: {n_train} training, {n_val} validation contexts")

    # Step 3: Encode to tensors on CPU (single encoding pass)
    # Tensors are kept on CPU to avoid VRAM exhaustion; batches are moved to GPU on demand
    _LOG.info("Encoding to tensors...")
    seq_len_max = tgt_stats["seq_len"]["max"]
    cpu_device = torch.device("cpu")

    train_tensors = _encode_sequential_to_tensors(
        train_tgt,
        tgt_stats,
        tgt_context_key,
        train_ctx,
        ctx_stats,
        ctx_primary_key,
        seq_len_max,
        cpu_device,
    )
    val_tensors = _encode_sequential_to_tensors(
        val_tgt,
        tgt_stats,
        tgt_context_key,
        val_ctx,
        ctx_stats,
        ctx_primary_key,
        seq_len_max,
        cpu_device,
    )

    # Step 4: Build model config
    tgt_cardinalities = get_cardinalities(tgt_stats)
    ctx_cardinalities = get_cardinalities(ctx_stats) if ctx_stats else {}
    ctx_seq_len_median = get_ctx_sequence_length(ctx_stats, key="median") if ctx_stats else {}

    model_config: ModelConfig = {
        "tgt_cardinalities": tgt_cardinalities,
        "ctx_cardinalities": ctx_cardinalities,
        "is_sequential": True,
        "trn_cnt": n_train,
        "val_cnt": n_val,
        "tgt_seq_len_median": tgt_stats["seq_len"]["median"],
        "tgt_seq_len_max": seq_len_max,
        "ctx_seq_len_median": ctx_seq_len_median,
        "empirical_probs": None,
    }

    # Step 5: Create batch iterators
    if batch_size is None:
        batch_size = min(n_train, 512)

    def make_batch_iter(
        tensor_dict: dict[str, torch.Tensor], batch_sz: int, target_device: torch.device, shuffle: bool = False
    ):
        """Create iterator over batches from pre-loaded CPU tensors.

        Tensors are stored on CPU and moved to GPU per-batch to avoid VRAM exhaustion.
        This allows training on datasets larger than GPU memory while still benefiting
        from single-pass encoding.
        """
        n = next(iter(tensor_dict.values())).shape[0]
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            batch_indices = indices[start:end]
            # Move only this batch to target device (GPU)
            yield {k: v[batch_indices].to(target_device, non_blocking=True) for k, v in tensor_dict.items()}

    # Step 6: Train
    with tempfile.TemporaryDirectory(prefix="train_seq_") as workspace_dir:
        workspace_path = Path(workspace_dir)
        model_store = workspace_path / "ModelStore"

        # Write stats
        tgt_stats_dir = model_store / "tgt-stats"
        tgt_stats_dir.mkdir(parents=True)
        with open(tgt_stats_dir / "stats.json", "w") as f:
            json.dump(tgt_stats, f)

        if ctx_stats:
            ctx_stats_dir = model_store / "ctx-stats"
            ctx_stats_dir.mkdir(parents=True)
            with open(ctx_stats_dir / "stats.json", "w") as f:
                json.dump(ctx_stats, f)

        _LOG.info(f"Training for up to {max_epochs} epochs...")

        train_internal(
            train_tensors=make_batch_iter(train_tensors, batch_size, device, shuffle=True),
            val_tensors=make_batch_iter(val_tensors, batch_size, device, shuffle=False),
            model_config=model_config,
            workspace_dir=workspace_dir,
            model=f"MOSTLY_AI/{_model_size_name(model_size)}",
            max_training_time=max_training_time,
            max_epochs=max_epochs,
            batch_size=batch_size,
            enable_flexible_generation=enable_flexible_generation,
            device=device,
            on_epoch=on_epoch,
        )

        # Step 7: Create artifact
        weights_path = model_store / "model-data" / "model-weights.pt"
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        artifact = ModelArtifact.from_state_dict(
            state_dict=state_dict,
            is_sequential=True,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            tgt_stats=minimize_stats(tgt_stats),
            ctx_stats=minimize_stats(ctx_stats) if ctx_stats else {},
            tgt_seq_len_min=tgt_stats["seq_len"]["min"],
            tgt_seq_len_max=seq_len_max,
            tgt_seq_len_median=tgt_stats["seq_len"]["median"],
            ctx_seq_len_median=ctx_seq_len_median,
            tgt_context_key=tgt_context_key,
            ctx_primary_key=ctx_primary_key,
            enable_flexible_generation=enable_flexible_generation,
        )

    _LOG.info(f"Training complete. Artifact size: {len(artifact.to_bytes()):,} bytes")
    return artifact


def _model_size_name(size: str) -> str:
    """Convert single-letter size to full name."""
    return {"S": "Small", "M": "Medium", "L": "Large"}.get(size.upper(), "Medium")


def _encode_df_to_tensors(
    df: pd.DataFrame,
    tgt_stats: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Encode DataFrame directly to GPU tensors.

    This avoids the per-batch encoding overhead by encoding the full
    dataset once and keeping it on GPU.
    """
    # Encode using existing function
    encoded_df, _, _ = encode_df(
        df=df,
        stats=tgt_stats,
        n_jobs=1,
    )

    # Convert to tensors on target device
    tensors = {}
    for col in encoded_df.columns:
        arr = encoded_df[col].to_numpy()
        tensors[col] = torch.tensor(arr, dtype=torch.int64, device=device).unsqueeze(-1)

    return tensors


def _encode_sequential_to_tensors(
    tgt_df: pd.DataFrame,
    tgt_stats: dict,
    tgt_context_key: str,
    ctx_df: pd.DataFrame | None,
    ctx_stats: dict | None,
    ctx_primary_key: str | None,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Encode sequential data to padded tensors.

    Groups target rows by context, encodes, and pads to uniform length.
    Also adds positional columns (sidx, slen, ridx) required for sequential models.

    In the original implementation, positional columns are added by _enrich_positional_columns()
    after encode_df() but before flatten_frame(). Since we're building tensors directly,
    we create these positional tensors here instead.
    """
    from mostlyai.engine._common import (
        CTXFLT,
        CTXSEQ,
        RIDX_SUB_COLUMN_PREFIX,
        SIDX_RIDX_DIGIT_ENCODING_THRESHOLD,
        SIDX_SUB_COLUMN_PREFIX,
        SLEN_SUB_COLUMN_PREFIX,
        TGT,
    )
    from mostlyai.engine._tabular.common import pad_ctx_sequences

    tensors = {}

    # Encode target data
    tgt_encoded, _, encoded_context_key = encode_df(
        df=tgt_df,
        stats=tgt_stats,
        tgt_context_key=tgt_context_key,
        n_jobs=1,
    )

    # Get unique contexts in order of first appearance
    context_ids = tgt_df[tgt_context_key].unique()
    n_contexts = len(context_ids)

    # Create mapping from context_id to context index (0, 1, 2, ...)
    ctx_id_to_idx = {ctx_id: idx for idx, ctx_id in enumerate(context_ids)}

    # Map each row to its context index
    row_ctx_indices = tgt_encoded[encoded_context_key].map(ctx_id_to_idx).to_numpy()

    # Calculate position of each row within its context (0, 1, 2, ... for each context)
    positions_within_ctx = tgt_encoded.groupby(encoded_context_key, sort=False).cumcount().to_numpy()

    # Get sequence lengths for each context (capped at max_seq_len)
    seq_lengths_series = tgt_df.groupby(tgt_context_key).size()
    seq_lengths_arr = np.array([min(seq_lengths_series.get(ctx_id, 0), max_seq_len) for ctx_id in context_ids])

    # +1 accounts for the padding row added per sequence
    padded_seq_len = max_seq_len + 1

    # Get target columns
    tgt_cols = [c for c in tgt_encoded.columns if c.startswith(TGT)]

    # ==========================================================================
    # VECTORIZED: Fill all columns using advanced indexing
    # ==========================================================================
    # Instead of: for col in cols: for i, ctx in enumerate(contexts): ...
    # We use: padded[row_ctx_indices, positions_within_ctx] = values

    for col in tgt_cols:
        col_data = tgt_encoded[col].to_numpy()

        # Create padded array on CPU first, then convert to tensor
        padded = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)

        # Only include rows where position < max_seq_len
        valid_mask = positions_within_ctx < max_seq_len
        valid_ctx_indices = row_ctx_indices[valid_mask]
        valid_positions = positions_within_ctx[valid_mask]
        valid_values = col_data[valid_mask]

        # Vectorized assignment using advanced indexing
        padded[valid_ctx_indices, valid_positions, 0] = valid_values

        tensors[col] = torch.tensor(padded, dtype=torch.int64, device=device)

    # ==========================================================================
    # Add positional columns (sidx, slen, ridx)
    # ==========================================================================
    # These columns encode position metadata that the model uses to understand
    # where each token sits within a sequence:
    #
    # - sidx (sequence index): Position from start of sequence (0, 1, 2, ...)
    #   Helps model learn patterns that depend on absolute position
    #
    # - slen (sequence length): Total length of the sequence (same for all positions)
    #   Helps model understand sequence scale and adjust predictions accordingly
    #
    # - ridx (reverse index): Position from end of sequence (counts down to 0)
    #   Helps model learn end-of-sequence patterns (e.g., "2 items left")
    #
    # IMPORTANT: The original implementation adds a padding row to each sequence
    # (via pad_tgt_sequences), then computes positional columns INCLUDING the padding row.
    # For a sequence of length N:
    #   - After padding: N+1 rows
    #   - sidx = [0, 1, 2, ..., N-1, N]  (N+1 values)
    #   - slen = [N, N, N, ..., N, N]    (all equal to original length)
    #   - ridx = [N, N-1, N-2, ..., 1, 0] (N+1 values, padding row has ridx=0)
    #
    # The padding row (with sidx=N, slen=N, ridx=0, and data=0) teaches the model
    # to recognize end-of-sequence.
    #
    # Encoding format depends on max_seq_len:
    # - For short sequences (< 100): Single categorical column (e.g., tgt:/__sidx_cat)
    # - For long sequences (>= 100): Digit encoding (e.g., tgt:/__sidx_E0, tgt:/__sidx_E1)
    # ==========================================================================

    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # Short sequences: single categorical column per positional encoding
        sidx_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)
        slen_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)
        ridx_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)

        for i in range(n_contexts):
            seq_len = seq_lengths_arr[i]
            if seq_len > 0:
                padded_len = seq_len + 1
                # sidx: 0, 1, 2, ..., seq_len
                sidx_arr[i, :padded_len, 0] = np.arange(padded_len)
                # slen: seq_len for all positions
                slen_arr[i, :padded_len, 0] = seq_len
                # ridx: seq_len, seq_len-1, ..., 1, 0
                ridx_arr[i, :padded_len, 0] = np.arange(seq_len, -1, -1)

        tensors[f"{SIDX_SUB_COLUMN_PREFIX}cat"] = torch.tensor(sidx_arr, dtype=torch.int64, device=device)
        tensors[f"{SLEN_SUB_COLUMN_PREFIX}cat"] = torch.tensor(slen_arr, dtype=torch.int64, device=device)
        tensors[f"{RIDX_SUB_COLUMN_PREFIX}cat"] = torch.tensor(ridx_arr, dtype=torch.int64, device=device)
    else:
        # Long sequences: digit encoding (each digit position is a separate column)
        # E.g., for max_seq_len=150, we get E2, E1, E0 columns representing hundreds, tens, ones
        n_digits = len(str(max_seq_len))

        for d in range(n_digits):
            exp = n_digits - d - 1
            divisor = 10**exp

            sidx_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)
            slen_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)
            ridx_arr = np.zeros((n_contexts, padded_seq_len, 1), dtype=np.int64)

            for i in range(n_contexts):
                seq_len = seq_lengths_arr[i]
                if seq_len > 0:
                    padded_len = seq_len + 1
                    sidx_vals = np.arange(padded_len)
                    ridx_vals = np.arange(seq_len, -1, -1)

                    sidx_arr[i, :padded_len, 0] = (sidx_vals // divisor) % 10
                    slen_arr[i, :padded_len, 0] = (seq_len // divisor) % 10
                    ridx_arr[i, :padded_len, 0] = (ridx_vals // divisor) % 10

            tensors[f"{SIDX_SUB_COLUMN_PREFIX}E{exp}"] = torch.tensor(sidx_arr, dtype=torch.int64, device=device)
            tensors[f"{SLEN_SUB_COLUMN_PREFIX}E{exp}"] = torch.tensor(slen_arr, dtype=torch.int64, device=device)
            tensors[f"{RIDX_SUB_COLUMN_PREFIX}E{exp}"] = torch.tensor(ridx_arr, dtype=torch.int64, device=device)

    # Encode context data if provided
    if ctx_df is not None and ctx_stats is not None:
        ctx_encoded, _, _ = encode_df(
            df=ctx_df,
            stats=ctx_stats,
            ctx_primary_key=ctx_primary_key,
            n_jobs=1,
        )

        # Reorder to match target context order
        ctx_id_to_idx_map = {cid: i for i, cid in enumerate(ctx_df[ctx_primary_key])}
        ctx_order = [ctx_id_to_idx_map.get(cid, 0) for cid in context_ids]

        ctx_encoded = pad_ctx_sequences(ctx_encoded)

        for col in ctx_encoded.columns:
            if col.startswith(CTXFLT):
                arr = ctx_encoded[col].iloc[ctx_order].to_numpy()
                tensors[col] = torch.tensor(arr, dtype=torch.int64, device=device).unsqueeze(-1)
            elif col.startswith(CTXSEQ):
                # Handle nested sequences
                nested = [
                    torch.tensor(ctx_encoded[col].iloc[ctx_order[i]], dtype=torch.int64, device=device)
                    for i in range(n_contexts)
                ]
                tensors[col] = torch.unsqueeze(
                    torch.nested.as_nested_tensor(nested, device=device),
                    dim=-1,
                )

    return tensors
