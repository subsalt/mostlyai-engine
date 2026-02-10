"""
Core generation functions that work with ModelArtifact instead of workspace.

This module contains generation logic that operates entirely in memory without
disk I/O, designed for distributed cluster environments.
"""

import logging
import random
import uuid
from typing import Literal

import pandas as pd
import torch

from mostlyai.engine._artifact import ModelArtifact
from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    DEFAULT_HAS_RIDX,
    DEFAULT_HAS_SDEC,
    DEFAULT_HAS_SLEN,
    POSITIONAL_COLUMN,
    RIDX_SUB_COLUMN_PREFIX,
    SDEC_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    encode_positional_column,
    get_argn_name,
    get_columns_from_cardinalities,
    get_sub_columns_from_cardinalities,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_SUB_COL_SUFFIX,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_DISCRETE_SUB_COL_SUFFIX,
    NUMERIC_DISCRETE_UNKNOWN_TOKEN,
)
from mostlyai.engine._tabular.argn import FlatModel, ModelSize, SequentialModel
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.generation_gpu import (
    compute_sequence_continue_mask_torch,
    decode_positional_column_torch,
)
from mostlyai.engine.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)

DUMMY_CONTEXT_KEY = "__dummy_context_key"


def _compute_fixed_probs_for_rare_suppression(tgt_stats: dict) -> dict:
    """
    Compute fixed probabilities to suppress rare/unknown tokens.

    This ensures the model doesn't generate _RARE_ tokens for categorical
    columns or unknown tokens for numeric columns.

    Args:
        tgt_stats: Target statistics dict

    Returns:
        Dict mapping sub_column names to code -> probability dicts
    """
    fixed_probs = {}

    for col, col_stats in tgt_stats.get("columns", {}).items():
        encoding_type = col_stats.get("encoding_type", "")
        codes = col_stats.get("codes", {})

        if not codes:
            continue

        # Get ARGN name prefix for this column
        argn_prefix = get_argn_name(
            argn_processor=col_stats.get(ARGN_PROCESSOR, "tgt"),
            argn_table=col_stats.get(ARGN_TABLE, "t0"),
            argn_column=col_stats.get(ARGN_COLUMN, "c0"),
            argn_sub_column="",
        )

        # Suppress _RARE_ for categorical columns
        if encoding_type == ModelEncodingType.tabular_categorical.value:
            if CATEGORICAL_UNKNOWN_TOKEN in codes:
                sub_col = f"{argn_prefix}{CATEGORICAL_SUB_COL_SUFFIX}"
                fixed_probs[sub_col] = {codes[CATEGORICAL_UNKNOWN_TOKEN]: 0.0}

        # Suppress unknown for numeric_discrete
        elif encoding_type == ModelEncodingType.tabular_numeric_discrete.value:
            if NUMERIC_DISCRETE_UNKNOWN_TOKEN in codes:
                sub_col = f"{argn_prefix}{NUMERIC_DISCRETE_SUB_COL_SUFFIX}"
                fixed_probs[sub_col] = {codes[NUMERIC_DISCRETE_UNKNOWN_TOKEN]: 0.0}

        # Suppress unknown for numeric_binned
        elif encoding_type == ModelEncodingType.tabular_numeric_binned.value:
            if NUMERIC_BINNED_UNKNOWN_TOKEN in codes:
                sub_col = f"{argn_prefix}{NUMERIC_BINNED_SUB_COL_SUFFIX}"
                fixed_probs[sub_col] = {codes[NUMERIC_BINNED_UNKNOWN_TOKEN]: 0.0}

    return fixed_probs


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve device specification to torch.device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _generate_primary_keys(size: int, type: Literal["uuid", "int"] = "uuid") -> pd.Series:
    """Generate primary keys for synthetic data."""
    if type == "uuid":
        return pd.Series(
            [f"mostly{str(uuid.UUID(int=random.getrandbits(128), version=4))[6:]}" for _ in range(size)],
            dtype="string",
        )
    else:
        return pd.Series(range(size), dtype="int")


def _create_model_from_artifact(
    artifact: ModelArtifact,
    device: torch.device,
    column_order: list[str] | None = None,
) -> FlatModel | SequentialModel:
    """
    Reconstruct model from artifact and load weights.

    Args:
        artifact: ModelArtifact containing weights and config
        device: Device to load model on
        column_order: Optional column order for generation

    Returns:
        Initialized model ready for inference
    """
    model_sizes = {"S": ModelSize.S, "M": ModelSize.M, "L": ModelSize.L}
    model_size = model_sizes.get(artifact.model_size, ModelSize.M)

    if artifact.is_sequential:
        model = SequentialModel(
            tgt_cardinalities=artifact.tgt_cardinalities,
            tgt_seq_len_median=artifact.tgt_seq_len_median or 1,
            tgt_seq_len_max=artifact.tgt_seq_len_max or 1,
            ctx_cardinalities=artifact.ctx_cardinalities,
            ctxseq_len_median=artifact.ctx_seq_len_median,
            model_size=model_size,
            column_order=column_order,
            device=device,
        )
    else:
        model = FlatModel(
            tgt_cardinalities=artifact.tgt_cardinalities,
            ctx_cardinalities=artifact.ctx_cardinalities,
            ctxseq_len_median=artifact.ctx_seq_len_median,
            model_size=model_size,
            column_order=column_order,
            device=device,
        )

    # Load weights
    state_dict = artifact.get_state_dict(device=str(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _LOG.info(f"Created {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def _encode_context_data(
    ctx_data: pd.DataFrame,
    ctx_stats: dict,
    ctx_primary_key: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, str]:
    """
    Encode context data to tensors for generation.

    Args:
        ctx_data: Context DataFrame
        ctx_stats: Context column statistics
        ctx_primary_key: Primary key column in context
        device: Device for tensors

    Returns:
        Tuple of (context_tensors, encoded_dataframe, encoded_primary_key)
    """
    from mostlyai.engine._tabular.common import pad_ctx_sequences

    # Encode context data
    ctx_encoded, ctx_primary_key_encoded, _ = encode_df(
        df=ctx_data,
        stats=ctx_stats,
        ctx_primary_key=ctx_primary_key,
    )

    # Pad empty sequences
    ctx_encoded = pad_ctx_sequences(ctx_encoded)

    # Build flat context inputs (CTXFLT/*)
    ctxflt_inputs = {
        col: torch.unsqueeze(
            torch.as_tensor(ctx_encoded[col].to_numpy(), device=device).type(torch.int),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXFLT)
    }

    # Build sequential context inputs (CTXSEQ/*)
    ctxseq_inputs = {
        col: torch.unsqueeze(
            torch.nested.as_nested_tensor(
                [torch.as_tensor(t, device=device).type(torch.int) for t in ctx_encoded[col]],
                device=device,
            ),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXSEQ)
    }

    return (ctxflt_inputs | ctxseq_inputs), ctx_encoded, ctx_primary_key_encoded


def _decode_df(
    df_encoded: pd.DataFrame,
    stats: dict,
    context_key: str | None = None,
) -> pd.DataFrame:
    """
    Decode encoded DataFrame back to original values.

    Args:
        df_encoded: Encoded DataFrame with integer codes
        stats: Column statistics for decoding
        context_key: Optional context key column to preserve

    Returns:
        Decoded DataFrame with original values
    """
    from mostlyai.engine._common import ARGN_COLUMN, ARGN_PROCESSOR, ARGN_TABLE, get_argn_name
    from mostlyai.engine._encoding_types.tabular.categorical import decode_categorical
    from mostlyai.engine._encoding_types.tabular.character import decode_character
    from mostlyai.engine._encoding_types.tabular.datetime import decode_datetime
    from mostlyai.engine._encoding_types.tabular.itt import decode_itt
    from mostlyai.engine._encoding_types.tabular.lat_long import decode_latlong
    from mostlyai.engine._encoding_types.tabular.numeric import decode_numeric
    from mostlyai.engine.domain import ModelEncodingType

    columns = []

    # Preserve context key
    if context_key and context_key in df_encoded.columns:
        columns.append(df_encoded[context_key])

    for column, column_stats in stats["columns"].items():
        if column_stats.keys() == {"encoding_type"}:
            # Training data was empty
            values = pd.Series(data=[], name=column, dtype="object")
            columns.append(values)
            continue

        # Get sub-column names
        sub_columns = [
            get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
                argn_column=column_stats[ARGN_COLUMN],
                argn_sub_column=sub_col,
            )
            for sub_col in column_stats["cardinalities"].keys()
        ]

        # Extract column-specific sub_columns
        df_encoded_col = df_encoded[sub_columns]

        # Remove column prefixes before decoding
        df_encoded_col.columns = [
            ocol.replace(
                get_argn_name(
                    argn_processor=column_stats[ARGN_PROCESSOR],
                    argn_table=column_stats[ARGN_TABLE],
                    argn_column=column_stats[ARGN_COLUMN],
                    argn_sub_column="",
                ),
                "",
            )
            for ocol in df_encoded_col.columns
        ]

        # Decode based on encoding type
        encoding_type = column_stats["encoding_type"]
        if encoding_type == ModelEncodingType.tabular_categorical.value:
            values = decode_categorical(df_encoded_col, column_stats)
        elif encoding_type in (
            ModelEncodingType.tabular_numeric_auto.value,
            ModelEncodingType.tabular_numeric_digit.value,
            ModelEncodingType.tabular_numeric_discrete.value,
            ModelEncodingType.tabular_numeric_binned.value,
        ):
            values = decode_numeric(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_datetime.value:
            values = decode_datetime(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_datetime_relative.value:
            values = decode_itt(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_character.value:
            values = decode_character(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_lat_long.value:
            values = decode_latlong(df_encoded_col, column_stats)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        values.name = column
        columns.append(values)

    return pd.concat(columns, axis=1)


@torch.no_grad()
def generate_flat_core(
    artifact: ModelArtifact,
    sample_size: int,
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic flat (non-sequential) data from a ModelArtifact.

    This is the core implementation that works entirely in memory without
    workspace file I/O.

    Args:
        artifact: ModelArtifact containing model weights and stats
        sample_size: Number of samples to generate
        ctx_data: Optional context data for conditional generation
        seed_data: Optional seed data to condition generation
        sampling_temperature: Temperature for sampling (higher = more random)
        sampling_top_p: Nucleus sampling probability threshold
        device: Device for generation ("cuda" or "cpu")
        batch_size: Batch size for generation (auto-determined if None)

    Returns:
        DataFrame with generated synthetic data
    """
    if artifact.is_sequential:
        raise ValueError("Use generate_sequential_core() for sequential models")

    device = _resolve_device(device)
    _LOG.info(f"Generating {sample_size} samples on {device}")

    # Get column info from artifact
    tgt_stats = artifact.tgt_stats
    ctx_stats = artifact.ctx_stats
    tgt_context_key = artifact.tgt_context_key
    ctx_primary_key = artifact.ctx_primary_key
    tgt_primary_key = artifact.tgt_primary_key

    column_order = get_columns_from_cardinalities(artifact.tgt_cardinalities)
    tgt_sub_columns = get_sub_columns_from_cardinalities(artifact.tgt_cardinalities)

    # Compute fixed probs to suppress rare/unknown tokens
    fixed_probs = _compute_fixed_probs_for_rare_suppression(tgt_stats)

    # Create model
    model = _create_model_from_artifact(artifact, device, column_order)

    # Determine if we have context
    has_context = bool(ctx_stats and ctx_stats.get("columns"))

    # Handle context data
    if has_context and ctx_data is not None:
        # Encode provided context
        ctx_data = ctx_data.reset_index(drop=True)
        sample_size = min(sample_size, len(ctx_data))
        ctx_data = ctx_data.head(sample_size)

        ctx_inputs, ctx_encoded, ctx_pk_encoded = _encode_context_data(
            ctx_data, ctx_stats, ctx_primary_key, device
        )
        ctx_keys = ctx_encoded[ctx_pk_encoded].rename(tgt_context_key)
    else:
        # Create dummy context
        ctx_primary_key = tgt_context_key or DUMMY_CONTEXT_KEY
        tgt_context_key = ctx_primary_key
        ctx_keys = _generate_primary_keys(sample_size, type="int")
        ctx_keys.rename(tgt_context_key, inplace=True)
        ctx_inputs = {}

    # Handle seed data
    seed_encoded = pd.DataFrame()
    if seed_data is not None and len(seed_data) > 0:
        # Link seed data to context for flat generation
        if tgt_context_key not in seed_data.columns:
            seed_data = seed_data.assign(**{tgt_context_key: ctx_keys.values[: len(seed_data)]})

        seed_encoded, _, _ = encode_df(
            df=seed_data,
            stats={"columns": tgt_stats.get("columns", {}), "is_sequential": False},
            tgt_context_key=tgt_context_key,
        )

    # Determine batch size
    if batch_size is None:
        batch_size = min(sample_size, 8192)

    # Generate in batches
    results = []
    seed_results = []
    num_batches = (sample_size + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, sample_size)
        batch_len = end - start

        batch_ctx_keys = ctx_keys.iloc[start:end].reset_index(drop=True)

        # Prepare batch context inputs
        if has_context and ctx_inputs:
            batch_ctx_inputs = {k: v[start:end] for k, v in ctx_inputs.items()}
        else:
            batch_ctx_inputs = None

        # Prepare fixed values from seed
        fixed_values = {}
        batch_seed = pd.DataFrame()
        if len(seed_encoded) > 0 and seed_data is not None:
            batch_seed_mask = seed_data[tgt_context_key].isin(batch_ctx_keys)
            batch_seed = seed_data[batch_seed_mask].reset_index(drop=True)
            batch_seed_encoded = seed_encoded[batch_seed_mask].reset_index(drop=True)
            fixed_values = {
                col: torch.as_tensor(batch_seed_encoded[col].to_numpy(), device=device).type(torch.int)
                for col in batch_seed_encoded.columns
                if col in tgt_sub_columns
            }

        # Forward pass
        out_dct, _ = model(
            batch_ctx_inputs,
            mode="gen",
            batch_size=batch_len,
            fixed_probs=fixed_probs,
            fixed_values=fixed_values,
            temperature=sampling_temperature,
            top_p=sampling_top_p,
            column_order=column_order,
        )

        # Collect results
        batch_df = pd.concat(
            [batch_ctx_keys]
            + [
                pd.Series(out_dct[sub_col].detach().cpu().numpy(), dtype="int32", name=sub_col)
                for sub_col in artifact.tgt_cardinalities.keys()
            ],
            axis=1,
        )
        results.append(batch_df)
        seed_results.append(batch_seed)

    # Combine batches
    encoded_df = pd.concat(results, ignore_index=True)
    all_seed = pd.concat(seed_results, ignore_index=True) if any(len(s) > 0 for s in seed_results) else pd.DataFrame()

    # Decode
    decoded_df = _decode_df(
        encoded_df,
        {"columns": tgt_stats.get("columns", {}), "is_sequential": False},
        context_key=tgt_context_key,
    )

    # Restore seed values
    if len(all_seed) > 0:
        for col in all_seed.columns:
            if col != tgt_context_key and col in decoded_df.columns:
                decoded_df[col] = all_seed[col].values

    # Generate primary keys if needed
    if tgt_primary_key and tgt_primary_key not in decoded_df.columns:
        decoded_df[tgt_primary_key] = _generate_primary_keys(len(decoded_df), type="uuid")

    # Drop dummy context key if present
    if DUMMY_CONTEXT_KEY in decoded_df.columns:
        decoded_df = decoded_df.drop(columns=[DUMMY_CONTEXT_KEY])

    _LOG.info(f"Generated {len(decoded_df)} samples with columns: {list(decoded_df.columns)}")

    return decoded_df.reset_index(drop=True)


# ==============================================================================
# Sequential Generation Helpers
# ==============================================================================


def _reshape_pt_to_pandas(
    data: list[torch.Tensor],
    sub_cols: list[str],
    keys: list[pd.Series],
    key_name: str,
) -> pd.DataFrame:
    """
    Reshape list of tensors (one per sequence step) to a single DataFrame.

    Args:
        data: List of tensors, one per step. Shape: (batch_size, n_sub_cols, 1)
        sub_cols: List of sub-column names
        keys: List of key Series, one per step
        key_name: Name for the key column

    Returns:
        DataFrame with all steps concatenated
    """
    if len(data) == 0:
        return pd.DataFrame(columns=[key_name] + sub_cols)

    # Concatenate all sequence steps
    df = pd.concat(
        [
            pd.DataFrame(
                step_tensor.squeeze(-1).detach().cpu().numpy(),
                columns=sub_cols,
                dtype="int32",
            )
            for step_tensor in data
        ],
        axis=0,
    ).reset_index(drop=True)

    # Concatenate keys
    keys_series = pd.concat(keys, axis=0).rename(key_name).reset_index(drop=True)
    return pd.concat([keys_series, df], axis=1)


def _build_positional_fixed_values(
    seq_step: int,
    step_size: int,
    seq_len_max: int,
    seq_len_min: int,
    n_seed_steps: int,
    out_df: pd.DataFrame | None,
    has_slen: bool,
    has_ridx: bool,
    has_sdec: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build fixed values for positional columns (SIDX, SLEN, RIDX, SDEC).

    These columns track sequence position and are used to control when
    sequences should end.
    """
    fixed_values = {}

    # SIDX: sequence index (always set to current step)
    sidx = pd.Series([seq_step] * step_size)
    sidx_df = encode_positional_column(sidx, max_seq_len=seq_len_max, prefix=SIDX_SUB_COLUMN_PREFIX)
    for col in sidx_df.columns:
        fixed_values[col] = torch.unsqueeze(
            torch.as_tensor(sidx_df[col].to_numpy(), device=device).type(torch.int),
            dim=-1,
        )

    # SLEN: sequence length (propagate from first step)
    if has_slen and seq_step > 0 and out_df is not None:
        slen = out_df[SLEN_SUB_COLUMN_PREFIX]
        slen_df = encode_positional_column(slen, max_seq_len=seq_len_max, prefix=SLEN_SUB_COLUMN_PREFIX)
        for col in slen_df.columns:
            fixed_values[col] = torch.unsqueeze(
                torch.as_tensor(slen_df[col].to_numpy(), dtype=torch.int64, device=device),
                dim=-1,
            )

    # RIDX: remaining index (propagate after seeded part, decrement each step)
    if has_ridx and seq_step > n_seed_steps and out_df is not None:
        ridx = (out_df[RIDX_SUB_COLUMN_PREFIX] - 1).clip(lower=0)
        ridx_df = encode_positional_column(ridx, max_seq_len=seq_len_max, prefix=RIDX_SUB_COLUMN_PREFIX)
        for col in ridx_df.columns:
            fixed_values[col] = torch.unsqueeze(
                torch.as_tensor(ridx_df[col].to_numpy(), dtype=torch.int64, device=device),
                dim=-1,
            )

    # SDEC: sequence decile (position within sequence as 0-9)
    if has_sdec:
        if seq_step > 0 and out_df is not None:
            slen = out_df[SLEN_SUB_COLUMN_PREFIX]
            sdec = ((10 * sidx / slen.clip(lower=1)).clip(upper=9).astype(int))
        else:
            sdec = pd.Series([0] * step_size)
        fixed_values[f"{SDEC_SUB_COLUMN_PREFIX}cat"] = torch.unsqueeze(
            torch.as_tensor(sdec.to_numpy(), device=device).type(torch.int),
            dim=-1,
        )

    return fixed_values


def _filter_sequence_state(
    include_mask: pd.Series,
    context: list | None,
    history: torch.Tensor,
    history_state: tuple,
) -> tuple[list | None, torch.Tensor, tuple]:
    """
    Filter sequence state tensors to only include continuing sequences.
    """
    mask = include_mask.values

    if context is not None:
        context = [
            c[mask, ...] if isinstance(c, torch.Tensor) else [sub_c[mask, ...] for sub_c in c]
            for c in context
        ]

    history = history[mask, ...]
    history_state = tuple(h[:, mask, ...] for h in history_state)

    return context, history, history_state


def _decode_sequential_df(
    df_encoded: pd.DataFrame,
    stats: dict,
    context_key: str | None = None,
    prev_steps: dict | None = None,
) -> pd.DataFrame:
    """
    Decode encoded sequential DataFrame back to original values.

    Similar to _decode_df but handles ITT (inter-transaction time) with prev_steps.
    """
    from mostlyai.engine._encoding_types.tabular.categorical import decode_categorical
    from mostlyai.engine._encoding_types.tabular.character import decode_character
    from mostlyai.engine._encoding_types.tabular.datetime import decode_datetime
    from mostlyai.engine._encoding_types.tabular.itt import decode_itt
    from mostlyai.engine._encoding_types.tabular.lat_long import decode_latlong
    from mostlyai.engine._encoding_types.tabular.numeric import decode_numeric

    columns = []

    if context_key and context_key in df_encoded.columns:
        columns.append(df_encoded[context_key])

    for column, column_stats in stats["columns"].items():
        if column_stats.keys() == {"encoding_type"}:
            columns.append(pd.Series(data=[], name=column, dtype="object"))
            continue

        # Get sub-column names
        sub_columns = [
            get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
                argn_column=column_stats[ARGN_COLUMN],
                argn_sub_column=sub_col,
            )
            for sub_col in column_stats["cardinalities"].keys()
        ]

        df_encoded_col = df_encoded[sub_columns]

        # Strip ARGN prefix for decoder
        df_encoded_col.columns = [
            ocol.replace(
                get_argn_name(
                    argn_processor=column_stats[ARGN_PROCESSOR],
                    argn_table=column_stats[ARGN_TABLE],
                    argn_column=column_stats[ARGN_COLUMN],
                    argn_sub_column="",
                ),
                "",
            )
            for ocol in df_encoded_col.columns
        ]

        # Track prev_steps for ITT
        prev_steps_col = None
        if prev_steps is not None:
            prev_steps[column] = prev_steps.get(column, {})
            prev_steps_col = prev_steps[column]

        # Decode based on encoding type
        encoding_type = column_stats["encoding_type"]
        if encoding_type == ModelEncodingType.tabular_categorical.value:
            values = decode_categorical(df_encoded_col, column_stats)
        elif encoding_type in (
            ModelEncodingType.tabular_numeric_auto.value,
            ModelEncodingType.tabular_numeric_digit.value,
            ModelEncodingType.tabular_numeric_discrete.value,
            ModelEncodingType.tabular_numeric_binned.value,
        ):
            values = decode_numeric(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_datetime.value:
            values = decode_datetime(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_datetime_relative.value:
            context_keys = df_encoded[context_key] if context_key in df_encoded.columns else None
            values = decode_itt(df_encoded_col, column_stats, context_keys, prev_steps_col)
        elif encoding_type == ModelEncodingType.tabular_character.value:
            values = decode_character(df_encoded_col, column_stats)
        elif encoding_type == ModelEncodingType.tabular_lat_long.value:
            values = decode_latlong(df_encoded_col, column_stats)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        values.name = column
        columns.append(values)

    return pd.concat(columns, axis=1)


def _restore_seed_values_sequential(
    decoded_df: pd.DataFrame,
    all_seed_dfs: list[pd.DataFrame],
    tgt_context_key: str,
) -> pd.DataFrame:
    """
    Restore original seed values in decoded DataFrame.

    For sequential data, we need to match by both context key and sequence index.
    """
    if not all_seed_dfs:
        return decoded_df

    all_seed = pd.concat(all_seed_dfs, ignore_index=True)
    if len(all_seed) == 0:
        return decoded_df

    # Add sequence index for matching
    decoded_df["__SEQ_IDX"] = decoded_df.groupby(tgt_context_key).cumcount()
    all_seed["__SEQ_IDX"] = all_seed.groupby(tgt_context_key).cumcount()

    # Merge to find matching rows
    df_overwrite = pd.merge(
        decoded_df[[tgt_context_key, "__SEQ_IDX"]].copy(),
        all_seed,
        on=[tgt_context_key, "__SEQ_IDX"],
        how="left",
        indicator="__INDICATOR",
    )

    # Overwrite with seed values
    seed_rows = df_overwrite["__INDICATOR"] == "both"
    for col in all_seed.columns:
        if col in [tgt_context_key, "__SEQ_IDX"]:
            continue
        if col in decoded_df.columns:
            decoded_df.loc[seed_rows.values, col] = df_overwrite.loc[seed_rows, col].values

    return decoded_df.drop(columns=["__SEQ_IDX"])


# ==============================================================================
# GPU-Optimized Helper Functions
# ==============================================================================


def _decode_and_filter_step_gpu(
    out_pt: torch.Tensor,
    tgt_sub_columns: list[str],
    seq_len_max: int,
    seq_len_min: int,
    n_seed_steps: int,
    seq_step: int,
    has_slen: bool,
    has_ridx: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    GPU-accelerated positional decoding and sequence filtering.

    This function replaces the CPU-bound pandas operations with GPU tensor operations:
    - Decodes SIDX/SLEN/RIDX positional columns on GPU
    - Computes sequence continuation mask on GPU
    - Filters tensors on GPU using boolean indexing

    Args:
        out_pt: Model output tensor (batch, n_sub_cols, 1)
        tgt_sub_columns: List of sub-column names
        seq_len_max: Maximum sequence length
        seq_len_min: Minimum sequence length
        n_seed_steps: Number of seed steps
        seq_step: Current sequence step
        has_slen: Whether SLEN columns are present
        has_ridx: Whether RIDX columns are present
        device: Torch device

    Returns:
        Tuple of (filtered_tensor, include_mask, next_step_size, sidx, slen, ridx)
    """
    from mostlyai.engine._common import SIDX_RIDX_DIGIT_ENCODING_THRESHOLD

    # Build sub-column index mapping
    sub_col_idx = {name: idx for idx, name in enumerate(tgt_sub_columns)}

    # Decode SIDX
    if seq_len_max < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # Categorical encoding - find the cat column
        sidx_idx = None
        for name, idx in sub_col_idx.items():
            if name.startswith(SIDX_SUB_COLUMN_PREFIX) and name.endswith("cat"):
                sidx_idx = idx
                break
        sidx = out_pt[:, sidx_idx, 0] if sidx_idx is not None else torch.zeros(out_pt.size(0), device=device)
    else:
        # Digit encoding
        sidx = decode_positional_column_torch(
            out_pt.squeeze(-1),
            seq_len_max,
            SIDX_SUB_COLUMN_PREFIX,
            sub_col_idx,
        )

    # Decode SLEN if present
    slen = None
    if has_slen:
        if seq_len_max < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
            slen_idx = None
            for name, idx in sub_col_idx.items():
                if name.startswith(SLEN_SUB_COLUMN_PREFIX) and name.endswith("cat"):
                    slen_idx = idx
                    break
            if slen_idx is not None:
                slen = out_pt[:, slen_idx, 0]
                slen = torch.clamp(slen, min=seq_len_min)
        else:
            slen = decode_positional_column_torch(
                out_pt.squeeze(-1),
                seq_len_max,
                SLEN_SUB_COLUMN_PREFIX,
                sub_col_idx,
            )
            slen = torch.clamp(slen, min=seq_len_min)

    # Decode RIDX if present
    ridx = None
    if has_ridx:
        if seq_len_max < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
            ridx_idx = None
            for name, idx in sub_col_idx.items():
                if name.startswith(RIDX_SUB_COLUMN_PREFIX) and name.endswith("cat"):
                    ridx_idx = idx
                    break
            if ridx_idx is not None:
                ridx = out_pt[:, ridx_idx, 0]
                ridx = torch.clamp(ridx, min=seq_len_min - seq_step, max=seq_len_max)
        else:
            ridx = decode_positional_column_torch(
                out_pt.squeeze(-1),
                seq_len_max,
                RIDX_SUB_COLUMN_PREFIX,
                sub_col_idx,
            )
            ridx = torch.clamp(ridx, min=seq_len_min - seq_step, max=seq_len_max)

    # Create dummy slen if not present (for mask computation)
    if slen is None:
        slen = sidx + 1

    # Compute continue mask
    include_mask = compute_sequence_continue_mask_torch(sidx, slen, ridx, n_seed_steps)

    # Filter tensor
    next_step_size = include_mask.sum().item()
    if next_step_size < out_pt.size(0):
        filtered_tensor = out_pt[include_mask]
    else:
        filtered_tensor = out_pt

    return filtered_tensor, include_mask, next_step_size, sidx, slen, ridx


# ==============================================================================
# Main Sequential Generation Function
# ==============================================================================


@torch.no_grad()
def generate_sequential_core(
    artifact: ModelArtifact,
    sample_size: int,
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic sequential (longitudinal) data from a ModelArtifact.

    This generates multi-row sequences per context record (e.g., transactions per user).

    Args:
        artifact: ModelArtifact containing model weights and stats
        sample_size: Number of contexts (sequences) to generate
        ctx_data: Optional context data for conditional generation
        seed_data: Optional seed data to condition initial sequence steps
        sampling_temperature: Temperature for sampling (higher = more random)
        sampling_top_p: Nucleus sampling probability threshold
        device: Device for generation ("cuda" or "cpu")
        batch_size: Batch size for generation (auto-determined if None)

    Returns:
        DataFrame with generated synthetic sequences
    """
    if not artifact.is_sequential:
        raise ValueError("Use generate_flat_core() for flat models")

    device = _resolve_device(device)
    _LOG.info(f"Generating {sample_size} sequences on {device}")

    # Extract config from artifact
    tgt_stats = artifact.tgt_stats
    ctx_stats = artifact.ctx_stats
    tgt_context_key = artifact.tgt_context_key
    ctx_primary_key = artifact.ctx_primary_key
    tgt_primary_key = artifact.tgt_primary_key

    seq_len_min = artifact.tgt_seq_len_min or 1
    seq_len_max = artifact.tgt_seq_len_max or 1

    column_order = get_columns_from_cardinalities(artifact.tgt_cardinalities)
    tgt_sub_columns = get_sub_columns_from_cardinalities(artifact.tgt_cardinalities)

    # Detect positional column variants
    has_slen = any(SLEN_SUB_COLUMN_PREFIX in k for k in artifact.tgt_cardinalities.keys())
    has_ridx = any(RIDX_SUB_COLUMN_PREFIX in k for k in artifact.tgt_cardinalities.keys())
    has_sdec = any(SDEC_SUB_COLUMN_PREFIX in k for k in artifact.tgt_cardinalities.keys())
    if not (has_slen or has_ridx or has_sdec):
        has_slen, has_ridx, has_sdec = DEFAULT_HAS_SLEN, DEFAULT_HAS_RIDX, DEFAULT_HAS_SDEC

    fixed_probs = _compute_fixed_probs_for_rare_suppression(tgt_stats)
    model = _create_model_from_artifact(artifact, device, column_order)

    # Prepare context
    has_context = bool(ctx_stats and ctx_stats.get("columns"))
    if has_context and ctx_data is not None:
        ctx_data = ctx_data.reset_index(drop=True).head(sample_size)
        sample_size = len(ctx_data)
        ctx_inputs, ctx_encoded, ctx_pk_encoded = _encode_context_data(
            ctx_data, ctx_stats, ctx_primary_key, device
        )
        ctx_keys = ctx_encoded[ctx_pk_encoded].rename(tgt_context_key)
    else:
        ctx_primary_key = tgt_context_key or DUMMY_CONTEXT_KEY
        tgt_context_key = ctx_primary_key
        ctx_keys = _generate_primary_keys(sample_size, type="int")
        ctx_keys.rename(tgt_context_key, inplace=True)
        ctx_inputs = {}

    # Prepare seed data
    if seed_data is None:
        seed_data = pd.DataFrame(columns=[tgt_context_key])
    if tgt_context_key not in seed_data.columns:
        seed_data = seed_data.assign(**{tgt_context_key: pd.Series(dtype="object")})

    # Encode seed data
    seed_grouped = None
    n_seed_steps = 0
    if len(seed_data) > 0:
        # Truncate to max length
        grouped = seed_data.groupby(tgt_context_key, group_keys=False)
        if grouped.size().max() > seq_len_max:
            seed_data = grouped.apply(lambda x: x.iloc[:seq_len_max]).reset_index(drop=True)
        seed_grouped = seed_data.groupby(tgt_context_key, sort=False)
        n_seed_steps = seed_grouped.size().max()

    # Generate
    if batch_size is None:
        batch_size = min(sample_size, 2048)

    all_step_tensors: list[torch.Tensor] = []
    all_step_keys: list[pd.Series] = []
    all_seed_dfs: list[pd.DataFrame] = []

    num_batches = (sample_size + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, sample_size)
        batch_ctx_keys = ctx_keys.iloc[start:end].reset_index(drop=True)
        step_size = len(batch_ctx_keys)

        # Prepare context for this batch
        if has_context and ctx_inputs:
            batch_ctx_inputs = {k: v[start:end] for k, v in ctx_inputs.items()}
            context = model.context_compressor(batch_ctx_inputs)
        else:
            context = None

        history = None
        history_state = None
        step_ctx_keys = batch_ctx_keys.copy()
        out_df: pd.DataFrame | None = None

        # Generate step by step
        for seq_step in range(seq_len_max):
            if step_size == 0:
                break

            # Get seed for this step
            seed_step = pd.DataFrame()
            seed_step_encoded = pd.DataFrame()
            if seed_grouped is not None and seq_step < n_seed_steps:
                seed_step = seed_grouped.nth(seq_step)
                seed_step = seed_step[seed_step.index.isin(step_ctx_keys)]
                if len(seed_step) > 0:
                    seed_step_encoded, _, _ = encode_df(
                        df=seed_step.reset_index(),
                        stats={"columns": tgt_stats.get("columns", {}), "is_sequential": True},
                        tgt_context_key=tgt_context_key,
                    )

            # Build fixed values
            fixed_values = _build_positional_fixed_values(
                seq_step=seq_step,
                step_size=step_size,
                seq_len_max=seq_len_max,
                seq_len_min=seq_len_min,
                n_seed_steps=n_seed_steps,
                out_df=out_df,
                has_slen=has_slen,
                has_ridx=has_ridx,
                has_sdec=has_sdec,
                device=device,
            )

            # Add seed values
            if len(seed_step_encoded) > 0:
                for col in seed_step_encoded.columns:
                    if col in tgt_sub_columns:
                        fixed_values[col] = torch.unsqueeze(
                            torch.as_tensor(seed_step_encoded[col].to_numpy(), device=device).type(torch.int),
                            dim=-1,
                        )

            # Forward pass
            out_dct, history, history_state = model(
                x=None,
                mode="gen",
                batch_size=step_size,
                fixed_probs=fixed_probs,
                fixed_values=fixed_values,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                history=history,
                history_state=history_state,
                context=context,
                column_order=column_order,
            )

            out_pt = torch.stack(list(out_dct.values()), dim=0).transpose(0, 1)

            # Decode positional columns and compute mask on GPU
            (
                filtered_pt,
                include_mask,
                next_step_size,
                sidx_decoded,
                slen_decoded,
                ridx_decoded,
            ) = _decode_and_filter_step_gpu(
                out_pt=out_pt,
                tgt_sub_columns=tgt_sub_columns,
                seq_len_max=seq_len_max,
                seq_len_min=seq_len_min,
                n_seed_steps=n_seed_steps,
                seq_step=seq_step,
                has_slen=has_slen,
                has_ridx=has_ridx,
                device=device,
            )

            # Create minimal DataFrame with only decoded positional columns
            # This is needed by _build_positional_fixed_values in the next iteration
            # We only transfer the decoded scalars to CPU, not the full tensor
            out_df_data = {SIDX_SUB_COLUMN_PREFIX: sidx_decoded.cpu().numpy()}
            if has_slen and slen_decoded is not None:
                out_df_data[SLEN_SUB_COLUMN_PREFIX] = slen_decoded.cpu().numpy()
            if has_ridx and ridx_decoded is not None:
                out_df_data[RIDX_SUB_COLUMN_PREFIX] = ridx_decoded.cpu().numpy()
            out_df = pd.DataFrame(out_df_data)

            # Filter for next iteration (must happen BEFORE storing results to match legacy behavior)
            # The include_mask determines which sequences should CONTINUE to the next step.
            # Sequences with RIDX=0 have completed and should NOT be included in this step's output.
            if step_size > next_step_size or next_step_size == 0:
                step_size = next_step_size
                # Convert mask to pandas for compatibility with step_ctx_keys filtering
                include_mask_cpu = include_mask.cpu().numpy()
                step_ctx_keys = step_ctx_keys[include_mask_cpu].reset_index(drop=True)
                out_df = out_df[include_mask_cpu].reset_index(drop=True)
                out_pt = filtered_pt
                if len(seed_step) > 0:
                    seed_step = seed_step[seed_step[tgt_context_key].isin(step_ctx_keys)].reset_index(drop=True)
                if step_size > 0:
                    # For _filter_sequence_state, create a pandas Series wrapper
                    include_mask_series = pd.Series(include_mask_cpu)
                    context, history, history_state = _filter_sequence_state(
                        include_mask_series, context, history, history_state
                    )
            else:
                out_df = out_df[include_mask.cpu().numpy()].reset_index(drop=True)
                out_pt = filtered_pt

            # Store results AFTER filtering (to exclude sequences that ended this step)
            all_step_tensors.append(out_pt)
            all_step_keys.append(step_ctx_keys)
            all_seed_dfs.append(seed_step.reset_index(drop=True) if len(seed_step) > 0 else pd.DataFrame())

            # Exit if no sequences remain
            if step_size == 0:
                break

    # Combine and decode all results
    if len(all_step_tensors) == 0:
        return pd.DataFrame(columns=[tgt_context_key])

    encoded_df = _reshape_pt_to_pandas(all_step_tensors, tgt_sub_columns, all_step_keys, tgt_context_key)
    encoded_df = encoded_df.drop(
        columns=[c for c in encoded_df.columns if c.startswith(POSITIONAL_COLUMN)],
        errors="ignore",
    )

    _LOG.info(f"Decoding {len(encoded_df)} rows")
    decoded_df = _decode_sequential_df(
        encoded_df,
        {"columns": tgt_stats.get("columns", {}), "is_sequential": True},
        context_key=tgt_context_key,
        prev_steps={},
    )

    decoded_df = _restore_seed_values_sequential(decoded_df, all_seed_dfs, tgt_context_key)

    if tgt_primary_key and tgt_primary_key not in decoded_df.columns:
        decoded_df[tgt_primary_key] = _generate_primary_keys(len(decoded_df), type="uuid")

    if DUMMY_CONTEXT_KEY in decoded_df.columns:
        decoded_df = decoded_df.drop(columns=[DUMMY_CONTEXT_KEY])

    n_sequences = decoded_df[tgt_context_key].nunique() if tgt_context_key in decoded_df.columns else 0
    _LOG.info(f"Generated {len(decoded_df)} rows across {n_sequences} sequences")

    return decoded_df.reset_index(drop=True)
