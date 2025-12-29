"""
In-memory statistics computation for training without workspace I/O.

This module provides functions to compute column statistics directly from
DataFrames, designed for distributed cluster environments where data
streams through Ray Data pipelines.
"""

import logging
from typing import Any

import pandas as pd

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    TABLE_COLUMN_INFIX,
    TGT,
    is_a_list,
    is_sequential,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    analyze_categorical,
    analyze_reduce_categorical,
)
from mostlyai.engine._encoding_types.tabular.character import (
    analyze_character,
    analyze_reduce_character,
)
from mostlyai.engine._encoding_types.tabular.datetime import (
    analyze_datetime,
    analyze_reduce_datetime,
)
from mostlyai.engine._encoding_types.tabular.itt import (
    analyze_itt,
    analyze_reduce_itt,
)
from mostlyai.engine._encoding_types.tabular.lat_long import (
    analyze_latlong,
    analyze_reduce_latlong,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    analyze_numeric,
    analyze_reduce_numeric,
)
from mostlyai.engine.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)


def compute_stats(
    tgt_data: pd.DataFrame,
    tgt_encoding_types: dict[str, ModelEncodingType | str],
    *,
    tgt_context_key: str | None = None,
    ctx_data: pd.DataFrame | None = None,
    ctx_primary_key: str | None = None,
    ctx_encoding_types: dict[str, ModelEncodingType | str] | None = None,
) -> tuple[dict, dict | None]:
    """
    Compute column statistics directly from DataFrames.

    This is a simplified, in-memory alternative to the workspace-based analyze()
    function. It skips value protection (rare category replacement) and differential
    privacy, producing stats suitable for training and generation.

    Args:
        tgt_data: Target DataFrame for training
        tgt_encoding_types: Dict mapping column names to encoding types
        tgt_context_key: Column linking target to context (for sequential data)
        ctx_data: Optional context DataFrame
        ctx_primary_key: Primary key column in context
        ctx_encoding_types: Dict mapping context column names to encoding types

    Returns:
        Tuple of (tgt_stats, ctx_stats). ctx_stats is None if no context provided.

    Example:
        >>> tgt_stats, ctx_stats = compute_stats(
        ...     tgt_data=transactions_df,
        ...     tgt_encoding_types={"amount": "tabular_numeric_auto", "category": "tabular_categorical"},
        ...     tgt_context_key="user_id",
        ...     ctx_data=users_df,
        ...     ctx_primary_key="user_id",
        ...     ctx_encoding_types={"age": "tabular_numeric_auto"},
        ... )
    """
    # Normalize encoding types to ModelEncodingType
    tgt_encoding_types = _normalize_encoding_types(tgt_encoding_types)
    if ctx_encoding_types:
        ctx_encoding_types = _normalize_encoding_types(ctx_encoding_types)

    has_context = ctx_data is not None and ctx_encoding_types is not None

    # Compute target stats
    tgt_stats = _compute_table_stats(
        df=tgt_data,
        encoding_types=tgt_encoding_types,
        context_key=tgt_context_key,
        mode="tgt",
    )

    # Add sequence length stats
    if tgt_context_key:
        seq_lens = tgt_data.groupby(tgt_context_key).size()
        tgt_stats["seq_len"] = {
            "min": int(seq_lens.min()),
            "max": int(seq_lens.max()),
            "median": int(seq_lens.median()),
            "value_protection": False,
        }
        tgt_stats["is_sequential"] = tgt_stats["seq_len"]["max"] > 1
    else:
        tgt_stats["seq_len"] = {"min": 1, "max": 1, "median": 1, "value_protection": False}
        tgt_stats["is_sequential"] = False

    # Add record counts (use context if available, otherwise unique context keys)
    if has_context:
        n_records = len(ctx_data)
    elif tgt_context_key:
        n_records = tgt_data[tgt_context_key].nunique()
    else:
        n_records = len(tgt_data)

    # Simple 90/10 split assumption
    tgt_stats["no_of_training_records"] = int(n_records * 0.9)
    tgt_stats["no_of_validation_records"] = n_records - tgt_stats["no_of_training_records"]

    # Add keys
    tgt_stats["keys"] = {
        "context_key": tgt_context_key,
        "primary_key": None,
    }
    tgt_stats["value_protection_epsilon_spent"] = None

    # Compute context stats if provided
    ctx_stats = None
    if has_context:
        ctx_stats = _compute_table_stats(
            df=ctx_data,
            encoding_types=ctx_encoding_types,
            context_key=None,
            mode="ctx",
        )
        ctx_stats["keys"] = {
            "primary_key": ctx_primary_key,
            "root_key": None,
        }
        ctx_stats["value_protection_epsilon_spent"] = None

    return tgt_stats, ctx_stats


def _normalize_encoding_types(
    encoding_types: dict[str, ModelEncodingType | str]
) -> dict[str, ModelEncodingType]:
    """Convert encoding types to ModelEncodingType."""
    result = {}
    for col, enc_type in encoding_types.items():
        if isinstance(enc_type, ModelEncodingType):
            result[col] = enc_type
        elif isinstance(enc_type, str):
            result[col] = ModelEncodingType(enc_type)
        else:
            raise ValueError(f"Invalid encoding type for {col}: {enc_type}")
    return result


def _compute_table_stats(
    df: pd.DataFrame,
    encoding_types: dict[str, ModelEncodingType],
    context_key: str | None,
    mode: str,
) -> dict:
    """Compute stats for a single table (target or context)."""
    stats: dict[str, Any] = {"columns": {}}

    # Create dummy keys for analysis
    if context_key and context_key in df.columns:
        context_keys = df[context_key].rename("__ckey")
    else:
        context_keys = pd.Series(range(len(df)), name="__ckey")

    root_keys = pd.Series(range(len(df)), name="__rkey")

    # Build ARGN identifiers
    unique_tables = _get_unique_tables(encoding_types.keys())

    for col_idx, (column, encoding_type) in enumerate(encoding_types.items()):
        values = df[column]

        # Analyze column using existing functions
        col_stats = _analyze_column(values, encoding_type, root_keys, context_keys)

        # Reduce (finalize) stats without value protection
        col_stats = _reduce_column_stats(col_stats, encoding_type)

        # Add ARGN identifiers
        table_name = _get_table(column)
        table_idx = unique_tables.index(table_name) if table_name in unique_tables else 0

        col_stats[ARGN_PROCESSOR] = _get_argn_processor(mode, is_flat=True)
        col_stats[ARGN_TABLE] = f"t{table_idx}"
        col_stats[ARGN_COLUMN] = f"c{col_idx}"

        # Set encoding_type if not already set by reduce function
        # (numeric reduce functions set the resolved type, others don't)
        if "encoding_type" not in col_stats:
            col_stats["encoding_type"] = encoding_type.value

        # Mark no value protection
        col_stats["value_protection"] = False

        stats["columns"][column] = col_stats
        _LOG.debug(f"Analyzed column `{column}`: {col_stats['encoding_type']}, cardinalities={col_stats.get('cardinalities', {})}")

    return stats


def _get_table(column_name: str) -> str:
    """Extract table name from qualified column name."""
    if TABLE_COLUMN_INFIX in column_name:
        return column_name.split(TABLE_COLUMN_INFIX)[0]
    return "t0"


def _get_unique_tables(column_names) -> list[str]:
    """Get unique table names preserving order."""
    tables = [_get_table(c) for c in column_names]
    return list(dict.fromkeys(tables))


def _get_argn_processor(mode: str, is_flat: bool) -> str:
    """Get ARGN processor identifier."""
    if mode == "tgt":
        return TGT
    return CTXFLT if is_flat else CTXSEQ


def _analyze_column(
    values: pd.Series,
    encoding_type: ModelEncodingType,
    root_keys: pd.Series,
    context_keys: pd.Series,
) -> dict:
    """Analyze a single column using existing analysis functions."""
    if values.empty:
        return {"encoding_type": encoding_type.value}

    # Handle sequential columns (lists)
    if is_sequential(values):
        non_empties = values.apply(lambda v: len(v) if is_a_list(v) else 1) > 0
        df = pd.concat([values[non_empties], root_keys[non_empties], context_keys[non_empties]], axis=1)
        df = df.explode(values.name).reset_index(drop=True)
        values = df[values.name]
        root_keys = df[root_keys.name]
        context_keys = df[context_keys.name]

    # Dispatch to encoding-type specific analysis
    if encoding_type == ModelEncodingType.tabular_categorical:
        return analyze_categorical(values, root_keys, context_keys)
    elif encoding_type in (
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_digit,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
    ):
        return analyze_numeric(values, root_keys, context_keys, encoding_type)
    elif encoding_type == ModelEncodingType.tabular_datetime:
        return analyze_datetime(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_datetime_relative:
        return analyze_itt(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_character:
        return analyze_character(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_lat_long:
        return analyze_latlong(values, root_keys, context_keys)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def _reduce_column_stats(
    partial_stats: dict,
    encoding_type: ModelEncodingType,
) -> dict:
    """Reduce (finalize) column stats without value protection."""
    # Wrap in list as reduce functions expect multiple partitions
    stats_list = [partial_stats]

    # Dispatch to encoding-type specific reduce
    if encoding_type == ModelEncodingType.tabular_categorical:
        return analyze_reduce_categorical(stats_list, value_protection=False)
    elif encoding_type in (
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_digit,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
    ):
        return analyze_reduce_numeric(stats_list, value_protection=False, encoding_type=encoding_type)
    elif encoding_type == ModelEncodingType.tabular_datetime:
        return analyze_reduce_datetime(stats_list, value_protection=False)
    elif encoding_type == ModelEncodingType.tabular_datetime_relative:
        return analyze_reduce_itt(stats_list, value_protection=False)
    elif encoding_type == ModelEncodingType.tabular_character:
        return analyze_reduce_character(stats_list, value_protection=False)
    elif encoding_type == ModelEncodingType.tabular_lat_long:
        return analyze_reduce_latlong(stats_list, value_protection=False)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
