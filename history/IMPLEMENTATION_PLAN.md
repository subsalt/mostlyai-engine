# Slim ARGN Implementation Plan

## Executive Summary

This document outlines the implementation plan for optimizing the mostlyai-engine fork for distributed cluster environments. The primary goals are:

1. **Eliminate disk I/O during inference** - Replace workspace-based generation with in-memory artifact-based generation
2. **Reduce model artifact size** - Remove training-only state (optimizer, scheduler, progress logs)
3. **Streamline training** - Remove privacy protection overhead, simplify stats computation
4. **Enable distributed training** - Support Ray Data streaming for large datasets

## Architecture Overview

### Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DataFrame ──► split() ──► parquet files ──► analyze() ──► stats JSON       │
│                    │                              │                          │
│                    ▼                              ▼                          │
│              workspace/                    workspace/                        │
│              OriginalData/                 ModelStore/                       │
│              tgt-data/                     tgt-stats/                        │
│                    │                              │                          │
│                    ▼                              ▼                          │
│              encode() ──► encoded parquet ──► train() ──► model weights      │
│                                                   │                          │
│                                                   ▼                          │
│                                            workspace/                        │
│                                            ModelStore/                       │
│                                            model-data/                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  tarball ──► extract to temp dir ──► engine.generate(workspace_dir)         │
│                                              │                               │
│                                              ▼                               │
│                                       read stats JSON                        │
│                                       load model weights                     │
│                                       encode context (if any)                │
│                                       generate batches                       │
│                                       decode to DataFrame                    │
│                                       write parquet                          │
│                                              │                               │
│                                              ▼                               │
│                                       read parquet ──► return DataFrame      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DataFrame/Iterator ──► compute_stats() ──► stats dict (in memory)          │
│         │                                        │                           │
│         ▼                                        ▼                           │
│    encode_dataframe() ──► encoded DataFrame ──► train_core() ──► weights    │
│                                                      │                       │
│                                                      ▼                       │
│                                              ModelArtifact                   │
│                                              (weights + stats + config)      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ModelArtifact ──► reconstruct model ──► generate_core() ──► DataFrame      │
│       │                   │                    │                             │
│       │                   ▼                    ▼                             │
│       │            load weights          encode context                      │
│       │            into model            generate batches                    │
│       │                                  decode in memory                    │
│       │                                        │                             │
│       ▼                                        ▼                             │
│  All in memory, no disk I/O            return DataFrame directly             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Inference Optimization

**Goal**: Eliminate disk I/O during inference, reduce artifact size by ~50%

**Estimated Scope**: ~500-700 lines of new code, ~200 lines of modifications

### 1.1 ModelArtifact Data Structure

Create a new data structure that contains everything needed for inference, nothing more.

```python
# File: mostlyai/engine/_artifact.py

from dataclasses import dataclass, field
from typing import Any
import io
import torch
import json
import zlib

@dataclass
class ModelArtifact:
    """
    Minimal artifact containing everything needed for inference.

    This replaces the workspace-based storage with a single serializable object.
    """

    # === Model Weights ===
    weights: bytes  # Compressed torch state_dict

    # === Model Architecture ===
    is_sequential: bool
    model_size: str  # "S", "M", "L"
    tgt_cardinalities: dict[str, int]  # {"tgt:col1/cat": 10, ...}
    ctx_cardinalities: dict[str, int]  # {"ctxflt/col1/cat": 5, ...}
    enable_flexible_generation: bool = False

    # === Sequential Model Parameters (optional) ===
    tgt_seq_len_min: int | None = None
    tgt_seq_len_max: int | None = None
    tgt_seq_len_median: int | None = None
    ctx_seq_len_median: dict[str, int] = field(default_factory=dict)

    # === Statistics for Encoding/Decoding ===
    tgt_stats: dict[str, Any] = field(default_factory=dict)  # Minimal stats per column
    ctx_stats: dict[str, Any] = field(default_factory=dict)

    # === Key Columns ===
    tgt_primary_key: str | None = None
    tgt_context_key: str | None = None
    ctx_primary_key: str | None = None

    # === Metadata ===
    version: int = 1

    def to_bytes(self) -> bytes:
        """Serialize artifact to bytes for storage."""
        # Separate weights (already compressed) from metadata
        metadata = {
            "version": self.version,
            "is_sequential": self.is_sequential,
            "model_size": self.model_size,
            "tgt_cardinalities": self.tgt_cardinalities,
            "ctx_cardinalities": self.ctx_cardinalities,
            "enable_flexible_generation": self.enable_flexible_generation,
            "tgt_seq_len_min": self.tgt_seq_len_min,
            "tgt_seq_len_max": self.tgt_seq_len_max,
            "tgt_seq_len_median": self.tgt_seq_len_median,
            "ctx_seq_len_median": self.ctx_seq_len_median,
            "tgt_stats": self.tgt_stats,
            "ctx_stats": self.ctx_stats,
            "tgt_primary_key": self.tgt_primary_key,
            "tgt_context_key": self.tgt_context_key,
            "ctx_primary_key": self.ctx_primary_key,
        }
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        metadata_compressed = zlib.compress(metadata_bytes, level=6)

        # Pack format: [metadata_len (4 bytes)][metadata][weights]
        metadata_len = len(metadata_compressed)
        return metadata_len.to_bytes(4, "little") + metadata_compressed + self.weights

    @classmethod
    def from_bytes(cls, data: bytes) -> "ModelArtifact":
        """Deserialize artifact from bytes."""
        metadata_len = int.from_bytes(data[:4], "little")
        metadata_compressed = data[4:4+metadata_len]
        weights = data[4+metadata_len:]

        metadata_bytes = zlib.decompress(metadata_compressed)
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        return cls(weights=weights, **metadata)

    @classmethod
    def from_workspace(cls, workspace_dir: str | Path) -> "ModelArtifact":
        """
        Load artifact from existing workspace directory.

        This enables migration from the old format to the new format.
        """
        # Implementation in section 1.2
        ...

    def get_weights_state_dict(self, device: str = "cpu") -> dict:
        """Decompress and load weights as state_dict."""
        weights_decompressed = zlib.decompress(self.weights)
        buffer = io.BytesIO(weights_decompressed)
        return torch.load(buffer, map_location=device, weights_only=True)
```

### 1.2 Stats Minimization

Current stats JSON contains many fields only needed for privacy protection. Create minimal stats structure.

**Current stats structure (example categorical column):**
```json
{
  "encoding_type": "tabular_categorical",
  "has_nan": true,
  "cnt_values": {"A": 1000, "B": 500, ...},  // REMOVE: only for value protection
  "no_of_rare_categories": 5,                // REMOVE: only for logging
  "value_protection": true,                  // REMOVE: flag we don't need
  "codes": {"_RARE_": 0, "<<NULL>>": 1, "A": 2, "B": 3, ...},
  "cardinalities": {"cat": 100},
  "argn_processor": "tgt",
  "argn_table": "t0",
  "argn_column": "c0"
}
```

**Minimal stats structure:**
```json
{
  "encoding_type": "tabular_categorical",
  "codes": {"_RARE_": 0, "<<NULL>>": 1, "A": 2, "B": 3, ...},
  "cardinalities": {"cat": 100},
  "argn_processor": "tgt",
  "argn_table": "t0",
  "argn_column": "c0"
}
```

```python
# File: mostlyai/engine/_artifact.py (continued)

def minimize_column_stats(column_stats: dict) -> dict:
    """
    Extract only the fields needed for encoding/decoding.

    Removes value protection metadata, raw counts, and other training-only fields.
    """
    essential_fields = {
        "encoding_type",
        "codes",           # Category -> int mapping
        "cardinalities",   # Vocab sizes per sub-column
        "argn_processor",  # tgt, ctxflt, ctxseq
        "argn_table",
        "argn_column",
        # Numeric fields
        "bins",            # Bin edges for binned encoding
        "min", "max",      # Range for digit encoding
        "has_nan", "has_na",  # Null handling
        # Datetime fields
        "epoch",
        "resolution",
        # Character fields
        "charset",
        "max_len",
        # Lat/long fields
        "precision",
        # Sequential context
        "seq_len",         # Only min/max/median needed
    }

    minimal = {}
    for key, value in column_stats.items():
        if key in essential_fields:
            # For seq_len, only keep min/max/median
            if key == "seq_len" and isinstance(value, dict):
                minimal[key] = {
                    k: v for k, v in value.items()
                    if k in ("min", "max", "median")
                }
            else:
                minimal[key] = value

    return minimal


def minimize_stats(full_stats: dict) -> dict:
    """
    Minimize full stats dict (as returned by analyze()).

    Keeps only what's needed for encoding/decoding.
    """
    minimal = {
        "columns": {
            col_name: minimize_column_stats(col_stats)
            for col_name, col_stats in full_stats.get("columns", {}).items()
        }
    }

    # Keep sequence length info for sequential models
    if "seq_len" in full_stats:
        minimal["seq_len"] = {
            k: v for k, v in full_stats["seq_len"].items()
            if k in ("min", "max", "median")
        }

    # Keep key column names
    if "keys" in full_stats:
        minimal["keys"] = full_stats["keys"]

    # Keep is_sequential flag
    if "is_sequential" in full_stats:
        minimal["is_sequential"] = full_stats["is_sequential"]

    return minimal
```

### 1.3 Workspace-to-Artifact Migration

```python
# File: mostlyai/engine/_artifact.py (continued)

@classmethod
def from_workspace(cls, workspace_dir: str | Path) -> "ModelArtifact":
    """
    Load artifact from existing workspace directory.

    This enables migration from the old tarball format to the new artifact format.
    """
    from mostlyai.engine._workspace import Workspace
    import zlib
    import io

    workspace = Workspace(Path(workspace_dir))

    # Load stats
    tgt_stats_full = workspace.tgt_stats.read()
    ctx_stats_full = (
        workspace.ctx_stats.read()
        if workspace.ctx_stats.path.exists()
        else {}
    )

    # Load model configs
    model_configs = workspace.model_configs.read()

    # Load and compress weights
    weights_path = workspace.model_tabular_weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    weights_compressed = zlib.compress(buffer.getvalue(), level=6)

    # Extract cardinalities
    from mostlyai.engine._common import get_cardinalities, get_sequence_length_stats

    is_sequential = tgt_stats_full.get("is_sequential", False)
    tgt_cardinalities = get_cardinalities(tgt_stats_full)
    ctx_cardinalities = get_cardinalities(ctx_stats_full) if ctx_stats_full else {}

    # Build artifact
    artifact = cls(
        weights=weights_compressed,
        is_sequential=is_sequential,
        model_size=model_configs.get("model_id", "MOSTLY_AI/Medium").split("/")[-1][0],  # "M"
        tgt_cardinalities=tgt_cardinalities,
        ctx_cardinalities=ctx_cardinalities,
        enable_flexible_generation=model_configs.get("enable_flexible_generation", False),
        tgt_stats=minimize_stats(tgt_stats_full),
        ctx_stats=minimize_stats(ctx_stats_full) if ctx_stats_full else {},
        tgt_primary_key=tgt_stats_full.get("keys", {}).get("primary_key"),
        tgt_context_key=tgt_stats_full.get("keys", {}).get("context_key"),
        ctx_primary_key=ctx_stats_full.get("keys", {}).get("primary_key") if ctx_stats_full else None,
    )

    # Add sequential params if needed
    if is_sequential:
        seq_len_stats = get_sequence_length_stats(tgt_stats_full)
        artifact.tgt_seq_len_min = seq_len_stats.get("min", 1)
        artifact.tgt_seq_len_max = seq_len_stats.get("max", 1)
        artifact.tgt_seq_len_median = seq_len_stats.get("median", 1)

        if ctx_stats_full:
            from mostlyai.engine._common import get_ctx_sequence_length
            artifact.ctx_seq_len_median = get_ctx_sequence_length(ctx_stats_full, key="median")

    return artifact
```

### 1.4 In-Memory Generation Functions

The core of Phase 1: generate synthetic data without any disk I/O.

```python
# File: mostlyai/engine/generation.py (new functions)

import torch
import pandas as pd
from mostlyai.engine._artifact import ModelArtifact


@torch.no_grad()
def generate_flat(
    artifact: ModelArtifact,
    sample_size: int,
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic flat (non-sequential) data from a ModelArtifact.

    This is the in-memory replacement for generate() that eliminates disk I/O.

    Args:
        artifact: ModelArtifact containing model weights and stats
        sample_size: Number of samples to generate
        ctx_data: Optional context data (if model was trained with context)
        seed_data: Optional seed data to condition generation
        sampling_temperature: Temperature for sampling (higher = more random)
        sampling_top_p: Nucleus sampling probability threshold
        device: Device for generation ("cuda" or "cpu")
        batch_size: Batch size for generation (auto-determined if None)

    Returns:
        DataFrame with generated synthetic data

    Example:
        >>> artifact = ModelArtifact.from_bytes(model_bytes)
        >>> df = generate_flat(artifact, sample_size=1000, device="cuda")
    """
    if artifact.is_sequential:
        raise ValueError("Use generate_sequential() for sequential models")

    # Implementation details in section 1.5
    ...


@torch.no_grad()
def generate_sequential(
    artifact: ModelArtifact,
    *,
    ctx_data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    seed_data: pd.DataFrame | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic sequential (longitudinal) data from a ModelArtifact.

    For sequential models, sample_size refers to the number of sequences (entities),
    not the total number of rows. The actual row count depends on generated sequence
    lengths.

    Args:
        artifact: ModelArtifact containing model weights and stats
        ctx_data: Context data for conditional generation (required if model has context)
        sample_size: Number of sequences to generate (defaults to len(ctx_data) or training size)
        seed_data: Optional seed data to condition generation
        sampling_temperature: Temperature for sampling
        sampling_top_p: Nucleus sampling probability threshold
        device: Device for generation
        batch_size: Batch size for generation

    Returns:
        DataFrame with generated synthetic data (multiple rows per sequence)

    Example:
        >>> # Generate entities first
        >>> entities_artifact = ModelArtifact.from_bytes(entities_bytes)
        >>> entities_df = generate_flat(entities_artifact, sample_size=1000)
        >>>
        >>> # Generate events conditioned on entities
        >>> events_artifact = ModelArtifact.from_bytes(events_bytes)
        >>> events_df = generate_sequential(events_artifact, ctx_data=entities_df)
    """
    if not artifact.is_sequential:
        raise ValueError("Use generate_flat() for flat models")

    # Implementation details in section 1.5
    ...
```

### 1.5 Generation Core Implementation

Extract and refactor the core generation logic from `_tabular/generation.py` to work with artifacts.

```python
# File: mostlyai/engine/_tabular/generation_core.py

"""
Core generation functions that work with ModelArtifact instead of workspace.

This module contains the refactored generation logic extracted from generation.py,
modified to work entirely in memory without disk I/O.
"""

import logging
import torch
import pandas as pd
import numpy as np
from typing import Any

from mostlyai.engine._artifact import ModelArtifact
from mostlyai.engine._tabular.argn import FlatModel, SequentialModel, ModelSize
from mostlyai.engine._common import (
    get_columns_from_cardinalities,
    get_sub_columns_from_cardinalities,
)

_LOG = logging.getLogger(__name__)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve device specification to torch.device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _create_model_from_artifact(
    artifact: ModelArtifact,
    device: torch.device,
    column_order: list[str] | None = None,
) -> FlatModel | SequentialModel:
    """
    Reconstruct model from artifact and load weights.

    This replaces create_and_load_model() which requires workspace.
    """
    model_sizes = {"S": ModelSize.S, "M": ModelSize.M, "L": ModelSize.L}
    model_size = model_sizes.get(artifact.model_size, ModelSize.M)

    model_kwargs = {
        "tgt_cardinalities": artifact.tgt_cardinalities,
        "ctx_cardinalities": artifact.ctx_cardinalities,
        "ctxseq_len_median": artifact.ctx_seq_len_median,
        "model_size": model_size,
        "column_order": column_order,
        "device": device,
        "with_dp": False,
        "empirical_probs_for_predictor_init": None,
    }

    if artifact.is_sequential:
        model = SequentialModel(
            **model_kwargs,
            tgt_seq_len_median=artifact.tgt_seq_len_median or 1,
            tgt_seq_len_max=artifact.tgt_seq_len_max or 1,
        )
    else:
        model = FlatModel(**model_kwargs)

    # Load weights
    state_dict = artifact.get_weights_state_dict(device=str(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def _encode_context_data(
    ctx_data: pd.DataFrame,
    ctx_stats: dict,
    ctx_primary_key: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, str]:
    """
    Encode context data to tensors for generation.

    This is extracted from prepare_context_inputs() to work without workspace.
    """
    from mostlyai.engine._tabular.encoding import encode_df
    from mostlyai.engine._tabular.common import prepare_context_inputs_from_encoded

    # Encode the context DataFrame
    ctx_encoded, ctx_primary_key_encoded, _ = encode_df(
        df=ctx_data,
        stats=ctx_stats,
        ctx_primary_key=ctx_primary_key,
    )

    # Convert to tensors
    ctx_inputs = prepare_context_inputs_from_encoded(
        ctx_encoded=ctx_encoded,
        ctx_stats=ctx_stats,
        device=device,
        ctx_primary_key_encoded=ctx_primary_key_encoded,
    )

    return ctx_inputs, ctx_encoded, ctx_primary_key_encoded


def _decode_generated_data(
    encoded_df: pd.DataFrame,
    tgt_stats: dict,
    context_key: str | None = None,
) -> pd.DataFrame:
    """
    Decode generated encoded data back to original values.

    This wraps _decode_df() from generation.py.
    """
    from mostlyai.engine._tabular.generation import _decode_df

    return _decode_df(
        df_encoded=encoded_df,
        stats=tgt_stats,
        context_key=context_key,
    )


def generate_flat_core(
    artifact: ModelArtifact,
    sample_size: int,
    ctx_data: pd.DataFrame | None,
    seed_data: pd.DataFrame | None,
    sampling_temperature: float,
    sampling_top_p: float,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """
    Core implementation of flat data generation.

    This contains the main generation loop, extracted and refactored
    from the original generate() function.
    """
    tgt_stats = artifact.tgt_stats
    ctx_stats = artifact.ctx_stats
    tgt_context_key = artifact.tgt_context_key
    ctx_primary_key = artifact.ctx_primary_key
    tgt_primary_key = artifact.tgt_primary_key

    # Get column info
    tgt_sub_columns = get_sub_columns_from_cardinalities(artifact.tgt_cardinalities)
    column_order = get_columns_from_cardinalities(artifact.tgt_cardinalities)

    # Create model
    model = _create_model_from_artifact(artifact, device, column_order)

    # Handle context
    has_context = bool(ctx_stats and ctx_stats.get("columns"))

    if has_context and ctx_data is not None:
        # Encode provided context
        ctx_inputs, ctx_encoded, ctx_pk_encoded = _encode_context_data(
            ctx_data, ctx_stats, ctx_primary_key, device
        )
        ctx_keys = ctx_encoded[ctx_pk_encoded].rename(tgt_context_key)
        sample_size = min(sample_size, len(ctx_data))
    else:
        # No context - create dummy keys
        from mostlyai.engine._tabular.generation import _generate_primary_keys, DUMMY_CONTEXT_KEY
        ctx_primary_key = tgt_context_key or DUMMY_CONTEXT_KEY
        tgt_context_key = ctx_primary_key
        ctx_keys = _generate_primary_keys(sample_size, type="int")
        ctx_keys.rename(tgt_context_key, inplace=True)
        ctx_inputs = {}

    # Encode seed data if provided
    seed_encoded = pd.DataFrame()
    if seed_data is not None and len(seed_data) > 0:
        from mostlyai.engine._tabular.encoding import encode_df
        seed_encoded, _, _ = encode_df(
            df=seed_data,
            stats={"columns": tgt_stats.get("columns", {}), "is_sequential": False},
            tgt_context_key=tgt_context_key,
        )

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
            batch_ctx_inputs = {
                k: v[start:end] for k, v in ctx_inputs.items()
            }
        else:
            batch_ctx_inputs = {}

        # Prepare fixed values from seed
        fixed_values = {}
        batch_seed = pd.DataFrame()
        if len(seed_encoded) > 0:
            batch_seed_mask = seed_data[tgt_context_key].isin(batch_ctx_keys)
            batch_seed = seed_data[batch_seed_mask].reset_index(drop=True)
            batch_seed_encoded = seed_encoded[batch_seed_mask].reset_index(drop=True)
            fixed_values = {
                col: torch.as_tensor(
                    batch_seed_encoded[col].to_numpy(),
                    device=device
                ).type(torch.int)
                for col in batch_seed_encoded.columns
                if col in tgt_sub_columns
            }

        # Forward pass
        out_dct, _ = model(
            batch_ctx_inputs if batch_ctx_inputs else None,
            mode="gen",
            batch_size=batch_len,
            fixed_probs={},  # No rebalancing/imputation
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
    all_seed = pd.concat(seed_results, ignore_index=True) if seed_results else pd.DataFrame()

    # Decode
    decoded_df = _decode_generated_data(
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
        from mostlyai.engine._tabular.generation import _generate_primary_keys
        decoded_df[tgt_primary_key] = _generate_primary_keys(len(decoded_df), type="uuid")

    # Drop dummy context key if present
    if DUMMY_CONTEXT_KEY in decoded_df.columns:
        decoded_df = decoded_df.drop(columns=[DUMMY_CONTEXT_KEY])

    return decoded_df.reset_index(drop=True)


def generate_sequential_core(
    artifact: ModelArtifact,
    ctx_data: pd.DataFrame | None,
    sample_size: int | None,
    seed_data: pd.DataFrame | None,
    sampling_temperature: float,
    sampling_top_p: float,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """
    Core implementation of sequential data generation.

    This is more complex due to the autoregressive sequence generation.
    The implementation follows the original generate() but without disk I/O.
    """
    # This is a longer implementation - key points:
    # 1. Process context to get compressed representation
    # 2. Loop over sequence steps (up to tgt_seq_len_max)
    # 3. At each step, generate next token conditioned on history
    # 4. Track which sequences are complete (RIDX=0)
    # 5. Decode all accumulated tokens at the end

    # Full implementation follows the pattern in generation.py lines 1031-1221
    # but using artifact instead of workspace

    ...  # Detailed implementation similar to flat but with sequence loop
```

### 1.6 Public API Updates

Update `__init__.py` to expose new functions:

```python
# File: mostlyai/engine/__init__.py (additions)

from mostlyai.engine._artifact import ModelArtifact
from mostlyai.engine.generation import generate_flat, generate_sequential

__all__ = [
    # ... existing exports ...

    # New artifact-based API
    "ModelArtifact",
    "generate_flat",
    "generate_sequential",
]
```

### 1.7 Testing Checkpoint: Phase 1

Create test script to validate artifact-based generation matches workspace-based generation.

```python
# File: mostlyai/engine/slim-argn.py

"""
Test script for slim ARGN implementation.

Run with: ./cluster-run.sh slim-argn.py

This script tests the artifact-based generation against the workspace-based
generation to ensure equivalence.
"""

import tempfile
import numpy as np
import pandas as pd
import torch

# Test configuration
SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def create_test_data(n_samples: int = 5000) -> pd.DataFrame:
    """Create synthetic test data for training."""
    np.random.seed(RANDOM_SEED)

    return pd.DataFrame({
        "cat_col": np.random.choice(["A", "B", "C", "D"], n_samples),
        "num_col": np.random.randn(n_samples) * 10 + 50,
        "int_col": np.random.randint(0, 100, n_samples),
    })


def test_phase1_artifact_generation():
    """
    Test Phase 1: Artifact-based generation.

    1. Train a model using the standard workflow
    2. Create ModelArtifact from workspace
    3. Generate using both methods
    4. Compare results statistically
    """
    from mostlyai import engine
    from mostlyai.engine import ModelArtifact, generate_flat
    from mostlyai.engine.domain import ModelEncodingType
    from mostlyai.engine.random_state import set_random_state

    print("=" * 60)
    print("PHASE 1 TEST: Artifact-based Generation")
    print("=" * 60)

    # Create test data
    print("\n[1/5] Creating test data...")
    df = create_test_data()
    print(f"  Created DataFrame with shape {df.shape}")

    # Train using standard workflow
    print("\n[2/5] Training model with standard workflow...")
    with tempfile.TemporaryDirectory(prefix="slim_argn_test_") as workspace_dir:
        encoding_types = {
            "cat_col": ModelEncodingType.tabular_categorical,
            "num_col": ModelEncodingType.tabular_numeric_auto,
            "int_col": ModelEncodingType.tabular_numeric_discrete,
        }

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types=encoding_types,
        )
        engine.analyze(
            workspace_dir=workspace_dir,
            value_protection=False,
        )
        engine.encode(workspace_dir=workspace_dir)
        engine.train(
            workspace_dir=workspace_dir,
            max_epochs=5,  # Short training for testing
            enable_flexible_generation=False,
        )

        print("  Training complete!")

        # Generate using workspace method
        print("\n[3/5] Generating with workspace method...")
        set_random_state(RANDOM_SEED)
        engine.generate(
            workspace_dir=workspace_dir,
            sample_size=SAMPLE_SIZE,
        )
        workspace_result = pd.read_parquet(f"{workspace_dir}/SyntheticData")
        print(f"  Generated {len(workspace_result)} samples")

        # Create artifact from workspace
        print("\n[4/5] Creating ModelArtifact from workspace...")
        artifact = ModelArtifact.from_workspace(workspace_dir)

        artifact_bytes = artifact.to_bytes()
        print(f"  Artifact size: {len(artifact_bytes):,} bytes")

        # Verify roundtrip
        artifact_restored = ModelArtifact.from_bytes(artifact_bytes)
        print(f"  Artifact roundtrip: OK")

        # Generate using artifact method
        print("\n[5/5] Generating with artifact method...")
        set_random_state(RANDOM_SEED)
        artifact_result = generate_flat(
            artifact=artifact_restored,
            sample_size=SAMPLE_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"  Generated {len(artifact_result)} samples")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    # Check columns match
    ws_cols = set(workspace_result.columns)
    art_cols = set(artifact_result.columns)
    print(f"\nColumn match: {ws_cols == art_cols}")
    if ws_cols != art_cols:
        print(f"  Workspace only: {ws_cols - art_cols}")
        print(f"  Artifact only: {art_cols - ws_cols}")

    # Check categorical distribution
    for col in ["cat_col"]:
        ws_dist = workspace_result[col].value_counts(normalize=True).sort_index()
        art_dist = artifact_result[col].value_counts(normalize=True).sort_index()
        max_diff = abs(ws_dist - art_dist).max()
        print(f"\n{col} distribution max diff: {max_diff:.4f}")

    # Check numeric statistics
    for col in ["num_col", "int_col"]:
        ws_mean = workspace_result[col].mean()
        art_mean = artifact_result[col].mean()
        ws_std = workspace_result[col].std()
        art_std = artifact_result[col].std()
        print(f"\n{col}:")
        print(f"  Workspace: mean={ws_mean:.2f}, std={ws_std:.2f}")
        print(f"  Artifact:  mean={art_mean:.2f}, std={art_std:.2f}")

    print("\n" + "=" * 60)
    print("PHASE 1 TEST COMPLETE")
    print("=" * 60)

    return True


def test_phase1_size_comparison():
    """
    Compare artifact size vs tarball size.
    """
    import base64
    import io
    import tarfile

    from mostlyai import engine
    from mostlyai.engine import ModelArtifact
    from mostlyai.engine.domain import ModelEncodingType

    print("\n" + "=" * 60)
    print("SIZE COMPARISON TEST")
    print("=" * 60)

    df = create_test_data()

    with tempfile.TemporaryDirectory(prefix="slim_argn_size_") as workspace_dir:
        encoding_types = {
            "cat_col": ModelEncodingType.tabular_categorical,
            "num_col": ModelEncodingType.tabular_numeric_auto,
            "int_col": ModelEncodingType.tabular_numeric_discrete,
        }

        engine.split(workspace_dir=workspace_dir, tgt_data=df, tgt_encoding_types=encoding_types)
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)
        engine.encode(workspace_dir=workspace_dir)
        engine.train(workspace_dir=workspace_dir, max_epochs=5, enable_flexible_generation=False)

        # Create tarball (old method)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(f"{workspace_dir}/ModelStore", arcname=".")
        tarball_size = len(buf.getvalue())

        # Create artifact (new method)
        artifact = ModelArtifact.from_workspace(workspace_dir)
        artifact_size = len(artifact.to_bytes())

        print(f"\nTarball size:  {tarball_size:,} bytes")
        print(f"Artifact size: {artifact_size:,} bytes")
        print(f"Reduction:     {(1 - artifact_size/tarball_size)*100:.1f}%")

    return True


if __name__ == "__main__":
    print("Starting Slim ARGN Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run tests
    test_phase1_artifact_generation()
    test_phase1_size_comparison()

    print("\n\nAll tests completed!")
```

---

## Phase 2: Training Optimization

**Goal**: Eliminate disk I/O during training, simplify stats computation

**Estimated Scope**: ~400-500 lines of new code, ~300 lines of modifications

### 2.1 Streamlined Stats Computation

Replace `analyze()` with in-memory stats computation.

```python
# File: mostlyai/engine/stats.py

"""
In-memory statistics computation for training.

This replaces the file-based analyze() workflow with direct computation
that returns stats dicts without disk I/O.
"""

import pandas as pd
import numpy as np
from typing import Any

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine._encoding_types.tabular.categorical import (
    analyze_categorical,
    analyze_reduce_categorical,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    analyze_numeric,
    analyze_reduce_numeric,
)
from mostlyai.engine._encoding_types.tabular.datetime import (
    analyze_datetime,
    analyze_reduce_datetime,
)
from mostlyai.engine._encoding_types.tabular.character import (
    analyze_character,
    analyze_reduce_character,
)


def compute_stats(
    df: pd.DataFrame,
    encoding_types: dict[str, ModelEncodingType],
    context_key: str | None = None,
    primary_key: str | None = None,
) -> dict[str, Any]:
    """
    Compute column statistics from a DataFrame in memory.

    This is a simplified replacement for analyze() that:
    - Works entirely in memory (no file I/O)
    - Skips value protection (not needed for your use case)
    - Returns stats dict directly

    Args:
        df: DataFrame to analyze
        encoding_types: Dict mapping column names to encoding types
        context_key: Optional column name for grouping (sequential data)
        primary_key: Optional primary key column name

    Returns:
        Stats dict in the same format as analyze() output
    """
    stats: dict[str, Any] = {
        "columns": {},
        "keys": {},
    }

    if primary_key:
        stats["keys"]["primary_key"] = primary_key
    if context_key:
        stats["keys"]["context_key"] = context_key

    # Compute per-column stats
    root_keys = pd.Series(range(len(df)), name="__root")

    for col_idx, (column, encoding_type) in enumerate(encoding_types.items()):
        if column not in df.columns:
            continue

        values = df[column]

        # Analyze column
        if encoding_type == ModelEncodingType.tabular_categorical:
            partial_stats = analyze_categorical(values, root_keys, None)
            col_stats = analyze_reduce_categorical([partial_stats], value_protection=False)
        elif encoding_type in (
            ModelEncodingType.tabular_numeric_auto,
            ModelEncodingType.tabular_numeric_digit,
            ModelEncodingType.tabular_numeric_discrete,
            ModelEncodingType.tabular_numeric_binned,
        ):
            partial_stats = analyze_numeric(values, root_keys, None, encoding_type)
            col_stats = analyze_reduce_numeric(
                [partial_stats],
                value_protection=False,
                encoding_type=encoding_type,
            )
        elif encoding_type == ModelEncodingType.tabular_datetime:
            partial_stats = analyze_datetime(values, root_keys, None)
            col_stats = analyze_reduce_datetime([partial_stats], value_protection=False)
        elif encoding_type == ModelEncodingType.tabular_character:
            partial_stats = analyze_character(values, root_keys, None)
            col_stats = analyze_reduce_character([partial_stats], value_protection=False)
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")

        # Add ARGN identifiers
        col_stats["encoding_type"] = encoding_type
        col_stats["argn_processor"] = "tgt"  # or "ctxflt"/"ctxseq" for context
        col_stats["argn_table"] = "t0"
        col_stats["argn_column"] = f"c{col_idx}"

        stats["columns"][column] = col_stats

    # Compute sequence length stats
    if context_key and context_key in df.columns:
        seq_lens = df.groupby(context_key).size()
        stats["seq_len"] = {
            "min": int(seq_lens.min()),
            "max": int(seq_lens.max()),
            "median": int(seq_lens.median()),
        }
        stats["is_sequential"] = stats["seq_len"]["max"] > 1
    else:
        stats["seq_len"] = {"min": 1, "max": 1, "median": 1}
        stats["is_sequential"] = False

    # Record counts
    if context_key:
        n_records = df[context_key].nunique()
    else:
        n_records = len(df)

    # 80/20 split assumption (matching default split behavior)
    stats["no_of_training_records"] = int(n_records * 0.8)
    stats["no_of_validation_records"] = int(n_records * 0.2)

    return stats
```

### 2.2 End-to-End Training Functions

```python
# File: mostlyai/engine/training.py (new functions)

def train_flat(
    data: pd.DataFrame,
    encoding_types: dict[str, ModelEncodingType],
    *,
    primary_key: str | None = None,
    model_size: str = "M",
    max_epochs: int = 100,
    max_training_time: float = 14400.0,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    validation_split: float = 0.2,
    random_state: int | None = None,
) -> ModelArtifact:
    """
    Train a flat TabularARGN model and return a ModelArtifact.

    This is the end-to-end training function that:
    - Computes stats in memory
    - Encodes data in memory
    - Trains the model
    - Returns a minimal artifact (no disk I/O)

    Args:
        data: Training DataFrame
        encoding_types: Dict mapping column names to encoding types
        primary_key: Optional primary key column name
        model_size: Model size ("S", "M", or "L")
        max_epochs: Maximum training epochs
        max_training_time: Maximum training time in minutes
        batch_size: Batch size (auto-determined if None)
        device: Training device
        validation_split: Fraction of data for validation
        random_state: Random seed for reproducibility

    Returns:
        ModelArtifact containing trained weights and minimal stats

    Example:
        >>> artifact = train_flat(
        ...     data=df,
        ...     encoding_types={"col1": ModelEncodingType.tabular_categorical},
        ...     max_epochs=50,
        ... )
        >>> samples = generate_flat(artifact, sample_size=1000)
    """
    ...


def train_sequential(
    tgt_data: pd.DataFrame,
    tgt_encoding_types: dict[str, ModelEncodingType],
    tgt_context_key: str,
    *,
    ctx_data: pd.DataFrame | None = None,
    ctx_encoding_types: dict[str, ModelEncodingType] | None = None,
    ctx_primary_key: str | None = None,
    model_size: str = "M",
    max_epochs: int = 100,
    max_training_time: float = 14400.0,
    max_sequence_window: int = 100,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    validation_split: float = 0.2,
    random_state: int | None = None,
) -> ModelArtifact:
    """
    Train a sequential TabularARGN model and return a ModelArtifact.

    For longitudinal data with variable-length sequences per entity.

    Args:
        tgt_data: Target (sequential) DataFrame
        tgt_encoding_types: Encoding types for target columns
        tgt_context_key: Column linking target rows to context entities
        ctx_data: Optional context (entity) DataFrame
        ctx_encoding_types: Encoding types for context columns
        ctx_primary_key: Primary key column in context data
        model_size: Model size ("S", "M", or "L")
        max_epochs: Maximum training epochs
        max_training_time: Maximum training time in minutes
        max_sequence_window: Maximum sequence length for training
        batch_size: Batch size
        device: Training device
        validation_split: Fraction for validation
        random_state: Random seed

    Returns:
        ModelArtifact for the sequential model
    """
    ...
```

### 2.3 Testing Checkpoint: Phase 2

```python
# Addition to slim-argn.py

def test_phase2_training():
    """
    Test Phase 2: Streamlined training.

    Compare train_flat() output with standard workflow.
    """
    from mostlyai.engine import train_flat, generate_flat
    from mostlyai.engine.domain import ModelEncodingType

    print("\n" + "=" * 60)
    print("PHASE 2 TEST: Streamlined Training")
    print("=" * 60)

    df = create_test_data()
    encoding_types = {
        "cat_col": ModelEncodingType.tabular_categorical,
        "num_col": ModelEncodingType.tabular_numeric_auto,
        "int_col": ModelEncodingType.tabular_numeric_discrete,
    }

    # Train using new interface
    print("\n[1/3] Training with train_flat()...")
    artifact = train_flat(
        data=df,
        encoding_types=encoding_types,
        max_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  Training complete! Artifact size: {len(artifact.to_bytes()):,} bytes")

    # Generate samples
    print("\n[2/3] Generating samples...")
    samples = generate_flat(artifact, sample_size=SAMPLE_SIZE)
    print(f"  Generated {len(samples)} samples")

    # Validate output
    print("\n[3/3] Validating output...")
    assert set(samples.columns) == set(df.columns) - {"__any_removed_keys__"}
    print("  Columns match: OK")

    print("\nPHASE 2 TEST COMPLETE")
    return True
```

---

## Phase 3: Distributed Training Support

**Goal**: Enable Ray Data streaming for large datasets

**Estimated Scope**: ~300-400 lines of new code

### 3.1 Distributed Stats Computation

```python
# File: mostlyai/engine/distributed.py

"""
Distributed training support for Ray Data pipelines.

These functions enable:
1. Distributed stats computation via map/reduce
2. Distributed encoding via map_batches
3. Streaming tensor delivery to training
"""

def compute_partial_stats(
    batch: pd.DataFrame,
    encoding_types: dict[str, ModelEncodingType],
    context_key: str | None = None,
) -> dict[str, Any]:
    """
    Compute partial statistics from a batch.

    Use with Ray Data map_batches() to compute stats in parallel:

        partial_stats = dataset.map_batches(
            lambda b: compute_partial_stats(b, encoding_types),
            batch_format="pandas",
        )
    """
    ...


def combine_partial_stats(
    stats_list: list[dict[str, Any]],
    encoding_types: dict[str, ModelEncodingType],
) -> dict[str, Any]:
    """
    Combine partial statistics into final stats.

    Use after collecting partial stats:

        all_partial = partial_stats.take_all()
        final_stats = combine_partial_stats(all_partial, encoding_types)
    """
    ...


def encode_batch_for_ray(
    batch: pd.DataFrame,
    stats: dict[str, Any],
    context_key: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Encode a batch for Ray Data map_batches().

    Returns dict of numpy arrays (Ray Data compatible).

        encoded_ds = dataset.map_batches(
            lambda b: encode_batch_for_ray(b, stats),
            batch_format="pandas",
        )
    """
    ...
```

### 3.2 Ray Data Integration Example

```python
# Example usage in your trainer.py

def train_with_ray_data(
    dataset: ray.data.Dataset,
    encoding_types: dict[str, ModelEncodingType],
    max_epochs: int = 100,
) -> ModelArtifact:
    """
    Train using Ray Data streaming pipeline.
    """
    from mostlyai.engine.distributed import (
        compute_partial_stats,
        combine_partial_stats,
        encode_batch_for_ray,
    )
    from mostlyai.engine import (
        train,
        build_model_config,
        prepare_flat_batch,
    )
    from functools import partial

    # Step 1: Compute stats distributedly
    partial_stats_ds = dataset.map_batches(
        partial(compute_partial_stats, encoding_types=encoding_types),
        batch_format="pandas",
    )
    final_stats = combine_partial_stats(
        partial_stats_ds.take_all(),
        encoding_types,
    )

    # Step 2: Encode distributedly
    encode_fn = partial(encode_batch_for_ray, stats=final_stats)
    encoded_ds = dataset.map_batches(encode_fn, batch_format="pandas")

    # Step 3: Split for training/validation
    train_ds, val_ds = encoded_ds.train_test_split(test_size=0.2)

    # Step 4: Create tensor iterators
    def tensor_iter(ds, device):
        for batch in ds.iter_batches(batch_format="numpy"):
            yield prepare_flat_batch(batch, device=device)

    # Step 5: Train
    config = build_model_config(final_stats)

    # ... use existing tensor interface training ...
```

---

## Implementation Order & Dependencies

```
Phase 1.1: ModelArtifact dataclass
    └── Phase 1.2: Stats minimization functions
        └── Phase 1.3: from_workspace() migration
            └── Phase 1.4: generate_flat() / generate_sequential()
                └── Phase 1.7: TEST CHECKPOINT ★

Phase 2.1: compute_stats() in-memory
    └── Phase 2.2: train_flat() / train_sequential()
        └── Phase 2.3: TEST CHECKPOINT ★

Phase 3.1: Distributed stats (compute_partial, combine)
    └── Phase 3.2: Ray Data integration
        └── Phase 3.3: TEST CHECKPOINT ★
```

---

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `mostlyai/engine/_artifact.py` | ModelArtifact dataclass and serialization |
| `mostlyai/engine/_tabular/generation_core.py` | In-memory generation functions |
| `mostlyai/engine/stats.py` | In-memory stats computation |
| `mostlyai/engine/distributed.py` | Ray Data integration (Phase 3) |
| `mostlyai/engine/slim-argn.py` | Test script for cluster validation |

### Modified Files
| File | Changes |
|------|---------|
| `mostlyai/engine/__init__.py` | Export new functions |
| `mostlyai/engine/generation.py` | Add generate_flat/generate_sequential wrappers |
| `mostlyai/engine/training.py` | Add train_flat/train_sequential wrappers |
| `mostlyai/engine/_tabular/common.py` | Extract helper functions for reuse |

### Files to Potentially Remove (Phase 2+)
| File | Reason |
|------|--------|
| `mostlyai/engine/splitting.py` | Replaced by in-memory processing |
| `mostlyai/engine/encoding.py` | Merged into stats.py / training flow |

---

## Migration Path for Existing Code

### Your TabularARGNModule Changes

```python
# generators/tabular_argn/model.py

class TabularARGNModule(BaseGeneratorModule):
    def __init__(
        self,
        model_data_tarball: str | None = None,  # Legacy
        model_artifact_bytes: bytes | None = None,  # New
        batch_size: int = 8_192,
        **kwargs
    ):
        ...
        self._model_data_tarball = model_data_tarball
        self._model_artifact_bytes = model_artifact_bytes

    def sample(self, n: int, ...) -> pd.DataFrame:
        if self._model_artifact_bytes:
            # New fast path
            from mostlyai.engine import ModelArtifact, generate_flat
            artifact = ModelArtifact.from_bytes(self._model_artifact_bytes)
            return generate_flat(artifact, sample_size=n, device="cuda")
        else:
            # Legacy tarball path
            ... # existing implementation

    def extra_serialization_params(self) -> dict:
        if self._model_artifact_bytes:
            return {
                "format": "artifact",
                "model_artifact_bytes": self._model_artifact_bytes,
            }
        else:
            return {
                "format": "tarball",
                "model_data_tarball": self._model_data_tarball,
            }
```

### Your Trainer Changes

```python
# generators/tabular_argn/trainer.py

def train(...) -> TabularARGNModule:
    from mostlyai.engine import train_flat
    from mostlyai.engine.domain import ModelEncodingType

    # New streamlined path
    artifact = train_flat(
        data=df,
        encoding_types=target_encoding_types,
        max_epochs=max_epochs,
        device="cuda",
    )

    return TabularARGNModule(
        model_artifact_bytes=artifact.to_bytes(),
        batch_size=approx_batch_size,
    )
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Inference latency (1K samples) | ~2-5s | <500ms | Time from artifact load to DataFrame |
| Artifact size | ~2x weights | ~1.1x weights | Bytes comparison |
| Training disk I/O | 5+ file operations | 0 | Count file writes during training |
| Memory efficiency | Load full parquet | Stream batches | Peak memory during training |

---

## Risk Mitigation

1. **Backward Compatibility**: Support both tarball and artifact formats during migration
2. **Validation**: Statistical comparison between old and new generation outputs
3. **Incremental Rollout**: Phase 1 (inference) can be deployed independently
4. **Fallback**: Keep workspace-based code paths available initially
