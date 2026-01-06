"""
ModelArtifact - Minimal serializable model representation for distributed inference.

This module provides a compact artifact format that contains everything needed
for inference without workspace file I/O. The artifact can be serialized to bytes
for storage in databases, object stores, or transmitted over the network.

Key design goals:
- Minimal size: Only store what's needed for inference
- Fast serialization: Single bytes object, no file I/O
- Self-contained: No external dependencies at inference time
"""

import io
import json
import logging
import zlib
from dataclasses import dataclass, field
from typing import Any

import torch

_LOG = logging.getLogger(__name__)

# Artifact format version - increment when making breaking changes
ARTIFACT_VERSION = 1


@dataclass
class ModelArtifact:
    """
    Minimal artifact containing everything needed for inference.

    This replaces the workspace-based storage with a single serializable object.
    Contains model weights, architecture parameters, and statistics for decoding.

    Attributes:
        weights: Compressed torch state_dict bytes
        is_sequential: Whether this is a sequential (longitudinal) model
        model_size: Model size identifier ("S", "M", or "L")
        tgt_cardinalities: Target column cardinalities (vocab sizes)
        ctx_cardinalities: Context column cardinalities
        tgt_stats: Target column statistics for encoding/decoding
        ctx_stats: Context column statistics (if any)
        tgt_seq_len_min: Minimum sequence length (sequential models only)
        tgt_seq_len_max: Maximum sequence length (sequential models only)
        tgt_seq_len_median: Median sequence length (sequential models only)
        ctx_seq_len_median: Context sequence length medians by table
        tgt_primary_key: Target primary key column name
        tgt_context_key: Target context key column name (links to ctx_primary_key)
        ctx_primary_key: Context primary key column name
        enable_flexible_generation: Whether model supports flexible column ordering

    Example:
        >>> # After training
        >>> artifact = ModelArtifact(
        ...     weights=compressed_weights,
        ...     is_sequential=False,
        ...     model_size="M",
        ...     tgt_cardinalities={"tgt:/col1__cat": 10},
        ...     ctx_cardinalities={},
        ...     tgt_stats={"columns": {...}},
        ... )
        >>>
        >>> # Serialize for storage
        >>> artifact_bytes = artifact.to_bytes()
        >>>
        >>> # Later, deserialize for inference
        >>> artifact = ModelArtifact.from_bytes(artifact_bytes)
        >>> df = generate_flat(artifact, sample_size=1000)
    """

    # === Model Weights ===
    weights: bytes  # Compressed torch state_dict

    # === Model Architecture ===
    is_sequential: bool
    model_size: str  # "S", "M", "L"
    tgt_cardinalities: dict[str, int]  # {"tgt:/col1__cat": 10, ...}
    ctx_cardinalities: dict[str, int] = field(default_factory=dict)
    enable_flexible_generation: bool = False

    # === Sequential Model Parameters ===
    tgt_seq_len_min: int | None = None
    tgt_seq_len_max: int | None = None
    tgt_seq_len_median: int | None = None
    ctx_seq_len_median: dict[str, int] = field(default_factory=dict)

    # === Statistics for Encoding/Decoding ===
    tgt_stats: dict[str, Any] = field(default_factory=dict)
    ctx_stats: dict[str, Any] = field(default_factory=dict)

    # === Key Columns ===
    tgt_primary_key: str | None = None
    tgt_context_key: str | None = None
    ctx_primary_key: str | None = None

    # === Metadata ===
    version: int = ARTIFACT_VERSION

    def to_bytes(self) -> bytes:
        """
        Serialize artifact to bytes for storage.

        Format: [metadata_len (4 bytes)][compressed metadata JSON][compressed weights]

        The metadata and weights are compressed separately to allow efficient
        access to metadata without decompressing the (larger) weights.

        Returns:
            Serialized artifact as bytes
        """
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

        # Compress metadata
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        metadata_compressed = zlib.compress(metadata_bytes, level=6)

        # Pack format: [metadata_len (4 bytes)][metadata][weights]
        metadata_len = len(metadata_compressed)
        return metadata_len.to_bytes(4, "little") + metadata_compressed + self.weights

    @classmethod
    def from_bytes(cls, data: bytes) -> "ModelArtifact":
        """
        Deserialize artifact from bytes.

        Args:
            data: Serialized artifact bytes from to_bytes()

        Returns:
            Reconstructed ModelArtifact

        Raises:
            ValueError: If artifact version is unsupported
        """
        # Unpack format
        metadata_len = int.from_bytes(data[:4], "little")
        metadata_compressed = data[4 : 4 + metadata_len]
        weights = data[4 + metadata_len :]

        # Decompress and parse metadata
        metadata_bytes = zlib.decompress(metadata_compressed)
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Version check
        version = metadata.get("version", 1)
        if version > ARTIFACT_VERSION:
            raise ValueError(
                f"Artifact version {version} is newer than supported version {ARTIFACT_VERSION}. "
                "Please upgrade mostlyai-engine."
            )

        return cls(
            weights=weights,
            is_sequential=metadata["is_sequential"],
            model_size=metadata["model_size"],
            tgt_cardinalities=metadata["tgt_cardinalities"],
            ctx_cardinalities=metadata.get("ctx_cardinalities", {}),
            enable_flexible_generation=metadata.get("enable_flexible_generation", False),
            tgt_seq_len_min=metadata.get("tgt_seq_len_min"),
            tgt_seq_len_max=metadata.get("tgt_seq_len_max"),
            tgt_seq_len_median=metadata.get("tgt_seq_len_median"),
            ctx_seq_len_median=metadata.get("ctx_seq_len_median", {}),
            tgt_stats=metadata.get("tgt_stats", {}),
            ctx_stats=metadata.get("ctx_stats", {}),
            tgt_primary_key=metadata.get("tgt_primary_key"),
            tgt_context_key=metadata.get("tgt_context_key"),
            ctx_primary_key=metadata.get("ctx_primary_key"),
            version=version,
        )

    def get_state_dict(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """
        Decompress and load weights as state_dict.

        Args:
            device: Device to load tensors to ("cpu", "cuda", etc.)

        Returns:
            PyTorch state_dict ready for model.load_state_dict()
        """
        weights_decompressed = zlib.decompress(self.weights)
        buffer = io.BytesIO(weights_decompressed)
        return torch.load(buffer, map_location=device, weights_only=True)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        is_sequential: bool,
        model_size: str,
        tgt_cardinalities: dict[str, int],
        tgt_stats: dict[str, Any],
        ctx_cardinalities: dict[str, int] | None = None,
        ctx_stats: dict[str, Any] | None = None,
        tgt_seq_len_min: int | None = None,
        tgt_seq_len_max: int | None = None,
        tgt_seq_len_median: int | None = None,
        ctx_seq_len_median: dict[str, int] | None = None,
        tgt_primary_key: str | None = None,
        tgt_context_key: str | None = None,
        ctx_primary_key: str | None = None,
        enable_flexible_generation: bool = False,
        compression_level: int = 6,
    ) -> "ModelArtifact":
        """
        Create artifact from a PyTorch state_dict and metadata.

        This is the primary way to create artifacts after training.

        Args:
            state_dict: PyTorch model state_dict
            is_sequential: Whether this is a sequential model
            model_size: Model size ("S", "M", "L")
            tgt_cardinalities: Target cardinalities
            tgt_stats: Target statistics for encoding/decoding
            ctx_cardinalities: Context cardinalities (optional)
            ctx_stats: Context statistics (optional)
            tgt_seq_len_min: Min sequence length (sequential only)
            tgt_seq_len_max: Max sequence length (sequential only)
            tgt_seq_len_median: Median sequence length (sequential only)
            ctx_seq_len_median: Context sequence length medians
            tgt_primary_key: Target primary key column
            tgt_context_key: Target context key column
            ctx_primary_key: Context primary key column
            enable_flexible_generation: Support flexible column ordering
            compression_level: zlib compression level (1-9, default 6)

        Returns:
            New ModelArtifact instance
        """
        # Serialize and compress state_dict
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        weights_compressed = zlib.compress(buffer.getvalue(), level=compression_level)

        return cls(
            weights=weights_compressed,
            is_sequential=is_sequential,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities or {},
            enable_flexible_generation=enable_flexible_generation,
            tgt_seq_len_min=tgt_seq_len_min,
            tgt_seq_len_max=tgt_seq_len_max,
            tgt_seq_len_median=tgt_seq_len_median,
            ctx_seq_len_median=ctx_seq_len_median or {},
            tgt_stats=tgt_stats,
            ctx_stats=ctx_stats or {},
            tgt_primary_key=tgt_primary_key,
            tgt_context_key=tgt_context_key,
            ctx_primary_key=ctx_primary_key,
        )

    def __repr__(self) -> str:
        weights_size = len(self.weights)
        n_tgt_cols = len(set(k.split("__")[0] for k in self.tgt_cardinalities.keys()))
        n_ctx_cols = len(set(k.split("__")[0] for k in self.ctx_cardinalities.keys()))
        return (
            f"ModelArtifact("
            f"version={self.version}, "
            f"is_sequential={self.is_sequential}, "
            f"model_size='{self.model_size}', "
            f"tgt_cols={n_tgt_cols}, "
            f"ctx_cols={n_ctx_cols}, "
            f"weights={weights_size:,} bytes"
            f")"
        )


# =============================================================================
# Stats Minimization Functions
# =============================================================================

# Fields to REMOVE from column stats (not needed for encoding/decoding)
_REMOVABLE_FIELDS = {
    "value_protection",      # Privacy flag - not needed
    "no_of_rare_categories", # Stats only - not needed for encoding
    "cnt_values",            # Raw counts - only for value protection
}

# Fields to keep for each encoding type
_ESSENTIAL_FIELDS_BY_TYPE = {
    # Common fields needed by all types
    "_common": {
        "encoding_type",
        "cardinalities",
        "argn_processor",
        "argn_table",
        "argn_column",
    },
    # Categorical
    "TABULAR_CATEGORICAL": {
        "codes",  # Value -> int mapping
    },
    # Numeric discrete (each unique value is a category)
    "TABULAR_NUMERIC_DISCRETE": {
        "codes",
        "min_decimal",  # For dtype conversion
    },
    # Numeric binned
    "TABULAR_NUMERIC_BINNED": {
        "codes",        # Special tokens (UNK, MIN, MAX)
        "bins",         # Bin edges
        "min_decimal",  # Decimal precision
    },
    # Numeric digit (digit-by-digit encoding)
    "TABULAR_NUMERIC_DIGIT": {
        "has_nan",
        "has_neg",
        "min_digits",   # Per-digit min values
        "max_digits",   # Per-digit max values
        "min_decimal",  # Precision range
        "max_decimal",
        "min",          # Value range for clipping
        "max",
    },
    # Datetime
    "TABULAR_DATETIME": {
        "has_nan",
        "has_time",
        "has_ms",
        "min_values",  # Per-component min
        "max_values",  # Per-component max
        "min",         # Overall min date string
        "max",         # Overall max date string
    },
    # Relative datetime (inter-transaction time)
    "TABULAR_DATETIME_RELATIVE": {
        "has_nan",
        "min_decimal",
        "max_decimal",
        "bins",
        "codes",
    },
    # Character (character-level encoding)
    "TABULAR_CHARACTER": {
        "charset",
        "max_len",
        "codes",
    },
    # Lat/Long
    "TABULAR_LAT_LONG": {
        "precision",
        "min_lat", "max_lat",
        "min_long", "max_long",
    },
}


def minimize_column_stats(column_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Extract only the fields needed for encoding/decoding from column stats.

    Removes value protection metadata, raw counts, and other training-only fields.

    Args:
        column_stats: Full column statistics dict from analyze()

    Returns:
        Minimized column statistics with only essential fields
    """
    encoding_type = column_stats.get("encoding_type", "")

    # Start with common fields
    essential_fields = set(_ESSENTIAL_FIELDS_BY_TYPE["_common"])

    # Add type-specific fields
    if encoding_type in _ESSENTIAL_FIELDS_BY_TYPE:
        essential_fields |= _ESSENTIAL_FIELDS_BY_TYPE[encoding_type]

    # Build minimized stats
    minimal: dict[str, Any] = {}
    for key, value in column_stats.items():
        # Skip explicitly removable fields
        if key in _REMOVABLE_FIELDS:
            continue

        # Include if it's an essential field or we don't know about this encoding type
        # (better to include unknown fields than break decoding)
        if key in essential_fields or encoding_type not in _ESSENTIAL_FIELDS_BY_TYPE:
            # Special handling for seq_len - only keep min/max/median
            if key == "seq_len" and isinstance(value, dict):
                minimal[key] = {
                    k: v for k, v in value.items()
                    if k in ("min", "max", "median")
                }
            else:
                minimal[key] = value

    return minimal


def minimize_stats(full_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Minimize full stats dict (as returned by analyze()).

    Keeps only what's needed for encoding/decoding, stripping privacy-related
    and training-only metadata.

    Args:
        full_stats: Full statistics dict from analyze()

    Returns:
        Minimized statistics dict

    Example:
        >>> full_stats = workspace.tgt_stats.read()
        >>> minimal = minimize_stats(full_stats)
        >>> # minimal is ~30-50% smaller and contains only essential fields
    """
    minimal: dict[str, Any] = {}

    # Minimize each column's stats
    if "columns" in full_stats:
        minimal["columns"] = {
            col_name: minimize_column_stats(col_stats)
            for col_name, col_stats in full_stats["columns"].items()
        }

    # Keep sequence length info (for sequential models)
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
