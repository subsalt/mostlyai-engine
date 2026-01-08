# PyTorch Training CPU Performance Optimization

## Overview

Optimize two critical CPU performance bottlenecks identified via flame graph profiling in the mostlyai-engine training pipeline. The bottlenecks cause significant slowdowns during data preprocessing and tensor encoding, particularly affecting:

1. **One-time statistics computation** (~10-30% of total training time for large datasets)
2. **Sequential tensor encoding** (~20-50% of total training time for many-context datasets)

This plan targets **10x+ speedup** for the critical hotspots through vectorization, pre-computation, and elimination of redundant pandas operations.

## Problem Statement

### Hotspot #1: `split_sub_columns_digit` (One-time Statistics)

**Location**: `mostlyai/engine/_encoding_types/tabular/numeric.py:121-155`

**Root Cause**: The function uses `pandas.Series.apply()` with `np.format_float_positional()` - a Python-level loop that calls a C function per element. This is 50-1300x slower than vectorized operations.

```python
# CURRENT (SLOW) - line 137-144
values_str = (
    values.abs()
    .apply(lambda x: np.format_float_positional(x, unique=True, pad_left=50, pad_right=20, precision=20))
    .astype("string[pyarrow]")
    .replace("nan", pd.NA)
)
```

**Call Path**: `train_sequential` → `compute_stats` → `_compute_table_stats` → `_analyze_column` → `analyze_numeric` → `split_sub_columns_digit`

**Impact**: Called once per numeric column with digit encoding. For a dataset with 50 numeric columns and 1M rows, this adds ~5-10 minutes of preprocessing time.

### Hotspot #2: `_encode_sequential_to_tensors` (Training Loop Encoding)

**Location**: `mostlyai/engine/_train.py:504-514`

**Root Cause**: Per-context loop with `groupby.get_group()` - O(n_contexts) pandas groupby operations, each with indexing overhead.

```python
# CURRENT (SLOW) - line 504-512
for i, ctx_id in enumerate(context_ids):
    try:
        group = tgt_grouped.get_group(ctx_id)  # Expensive pandas operation per context
        seq = group[col].to_numpy()
        seq_len = min(len(seq), max_seq_len)
        padded[i, :seq_len, 0] = torch.tensor(seq[:seq_len], dtype=torch.int64, device=device)
    except KeyError:
        pass
```

**Call Path**: `train_sequential` → `_encode_sequential_to_tensors` → per-context loop

**Impact**: For 100K contexts with 100 rows each, this adds ~30-60 seconds per tensor column. With 50 columns, that's 25-50 minutes of preprocessing.

## Proposed Solution

### Phase 1: Vectorize `split_sub_columns_digit`

Replace the `apply()` pattern with a Numba JIT-compiled function that maintains exact floating-point formatting behavior.

**Approach**:
1. Extract digit positions using vectorized arithmetic operations
2. Use Numba for the core digit extraction loop
3. Maintain identical output format for backwards compatibility

### Phase 2: Pre-compute Group Indices for Tensor Encoding

Replace per-context `get_group()` calls with pre-computed indices using `groupby.indices`.

**Approach**:
1. Pre-compute `indices_map = tgt_grouped.indices` once
2. Convert DataFrame columns to NumPy arrays before looping
3. Use direct array indexing instead of pandas DataFrame operations

### Phase 3: Optimize Supporting Operations

Address secondary bottlenecks identified in the flame graphs:
- Batch type conversions
- Reduce intermediate allocations
- Optimize Arrow→NumPy conversions

## Technical Approach

### Architecture

```
Current Flow (Slow):
train_sequential()
    ├── compute_stats()
    │   └── _analyze_column() × N columns
    │       └── split_sub_columns_digit()  ← APPLY() BOTTLENECK
    └── _encode_sequential_to_tensors()
        └── for ctx_id in context_ids:     ← O(N) GROUPBY BOTTLENECK
            └── get_group(ctx_id)

Optimized Flow (Fast):
train_sequential()
    ├── compute_stats()
    │   └── _analyze_column() × N columns
    │       └── split_sub_columns_digit_vectorized()  ← NUMBA JIT
    └── _encode_sequential_to_tensors()
        ├── indices_map = grouped.indices  ← PRE-COMPUTE ONCE
        ├── data_np = df[cols].to_numpy()  ← CONVERT ONCE
        └── for ctx_id in context_ids:
            └── data_np[indices_map[ctx_id]]  ← NUMPY INDEXING
```

### Implementation Phases

#### Phase 1: Vectorize `split_sub_columns_digit` (HIGH IMPACT)

**Target Speedup**: 50-100x
**Risk Level**: Medium (must maintain exact output format)

**Tasks:**

- [ ] **1.1** Create benchmark suite for `split_sub_columns_digit`
  - Test cases: 1K, 10K, 100K, 1M rows
  - Edge cases: all NaN, all integers, scientific notation, extreme values
  - File: `tests/benchmarks/test_split_sub_columns_digit_benchmark.py`

- [ ] **1.2** Implement Numba-accelerated digit extraction
  - Use `@numba.njit` for the core digit extraction loop
  - Maintain exact `np.format_float_positional` behavior
  - File: `mostlyai/engine/_encoding_types/tabular/numeric.py`

```python
# Proposed implementation sketch
@numba.njit(parallel=True)
def extract_digits_vectorized(values, max_decimal, min_decimal):
    """Extract digit columns from float array without string conversion."""
    n = len(values)
    n_digits = max_decimal - min_decimal + 1
    result = np.zeros((n, n_digits), dtype=np.int8)

    for i in numba.prange(n):
        if np.isnan(values[i]):
            continue
        # Extract digits via arithmetic
        abs_val = abs(values[i])
        for d in range(n_digits):
            power = max_decimal - d
            digit = int(abs_val / (10 ** power)) % 10
            result[i, d] = digit

    return result
```

- [ ] **1.3** Add correctness tests comparing vectorized vs. original output
  - Bit-identical output validation for all test cases
  - File: `tests/test_numeric_encoding.py`

- [ ] **1.4** Integrate and benchmark end-to-end
  - Measure `compute_stats` time reduction
  - Validate no regression in model quality

#### Phase 2: Pre-compute Group Indices (HIGH IMPACT)

**Target Speedup**: 10-50x
**Risk Level**: Low (well-established pattern)

**Tasks:**

- [ ] **2.1** Create benchmark suite for `_encode_sequential_to_tensors`
  - Test cases: 1K, 10K, 100K contexts with varying sequence lengths
  - File: `tests/benchmarks/test_encode_sequential_benchmark.py`

- [ ] **2.2** Implement pre-computed indices approach
  - Use `groupby.indices` instead of repeated `get_group()`
  - Convert DataFrame columns to NumPy before the loop
  - File: `mostlyai/engine/_train.py`

```python
# Proposed implementation sketch
def _encode_sequential_to_tensors_optimized(...):
    # Pre-compute group indices ONCE
    indices_map = tgt_encoded.groupby(encoded_context_key, sort=False).indices

    # Convert to numpy arrays ONCE
    tgt_data = {col: tgt_encoded[col].to_numpy() for col in tgt_cols}

    for col in tgt_cols:
        padded = torch.zeros((n_contexts, padded_seq_len, 1), dtype=torch.int64, device=device)

        for i, ctx_id in enumerate(context_ids):
            if ctx_id not in indices_map:
                continue  # Empty sequence

            indices = indices_map[ctx_id]
            seq = tgt_data[col][indices]
            seq_len = min(len(seq), max_seq_len)
            padded[i, :seq_len, 0] = torch.tensor(seq[:seq_len], dtype=torch.int64, device=device)

        tensors[col] = padded
```

- [ ] **2.3** Further optimize with batch tensor construction
  - Use numpy advanced indexing to build all sequences at once
  - Consider nested tensors or ragged arrays if supported

- [ ] **2.4** Add correctness tests comparing optimized vs. original output
  - Tensor equality validation for all test cases
  - File: `tests/test_train.py`

- [ ] **2.5** Integrate and benchmark end-to-end
  - Measure `_encode_sequential_to_tensors` time reduction
  - Validate no regression in model quality

#### Phase 3: Secondary Optimizations (MEDIUM IMPACT)

**Target Speedup**: 2-5x additional
**Risk Level**: Low

**Tasks:**

- [ ] **3.1** Batch type conversions in `analyze_numeric`
  - Current: Multiple separate `astype()` calls
  - Optimized: Single `astype(dict)` call where possible
  - File: `mostlyai/engine/_encoding_types/tabular/numeric.py`

- [ ] **3.2** Optimize Arrow→NumPy conversions
  - Use `zero_copy_only=True` where data permits
  - Pre-validate zero-copy compatibility
  - File: `mostlyai/engine/_encoding_types/tabular/numeric.py`

- [ ] **3.3** Reduce quantile computation overhead in `analyze_numeric`
  - Current: 1001 quantiles computed (line 176-180)
  - Optimized: Use approximate quantiles for large datasets
  - File: `mostlyai/engine/_encoding_types/tabular/numeric.py:176`

- [ ] **3.4** Profile and address any remaining hotspots
  - Re-run flame graph profiling after Phase 1 & 2
  - Document findings and create follow-up issues

## Acceptance Criteria

### Functional Requirements

- [ ] All existing tests pass without modification
- [ ] No regression in generated synthetic data quality
- [ ] Model checkpoints remain compatible
- [ ] Stats JSON format unchanged

### Non-Functional Requirements

- [ ] `split_sub_columns_digit` achieves **>10x speedup** on 100K+ rows
- [ ] `_encode_sequential_to_tensors` achieves **>5x speedup** on 10K+ contexts
- [ ] Memory overhead from pre-computation **<20% of dataset size**
- [ ] No new external dependencies (Numba already used in codebase)

### Quality Gates

- [ ] Unit test coverage for all new code paths
- [ ] Benchmark tests added to CI pipeline
- [ ] Code review approval from 2+ reviewers
- [ ] Performance validated on representative production dataset

## Success Metrics

| Metric | Current | Target | Measurement Method |
|--------|---------|--------|-------------------|
| `split_sub_columns_digit` time (100K rows) | ~10s | <1s | Benchmark test |
| `_encode_sequential_to_tensors` time (10K contexts) | ~30s | <5s | Benchmark test |
| Total `train_sequential` preprocessing | ~5 min | <1 min | End-to-end benchmark |
| Memory overhead | Baseline | <+20% | Memory profiler |
| Synthetic data quality | Baseline | No degradation | Distribution tests |

## Dependencies & Prerequisites

### Required Before Starting

- [ ] Access to representative benchmark dataset (100K+ rows, 10K+ contexts)
- [ ] Baseline performance measurements documented
- [ ] Numba installed and tested in dev environment

### Blocked By

- None (can start immediately)

### Blocks

- Future distributed training optimizations
- CUDA acceleration of preprocessing

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Floating-point precision differences | Medium | High | Extensive correctness tests with edge cases |
| Memory explosion on large datasets | Low | High | Add memory budget checks, fallback to slow path |
| Numba compilation overhead | Low | Low | Pre-warm JIT, measure cold vs warm |
| Backwards compatibility breakage | Low | High | Bit-identical output validation |

## Resource Requirements

- **Developer time**: ~3-5 days for Phase 1 & 2, ~2 days for Phase 3
- **Testing resources**: Representative benchmark dataset
- **Infrastructure**: CI pipeline for automated benchmarks

## Future Considerations

1. **GPU Acceleration**: Move digit extraction to CUDA for further speedup
2. **Distributed Processing**: Parallelize across partitions in Ray Data
3. **Lazy Evaluation**: Use Polars for data preparation (3-30x faster than pandas)
4. **Memory-Mapped Tensors**: Support datasets larger than RAM

## References

### Internal References

- Hotspot #1: `mostlyai/engine/_encoding_types/tabular/numeric.py:121-155`
- Hotspot #2: `mostlyai/engine/_train.py:504-514`
- Call path: `mostlyai/engine/_stats.py:196` → `numeric.py:202`
- Training entry: `mostlyai/engine/_train.py:201` (`train_sequential`)

### External References

- [Pandas Enhancing Performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Numba User Guide](https://numba.readthedocs.io/en/stable/user/index.html)
- [PyTorch DataLoader Optimization](https://docs.pytorch.org/docs/stable/data.html)
- [numpy-groupies library](https://github.com/ml31415/numpy-groupies)

### Flame Graph Analysis

- **Graph 1 (Statistics)**: Heavy `apply()` in `split_sub_columns_digit`, `astype` conversions
- **Graph 2 (Encoding)**: Per-context `get_group()` loop, pandas indexing overhead

---

*Plan created: 2026-01-08*
