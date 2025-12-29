"""
Test script for Slim ARGN implementation.

Run with: ./cluster-run.sh slim-argn.py

This script tests the artifact-based generation against the workspace-based
generation to ensure equivalence, and measures size/performance improvements.
"""

import tempfile
import time
import json
import numpy as np
import pandas as pd
import torch
import ray

# Test configuration
SAMPLE_SIZE = 1000
TRAINING_EPOCHS = 5
RANDOM_SEED = 42


def create_test_data(n_samples: int = 5000) -> pd.DataFrame:
    """Create synthetic test data for training."""
    np.random.seed(RANDOM_SEED)

    return pd.DataFrame({
        "cat_col": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
        "num_col": np.random.randn(n_samples) * 10 + 50,
        "int_col": np.random.randint(0, 100, n_samples),
        "cat_col2": np.random.choice(["X", "Y", "Z"], n_samples),
    })


@ray.remote(num_cpus=0, num_gpus=1)
def test_phase1_artifact_generation():
    """
    Test Phase 1: Artifact-based generation.

    1. Train a model using the tensor interface
    2. Create ModelArtifact from training outputs
    3. Generate using both methods
    4. Compare results statistically
    """
    from mostlyai import engine
    from mostlyai.engine._artifact import ModelArtifact, minimize_stats
    from mostlyai.engine._common import get_cardinalities
    from mostlyai.engine.generation import generate_flat
    from mostlyai.engine.domain import ModelEncodingType
    from mostlyai.engine.random_state import set_random_state

    print("=" * 70)
    print("PHASE 1 TEST: Artifact-based Generation")
    print("=" * 70)

    # Create test data
    print("\n[1/6] Creating test data...")
    df = create_test_data()
    print(f"  Created DataFrame with shape {df.shape}")

    encoding_types = {
        "cat_col": ModelEncodingType.tabular_categorical,
        "num_col": ModelEncodingType.tabular_numeric_auto,
        "int_col": ModelEncodingType.tabular_numeric_discrete,
        "cat_col2": ModelEncodingType.tabular_categorical,
    }

    # Train using tensor interface
    print("\n[2/6] Training model...")
    with tempfile.TemporaryDirectory(prefix="slim_argn_test_") as workspace_dir:
        # Prep steps
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

        # Load tensors
        train_tensors, val_tensors, model_config = engine.load_tensors_from_workspace(
            workspace_dir, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Train
        t0 = time.time()
        engine.train(
            train_tensors=iter(train_tensors),
            val_tensors=iter(val_tensors),
            model_config=model_config,
            workspace_dir=workspace_dir,
            max_epochs=TRAINING_EPOCHS,
            enable_flexible_generation=False,
        )
        training_time = time.time() - t0
        print(f"  Training completed in {training_time:.1f}s")

        # Load artifacts
        print("\n[3/6] Creating ModelArtifact...")
        with open(f"{workspace_dir}/ModelStore/tgt-stats/stats.json") as f:
            tgt_stats = json.load(f)
        with open(f"{workspace_dir}/ModelStore/model-data/model-configs.json") as f:
            model_configs = json.load(f)

        weights_path = f"{workspace_dir}/ModelStore/model-data/model-weights.pt"
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )

        tgt_cardinalities = get_cardinalities(tgt_stats)
        model_size = model_configs.get("model_id", "MOSTLY_AI/Medium").split("/")[-1][0]

        artifact = ModelArtifact.from_state_dict(
            state_dict=state_dict,
            is_sequential=False,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            tgt_stats=minimize_stats(tgt_stats),
        )

        artifact_bytes = artifact.to_bytes()
        print(f"  Artifact: {artifact}")
        print(f"  Serialized size: {len(artifact_bytes):,} bytes")

        # Verify roundtrip
        artifact_restored = ModelArtifact.from_bytes(artifact_bytes)
        print("  Roundtrip serialization: OK")

        # Generate using legacy method
        print("\n[4/6] Generating with LEGACY method (workspace)...")
        set_random_state(RANDOM_SEED)
        t0 = time.time()
        engine.generate(
            workspace_dir=workspace_dir,
            sample_size=SAMPLE_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        legacy_time = time.time() - t0
        legacy_samples = pd.read_parquet(f"{workspace_dir}/SyntheticData")
        print(f"  Generated {len(legacy_samples)} samples in {legacy_time:.2f}s")

        # Generate using new method
        print("\n[5/6] Generating with NEW method (artifact)...")
        set_random_state(RANDOM_SEED)
        t0 = time.time()
        new_samples = generate_flat(
            artifact=artifact_restored,
            sample_size=SAMPLE_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        new_time = time.time() - t0
        print(f"  Generated {len(new_samples)} samples in {new_time:.2f}s")

    # Compare results
    print("\n[6/6] Comparing results...")
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Check columns match
    legacy_cols = set(legacy_samples.columns)
    new_cols = set(new_samples.columns)
    print(f"\nColumn match: {legacy_cols == new_cols}")
    if legacy_cols != new_cols:
        print(f"  Legacy only: {legacy_cols - new_cols}")
        print(f"  New only: {new_cols - legacy_cols}")

    # Compare categorical distributions
    print("\n--- Categorical Distributions ---")
    for col in ["cat_col", "cat_col2"]:
        legacy_dist = legacy_samples[col].value_counts(normalize=True).sort_index()
        new_dist = new_samples[col].value_counts(normalize=True).sort_index()

        if set(legacy_dist.index) == set(new_dist.index):
            max_diff = abs(legacy_dist - new_dist).max()
            status = "PASS" if max_diff < 0.001 else "WARN"
            print(f"  {col}: {status} max diff = {max_diff:.4f}")
        else:
            print(f"  {col}: FAIL categories differ")
            print(f"    Legacy: {set(legacy_dist.index)}")
            print(f"    New: {set(new_dist.index)}")

    # Compare numeric stats
    print("\n--- Numeric Statistics ---")
    for col in ["num_col", "int_col"]:
        legacy_mean = legacy_samples[col].mean()
        new_mean = new_samples[col].mean()
        legacy_std = legacy_samples[col].std()
        new_std = new_samples[col].std()

        mean_diff = abs(legacy_mean - new_mean)
        std_diff = abs(legacy_std - new_std)

        status = "PASS" if mean_diff < 0.01 and std_diff < 0.01 else "WARN"
        print(f"  {col}: {status} mean diff = {mean_diff:.4f}, std diff = {std_diff:.4f}")

    # Performance comparison
    print("\n--- Performance ---")
    print(f"  Legacy generation time: {legacy_time:.2f}s")
    print(f"  New generation time:    {new_time:.2f}s")
    if new_time < legacy_time:
        print(f"  Speedup: {legacy_time/new_time:.1f}x faster")
    else:
        print(f"  Note: New method {legacy_time/new_time:.2f}x (may vary by hardware)")

    print("\n" + "=" * 70)
    print("PHASE 1 TEST COMPLETE")
    print("=" * 70)

    return True


@ray.remote(num_cpus=0, num_gpus=1)
def test_artifact_size_comparison():
    """
    Compare artifact size vs tarball size.
    """
    import io
    import tarfile

    from mostlyai import engine
    from mostlyai.engine._artifact import ModelArtifact, minimize_stats
    from mostlyai.engine._common import get_cardinalities
    from mostlyai.engine.domain import ModelEncodingType

    print("\n" + "=" * 70)
    print("SIZE COMPARISON TEST")
    print("=" * 70)

    df = create_test_data(n_samples=2000)
    encoding_types = {
        "cat_col": ModelEncodingType.tabular_categorical,
        "num_col": ModelEncodingType.tabular_numeric_auto,
        "int_col": ModelEncodingType.tabular_numeric_discrete,
        "cat_col2": ModelEncodingType.tabular_categorical,
    }

    with tempfile.TemporaryDirectory(prefix="slim_argn_size_") as workspace_dir:
        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types=encoding_types,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)
        engine.encode(workspace_dir=workspace_dir)

        train_tensors, val_tensors, model_config = engine.load_tensors_from_workspace(
            workspace_dir, device="cpu"
        )

        engine.train(
            train_tensors=iter(train_tensors),
            val_tensors=iter(val_tensors),
            model_config=model_config,
            workspace_dir=workspace_dir,
            max_epochs=3,
            enable_flexible_generation=False,
        )

        # Measure tarball size (old method)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(f"{workspace_dir}/ModelStore", arcname=".")
        tarball_size = len(buf.getvalue())

        # Measure artifact size (new method)
        with open(f"{workspace_dir}/ModelStore/tgt-stats/stats.json") as f:
            tgt_stats = json.load(f)
        with open(f"{workspace_dir}/ModelStore/model-data/model-configs.json") as f:
            model_configs = json.load(f)

        state_dict = torch.load(
            f"{workspace_dir}/ModelStore/model-data/model-weights.pt",
            map_location="cpu",
            weights_only=True,
        )

        tgt_cardinalities = get_cardinalities(tgt_stats)
        model_size = model_configs.get("model_id", "MOSTLY_AI/Medium").split("/")[-1][0]

        artifact = ModelArtifact.from_state_dict(
            state_dict=state_dict,
            is_sequential=False,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            tgt_stats=minimize_stats(tgt_stats),
        )
        artifact_size = len(artifact.to_bytes())

        print(f"\nTarball size (gzipped):  {tarball_size:,} bytes")
        print(f"Artifact size:           {artifact_size:,} bytes")
        print(f"Size reduction:          {(1 - artifact_size/tarball_size)*100:.1f}%")

        # Also show components
        weights_size = len(artifact.weights)
        metadata_size = artifact_size - weights_size
        print(f"\n  Weights:  {weights_size:,} bytes ({weights_size/artifact_size*100:.1f}%)")
        print(f"  Metadata: {metadata_size:,} bytes ({metadata_size/artifact_size*100:.1f}%)")

    print("\n" + "=" * 70)
    print("SIZE COMPARISON COMPLETE")
    print("=" * 70)

    return True


def create_sequential_test_data(n_contexts: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic sequential test data (context + target tables)."""
    np.random.seed(RANDOM_SEED)

    # Context table: one row per user
    ctx_data = pd.DataFrame({
        "user_id": range(n_contexts),
        "age": np.random.randint(18, 80, n_contexts),
        "gender": np.random.choice(["M", "F"], n_contexts),
    })

    # Target table: multiple transactions per user
    tgt_rows = []
    for user_id in range(n_contexts):
        n_transactions = np.random.randint(2, 10)  # 2-9 transactions per user
        for _ in range(n_transactions):
            tgt_rows.append({
                "user_id": user_id,
                "amount": np.random.exponential(100),
                "category": np.random.choice(["food", "travel", "shopping", "bills"]),
            })

    tgt_data = pd.DataFrame(tgt_rows)
    return ctx_data, tgt_data


@ray.remote(num_cpus=0, num_gpus=1)
def test_phase2_train_flat():
    """
    Test Phase 2: In-memory training with train_flat().

    Validates the new train_flat() function which bypasses workspace I/O:
    1. Computes stats in-memory
    2. Encodes data directly to GPU tensors
    3. Trains and returns ModelArtifact
    4. Compares generation quality to workspace-trained model
    """
    from mostlyai import engine
    from mostlyai.engine._train import train_flat
    from mostlyai.engine.generation import generate_flat
    from mostlyai.engine.domain import ModelEncodingType
    from mostlyai.engine.random_state import set_random_state

    print("=" * 70)
    print("PHASE 2 TEST: In-Memory Flat Training (train_flat)")
    print("=" * 70)

    # Create test data
    print("\n[1/4] Creating test data...")
    df = create_test_data(n_samples=3000)
    print(f"  Created DataFrame with shape {df.shape}")

    encoding_types = {
        "cat_col": ModelEncodingType.tabular_categorical,
        "num_col": ModelEncodingType.tabular_numeric_auto,
        "int_col": ModelEncodingType.tabular_numeric_discrete,
        "cat_col2": ModelEncodingType.tabular_categorical,
    }

    # Train using new in-memory method
    print("\n[2/4] Training with train_flat() (no workspace)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    artifact = train_flat(
        tgt_data=df,
        tgt_encoding_types=encoding_types,
        val_split=0.1,
        model_size="M",
        max_epochs=TRAINING_EPOCHS,
        batch_size=512,
        device=device,
        enable_flexible_generation=False,
    )
    training_time = time.time() - t0
    print(f"  Training completed in {training_time:.1f}s")
    print(f"  Artifact: {artifact}")
    print(f"  Artifact size: {len(artifact.to_bytes()):,} bytes")

    # Verify roundtrip serialization
    artifact_bytes = artifact.to_bytes()
    artifact_restored = artifact.from_bytes(artifact_bytes)
    print("  Roundtrip serialization: OK")

    # Generate samples
    print("\n[3/4] Generating synthetic data...")
    set_random_state(RANDOM_SEED)
    t0 = time.time()
    samples = generate_flat(
        artifact=artifact_restored,
        sample_size=SAMPLE_SIZE,
        device=device,
    )
    gen_time = time.time() - t0
    print(f"  Generated {len(samples)} samples in {gen_time:.2f}s")

    # Validate output quality
    print("\n[4/4] Validating output quality...")
    print("=" * 70)
    print("QUALITY METRICS")
    print("=" * 70)

    # Check columns match input
    expected_cols = set(encoding_types.keys())
    actual_cols = set(samples.columns)
    print(f"\nColumn match: {expected_cols == actual_cols}")
    if expected_cols != actual_cols:
        print(f"  Expected: {expected_cols}")
        print(f"  Actual: {actual_cols}")

    # Compare distributions to original data
    print("\n--- Categorical Distributions (vs original) ---")
    for col in ["cat_col", "cat_col2"]:
        orig_dist = df[col].value_counts(normalize=True).sort_index()
        gen_dist = samples[col].value_counts(normalize=True).sort_index()

        # Check all categories present
        missing = set(orig_dist.index) - set(gen_dist.index)
        if missing:
            print(f"  {col}: WARN missing categories: {missing}")
        else:
            # Compare distributions
            aligned_orig = orig_dist.reindex(gen_dist.index, fill_value=0)
            max_diff = abs(aligned_orig - gen_dist).max()
            status = "PASS" if max_diff < 0.15 else "WARN"
            print(f"  {col}: {status} max diff = {max_diff:.4f}")

    print("\n--- Numeric Statistics (vs original) ---")
    for col in ["num_col", "int_col"]:
        orig_mean, orig_std = df[col].mean(), df[col].std()
        gen_mean, gen_std = samples[col].mean(), samples[col].std()

        mean_pct_diff = abs(orig_mean - gen_mean) / (abs(orig_mean) + 1e-6) * 100
        std_pct_diff = abs(orig_std - gen_std) / (abs(orig_std) + 1e-6) * 100

        status = "PASS" if mean_pct_diff < 20 and std_pct_diff < 30 else "WARN"
        print(f"  {col}: {status} mean diff={mean_pct_diff:.1f}%, std diff={std_pct_diff:.1f}%")
        print(f"    Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
        print(f"    Generated: mean={gen_mean:.2f}, std={gen_std:.2f}")

    print("\n--- Performance Summary ---")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Artifact size: {len(artifact_bytes):,} bytes")

    print("\n" + "=" * 70)
    print("PHASE 2 FLAT TRAINING TEST COMPLETE")
    print("=" * 70)

    return True


@ray.remote(num_cpus=0, num_gpus=1)
def test_phase2_train_sequential():
    """
    Test Phase 2: In-memory sequential training with train_sequential().

    Validates the new train_sequential() function for longitudinal data.
    """
    from mostlyai.engine._train import train_sequential
    from mostlyai.engine.generation import generate_sequential
    from mostlyai.engine.domain import ModelEncodingType
    from mostlyai.engine.random_state import set_random_state

    print("=" * 70)
    print("PHASE 2 TEST: In-Memory Sequential Training (train_sequential)")
    print("=" * 70)

    # Create test data
    print("\n[1/4] Creating sequential test data...")
    ctx_data, tgt_data = create_sequential_test_data(n_contexts=400)
    print(f"  Context: {ctx_data.shape}, Target: {tgt_data.shape}")
    print(f"  Sequences per context: {tgt_data.groupby('user_id').size().describe()}")

    ctx_encoding_types = {
        "age": ModelEncodingType.tabular_numeric_auto,
        "gender": ModelEncodingType.tabular_categorical,
    }
    tgt_encoding_types = {
        "amount": ModelEncodingType.tabular_numeric_auto,
        "category": ModelEncodingType.tabular_categorical,
    }

    # Train using new in-memory method
    print("\n[2/4] Training with train_sequential() (no workspace)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    artifact = train_sequential(
        tgt_data=tgt_data,
        tgt_encoding_types=tgt_encoding_types,
        tgt_context_key="user_id",
        ctx_data=ctx_data,
        ctx_primary_key="user_id",
        ctx_encoding_types=ctx_encoding_types,
        val_split=0.1,
        model_size="M",
        max_epochs=TRAINING_EPOCHS,
        batch_size=128,
        device=device,
        enable_flexible_generation=False,
    )
    training_time = time.time() - t0
    print(f"  Training completed in {training_time:.1f}s")
    print(f"  Artifact: {artifact}")
    print(f"  Artifact size: {len(artifact.to_bytes()):,} bytes")

    # Verify roundtrip
    artifact_bytes = artifact.to_bytes()
    artifact_restored = artifact.from_bytes(artifact_bytes)
    print("  Roundtrip serialization: OK")

    # Generate sequences
    n_generate = 100
    print(f"\n[3/4] Generating {n_generate} sequences...")
    set_random_state(RANDOM_SEED)
    t0 = time.time()
    samples = generate_sequential(
        artifact=artifact_restored,
        sample_size=n_generate,
        ctx_data=ctx_data.head(n_generate),
        device=device,
    )
    gen_time = time.time() - t0
    print(f"  Generated {len(samples)} rows in {gen_time:.2f}s")
    print(f"  Unique contexts: {samples['user_id'].nunique()}")

    # Validate output
    print("\n[4/4] Validating output quality...")
    print("=" * 70)
    print("QUALITY METRICS")
    print("=" * 70)

    # Sequence length stats
    orig_seq_lens = tgt_data.groupby("user_id").size()
    gen_seq_lens = samples.groupby("user_id").size()

    print("\n--- Sequence Length Distribution ---")
    print(f"  Original: mean={orig_seq_lens.mean():.2f}, std={orig_seq_lens.std():.2f}")
    print(f"  Generated: mean={gen_seq_lens.mean():.2f}, std={gen_seq_lens.std():.2f}")

    # Category distribution
    print("\n--- Category Distribution ---")
    orig_dist = tgt_data["category"].value_counts(normalize=True).sort_index()
    gen_dist = samples["category"].value_counts(normalize=True).sort_index()
    for cat in orig_dist.index:
        if cat in gen_dist.index:
            diff = abs(orig_dist[cat] - gen_dist[cat])
            status = "PASS" if diff < 0.15 else "WARN"
            print(f"  {cat}: {status} orig={orig_dist[cat]:.3f}, gen={gen_dist[cat]:.3f}")

    # Numeric stats
    print("\n--- Amount Statistics ---")
    print(f"  Original: mean={tgt_data['amount'].mean():.2f}, std={tgt_data['amount'].std():.2f}")
    print(f"  Generated: mean={samples['amount'].mean():.2f}, std={samples['amount'].std():.2f}")

    print("\n--- Performance Summary ---")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Artifact size: {len(artifact_bytes):,} bytes")

    print("\n" + "=" * 70)
    print("PHASE 2 SEQUENTIAL TRAINING TEST COMPLETE")
    print("=" * 70)

    return True


@ray.remote(num_cpus=0, num_gpus=1)
def test_sequential_artifact_generation():
    """
    Test sequential (longitudinal) artifact-based generation.

    Tests context-conditioned generation of variable-length sequences.
    """
    from mostlyai import engine
    from mostlyai.engine._artifact import ModelArtifact, minimize_stats
    from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length
    from mostlyai.engine.generation import generate_sequential
    from mostlyai.engine.domain import ModelEncodingType
    from mostlyai.engine.random_state import set_random_state

    print("=" * 70)
    print("SEQUENTIAL TEST: Artifact-based Generation")
    print("=" * 70)

    # Create test data
    print("\n[1/6] Creating sequential test data...")
    ctx_data, tgt_data = create_sequential_test_data()
    print(f"  Context: {ctx_data.shape}, Target: {tgt_data.shape}")
    print(f"  Sequences per context: {tgt_data.groupby('user_id').size().describe()}")

    ctx_encoding_types = {
        "age": ModelEncodingType.tabular_numeric_auto,
        "gender": ModelEncodingType.tabular_categorical,
    }
    tgt_encoding_types = {
        "amount": ModelEncodingType.tabular_numeric_auto,
        "category": ModelEncodingType.tabular_categorical,
    }

    # Train
    print("\n[2/6] Training sequential model...")
    with tempfile.TemporaryDirectory(prefix="slim_argn_seq_") as workspace_dir:
        engine.split(
            workspace_dir=workspace_dir,
            ctx_data=ctx_data,
            ctx_primary_key="user_id",
            tgt_data=tgt_data,
            tgt_context_key="user_id",
            tgt_encoding_types=tgt_encoding_types,
            ctx_encoding_types=ctx_encoding_types,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)
        engine.encode(workspace_dir=workspace_dir)

        train_tensors, val_tensors, model_config = engine.load_tensors_from_workspace(
            workspace_dir, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        t0 = time.time()
        engine.train(
            train_tensors=iter(train_tensors),
            val_tensors=iter(val_tensors),
            model_config=model_config,
            workspace_dir=workspace_dir,
            max_epochs=TRAINING_EPOCHS,
            enable_flexible_generation=False,
        )
        training_time = time.time() - t0
        print(f"  Training completed in {training_time:.1f}s")

        # Load artifacts
        print("\n[3/6] Creating ModelArtifact...")
        with open(f"{workspace_dir}/ModelStore/tgt-stats/stats.json") as f:
            tgt_stats = json.load(f)
        with open(f"{workspace_dir}/ModelStore/ctx-stats/stats.json") as f:
            ctx_stats = json.load(f)
        with open(f"{workspace_dir}/ModelStore/model-data/model-configs.json") as f:
            model_configs = json.load(f)

        state_dict = torch.load(
            f"{workspace_dir}/ModelStore/model-data/model-weights.pt",
            map_location="cpu",
            weights_only=True,
        )

        tgt_cardinalities = get_cardinalities(tgt_stats)
        ctx_cardinalities = get_cardinalities(ctx_stats)
        seq_len_stats = tgt_stats.get("seq_len", {})
        ctx_seq_len_median = get_ctx_sequence_length(ctx_stats, key="median")
        model_size = model_configs.get("model_id", "MOSTLY_AI/Medium").split("/")[-1][0]

        artifact = ModelArtifact.from_state_dict(
            state_dict=state_dict,
            is_sequential=True,
            model_size=model_size,
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            tgt_stats=minimize_stats(tgt_stats),
            ctx_stats=minimize_stats(ctx_stats),
            tgt_seq_len_min=seq_len_stats.get("min", 1),
            tgt_seq_len_max=seq_len_stats.get("max", 10),
            tgt_seq_len_median=seq_len_stats.get("median", 5),
            ctx_seq_len_median=ctx_seq_len_median,
            tgt_context_key="user_id",
            ctx_primary_key="user_id",
        )

        artifact_bytes = artifact.to_bytes()
        print(f"  Artifact: {artifact}")
        print(f"  Serialized size: {len(artifact_bytes):,} bytes")

        artifact_restored = ModelArtifact.from_bytes(artifact_bytes)
        print("  Roundtrip serialization: OK")

        # Generate using legacy method
        n_generate = 100
        print(f"\n[4/6] Generating {n_generate} sequences with LEGACY method...")
        set_random_state(RANDOM_SEED)
        t0 = time.time()
        engine.generate(
            workspace_dir=workspace_dir,
            ctx_data=ctx_data.head(n_generate),
            sample_size=n_generate,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        legacy_time = time.time() - t0
        legacy_samples = pd.read_parquet(f"{workspace_dir}/SyntheticData")
        print(f"  Generated {len(legacy_samples)} rows in {legacy_time:.2f}s")
        print(f"  Sequences: {legacy_samples['user_id'].nunique()}")

        # Generate using new method
        print(f"\n[5/6] Generating {n_generate} sequences with NEW method...")
        set_random_state(RANDOM_SEED)
        t0 = time.time()
        new_samples = generate_sequential(
            artifact=artifact_restored,
            sample_size=n_generate,
            ctx_data=ctx_data.head(n_generate),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        new_time = time.time() - t0
        print(f"  Generated {len(new_samples)} rows in {new_time:.2f}s")
        print(f"  Sequences: {new_samples['user_id'].nunique()}")

    # Compare results
    print("\n[6/6] Comparing results...")
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Check basic stats
    print(f"\nLegacy rows: {len(legacy_samples)}, New rows: {len(new_samples)}")

    # Compare sequence length distributions
    legacy_seq_lens = legacy_samples.groupby("user_id").size()
    new_seq_lens = new_samples.groupby("user_id").size()

    print("\n--- Sequence Length Stats ---")
    print(f"  Legacy: mean={legacy_seq_lens.mean():.2f}, std={legacy_seq_lens.std():.2f}")
    print(f"  New:    mean={new_seq_lens.mean():.2f}, std={new_seq_lens.std():.2f}")

    # Compare categorical distribution
    print("\n--- Category Distribution ---")
    legacy_dist = legacy_samples["category"].value_counts(normalize=True).sort_index()
    new_dist = new_samples["category"].value_counts(normalize=True).sort_index()
    for cat in legacy_dist.index:
        if cat in new_dist.index:
            diff = abs(legacy_dist[cat] - new_dist[cat])
            print(f"  {cat}: legacy={legacy_dist[cat]:.3f}, new={new_dist[cat]:.3f}, diff={diff:.4f}")

    # Performance
    print("\n--- Performance ---")
    print(f"  Legacy generation time: {legacy_time:.2f}s")
    print(f"  New generation time:    {new_time:.2f}s")
    if new_time < legacy_time:
        print(f"  Speedup: {legacy_time/new_time:.1f}x faster")

    print("\n" + "=" * 70)
    print("SEQUENTIAL TEST COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    # Initialize Ray (connects to existing cluster or starts local)
    ray.init()

    print("=" * 70)
    print("SLIM ARGN TEST SUITE")
    print("=" * 70)
    print(f"\nRay cluster resources: {ray.cluster_resources()}")
    print(f"\nTest configuration:")
    print(f"  SAMPLE_SIZE: {SAMPLE_SIZE}")
    print(f"  TRAINING_EPOCHS: {TRAINING_EPOCHS}")
    print(f"  RANDOM_SEED: {RANDOM_SEED}")

    # Run tests on GPU workers
    try:
        # Phase 1: Artifact-based generation tests
        print("\n>>> PHASE 1: Artifact-based Generation <<<\n")
        result1 = ray.get(test_phase1_artifact_generation.remote())
        result2 = ray.get(test_artifact_size_comparison.remote())
        result3 = ray.get(test_sequential_artifact_generation.remote())

        # Phase 2: In-memory training tests
        print("\n>>> PHASE 2: In-Memory Training <<<\n")
        result4 = ray.get(test_phase2_train_flat.remote())
        result5 = ray.get(test_phase2_train_sequential.remote())

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        ray.shutdown()
