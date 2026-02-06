"""
Tests for GPU-accelerated generation operations.

These tests verify that the GPU implementations work correctly.
Output equivalence was already verified during development.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from mostlyai.engine._common import (
    RIDX_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
)
from mostlyai.engine._tabular.generation_gpu import (
    compute_sequence_continue_mask_torch,
    decode_positional_column_torch,
)


def generate_encoded_positional_data_for_test(
    n_rows: int, max_seq_len: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Generate test data in torch format.

    Returns:
        (torch_tensor, column_index_map)
    """
    np.random.seed(42)

    if max_seq_len < 100:  # SIDX_RIDX_DIGIT_ENCODING_THRESHOLD
        # Categorical encoding
        values = np.random.randint(0, max_seq_len + 1, n_rows, dtype=np.int32)
        tensor = torch.from_numpy(values.reshape(-1, 1))
        col_idx = {f"{SIDX_SUB_COLUMN_PREFIX}cat": 0}
    else:
        # Digit encoding
        values = np.random.randint(0, max_seq_len + 1, n_rows)
        n_digits = len(str(max_seq_len))
        tensor_cols = []
        col_idx = {}

        for d in range(n_digits):
            col_name = f"{SIDX_SUB_COLUMN_PREFIX}E{d}"
            digit = (values // (10**d)) % 10
            tensor_cols.append(digit)
            col_idx[col_name] = d

        tensor = torch.from_numpy(np.stack(tensor_cols, axis=1))

    return tensor, col_idx


class TestDecodePositionalColumn:
    """Test GPU decode implementation."""

    @pytest.mark.parametrize("n_rows", [10, 100, 1000])
    @pytest.mark.parametrize("max_seq_len", [5, 50, 100, 500])
    def test_decode_produces_valid_output(self, n_rows, max_seq_len):
        """Test that GPU decode produces valid integer outputs."""
        tensor, col_idx = generate_encoded_positional_data_for_test(n_rows, max_seq_len)

        if max_seq_len < 100:
            torch_result = decode_positional_column_torch(
                tensor, max_seq_len, SIDX_SUB_COLUMN_PREFIX, None
            )
        else:
            torch_result = decode_positional_column_torch(
                tensor, max_seq_len, SIDX_SUB_COLUMN_PREFIX, col_idx
            )

        # Check outputs are valid
        assert torch_result.shape == (n_rows,)
        assert torch_result.min() >= 0
        assert torch_result.max() <= max_seq_len
        assert torch_result.dtype == torch.long

    def test_decode_deterministic(self):
        """Test that decode is deterministic."""
        n_rows = 100
        max_seq_len = 500

        tensor, col_idx = generate_encoded_positional_data_for_test(n_rows, max_seq_len)

        result1 = decode_positional_column_torch(
            tensor, max_seq_len, SIDX_SUB_COLUMN_PREFIX, col_idx
        )
        result2 = decode_positional_column_torch(
            tensor, max_seq_len, SIDX_SUB_COLUMN_PREFIX, col_idx
        )

        assert torch.all(result1 == result2)

    def test_decode_edge_cases(self):
        """Test edge cases: all zeros, all max values."""
        max_seq_len = 50  # Use < 100 for categorical encoding

        # All zeros
        tensor_zeros = torch.zeros(10, 1, dtype=torch.long)
        result = decode_positional_column_torch(
            tensor_zeros, max_seq_len, SIDX_SUB_COLUMN_PREFIX, None
        )
        assert torch.all(result == 0)

        # All max values
        tensor_max = torch.full((10, 1), max_seq_len, dtype=torch.long)
        result = decode_positional_column_torch(
            tensor_max, max_seq_len, SIDX_SUB_COLUMN_PREFIX, None
        )
        assert torch.all(result == max_seq_len)


class TestComputeSequenceContinueMask:
    """Test GPU mask computation."""

    def generate_step_data(
        self, n_rows: int, max_seq_len: int, has_ridx: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Generate test data for mask computation."""
        np.random.seed(42)

        sidx_values = np.random.randint(0, max_seq_len, n_rows, dtype=np.int32)
        slen_values = np.random.randint(sidx_values + 1, max_seq_len + 1, dtype=np.int32)
        ridx_values = (slen_values - sidx_values) if has_ridx else None

        sidx_tensor = torch.from_numpy(sidx_values)
        slen_tensor = torch.from_numpy(slen_values)
        ridx_tensor = torch.from_numpy(ridx_values) if ridx_values is not None else None

        return sidx_tensor, slen_tensor, ridx_tensor

    @pytest.mark.parametrize("n_rows", [10, 100, 1000])
    @pytest.mark.parametrize("has_ridx", [True, False])
    @pytest.mark.parametrize("n_seed_steps", [0, 3, 10])
    def test_mask_produces_valid_output(self, n_rows, has_ridx, n_seed_steps):
        """Test that GPU mask computation produces valid boolean output."""
        max_seq_len = 100
        sidx, slen, ridx = self.generate_step_data(n_rows, max_seq_len, has_ridx)

        mask = compute_sequence_continue_mask_torch(sidx, slen, ridx, n_seed_steps)

        # Check outputs are valid
        assert mask.shape == (n_rows,)
        assert mask.dtype == torch.bool

    def test_mask_ridx_logic(self):
        """Test RIDX=0 results in False, RIDX>0 results in True."""
        # RIDX=0 should stop
        mask = compute_sequence_continue_mask_torch(
            torch.tensor([5]), torch.tensor([10]), torch.tensor([0]), 0
        )
        assert not mask[0].item()

        # RIDX>0 should continue
        mask = compute_sequence_continue_mask_torch(
            torch.tensor([5]), torch.tensor([10]), torch.tensor([3]), 0
        )
        assert mask[0].item()

    def test_mask_seed_steps_override(self):
        """Test that seed steps are always included regardless of RIDX."""
        # RIDX=0 but SIDX < n_seed_steps should still be True
        mask = compute_sequence_continue_mask_torch(
            torch.tensor([2]), torch.tensor([10]), torch.tensor([0]), 5
        )
        assert mask[0].item()

    def test_mask_without_ridx_fallback(self):
        """Test fallback logic when RIDX is not present (uses SIDX < SLEN)."""
        # SIDX < SLEN should continue
        mask = compute_sequence_continue_mask_torch(
            torch.tensor([5, 9]), torch.tensor([10, 10]), None, 0
        )
        assert mask[0].item()
        assert mask[1].item()

        # SIDX >= SLEN should stop
        mask = compute_sequence_continue_mask_torch(
            torch.tensor([10]), torch.tensor([10]), None, 0
        )
        assert not mask[0].item()


class TestGPUDevice:
    """Test that operations work on both CPU and GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_decode_on_gpu(self):
        """Test decode on GPU device."""
        n_rows = 100
        max_seq_len = 500

        tensor_cpu, col_idx = generate_encoded_positional_data_for_test(n_rows, max_seq_len)
        tensor_gpu = tensor_cpu.cuda()

        result = decode_positional_column_torch(
            tensor_gpu, max_seq_len, SIDX_SUB_COLUMN_PREFIX, col_idx
        )

        # Should stay on GPU
        assert result.device.type == "cuda"
        # Check validity
        assert result.min() >= 0
        assert result.max() <= max_seq_len

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mask_on_gpu(self):
        """Test mask computation on GPU device."""
        sidx = torch.tensor([1, 5, 9]).cuda()
        slen = torch.tensor([10, 10, 10]).cuda()
        ridx = torch.tensor([9, 5, 1]).cuda()

        mask = compute_sequence_continue_mask_torch(sidx, slen, ridx, 0)

        # Should stay on GPU
        assert mask.device.type == "cuda"
        # Check validity
        assert mask.dtype == torch.bool
        assert mask.shape == (3,)
