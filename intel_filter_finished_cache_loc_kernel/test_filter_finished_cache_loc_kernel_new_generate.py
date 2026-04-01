# File Name: ut_filter_finished_cache_loc_kernel.py
import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_filter_finished_cache_loc_kernel import filter_finished_cache_loc_kernel


@triton.jit
def filter_finished_cache_loc_kernel_triton(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


class TestFilterFinishedCacheLocKernel(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        # Skip tests if XPU is not available and Triton is expected to work
        if self.device != "xpu":
            self.skipTest("XPU not available for Triton kernel testing")

    def _run_comparison_test(self, batch_size, num_verify_tokens, max_total_tokens, 
                             tgt_cache_loc_data, accept_length_data, accept_length_filter_data):
        # Prepare inputs on CPU for the extension
        out_cache_loc_cpu = torch.zeros(max_total_tokens, dtype=torch.int32, device="cpu")
        tgt_cache_loc_cpu = torch.tensor(tgt_cache_loc_data, dtype=torch.int32, device="cpu")
        accept_length_cpu = torch.tensor(accept_length_data, dtype=torch.int32, device="cpu")
        accept_length_filter_cpu = torch.tensor(accept_length_filter_data, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        out_cache_loc_xpu = torch.zeros(max_total_tokens, dtype=torch.int32, device=self.device)
        tgt_cache_loc_xpu = tgt_cache_loc_cpu.to(self.device)
        accept_length_xpu = accept_length_cpu.to(self.device)
        accept_length_filter_xpu = accept_length_filter_cpu.to(self.device)

        # Run CPU extension
        filter_finished_cache_loc_kernel(
            out_cache_loc_cpu,
            tgt_cache_loc_cpu,
            accept_length_cpu,
            accept_length_filter_cpu,
            batch_size,
            num_verify_tokens
        )

        # Run Triton version
        grid = (batch_size,)
        filter_finished_cache_loc_kernel_triton[grid](
            out_cache_loc_xpu,
            tgt_cache_loc_xpu,
            accept_length_xpu,
            accept_length_filter_xpu,
            bs_upper=batch_size,
            num_verify_tokens_upper=num_verify_tokens
        )

        # Compare results (move Triton result to CPU for comparison)
        out_cache_loc_triton_cpu = out_cache_loc_xpu.cpu()
        return torch.equal(out_cache_loc_cpu, out_cache_loc_triton_cpu), out_cache_loc_cpu, out_cache_loc_triton_cpu

    def test_basic_functionality_cpu_extension_vs_triton(self):
        # Test basic functionality with small tensors
        batch_size = 8  # Power of 2 to satisfy Triton requirement
        num_verify_tokens = 8  # Power of 2 to satisfy Triton requirement
        max_total_tokens = 16

        tgt_cache_loc_data = list(range(0, max_total_tokens))
        accept_length_data = [3, 2, 1, 4, 2, 3, 1, 2]
        accept_length_filter_data = [2, 3, 1, 3, 2, 2, 1, 1]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_boundary_values_cpu_extension_vs_triton(self):
        # Test with boundary values using power-of-2 dimensions
        batch_size = 4  # Power of 2
        num_verify_tokens = 8  # Power of 2
        max_total_tokens = 16

        tgt_cache_loc_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        accept_length_data = [2, 1, 3, 2]
        accept_length_filter_data = [3, 1, 2, 2]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_with_zero_accept_lengths_cpu_extension_vs_triton(self):
        # Test with zero accept lengths using power-of-2 dimensions
        batch_size = 8  # Power of 2
        num_verify_tokens = 8  # Power of 2
        max_total_tokens = 32

        tgt_cache_loc_data = list(range(0, max_total_tokens))
        accept_length_data = [0, 1, 2, 0, 3, 1, 0, 2]
        accept_length_filter_data = [0, 1, 2, 0, 3, 1, 0, 2]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_large_batch_cpu_extension_vs_triton(self):
        # Test with larger batch size using power-of-2 dimensions
        batch_size = 16  # Power of 2
        num_verify_tokens = 16  # Power of 2
        max_total_tokens = 64

        tgt_cache_loc_data = torch.randint(0, 100, (max_total_tokens,)).tolist()
        accept_length_data = torch.randint(1, 5, (batch_size,)).tolist()
        accept_length_filter_data = torch.randint(1, 4, (batch_size,)).tolist()

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_empty_tensors(self):
        # Test with minimal valid inputs using power-of-2 dimensions
        batch_size = 1  # Power of 2
        num_verify_tokens = 1  # Power of 2
        max_total_tokens = 1

        tgt_cache_loc_data = [0]
        accept_length_data = [0]
        accept_length_filter_data = [0]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_max_values_cpu_extension_vs_triton(self):
        # Test with maximum values using power-of-2 dimensions
        batch_size = 8  # Power of 2
        num_verify_tokens = 8  # Power of 2
        max_total_tokens = 32

        tgt_cache_loc_data = [999999] * max_total_tokens
        accept_length_data = [5, 7, 3, 4, 6, 2, 8, 5]
        accept_length_filter_data = [4, 6, 2, 3, 5, 2, 7, 4]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_different_input_patterns(self):
        # Additional test with different patterns using power-of-2 dimensions
        batch_size = 4  # Power of 2
        num_verify_tokens = 4  # Power of 2
        max_total_tokens = 20

        # Alternating high/low values
        tgt_cache_loc_data = [i * 10 if i % 2 == 0 else i * 5 for i in range(max_total_tokens)]
        accept_length_data = [4, 3, 5, 2]
        accept_length_filter_data = [3, 2, 4, 1]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")

    def test_all_zeros_and_ones(self):
        # Test with all zeros and ones using power-of-2 dimensions
        batch_size = 2  # Power of 2
        num_verify_tokens = 4  # Power of 2
        max_total_tokens = 10

        tgt_cache_loc_data = [1] * max_total_tokens
        accept_length_data = [3, 2]
        accept_length_filter_data = [2, 3]

        equal, cpu_result, triton_result = self._run_comparison_test(
            batch_size, num_verify_tokens, max_total_tokens,
            tgt_cache_loc_data, accept_length_data, accept_length_filter_data
        )
        self.assertTrue(equal, f"Results differ:CPU: {cpu_result}Triton: {triton_result}")


if __name__ == "__main__":
    unittest.main()
