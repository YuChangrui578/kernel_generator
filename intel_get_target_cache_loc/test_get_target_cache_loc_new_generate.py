
# File Name: ut_get_target_cache_loc.py
import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_get_target_cache_loc import get_target_cache_loc


@triton.jit
def get_target_cache_loc_triton(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


class TestGetTargetCacheLoc(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def _calculate_output_sizes(self, accept_length, to_free_num_slots):
        """Helper to calculate correct output tensor sizes"""
        total_tgt_tokens = sum(accept_length.tolist()) + len(accept_length)
        total_free_slots = sum(to_free_num_slots.tolist())
        return total_tgt_tokens, total_free_slots

    def test_basic_functionality_cpu_extension_vs_triton(self):
        # Test basic functionality with small tensors
        bs_upper = 2
        num_verify_tokens = 5
        num_verify_tokens_upper = 8  # Power of 2
        accept_length_cpu = torch.tensor([2, 3], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([2, 4], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24]
        ], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton (since device is CPU)
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version on CPU
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Compare results
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_boundary_values_cpu_extension_vs_triton(self):
        # Test with boundary values
        bs_upper = 1
        num_verify_tokens = 3
        num_verify_tokens_upper = 4  # Power of 2
        accept_length_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([2], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([[100, 200, 300]], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Compare results
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_with_larger_batch_cpu_extension_vs_triton(self):
        # Test with larger batch size - updated to use power of 2 for num_verify_tokens_upper
        bs_upper = 2
        num_verify_tokens = 4
        num_verify_tokens_upper = 8  # Changed from 6 to 8 (power of 2)
        accept_lengths = [1, 2, 1]
        free_slot_counts = [1, 2, 1]
        accept_length_cpu = torch.tensor(accept_lengths, dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor(free_slot_counts, dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33]
        ], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_empty_case_cpu_extension_vs_triton(self):
        # Test with empty case (all lengths zero)
        bs_upper = 2
        num_verify_tokens = 3
        num_verify_tokens_upper = 4  # Power of 2
        accept_length_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([
            [10, 11, 12],
            [20, 21, 22]
        ], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_dtype_int64_cpu_extension_vs_triton(self):
        # Test with int64 dtype
        bs_upper = 2
        num_verify_tokens = 5
        num_verify_tokens_upper = 8  # Power of 2
        accept_length_cpu = torch.tensor([2, 3], dtype=torch.int64, device="cpu")
        to_free_num_slots_cpu = torch.tensor([2, 4], dtype=torch.int64, device="cpu")
        out_cache_loc_cpu = torch.tensor([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24]
        ], dtype=torch.int64, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int64, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int64, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_single_batch_edge_case(self):
        # Test with single batch edge case
        bs_upper = 1
        num_verify_tokens = 1
        num_verify_tokens_upper = 2  # Power of 2
        accept_length_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([[10]], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_different_dtypes(self):
        # Test with different dtypes to ensure compatibility
        for dtype in [torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):
                bs_upper = 2
                num_verify_tokens = 4
                num_verify_tokens_upper = 8
                accept_length_cpu = torch.tensor([1, 2], dtype=dtype, device="cpu")
                to_free_num_slots_cpu = torch.tensor([1, 2], dtype=dtype, device="cpu")
                out_cache_loc_cpu = torch.tensor([
                    [10, 11, 12, 13],
                    [20, 21, 22, 23]
                ], dtype=dtype, device="cpu")
                
                total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

                # Prepare inputs on CPU for the extension
                tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=dtype, device="cpu")
                to_free_slots_cpu = torch.zeros(total_free_slots, dtype=dtype, device="cpu")

                # Prepare inputs on CPU for Triton
                tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
                to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
                accept_length_triton = accept_length_cpu.clone().to(self.device)
                to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
                out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

                # Run CPU extension
                get_target_cache_loc(
                    tgt_cache_loc_cpu,
                    to_free_slots_cpu,
                    accept_length_cpu,
                    to_free_num_slots_cpu,
                    out_cache_loc_cpu,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    bs_upper
                )

                # Run Triton version
                grid = (bs_upper,)
                get_target_cache_loc_triton[grid](
                    tgt_cache_loc_triton,
                    to_free_slots_triton,
                    accept_length_triton,
                    to_free_num_slots_triton,
                    out_cache_loc_triton,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    bs_upper
                )
                tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
                to_free_slots_triton=to_free_slots_triton.to("cpu")
                # Compare results
                self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
                self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_large_accept_length(self):
        # Test with larger accept length values
        bs_upper = 1
        num_verify_tokens = 10
        num_verify_tokens_upper = 16  # Power of 2
        accept_length_cpu = torch.tensor([15], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([3], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.arange(10, dtype=torch.int32, device="cpu").unsqueeze(0)
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_random_data_consistency(self):
        # Test with random data to ensure general consistency
        torch.manual_seed(42)  # For reproducible tests
        for bs_upper in [1, 2, 4]:
            with self.subTest(bs_upper=bs_upper):
                num_verify_tokens = 6
                num_verify_tokens_upper = 8
                accept_length_cpu = torch.randint(0, min(num_verify_tokens, 8), (bs_upper,), dtype=torch.int32, device="cpu")
                to_free_num_slots_cpu = torch.randint(0, min(num_verify_tokens, 6), (bs_upper,), dtype=torch.int32, device="cpu")
                out_cache_loc_cpu = torch.randint(0, 1000, (bs_upper, num_verify_tokens), dtype=torch.int32, device="cpu")
                
                total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

                # Prepare inputs on CPU for the extension
                tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
                to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

                # Prepare inputs on CPU for Triton
                tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
                to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
                accept_length_triton = accept_length_cpu.clone().to(self.device)
                to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
                out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

                # Run CPU extension
                get_target_cache_loc(
                    tgt_cache_loc_cpu,
                    to_free_slots_cpu,
                    accept_length_cpu,
                    to_free_num_slots_cpu,
                    out_cache_loc_cpu,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    bs_upper
                )

                # Run Triton version
                grid = (bs_upper,)
                get_target_cache_loc_triton[grid](
                    tgt_cache_loc_triton,
                    to_free_slots_triton,
                    accept_length_triton,
                    to_free_num_slots_triton,
                    out_cache_loc_triton,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    bs_upper
                )
                tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
                to_free_slots_triton=to_free_slots_triton.to("cpu")
                # Compare results
                self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
                self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_minimal_case(self):
        # Test with minimal possible values
        bs_upper = 1
        num_verify_tokens = 1
        num_verify_tokens_upper = 2  # Power of 2
        accept_length_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([[42]], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_large_batch_size(self):
        # Test with a larger batch size to stress-test the implementation
        bs_upper = 8
        num_verify_tokens = 10
        num_verify_tokens_upper = 16  # Power of 2
        accept_length_cpu = torch.randint(1, 6, (bs_upper,), dtype=torch.int32, device="cpu")
        to_free_num_slots_cpu = torch.randint(1, 5, (bs_upper,), dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.randint(0, 100, (bs_upper, num_verify_tokens), dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))

    def test_very_large_accept_length(self):
        # Test with very large accept length relative to num_verify_tokens
        bs_upper = 1
        num_verify_tokens = 5
        num_verify_tokens_upper = 8  # Power of 2
        accept_length_cpu = torch.tensor([10], dtype=torch.int32, device="cpu")  # Larger than num_verify_tokens
        to_free_num_slots_cpu = torch.tensor([2], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int32, device="cpu")
        
        total_tgt_tokens, total_free_slots = self._calculate_output_sizes(accept_length_cpu, to_free_num_slots_cpu)

        # Prepare inputs on CPU for the extension
        tgt_cache_loc_cpu = torch.zeros(total_tgt_tokens, dtype=torch.int32, device="cpu")
        to_free_slots_cpu = torch.zeros(total_free_slots, dtype=torch.int32, device="cpu")

        # Prepare inputs on CPU for Triton
        tgt_cache_loc_triton = torch.zeros(total_tgt_tokens, dtype=torch.int32, device=self.device)
        to_free_slots_triton = torch.zeros(total_free_slots, dtype=torch.int32, device=self.device)
        accept_length_triton = accept_length_cpu.clone().to(self.device)
        to_free_num_slots_triton = to_free_num_slots_cpu.clone().to(self.device)
        out_cache_loc_triton = out_cache_loc_cpu.clone().to(self.device)

        # Run CPU extension
        get_target_cache_loc(
            tgt_cache_loc_cpu,
            to_free_slots_cpu,
            accept_length_cpu,
            to_free_num_slots_cpu,
            out_cache_loc_cpu,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )

        # Run Triton version
        grid = (bs_upper,)
        get_target_cache_loc_triton[grid](
            tgt_cache_loc_triton,
            to_free_slots_triton,
            accept_length_triton,
            to_free_num_slots_triton,
            out_cache_loc_triton,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        )
        tgt_cache_loc_triton=tgt_cache_loc_triton.to("cpu")
        to_free_slots_triton=to_free_slots_triton.to("cpu")
        # Compare results
        self.assertTrue(torch.equal(tgt_cache_loc_cpu, tgt_cache_loc_triton))
        self.assertTrue(torch.equal(to_free_slots_cpu, to_free_slots_triton))


if __name__ == "__main__":
    unittest.main()

