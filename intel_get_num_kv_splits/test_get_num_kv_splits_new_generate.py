import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_get_num_kv_splits import get_num_kv_splits


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


class TestGetNumKvSplits(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def test_basic_functionality_cpu_extension_vs_triton(self):
        # Test basic functionality with small tensors
        num_seq = 2
        num_group = 4
        num_head = 8
        num_kv_head = 2
        max_kv_splits = 8
        device_core_count = 16
        MAX_NUM_SEQ = 8

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results (move Triton result to CPU for comparison)
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")

    def test_boundary_values_cpu_extension_vs_triton(self):
        # Test with boundary values
        num_seq = 1
        num_group = 2
        num_head = 4
        num_kv_head = 1
        max_kv_splits = 4
        device_core_count = 8
        MAX_NUM_SEQ = 4

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")

    def test_large_sequence_lengths_cpu_extension_vs_triton(self):
        # Test with larger sequence lengths
        num_seq = 3
        num_group = 2
        num_head = 8
        num_kv_head = 4
        max_kv_splits = 16
        device_core_count = 32
        MAX_NUM_SEQ = 8

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([100, 200, 150], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")

    def test_maximal_splits_cpu_extension_vs_triton(self):
        # Test with maximum possible splits
        num_seq = 4
        num_group = 1
        num_head = 16
        num_kv_head = 1
        max_kv_splits = 32
        device_core_count = 64
        MAX_NUM_SEQ = 8

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([50, 100, 75, 120], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")

    def test_single_sequence_single_group_cpu_extension_vs_triton(self):
        # Test with single sequence and single group
        num_seq = 1
        num_group = 1
        num_head = 2
        num_kv_head = 1
        max_kv_splits = 2
        device_core_count = 4
        MAX_NUM_SEQ = 2

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([5], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")

    def test_various_head_configurations_cpu_extension_vs_triton(self):
        # Test with various head configurations
        num_seq = 2
        num_group = 3
        num_head = 12
        num_kv_head = 4
        max_kv_splits = 10
        device_core_count = 20
        MAX_NUM_SEQ = 4

        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([30, 60], dtype=torch.int32, device="cpu")
        num_kv_splits_cpu = torch.zeros(num_seq * num_group, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        num_kv_splits_triton = torch.zeros(num_seq * num_group, dtype=torch.int32, device=self.device)

        # Run CPU extension - make sure inputs are on CPU
        get_num_kv_splits(
            num_kv_splits_cpu,
            seq_lens_cpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Run Triton version
        grid = (1,)
        get_num_kv_splits_triton[grid](
            num_kv_splits_triton,
            seq_lens_xpu,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        )

        # Compare results
        num_kv_splits_triton_cpu = num_kv_splits_triton.cpu()
        self.assertTrue(torch.equal(num_kv_splits_cpu, num_kv_splits_triton_cpu),f"Results differ:CPU: {num_kv_splits_cpu}Triton: {num_kv_splits_triton_cpu}")


if __name__ == "__main__":
    unittest.main()
