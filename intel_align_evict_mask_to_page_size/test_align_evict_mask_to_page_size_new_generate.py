import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_align_evict_mask_to_page_size import align_evict_mask_to_page_size


@triton.jit
def align_evict_mask_to_page_size_reference(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


class TestAlignEvictMaskToPageSize(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        self.page_size = 16
        self.num_draft_tokens = 32
        self.block_size = 512

    def test_basic_functionality_cpu_extension_vs_triton(self):
        # Test basic functionality with small tensors
        batch_size = 2
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.randint(0, 2, (batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            self.page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=self.page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results (move Triton result to CPU for comparison)
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    def test_boundary_values_cpu_extension_vs_triton(self):
        # Test with boundary values
        batch_size = 1
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([5], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.ones((batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            self.page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=self.page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    def test_all_false_mask_cpu_extension_vs_triton(self):
        # Test with all False mask
        batch_size = 3
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([8, 16, 24], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.zeros((batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            self.page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=self.page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    def test_alternating_mask_cpu_extension_vs_triton(self):
        # Test with alternating True/False mask
        batch_size = 2
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([12, 18], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.zeros((batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        evict_mask_cpu[0, ::2] = True  # Alternate true/false for first batch
        evict_mask_cpu[1, 1::2] = True  # Alternate starting from second element for second batch
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            self.page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=self.page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    def test_different_page_sizes_cpu_extension_vs_triton(self):
        # Test with different page sizes
        batch_size = 2
        page_size = 8  # Different page size
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([7, 9], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.randint(0, 2, (batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    def test_single_sequence_cpu_extension_vs_triton(self):
        # Test with single sequence
        batch_size = 1
        
        # Prepare inputs on CPU for the extension
        seq_lens_cpu = torch.tensor([15], dtype=torch.int32, device="cpu")
        evict_mask_cpu = torch.randint(0, 2, (batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
        # Create a copy for the extension to modify
        evict_mask_cpu_ext = evict_mask_cpu.clone()

        # Prepare inputs on XPU for Triton
        seq_lens_xpu = seq_lens_cpu.to(self.device)
        evict_mask_xpu = evict_mask_cpu.to(self.device)
        
        # Create a copy for Triton to modify
        evict_mask_xpu_triton = evict_mask_xpu.clone()

        # Run CPU extension
        align_evict_mask_to_page_size(
            seq_lens_cpu,
            evict_mask_cpu_ext,
            self.page_size,
            self.num_draft_tokens,
            self.block_size
        )

        # Run Triton version
        grid = (batch_size,)
        align_evict_mask_to_page_size_reference[grid](
            seq_lens_xpu,
            evict_mask_xpu_triton,
            page_size=self.page_size,
            num_draft_tokens=self.num_draft_tokens,
            BLOCK_SIZE=self.block_size
        )

        # Compare results
        evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
        self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))

    # def test_large_seq_lengths_cpu_extension_vs_triton(self):
    #     # Test with large sequence lengths
    #     batch_size = 4
        
    #     # Prepare inputs on CPU for the extension
    #     seq_lens_cpu = torch.tensor([100, 200, 150, 300], dtype=torch.int32, device="cpu")
    #     evict_mask_cpu = torch.randint(0, 2, (batch_size, self.num_draft_tokens), dtype=torch.bool, device="cpu")
        
    #     # Create a copy for the extension to modify
    #     evict_mask_cpu_ext = evict_mask_cpu.clone()

    #     # Prepare inputs on XPU for Triton
    #     seq_lens_xpu = seq_lens_cpu.to(self.device)
    #     evict_mask_xpu = evict_mask_cpu.to(self.device)
        
    #     # Create a copy for Triton to modify
    #     evict_mask_xpu_triton = evict_mask_xpu.clone()

    #     # Run CPU extension
    #     align_evict_mask_to_page_size(
    #         seq_lens_cpu,
    #         evict_mask_cpu_ext,
    #         self.page_size,
    #         self.num_draft_tokens,
    #         self.block_size
    #     )

    #     # Run Triton version
    #     grid = (batch_size,)
    #     align_evict_mask_to_page_size_reference[grid](
    #         seq_lens_xpu,
    #         evict_mask_xpu_triton,
    #         page_size=self.page_size,
    #         num_draft_tokens=self.num_draft_tokens,
    #         BLOCK_SIZE=self.block_size
    #     )

    #     # Compare results
    #     evict_mask_triton_cpu = evict_mask_xpu_triton.cpu()
    #     self.assertTrue(torch.equal(evict_mask_cpu_ext, evict_mask_triton_cpu))


if __name__ == "__main__":
    unittest.main()
