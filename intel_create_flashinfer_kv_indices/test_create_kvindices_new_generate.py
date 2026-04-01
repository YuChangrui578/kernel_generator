
import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_create_flashinfer_kv_indices import create_flashinfer_kv_indices


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


class TestCreateFlashInferKvIndices(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def test_basic_functionality_cpu_extension_vs_triton(self):
        # Test basic functionality with small tensors
        max_batch = 2
        max_context_len = 10
        total_tokens = 15

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.randint(0, 100, (max_batch, max_context_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.arange(max_batch, dtype=torch.int32, device="cpu")
        page_kernel_lens_cpu = torch.tensor([5, 7], dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.tensor([0, 5], dtype=torch.int32, device="cpu")
        kv_start_idx_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        kv_indices_cpu = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        page_kernel_lens_xpu = page_kernel_lens_cpu.to(self.device)
        kv_indptr_xpu = kv_indptr_cpu.to(self.device)
        kv_start_idx_xpu = kv_start_idx_cpu.to(self.device)
        kv_indices_triton = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        # Run CPU extension
        create_flashinfer_kv_indices(
            req_to_token_cpu,
            req_pool_indices_cpu,
            page_kernel_lens_cpu,
            kv_indptr_cpu,
            kv_start_idx_cpu,
            kv_indices_cpu,
            req_to_token_cpu.stride()[0]
        )

        # Run Triton version
        grid = (max_batch,)
        create_flashinfer_kv_indices_triton[grid](
            req_to_token_xpu,
            req_pool_indices_xpu,
            page_kernel_lens_xpu,
            kv_indptr_xpu,
            kv_start_idx_xpu,
            kv_indices_triton,
            req_to_token_xpu.stride()[0]
        )

        # Compare results (move Triton result to CPU for comparison)
        kv_indices_triton_cpu = kv_indices_triton.cpu()
        self.assertTrue(torch.equal(kv_indices_cpu, kv_indices_triton_cpu),f"Results differ:CPU: {kv_indices_cpu}Triton: {kv_indices_triton_cpu}")

    def test_boundary_values_cpu_extension_vs_triton(self):
        # Test with boundary values
        max_batch = 1
        max_context_len = 5
        total_tokens = 3

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        page_kernel_lens_cpu = torch.tensor([3], dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        kv_start_idx_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        kv_indices_cpu = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        page_kernel_lens_xpu = page_kernel_lens_cpu.to(self.device)
        kv_indptr_xpu = kv_indptr_cpu.to(self.device)
        kv_start_idx_xpu = kv_start_idx_cpu.to(self.device)
        kv_indices_triton = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        # Run CPU extension
        create_flashinfer_kv_indices(
            req_to_token_cpu,
            req_pool_indices_cpu,
            page_kernel_lens_cpu,
            kv_indptr_cpu,
            kv_start_idx_cpu,
            kv_indices_cpu,
            req_to_token_cpu.stride()[0]
        )

        # Run Triton version
        grid = (max_batch,)
        create_flashinfer_kv_indices_triton[grid](
            req_to_token_xpu,
            req_pool_indices_xpu,
            page_kernel_lens_xpu,
            kv_indptr_xpu,
            kv_start_idx_xpu,
            kv_indices_triton,
            req_to_token_xpu.stride()[0]
        )

        # Compare results
        kv_indices_triton_cpu = kv_indices_triton.cpu()
        self.assertTrue(torch.equal(kv_indices_cpu, kv_indices_triton_cpu),f"Results differ:CPU: {kv_indices_cpu}Triton: {kv_indices_triton_cpu}")

    def test_with_start_idx_cpu_extension_vs_triton(self):
        # Test with non-zero start indices
        max_batch = 2
        max_context_len = 8
        total_tokens = 8

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.arange(max_batch * max_context_len, dtype=torch.int32, device="cpu").reshape(max_batch, max_context_len)
        req_pool_indices_cpu = torch.arange(max_batch, dtype=torch.int32, device="cpu")
        page_kernel_lens_cpu = torch.tensor([3, 5], dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.tensor([0, 3], dtype=torch.int32, device="cpu")
        kv_start_idx_cpu = torch.tensor([2, 1], dtype=torch.int32, device="cpu")  # Start from different positions
        kv_indices_cpu = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        page_kernel_lens_xpu = page_kernel_lens_cpu.to(self.device)
        kv_indptr_xpu = kv_indptr_cpu.to(self.device)
        kv_start_idx_xpu = kv_start_idx_cpu.to(self.device)
        kv_indices_triton = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        # Run CPU extension
        create_flashinfer_kv_indices(
            req_to_token_cpu,
            req_pool_indices_cpu,
            page_kernel_lens_cpu,
            kv_indptr_cpu,
            kv_start_idx_cpu,
            kv_indices_cpu,
            req_to_token_cpu.stride()[0]
        )

        # Run Triton version
        grid = (max_batch,)
        create_flashinfer_kv_indices_triton[grid](
            req_to_token_xpu,
            req_pool_indices_xpu,
            page_kernel_lens_xpu,
            kv_indptr_xpu,
            kv_start_idx_xpu,
            kv_indices_triton,
            req_to_token_xpu.stride()[0]
        )

        # Compare results
        kv_indices_triton_cpu = kv_indices_triton.cpu()
        self.assertTrue(torch.equal(kv_indices_cpu, kv_indices_triton_cpu),f"Results differ:CPU: {kv_indices_cpu}Triton: {kv_indices_triton_cpu}")

    def test_large_batch_cpu_extension_vs_triton(self):
        # Test with larger batch size
        max_batch = 5
        max_context_len = 20
        total_tokens = 50

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.randint(0, 1000, (max_batch, max_context_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.arange(max_batch, dtype=torch.int32, device="cpu")
        page_kernel_lens_cpu = torch.randint(1, 10, (max_batch,), dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.cumsum(torch.tensor([0] + page_kernel_lens_cpu[:-1].tolist()), 0, dtype=torch.int32, device="cpu")
        kv_start_idx_cpu = torch.zeros(max_batch, dtype=torch.int32, device="cpu")
        kv_indices_cpu = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        page_kernel_lens_xpu = page_kernel_lens_cpu.to(self.device)
        kv_indptr_xpu = kv_indptr_cpu.to(self.device)
        kv_start_idx_xpu = kv_start_idx_cpu.to(self.device)
        kv_indices_triton = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        # Run CPU extension
        create_flashinfer_kv_indices(
            req_to_token_cpu,
            req_pool_indices_cpu,
            page_kernel_lens_cpu,
            kv_indptr_cpu,
            kv_start_idx_cpu,
            kv_indices_cpu,
            req_to_token_cpu.stride()[0]
        )

        # Run Triton version
        grid = (max_batch,)
        create_flashinfer_kv_indices_triton[grid](
            req_to_token_xpu,
            req_pool_indices_xpu,
            page_kernel_lens_xpu,
            kv_indptr_xpu,
            kv_start_idx_xpu,
            kv_indices_triton,
            req_to_token_xpu.stride()[0]
        )

        # Compare results
        kv_indices_triton_cpu = kv_indices_triton.cpu()
        self.assertTrue(torch.equal(kv_indices_cpu, kv_indices_triton_cpu),f"Results differ:CPU: {kv_indices_cpu}Triton: {kv_indices_triton_cpu}")

    def test_empty_tensors(self):
        # Test with minimal valid inputs
        max_batch = 1
        max_context_len = 5
        total_tokens = 0

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.zeros((max_batch, max_context_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        page_kernel_lens_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        kv_start_idx_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        kv_indices_cpu = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for Triton
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        page_kernel_lens_xpu = page_kernel_lens_cpu.to(self.device)
        kv_indptr_xpu = kv_indptr_cpu.to(self.device)
        kv_start_idx_xpu = kv_start_idx_cpu.to(self.device)
        kv_indices_triton = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        # Run CPU extension
        create_flashinfer_kv_indices(
            req_to_token_cpu,
            req_pool_indices_cpu,
            page_kernel_lens_cpu,
            kv_indptr_cpu,
            kv_start_idx_cpu,
            kv_indices_cpu,
            req_to_token_cpu.stride()[0]
        )

        # Run Triton version
        grid = (max_batch,)
        create_flashinfer_kv_indices_triton[grid](
            req_to_token_xpu,
            req_pool_indices_xpu,
            page_kernel_lens_xpu,
            kv_indptr_xpu,
            kv_start_idx_xpu,
            kv_indices_triton,
            req_to_token_xpu.stride()[0]
        )

        # Compare results
        kv_indices_triton_cpu = kv_indices_triton.cpu()
        self.assertTrue(torch.equal(kv_indices_cpu, kv_indices_triton_cpu),f"Results differ:CPU: {kv_indices_cpu}Triton: {kv_indices_triton_cpu}")


if __name__ == "__main__":
    unittest.main()
