import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_generate_draft_decode_kv_indices import generate_draft_decode_kv_indices


class TestGenerateDraftDecodeKvIndices(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def test_basic_functionality_cpu_extension(self):
        # Test basic functionality with small tensors
        num_seqs = 2
        topk = 2
        speculative_num_steps = 3
        embedding_dim = 10
        pool_len = 100
        page_size = 1
        
        # Calculate shapes based on function parameters
        req_pool_indices_cpu = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token_cpu = torch.randint(0, 100, (num_seqs * pool_len, embedding_dim), dtype=torch.int32, device="cpu")
        paged_kernel_lens_cpu = torch.tensor([5, 7], dtype=torch.int32, device="cpu")
        
        # Initialize outputs
        kv_indices_cpu = torch.zeros(speculative_num_steps, num_seqs, topk, dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.zeros(num_seqs * topk, dtype=torch.int32, device="cpu")
        positions_cpu = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
        
        # Run CPU extension
        generate_draft_decode_kv_indices(
            req_pool_indices_cpu,
            req_to_token_cpu,
            paged_kernel_lens_cpu,
            kv_indices_cpu,
            kv_indptr_cpu,
            positions_cpu,
            pool_len,
            kv_indices_cpu.stride()[0],
            kv_indptr_cpu.stride()[0],
            num_seqs,
            speculative_num_steps,
            num_seqs * topk,
            page_size
        )

        # Expected behavior validation
        self.assertEqual(kv_indices_cpu.shape, (speculative_num_steps, num_seqs, topk))
        self.assertEqual(kv_indptr_cpu.shape, (num_seqs * topk,))
        self.assertTrue(kv_indices_cpu.dtype == torch.int32)
        self.assertTrue(kv_indptr_cpu.dtype == torch.int32)

    def test_boundary_values_cpu_extension(self):
        # Test with boundary values
        num_seqs = 1
        topk = 1
        speculative_num_steps = 1
        embedding_dim = 5
        pool_len = 50
        page_size = 1
        
        # Calculate shapes based on function parameters
        req_pool_indices_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        req_to_token_cpu = torch.arange(num_seqs * pool_len * embedding_dim, dtype=torch.int32, device="cpu").reshape(num_seqs * pool_len, embedding_dim)
        paged_kernel_lens_cpu = torch.tensor([3], dtype=torch.int32, device="cpu")
        
        # Initialize outputs
        kv_indices_cpu = torch.zeros(speculative_num_steps, num_seqs, topk, dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.zeros(num_seqs * topk, dtype=torch.int32, device="cpu")
        positions_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")

        # Run CPU extension
        generate_draft_decode_kv_indices(
            req_pool_indices_cpu,
            req_to_token_cpu,
            paged_kernel_lens_cpu,
            kv_indices_cpu,
            kv_indptr_cpu,
            positions_cpu,
            pool_len,
            kv_indices_cpu.stride()[0],
            kv_indptr_cpu.stride()[0],
            num_seqs,
            speculative_num_steps,
            num_seqs * topk,
            page_size
        )

        # Expected behavior validation
        self.assertEqual(kv_indices_cpu.shape, (speculative_num_steps, num_seqs, topk))
        self.assertEqual(kv_indptr_cpu.shape, (num_seqs * topk,))
        self.assertTrue(kv_indices_cpu.dtype == torch.int32)
        self.assertTrue(kv_indptr_cpu.dtype == torch.int32)

    def test_with_larger_params_cpu_extension(self):
        # Test with larger parameters
        num_seqs = 3
        topk = 2
        speculative_num_steps = 2
        embedding_dim = 8
        pool_len = 64
        page_size = 4
        
        # Calculate shapes based on function parameters
        req_pool_indices_cpu = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token_cpu = torch.randint(0, 200, (num_seqs * pool_len, embedding_dim), dtype=torch.int32, device="cpu")
        paged_kernel_lens_cpu = torch.tensor([10, 15, 8], dtype=torch.int32, device="cpu")
        
        # Initialize outputs
        kv_indices_cpu = torch.zeros(speculative_num_steps, num_seqs, topk, dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.zeros(num_seqs * topk, dtype=torch.int32, device="cpu")
        positions_cpu = torch.tensor([2, 4, 6, 8, 10, 12], dtype=torch.int32, device="cpu")

        # Run CPU extension
        generate_draft_decode_kv_indices(
            req_pool_indices_cpu,
            req_to_token_cpu,
            paged_kernel_lens_cpu,
            kv_indices_cpu,
            kv_indptr_cpu,
            positions_cpu,
            pool_len,
            kv_indices_cpu.stride()[0],
            kv_indptr_cpu.stride()[0],
            num_seqs,
            speculative_num_steps,
            num_seqs * topk,
            page_size
        )

        # Expected behavior validation
        self.assertEqual(kv_indices_cpu.shape, (speculative_num_steps, num_seqs, topk))
        self.assertEqual(kv_indptr_cpu.shape, (num_seqs * topk,))
        self.assertTrue(kv_indices_cpu.dtype == torch.int32)
        self.assertTrue(kv_indptr_cpu.dtype == torch.int32)

    def test_empty_tensors_cpu_extension(self):
        # Test with minimal valid inputs
        num_seqs = 1
        topk = 1
        speculative_num_steps = 1
        embedding_dim = 5
        pool_len = 10
        page_size = 1
        
        # Calculate shapes based on function parameters
        req_pool_indices_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        req_to_token_cpu = torch.zeros((num_seqs * pool_len, embedding_dim), dtype=torch.int32, device="cpu")
        paged_kernel_lens_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        
        # Initialize outputs
        kv_indices_cpu = torch.zeros(speculative_num_steps, num_seqs, topk, dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.zeros(num_seqs * topk, dtype=torch.int32, device="cpu")
        positions_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")

        # Run CPU extension
        generate_draft_decode_kv_indices(
            req_pool_indices_cpu,
            req_to_token_cpu,
            paged_kernel_lens_cpu,
            kv_indices_cpu,
            kv_indptr_cpu,
            positions_cpu,
            pool_len,
            kv_indices_cpu.stride()[0],
            kv_indptr_cpu.stride()[0],
            num_seqs,
            speculative_num_steps,
            num_seqs * topk,
            page_size
        )

        # Expected behavior validation
        self.assertEqual(kv_indices_cpu.shape, (speculative_num_steps, num_seqs, topk))
        self.assertEqual(kv_indptr_cpu.shape, (num_seqs * topk,))
        self.assertTrue(kv_indices_cpu.dtype == torch.int32)
        self.assertTrue(kv_indptr_cpu.dtype == torch.int32)

    def test_different_page_sizes_cpu_extension(self):
        # Test with different page sizes
        num_seqs = 2
        topk = 1
        speculative_num_steps = 2
        embedding_dim = 4
        pool_len = 32
        page_size = 8  # Different from default
        
        # Calculate shapes based on function parameters
        req_pool_indices_cpu = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token_cpu = torch.randint(0, 100, (num_seqs * pool_len, embedding_dim), dtype=torch.int32, device="cpu")
        paged_kernel_lens_cpu = torch.tensor([16, 20], dtype=torch.int32, device="cpu")
        
        # Initialize outputs
        kv_indices_cpu = torch.zeros(speculative_num_steps, num_seqs, topk, dtype=torch.int32, device="cpu")
        kv_indptr_cpu = torch.zeros(num_seqs * topk, dtype=torch.int32, device="cpu")
        positions_cpu = torch.tensor([5, 10], dtype=torch.int32, device="cpu")

        # Run CPU extension
        generate_draft_decode_kv_indices(
            req_pool_indices_cpu,
            req_to_token_cpu,
            paged_kernel_lens_cpu,
            kv_indices_cpu,
            kv_indptr_cpu,
            positions_cpu,
            pool_len,
            kv_indices_cpu.stride()[0],
            kv_indptr_cpu.stride()[0],
            num_seqs,
            speculative_num_steps,
            num_seqs * topk,
            page_size
        )

        # Expected behavior validation
        self.assertEqual(kv_indices_cpu.shape, (speculative_num_steps, num_seqs, topk))
        self.assertEqual(kv_indptr_cpu.shape, (num_seqs * topk,))
        self.assertTrue(kv_indices_cpu.dtype == torch.int32)
        self.assertTrue(kv_indptr_cpu.dtype == torch.int32)


if __name__ == "__main__":
    unittest.main()
