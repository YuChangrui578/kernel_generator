import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_assign_req_to_token_pool import assign_req_to_token_pool


class TestAssignReqToTokenPool(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def test_basic_functionality(self):
        # Test basic functionality with small tensors
        batch_size = 2
        num_requests = 2
        pool_len = 10
        total_tokens = 9

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.randint(0, 100, (num_requests, pool_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.arange(batch_size, dtype=torch.int32, device="cpu")
        start_offset_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        end_offset_cpu = torch.tensor([4, 5], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([0, 4], dtype=torch.int32, device="cpu")
        pool_len_cpu = pool_len
        bs_upper_cpu = 4  # Next power of 2 for batch_size=2
        
        token_pool_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for comparison
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        start_offset_xpu = start_offset_cpu.to(self.device)
        end_offset_xpu = end_offset_cpu.to(self.device)
        out_cache_loc_xpu = out_cache_loc_cpu.to(self.device)
        
        token_pool_xpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device=self.device)

        # Run CPU extension
        assign_req_to_token_pool(
            req_pool_indices_cpu,
            req_to_token_cpu,
            start_offset_cpu,
            end_offset_cpu,
            out_cache_loc_cpu,
            pool_len_cpu,
            bs_upper_cpu,
            token_pool_cpu
        )

        # Simulate the same operation on XPU for comparison
        for i in range(batch_size):
            req_idx = req_pool_indices_xpu[i].item()
            start = start_offset_xpu[i].item()
            end = end_offset_xpu[i].item()
            out_start = out_cache_loc_xpu[i].item()
            
            for j in range(start, end):
                if out_start + (j - start) < token_pool_xpu.size(1):
                    token_pool_xpu[req_idx][out_start + (j - start)] = req_to_token_xpu[req_idx][j]

        # Compare results by moving XPU result to CPU
        self.assertTrue(torch.equal(token_pool_cpu, token_pool_xpu.cpu()))

    def test_boundary_values(self):
        # Test with boundary values
        batch_size = 1
        num_requests = 1
        pool_len = 5
        total_tokens = 3

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        start_offset_cpu = torch.tensor([1], dtype=torch.int32, device="cpu")
        end_offset_cpu = torch.tensor([4], dtype=torch.int32, device="cpu")
        out_cache_loc_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
        pool_len_cpu = pool_len
        bs_upper_cpu = 1  # Next power of 2 for batch_size=1
        
        token_pool_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for comparison
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        start_offset_xpu = start_offset_cpu.to(self.device)
        end_offset_xpu = end_offset_cpu.to(self.device)
        out_cache_loc_xpu = out_cache_loc_cpu.to(self.device)
        
        token_pool_xpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device=self.device)

        # Run CPU extension
        assign_req_to_token_pool(
            req_pool_indices_cpu,
            req_to_token_cpu,
            start_offset_cpu,
            end_offset_cpu,
            out_cache_loc_cpu,
            pool_len_cpu,
            bs_upper_cpu,
            token_pool_cpu
        )

        # Simulate the same operation on XPU for comparison
        for i in range(batch_size):
            req_idx = req_pool_indices_xpu[i].item()
            start = start_offset_xpu[i].item()
            end = end_offset_xpu[i].item()
            out_start = out_cache_loc_xpu[i].item()
            
            for j in range(start, end):
                if out_start + (j - start) < token_pool_xpu.size(1):
                    token_pool_xpu[req_idx][out_start + (j - start)] = req_to_token_xpu[req_idx][j]

        # Compare results by moving XPU result to CPU
        self.assertTrue(torch.equal(token_pool_cpu, token_pool_xpu.cpu()))

    def test_with_start_end_offset(self):
        # Test with different start and end offsets
        batch_size = 2
        num_requests = 3
        pool_len = 8

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.arange(num_requests * pool_len, dtype=torch.int32, device="cpu").reshape(num_requests, pool_len)
        req_pool_indices_cpu = torch.tensor([0, 2], dtype=torch.int32, device="cpu")  # Only using first and third request
        start_offset_cpu = torch.tensor([2, 1], dtype=torch.int32, device="cpu")  # Start from different positions
        end_offset_cpu = torch.tensor([5, 6], dtype=torch.int32, device="cpu")    # End at different positions
        out_cache_loc_cpu = torch.tensor([0, 3], dtype=torch.int32, device="cpu") # Output locations
        pool_len_cpu = pool_len
        bs_upper_cpu = 4  # Next power of 2 for batch_size=2
        
        token_pool_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for comparison
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        start_offset_xpu = start_offset_cpu.to(self.device)
        end_offset_xpu = end_offset_cpu.to(self.device)
        out_cache_loc_xpu = out_cache_loc_cpu.to(self.device)
        
        token_pool_xpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device=self.device)

        # Run CPU extension
        assign_req_to_token_pool(
            req_pool_indices_cpu,
            req_to_token_cpu,
            start_offset_cpu,
            end_offset_cpu,
            out_cache_loc_cpu,
            pool_len_cpu,
            bs_upper_cpu,
            token_pool_cpu
        )

        # Simulate the same operation on XPU for comparison
        for i in range(batch_size):
            req_idx = req_pool_indices_xpu[i].item()
            start = start_offset_xpu[i].item()
            end = end_offset_xpu[i].item()
            out_start = out_cache_loc_xpu[i].item()
            
            for j in range(start, end):
                if out_start + (j - start) < token_pool_xpu.size(1):
                    token_pool_xpu[req_idx][out_start + (j - start)] = req_to_token_xpu[req_idx][j]

        # Compare results by moving XPU result to CPU
        self.assertTrue(torch.equal(token_pool_cpu, token_pool_xpu.cpu()))

    def test_large_batch(self):
        # Test with larger batch size
        batch_size = 5
        num_requests = 8
        pool_len = 20

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.randint(0, 1000, (num_requests, pool_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.randint(0, num_requests, (batch_size,), dtype=torch.int32, device="cpu")
        start_offset_cpu = torch.randint(0, pool_len//2, (batch_size,), dtype=torch.int32, device="cpu")
        end_offset_cpu = start_offset_cpu + torch.randint(1, pool_len//2, (batch_size,), dtype=torch.int32, device="cpu")
        # Ensure end_offset doesn't exceed pool_len
        end_offset_cpu = torch.min(end_offset_cpu, torch.tensor(pool_len, dtype=torch.int32))
        out_cache_loc_cpu = torch.cumsum(torch.tensor([0] + (end_offset_cpu - start_offset_cpu).tolist()[:-1]), 0, dtype=torch.int32)
        pool_len_cpu = pool_len
        bs_upper_cpu = 8  # Next power of 2 for batch_size=5
        
        token_pool_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for comparison
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        start_offset_xpu = start_offset_cpu.to(self.device)
        end_offset_xpu = end_offset_cpu.to(self.device)
        out_cache_loc_xpu = out_cache_loc_cpu.to(self.device)
        
        token_pool_xpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device=self.device)

        # Run CPU extension
        assign_req_to_token_pool(
            req_pool_indices_cpu,
            req_to_token_cpu,
            start_offset_cpu,
            end_offset_cpu,
            out_cache_loc_cpu,
            pool_len_cpu,
            bs_upper_cpu,
            token_pool_cpu
        )

        # Simulate the same operation on XPU for comparison
        for i in range(batch_size):
            req_idx = req_pool_indices_xpu[i].item()
            start = start_offset_xpu[i].item()
            end = end_offset_xpu[i].item()
            out_start = out_cache_loc_xpu[i].item()
            
            for j in range(start, end):
                if out_start + (j - start) < token_pool_xpu.size(1):
                    token_pool_xpu[req_idx][out_start + (j - start)] = req_to_token_xpu[req_idx][j]

        # Compare results by moving XPU result to CPU
        self.assertTrue(torch.equal(token_pool_cpu, token_pool_xpu.cpu()))

    def test_empty_tensors(self):
        # Test with empty ranges
        batch_size = 2
        num_requests = 3
        pool_len = 5

        # Prepare inputs on CPU for the extension
        req_to_token_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")
        req_pool_indices_cpu = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
        start_offset_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        end_offset_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")  # Empty ranges
        out_cache_loc_cpu = torch.tensor([0, 0], dtype=torch.int32, device="cpu")
        pool_len_cpu = pool_len
        bs_upper_cpu = 4  # Next power of 2 for batch_size=2
        
        token_pool_cpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device="cpu")

        # Prepare inputs on XPU for comparison
        req_to_token_xpu = req_to_token_cpu.to(self.device)
        req_pool_indices_xpu = req_pool_indices_cpu.to(self.device)
        start_offset_xpu = start_offset_cpu.to(self.device)
        end_offset_xpu = end_offset_cpu.to(self.device)
        out_cache_loc_xpu = out_cache_loc_cpu.to(self.device)
        
        token_pool_xpu = torch.zeros((num_requests, pool_len), dtype=torch.int32, device=self.device)

        # Run CPU extension
        assign_req_to_token_pool(
            req_pool_indices_cpu,
            req_to_token_cpu,
            start_offset_cpu,
            end_offset_cpu,
            out_cache_loc_cpu,
            pool_len_cpu,
            bs_upper_cpu,
            token_pool_cpu
        )

        # For empty ranges, the XPU simulation should also result in no changes
        # Compare results by moving XPU result to CPU
        self.assertTrue(torch.equal(token_pool_cpu, token_pool_xpu.cpu()))


if __name__ == "__main__":
    unittest.main()
