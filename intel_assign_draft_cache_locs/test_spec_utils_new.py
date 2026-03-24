import unittest

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import copy_all_layer_kv_cache_tiled
from sglang.srt.utils import next_power_of_2
from intel_assign_draft_cache_locs import assign_draft_cache_locs

BYTES_PER_TILE = 128


class TestSpecUtils(unittest.TestCase):

    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        self.k_cache = [
            torch.zeros((100, 1, 1), dtype=torch.float32, device=self.device)
        ]
        self.v_cache = [
            torch.zeros((100, 1, 1), dtype=torch.float32, device=self.device)
        ]
        self.k_cache[0][:11, 0, 0] = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.v_cache[0][:11, 0, 0] = torch.tensor(
            [-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.data_ptrs=torch.tensor([[self.k_cache[0].data_ptr()],[self.v_cache[0].data_ptr()]],device="xpu",dtype=torch.uint64)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_cache + self.v_cache
            ],
            device=self.device,
            dtype=torch.int64,
        )

    def test_assign_draft_cache_locs_single_seq(self):
        # Testing Setup: req_to_token starting from 4
        # 4,5,6,7,{8,9,10}, 8,9,10 is the last partial page, 3 tokens < page_size=4
        # next kv cache will be stored starting 11,12,13...
        device = self.device
        num_seqs = 1
        page_size = 4
        speculative_num_steps = 5
        topk = 8
        seq_lens_num = 7
        extend_lens_num = 61  # includes the duplicated last page
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device="cpu")
        req_to_token[0, :seq_lens_num] = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10], device="cpu"
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device="cpu")
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device="cpu")
        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device="cpu")
        out_cache_loc = torch.arange(11, 11 + extend_lens_num, device="cpu")
        last_page_lens = torch.tensor([3], dtype=torch.int32, device="cpu")
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        
        # Call the CPU-based function without [(num_seqs,)]
        assign_draft_cache_locs(
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )

        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected_source_cache_loc = torch.tensor(
            [8, 9, 10] * (topk - 1), device="cpu", dtype=torch.int32
        )
        assert torch.allclose(source_cache_loc, expected_source_cache_loc)

        # Convert CPU tensors back to XPU for copy operation
        target_cache_loc_xpu = target_cache_loc.to(device)
        source_cache_loc_xpu = source_cache_loc.to(device)
        
        copy_all_layer_kv_cache_tiled[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            target_cache_loc_xpu,
            source_cache_loc_xpu,
            len(target_cache_loc_xpu),
            next_power_of_2(len(target_cache_loc_xpu)),
            BYTES_PER_TILE,
        )
        assert torch.allclose(
            self.k_cache[0][16:19, 0, 0],
            torch.tensor(
                [0.8, 0.9, 1.0],
                dtype=torch.float32,
                device=device,
            ),
        )
        assert torch.allclose(
            self.v_cache[0][16:19, 0, 0],
            torch.tensor(
                [-0.8, -0.9, -1.0],
                dtype=torch.float32,
                device=device,
            ),
        )

    def test_assign_draft_cache_locs_multi_seq(self):
        device = self.device
        num_seqs = 3
        page_size = 4
        speculative_num_steps = 5
        topk = 8
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device="cpu")
        seq_lens = torch.tensor([8, 7, 5], dtype=torch.int32, device="cpu")
        extend_lens = torch.tensor([64, 64, 64], dtype=torch.int32, device="cpu")
        num_new_pages_per_topk = torch.tensor(
            [2, 2, 2], dtype=torch.int32, device="cpu"
        )
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device="cpu")
        req_to_token[0, :8] = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], device="cpu")
        req_to_token[1, :7] = torch.tensor([4, 5, 6, 7, 8, 9, 10], device="cpu")
        req_to_token[2, :5] = torch.tensor([4, 5, 6, 7, 8], device="cpu")
        last_page_lens = torch.tensor([0, 3, 1], dtype=torch.int32, device="cpu")
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        out_cache_loc = torch.arange(
            12, 12 + torch.sum(extend_lens), dtype=torch.int32, device="cpu"
        )
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        
        # Call the CPU-based function without [(num_seqs,)]
        assign_draft_cache_locs(
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        # fmt: off
        expected_out_cache_loc = torch.tensor([
            12,  13,  14,  15,  16,
            20,  21,  22,  23,  24,
            28,  29,  30,  31,  32,
            36,  37,  38,  39,  40,
            44,  45,  46,  47,  48,
            52,  53,  54,  55,  56,
            60,  61,  62,  63,  64,
            68,  69,  70,  71,  72,
            76,  77,  78,  79,  80,
            84,  85,  86,  87,  88,
            92,  93,  94,  95,  96,
            100, 101, 102, 103, 104,
            108, 109, 110, 111, 112,
            116, 117, 118, 119, 120,
            124, 125, 126, 127, 128,
            132, 133, 134, 135, 136,
            140, 141, 142, 143, 144,
            148, 149, 150, 151, 152,
            156, 157, 158, 159, 160,
            164, 165, 166, 167, 168,
            172, 173, 174, 175, 176,
            180, 181, 182, 183, 184,
            188, 189, 190, 191, 192,
            196, 197, 198, 199, 200
        ], device="cpu", dtype=torch.int32)
        expected_source_cache_loc = torch.tensor([8, 9, 10] * 7 + [8] * 7, device="cpu", dtype=torch.int32)
        expected_target_cache_loc = torch.tensor([
            81,  82,  83,  89,  90,  91,  97,  98,  99, 105, 106, 107, 113, 114,
           115, 121, 122, 123, 129, 130, 131, 147, 155, 163, 171, 179, 187, 195
        ], device="cpu", dtype=torch.int32)
        # fmt: on
        assert torch.allclose(out_cache_loc, expected_out_cache_loc)
        assert torch.allclose(source_cache_loc, expected_source_cache_loc)
        assert torch.allclose(target_cache_loc, expected_target_cache_loc)
        
        # Convert CPU tensors back to XPU for copy operation
        target_cache_loc_xpu = target_cache_loc.to(device)
        source_cache_loc_xpu = source_cache_loc.to(device)
        
        copy_all_layer_kv_cache_tiled[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            target_cache_loc_xpu,
            source_cache_loc_xpu,
            len(target_cache_loc_xpu),
            next_power_of_2(len(target_cache_loc_xpu)),
            BYTES_PER_TILE,
        )
        assert torch.allclose(
            self.k_cache[0][81:84, 0, 0],
            torch.tensor(
                [0.8, 0.9, 1.0],
                dtype=torch.float32,
                device=device,
            ),
        )
        assert torch.allclose(
            self.v_cache[0][81:84, 0, 0],
            torch.tensor(
                [-0.8, -0.9, -1.0],
                dtype=torch.float32,
                device=device,
            ),
        )

    def test_assign_draft_cache_locs_page_size_1(self):
        # Test to make sure page_size=1 not affected
        device = self.device
        num_seqs = 1
        page_size = 1
        speculative_num_steps = 5
        topk = 8
        seq_lens_num = 7
        extend_lens_num = topk * speculative_num_steps
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device="cpu")
        req_to_token[0, :seq_lens_num] = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10], device="cpu"
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device="cpu")
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device="cpu")
        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device="cpu")
        out_cache_loc = torch.arange(11, 11 + extend_lens_num, device="cpu")
        last_page_lens = torch.tensor([3], dtype=torch.int32, device="cpu")
        duplicate_cache_len = 0
        target_cache_loc = None
        source_cache_loc = None
        last_page_lens_cumsum = None
        
        # Call the CPU-based function without [(num_seqs,)]
        assign_draft_cache_locs(
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected_out_cache_loc = torch.arange(11, 11 + extend_lens_num, device="cpu")
        assert torch.allclose(out_cache_loc, expected_out_cache_loc)

    def test_assign_draft_cache_locs_page_size_gt_spec_steps(self):
        device = self.device
        num_seqs = 1
        page_size = 16
        speculative_num_steps = 4
        topk = 3
        seq_lens_num = 12
        pool_len = 256
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
        req_to_token = torch.zeros(
            (num_seqs, pool_len), dtype=torch.int32, device="cpu"
        )
        req_to_token[0, :seq_lens_num] = torch.arange(
            seq_lens_num, dtype=torch.int32, device="cpu"
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device="cpu")
        last_page_len = seq_lens_num % page_size
        last_page_lens = torch.tensor([last_page_len], dtype=torch.int32, device="cpu")
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        num_new_pages_per_topk_val = (
            last_page_len + speculative_num_steps + page_size - 1
        ) // page_size
        num_new_pages_per_topk = torch.tensor(
            [num_new_pages_per_topk_val], dtype=torch.int32, device="cpu"
        )
        extend_lens_num = num_new_pages_per_topk_val * page_size * topk
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device="cpu")
        out_cache_loc = torch.arange(
            2000, 2000 + extend_lens_num, dtype=torch.int32, device="cpu"
        )
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device="cpu"
        )
        
        # Call the CPU-based function without [(num_seqs,)]
        assign_draft_cache_locs(
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        trimmed = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected = []
        for topk_id in range(topk):
            start = seq_lens_num + topk_id * num_new_pages_per_topk_val * page_size
            expected.append(
                req_to_token[0, start : start + speculative_num_steps].clone()
            )
        expected_out_cache_loc = torch.cat(expected)
        assert torch.allclose(trimmed, expected_out_cache_loc)


if __name__ == "__main__":
    unittest.main()