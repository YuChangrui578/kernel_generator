import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from intel_create_extend_after_decode_spec_info import create_extend_after_decode_spec_info
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Unit test for CPU-based extend after decode spec info creation
register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

@triton.jit
def triton_impl(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


class TestCreateExtendAfterDecodeSpecInfo(unittest.TestCase):
    def setUp(self):
        self.device = "xpu" if torch.xpu.is_available() else "cpu"

    def _run_test(self, batch_size, bs_upper):
        # Create input tensors on XPU (default device)
        verified_id = torch.arange(batch_size * 2, dtype=torch.int32, device=get_device())
        seq_lens = torch.randint(1, 100, (batch_size,), dtype=torch.int32, device=get_device())
        accept_lens = torch.randint(1, 20, (batch_size,), dtype=torch.int32, device=get_device())
        
        # Calculate total positions needed based on accept_lens
        total_accept_len = torch.sum(accept_lens).item()
        positions = torch.zeros(total_accept_len, dtype=torch.int32, device=get_device())
        new_verified_id = torch.zeros(batch_size, dtype=torch.int32, device=get_device())

        # Reference implementation using native Triton (XPU-based)
        positions_xpu = positions.clone()
        new_verified_id_xpu = new_verified_id.clone()
        
        # Run native triton implementation on XPU
        
        # Calculate grid for triton
        grid = (batch_size,)
        
        # Execute triton kernel
        triton_impl[(batch_size,)](
            verified_id,
            seq_lens,
            accept_lens,
            positions_xpu,
            new_verified_id_xpu,
            bs_upper=bs_upper
        )
        
        # Prepare CPU inputs for the Intel CPU-based implementation
        verified_id_cpu = verified_id.cpu().to(torch.int32)  # Ensure correct dtype
        seq_lens_cpu = seq_lens.cpu().to(torch.int32)
        accept_lens_cpu = accept_lens.cpu().to(torch.int32)
        positions_cpu = torch.zeros(total_accept_len, dtype=torch.int32, device='cpu')
        new_verified_id_cpu = torch.zeros(batch_size, dtype=torch.int32, device='cpu')

        # Call the Intel CPU-based extension function
        create_extend_after_decode_spec_info(
            verified_id_cpu,
            seq_lens_cpu,
            accept_lens_cpu,
            positions_cpu,
            new_verified_id_cpu,
            bs_upper
        )

        # Compare results - convert XPU results back to CPU for comparison
        positions_ref_cpu = positions_xpu.cpu()
        new_verified_id_ref_cpu = new_verified_id_xpu.cpu()
        
        self.assertTrue(torch.equal(positions_ref_cpu, positions_cpu))
        self.assertTrue(torch.equal(new_verified_id_ref_cpu, new_verified_id_cpu))

    def test_create_extend_after_decode_spec_info_basic(self):
        # Basic test case
        self._run_test(batch_size=1, bs_upper=16)

    def test_create_extend_after_decode_spec_info_medium(self):
        # Medium batch size
        self._run_test(batch_size=32, bs_upper=64)

    def test_create_extend_after_decode_spec_info_large(self):
        # Large batch size
        self._run_test(batch_size=128, bs_upper=256)

    def test_create_extend_after_decode_spec_info_edge_cases(self):
        # Edge case: single element
        self._run_test(batch_size=1, bs_upper=1)
        
        # Edge case: small batch with large upper bound
        self._run_test(batch_size=2, bs_upper=100)


if __name__ == "__main__":
    unittest.main()
