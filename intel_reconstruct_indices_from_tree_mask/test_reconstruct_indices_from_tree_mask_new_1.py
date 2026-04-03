import sys

import pytest
import torch
import torch.nn.functional as F
from intel_reconstruct_indices_from_tree_mask import reconstruct_indices_from_tree_mask


def test_reconstruct_indices_from_tree_mask():
    bs = 1
    num_branch_token = 4
    seq_lens = torch.tensor([12], device="xpu", dtype=torch.int64)

    retrive_index = torch.full(
        (bs, num_branch_token), -1, device="xpu", dtype=torch.int64
    )
    retrive_next_token = torch.full(
        (bs, num_branch_token), -1, device="xpu", dtype=torch.int64
    )
    retrive_next_sibling = torch.full(
        (bs, num_branch_token), -1, device="xpu", dtype=torch.int64
    )
    positions = torch.empty((bs * num_branch_token), device="xpu", dtype=torch.int64)

    tree_mask = torch.tensor(
        [
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ],
        device="xpu",
        dtype=torch.int32,
    ).to(torch.bool)

    # Copy data to CPU for the CPU-based function
    tree_mask_cpu = tree_mask.cpu()
    seq_lens_cpu = seq_lens.cpu()
    positions_cpu = positions.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()

    reconstruct_indices_from_tree_mask(
        tree_mask_cpu,
        seq_lens_cpu,
        positions_cpu,  # mutable
        retrive_index_cpu,  # mutable
        retrive_next_token_cpu,  # mutable
        retrive_next_sibling_cpu,  # mutable
        bs,
        num_branch_token,
    )

    # Copy results back to xpu for comparison
    retrive_index_result = retrive_index_cpu.to("xpu")
    retrive_next_token_result = retrive_next_token_cpu.to("xpu")
    retrive_next_sibling_result = retrive_next_sibling_cpu.to("xpu")
    positions_result = positions_cpu.to("xpu")

    assert retrive_index_result.tolist() == [
        [0, 1, 2, 3],
    ], f"{retrive_index_result=}"
    assert retrive_next_token_result.tolist() == [
        [1, -1, 3, -1],
    ], f"{retrive_next_token_result=}"
    assert retrive_next_sibling_result.tolist() == [
        [-1, 2, -1, -1],
    ], f"{retrive_next_sibling_result=}"
    assert positions_result.tolist() == [
        12,
        13,
        13,
        14,
    ], f"{positions_result=}"


if __name__ == "__main__":
    test_reconstruct_indices_from_tree_mask()
    sys.exit(pytest.main([__file__]))
