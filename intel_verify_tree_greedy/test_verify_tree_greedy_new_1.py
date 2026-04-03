import sys

import pytest
import torch
import torch.nn.functional as F
from intel_verify_tree_greedy import verify_tree_greedy


def test_verify_tree_greedy():
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device="xpu",
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device="xpu",
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device="xpu",
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device="xpu",
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device="xpu")
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10

    target_predict = torch.argmax(target_logits, dim=-1)
    predict_shape = (12,)

    bs = candidates.shape[0]
    num_spec_step = 4

    predicts = torch.full(
        predict_shape, -1, dtype=torch.int32, device="xpu"
    )  # mutable
    accept_index = torch.full(
        (bs, num_spec_step), -1, dtype=torch.int32, device="xpu"
    )  # mutable
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device="xpu")  # mutable

    # Copy data to CPU for the CPU-based verify_tree_greedy function
    candidates_cpu = candidates.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()
    target_predict_cpu = target_predict.cpu()
    predicts_cpu = predicts.cpu()
    accept_index_cpu = accept_index.cpu()
    accept_token_num_cpu = accept_token_num.cpu()

    verify_tree_greedy(
        predicts_cpu,
        accept_index_cpu,
        accept_token_num_cpu,
        candidates_cpu,
        retrive_index_cpu,
        retrive_next_token_cpu,
        retrive_next_sibling_cpu,
        target_predict_cpu,
    )

    # Check the expected output using CPU tensors
    assert predicts_cpu.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index_cpu.tolist() == [
        [0, 3, 4, 5],
        [6, 10, 11, -1],
    ]
    assert accept_token_num_cpu.tolist() == [3, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
