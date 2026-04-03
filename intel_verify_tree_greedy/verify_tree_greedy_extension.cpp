// Filename: verify_tree_greedy_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

void verify_tree_greedy(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict);

PYBIND11_MODULE(intel_verify_tree_greedy, m) {
  m.def("verify_tree_greedy", &verify_tree_greedy, "Verify Tree Greedy CPU kernel");
}