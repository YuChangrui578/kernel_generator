// Filename: reconstruct_indices_from_tree_mask_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

// Forward declaration of the kernel function
void reconstruct_indices_from_tree_mask_kernel(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num);

void reconstruct_indices_from_tree_mask(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
    
    TORCH_CHECK(tree_mask.is_contiguous(), "tree_mask must be contiguous");
    TORCH_CHECK(verified_seq_len.is_contiguous(), "verified_seq_len must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(retrive_index.is_contiguous(), "retrive_index must be contiguous");
    TORCH_CHECK(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
    TORCH_CHECK(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
    
    TORCH_CHECK(tree_mask.dtype() == torch::kBool || tree_mask.dtype() == torch::kInt32 || tree_mask.dtype() == torch::kInt64,
                "tree_mask must be bool, int32, or int64 type");
    TORCH_CHECK(verified_seq_len.dtype() == torch::kInt64, "verified_seq_len must be int64 type");
    TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64 type");
    TORCH_CHECK(retrive_index.dtype() == torch::kInt64, "retrive_index must be int64 type");
    TORCH_CHECK(retrive_next_token.dtype() == torch::kInt64, "retrive_next_token must be int64 type");
    TORCH_CHECK(retrive_next_sibling.dtype() == torch::kInt64, "retrive_next_sibling must be int64 type");
    
    TORCH_CHECK(tree_mask.dim() == 1, "tree_mask should be 1D tensor with size [batch_size * draft_token_num * draft_token_num]");
    TORCH_CHECK(verified_seq_len.dim() == 1, "verified_seq_len should be 1D tensor with size [batch_size]");
    TORCH_CHECK(positions.dim() == 1, "positions should be 1D tensor with size [batch_size * draft_token_num]");
    
    TORCH_CHECK(verified_seq_len.size(0) == batch_size, "verified_seq_len size mismatch");
    TORCH_CHECK(positions.size(0) == batch_size * draft_token_num, "positions size mismatch");
    TORCH_CHECK(retrive_index.size(0) == batch_size, "retrive_index first dim size mismatch");
    TORCH_CHECK(retrive_index.size(1) == draft_token_num, "retrive_index second dim size mismatch");
    TORCH_CHECK(retrive_next_token.size(0) == batch_size, "retrive_next_token first dim size mismatch");
    TORCH_CHECK(retrive_next_token.size(1) == draft_token_num, "retrive_next_token second dim size mismatch");
    TORCH_CHECK(retrive_next_sibling.size(0) == batch_size, "retrive_next_sibling first dim size mismatch");
    TORCH_CHECK(retrive_next_sibling.size(1) == draft_token_num, "retrive_next_sibling second dim size mismatch");
    
    reconstruct_indices_from_tree_mask_kernel(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reconstruct_indices_from_tree_mask", &reconstruct_indices_from_tree_mask, 
        "Reconstruct indices from tree mask for speculative decoding on CPU",
        py::arg("tree_mask"), 
        py::arg("verified_seq_len"),
        py::arg("positions"),
        py::arg("retrive_index"),
        py::arg("retrive_next_token"),
        py::arg("retrive_next_sibling"),
        py::arg("batch_size"),
        py::arg("draft_token_num"));
}