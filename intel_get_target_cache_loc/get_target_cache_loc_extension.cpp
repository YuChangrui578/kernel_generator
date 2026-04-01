// Filename: get_target_cache_loc_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

// Forward declaration of the kernel function
void get_target_cache_loc_kernel(
    at::Tensor& tgt_cache_loc,
    at::Tensor& to_free_slots,
    const at::Tensor& accept_length,
    const at::Tensor& to_free_num_slots,
    const at::Tensor& out_cache_loc,
    int64_t num_verify_tokens,
    int64_t num_verify_tokens_upper,
    int64_t bs_upper
);

torch::Tensor get_target_cache_loc(
    torch::Tensor tgt_cache_loc,
    torch::Tensor to_free_slots,
    torch::Tensor accept_length,
    torch::Tensor to_free_num_slots,
    torch::Tensor out_cache_loc,
    int64_t num_verify_tokens,
    int64_t num_verify_tokens_upper,
    int64_t bs_upper
) {
    // Validate tensor properties
    TORCH_CHECK(tgt_cache_loc.is_contiguous(), "tgt_cache_loc must be contiguous");
    TORCH_CHECK(to_free_slots.is_contiguous(), "to_free_slots must be contiguous");
    TORCH_CHECK(accept_length.is_contiguous(), "accept_length must be contiguous");
    TORCH_CHECK(to_free_num_slots.is_contiguous(), "to_free_num_slots must be contiguous");
    TORCH_CHECK(out_cache_loc.is_contiguous(), "out_cache_loc must be contiguous");
    
    TORCH_CHECK(tgt_cache_loc.dtype() == to_free_slots.dtype() &&
                tgt_cache_loc.dtype() == accept_length.dtype() &&
                tgt_cache_loc.dtype() == to_free_num_slots.dtype() &&
                tgt_cache_loc.dtype() == out_cache_loc.dtype(),
                "All tensors must have the same dtype");
    
    TORCH_CHECK(tgt_cache_loc.device().is_cpu() && to_free_slots.device().is_cpu() &&
                accept_length.device().is_cpu() && to_free_num_slots.device().is_cpu() &&
                out_cache_loc.device().is_cpu(), "All tensors must be on CPU");

    // Additional validation checks
    TORCH_CHECK(accept_length.numel() >= bs_upper, "accept_length must have at least bs_upper elements");
    TORCH_CHECK(to_free_num_slots.numel() >= bs_upper, "to_free_num_slots must have at least bs_upper elements");
    TORCH_CHECK(out_cache_loc.size(0) >= bs_upper, "out_cache_loc must have at least bs_upper rows");
    TORCH_CHECK(out_cache_loc.size(1) >= num_verify_tokens, "out_cache_loc must have at least num_verify_tokens columns");

    // Call the actual kernel
    get_target_cache_loc_kernel(
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
        num_verify_tokens_upper,
        bs_upper
    );

    // Return dummy tensor as function is in-place
    return torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_target_cache_loc", &get_target_cache_loc, "get_target_cache_loc kernel");
}