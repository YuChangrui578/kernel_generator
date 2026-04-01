// Filename: assign_req_to_token_pool_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

void assign_req_to_token_pool_kernel(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    at::Tensor& token_pool,
    int64_t pool_len,
    int64_t bs_upper);

void assign_req_to_token_pool(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    int64_t pool_len,
    int64_t bs_upper,
    at::Tensor& token_pool) {
    
    // Validate input tensors are on CPU
    TORCH_CHECK(req_pool_indices.device().is_cpu(), "req_pool_indices must be on CPU");
    TORCH_CHECK(req_to_token.device().is_cpu(), "req_to_token must be on CPU");
    TORCH_CHECK(start_offset.device().is_cpu(), "start_offset must be on CPU");
    TORCH_CHECK(end_offset.device().is_cpu(), "end_offset must be on CPU");
    TORCH_CHECK(out_cache_loc.device().is_cpu(), "out_cache_loc must be on CPU");
    TORCH_CHECK(token_pool.device().is_cpu(), "token_pool must be on CPU");
    
    // Validate tensor data types
    TORCH_CHECK(req_pool_indices.dtype() == at::kInt, "req_pool_indices must be int32");
    TORCH_CHECK(req_to_token.dtype() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(start_offset.dtype() == at::kInt, "start_offset must be int32");
    TORCH_CHECK(end_offset.dtype() == at::kInt, "end_offset must be int32");
    TORCH_CHECK(out_cache_loc.dtype() == at::kInt, "out_cache_loc must be int32");
    TORCH_CHECK(token_pool.dtype() == at::kInt, "token_pool must be int32");
    
    assign_req_to_token_pool_kernel(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        token_pool,
        pool_len,
        bs_upper
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assign_req_to_token_pool", &assign_req_to_token_pool, "Assign request to token pool kernel");
}