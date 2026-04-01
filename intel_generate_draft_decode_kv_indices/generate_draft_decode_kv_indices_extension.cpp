// Filename: generate_draft_decode_kv_indices_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

void generate_draft_decode_kv_indices_kernel(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& paged_kernel_lens,
    at::Tensor& kv_indices,
    at::Tensor& kv_indptr,
    const at::Tensor& positions,
    int64_t pool_len,
    int64_t kv_indices_stride,
    int64_t kv_indptr_stride,
    int64_t bs_upper,
    int64_t iter_upper,
    int64_t num_tokens_upper,
    int64_t page_size
);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::tuple<at::Tensor, at::Tensor> generate_draft_decode_kv_indices(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& paged_kernel_lens,
    at::Tensor& kv_indices,
    at::Tensor& kv_indptr,
    const at::Tensor& positions,
    int64_t pool_len,
    int64_t kv_indices_stride,
    int64_t kv_indptr_stride,
    int64_t bs_upper,
    int64_t iter_upper,
    int64_t num_tokens_upper,
    int64_t page_size
) {
    // Input validation
    TORCH_CHECK(req_pool_indices.is_cuda() == false, "req_pool_indices must be on CPU");
    TORCH_CHECK(req_to_token.is_cuda() == false, "req_to_token must be on CPU");
    TORCH_CHECK(paged_kernel_lens.is_cuda() == false, "paged_kernel_lens must be on CPU");
    TORCH_CHECK(kv_indices.is_cuda() == false, "kv_indices must be on CPU");
    TORCH_CHECK(kv_indptr.is_cuda() == false, "kv_indptr must be on CPU");
    TORCH_CHECK(positions.is_cuda() == false, "positions must be on CPU");
    
    TORCH_CHECK(req_pool_indices.dtype() == torch::kInt32, "req_pool_indices must be int32");
    TORCH_CHECK(req_to_token.dtype() == torch::kInt32, "req_to_token must be int32");
    TORCH_CHECK(paged_kernel_lens.dtype() == torch::kInt32, "paged_kernel_lens must be int32");
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(positions.dtype() == torch::kInt32, "positions must be int32");
    
    CHECK_CONTIGUOUS(req_pool_indices);
    CHECK_CONTIGUOUS(req_to_token);
    CHECK_CONTIGUOUS(paged_kernel_lens);
    CHECK_CONTIGUOUS(kv_indices);
    CHECK_CONTIGUOUS(kv_indptr);
    CHECK_CONTIGUOUS(positions);
    
    // Call the actual kernel
    generate_draft_decode_kv_indices_kernel(
        req_pool_indices,
        req_to_token,
        paged_kernel_lens,
        kv_indices,
        kv_indptr,
        positions,
        pool_len,
        kv_indices_stride,
        kv_indptr_stride,
        bs_upper,
        iter_upper,
        num_tokens_upper,
        page_size
    );
    
    return std::make_tuple(kv_indices, kv_indptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_draft_decode_kv_indices", &generate_draft_decode_kv_indices, 
          "Generate draft decode KV indices kernel (CPU)");
}