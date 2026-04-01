// Filename: align_evict_mask_to_page_size_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

void align_evict_mask_to_page_size_kernel(
    const at::Tensor& seq_lens,
    at::Tensor& evict_mask,
    int64_t page_size,
    int64_t num_draft_tokens,
    int64_t block_size
);

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor align_evict_mask_to_page_size(
    torch::Tensor seq_lens,
    torch::Tensor evict_mask,
    int64_t page_size,
    int64_t num_draft_tokens,
    int64_t block_size
) {
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(evict_mask);
    
    AT_ASSERTM(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32 tensor");
    AT_ASSERTM(evict_mask.dtype() == torch::kBool, "evict_mask must be bool tensor");
    AT_ASSERTM(seq_lens.dim() == 1, "seq_lens must be 1D tensor");
    AT_ASSERTM(evict_mask.dim() == 2, "evict_mask must be 2D tensor");
    AT_ASSERTM(seq_lens.size(0) == evict_mask.size(0), "batch size mismatch between seq_lens and evict_mask");
    
    // Clone the input mask to avoid modifying the original tensor
    auto evict_mask_copy = evict_mask.clone();
    
    align_evict_mask_to_page_size_kernel(
        seq_lens,
        evict_mask_copy,
        page_size,
        num_draft_tokens,
        block_size
    );
    
    return evict_mask_copy;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("align_evict_mask_to_page_size", &align_evict_mask_to_page_size, "Align evict mask to page size CPU kernel");
}