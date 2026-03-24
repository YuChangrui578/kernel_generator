// Filename: assign_draft_cache_locs_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

// Forward declaration of the kernel implementation function.
at::Tensor assign_draft_cache_locs(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    const at::Tensor& extend_lens,
    const at::Tensor& num_new_pages_per_topk,
    const at::Tensor& out_cache_loc,
    const c10::optional<at::Tensor>& source_cache_loc,
    const c10::optional<at::Tensor>& target_cache_loc,
    const c10::optional<at::Tensor>& last_page_lens_cumsum,
    int64_t duplicate_cache_len,
    int64_t pool_len,
    int64_t topk,
    int64_t speculative_num_steps,
    int64_t page_size,
    int64_t bs_upper,
    int64_t iter_upper);

// Pybind wrapper function
at::Tensor assign_draft_cache_locs_pybind(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    const at::Tensor& extend_lens,
    const at::Tensor& num_new_pages_per_topk,
    const at::Tensor& out_cache_loc,
    const c10::optional<at::Tensor>& source_cache_loc,
    const c10::optional<at::Tensor>& target_cache_loc,
    const c10::optional<at::Tensor>& last_page_lens_cumsum,
    int64_t duplicate_cache_len,
    int64_t pool_len,
    int64_t topk,
    int64_t speculative_num_steps,
    int64_t page_size,
    int64_t bs_upper,
    int64_t iter_upper) {
    
    // Validate tensor dimensions and types
    TORCH_CHECK(req_pool_indices.is_cpu(), "req_pool_indices must be on CPU");
    TORCH_CHECK(req_to_token.is_cpu(), "req_to_token must be on CPU");
    TORCH_CHECK(seq_lens.is_cpu(), "seq_lens must be on CPU");
    TORCH_CHECK(extend_lens.is_cpu(), "extend_lens must be on CPU");
    TORCH_CHECK(num_new_pages_per_topk.is_cpu(), "num_new_pages_per_topk must be on CPU");
    TORCH_CHECK(out_cache_loc.is_cpu(), "out_cache_loc must be on CPU");
    
    // Ensure all tensor inputs are on CPU and have type Int32 (kInt)
    // to match the kernel implementation's expectation.
    auto req_pool_indices_i = req_pool_indices.to(at::kCPU).to(at::kInt);
    auto req_to_token_i = req_to_token.to(at::kCPU).to(at::kInt);
    auto seq_lens_i = seq_lens.to(at::kCPU).to(at::kInt);
    auto extend_lens_i = extend_lens.to(at::kCPU).to(at::kInt);
    auto num_new_pages_per_topk_i = num_new_pages_per_topk.to(at::kCPU).to(at::kInt);
    
    // out_cache_loc is an output buffer, but it is also read from.
    // We create an Int32 version to pass to the kernel.
    auto out_cache_loc_i = out_cache_loc.to(at::kCPU).to(at::kInt);

    // Handle optional tensors
    c10::optional<at::Tensor> source_cache_loc_i = c10::nullopt;
    if (source_cache_loc.has_value()) {
        source_cache_loc_i = source_cache_loc.value().to(at::kCPU).to(at::kInt);
    }

    c10::optional<at::Tensor> target_cache_loc_i = c10::nullopt;
    if (target_cache_loc.has_value()) {
        target_cache_loc_i = target_cache_loc.value().to(at::kCPU).to(at::kInt);
    }

    c10::optional<at::Tensor> last_page_lens_cumsum_i = c10::nullopt;
    if (last_page_lens_cumsum.has_value()) {
        last_page_lens_cumsum_i = last_page_lens_cumsum.value().to(at::kCPU).to(at::kInt);
    }

    // Call the directly linked implementation
    return assign_draft_cache_locs(
        req_pool_indices_i, 
        req_to_token_i, 
        seq_lens_i, 
        extend_lens_i, 
        num_new_pages_per_topk_i, 
        out_cache_loc_i, 
        source_cache_loc_i, 
        target_cache_loc_i, 
        last_page_lens_cumsum_i,
        duplicate_cache_len, 
        pool_len, 
        topk, 
        speculative_num_steps, 
        page_size, 
        bs_upper, 
        iter_upper
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Intel CPU optimized assign_draft_cache_locs operator";
    // Bind the wrapper function
    m.def("assign_draft_cache_locs", &assign_draft_cache_locs_pybind, "Assign draft cache locations");
}