// Filename: intel_assign_req_to_token_pool_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

template <typename scalar_t>
void assign_req_to_token_pool_kernel_impl(
    const int32_t* __restrict__ req_pool_indices_ptr,
    const int32_t* __restrict__ req_to_token_ptr,
    const int32_t* __restrict__ start_offset_ptr,
    const int32_t* __restrict__ end_offset_ptr,
    const int32_t* __restrict__ out_cache_loc_ptr,
    int32_t* __restrict__ token_pool_ptr,
    int64_t batch_size,
    int64_t num_requests,
    int64_t pool_len,
    int64_t bs_upper) {
    
    // Process each request in the batch
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // Get the request index for this batch item
            int32_t req_idx = req_pool_indices_ptr[i];
            
            // Validate request index bounds
            if (req_idx < 0 || req_idx >= num_requests) {
                continue;
            }
            
            // Get the start and end offsets for this request
            int32_t start = start_offset_ptr[i];
            int32_t end = end_offset_ptr[i];
            int32_t out_start = out_cache_loc_ptr[i];
            
            // Validate offsets
            if (start < 0 || end < start || start >= pool_len) {
                continue;
            }
            
            // Calculate the actual end position (bounded by pool length)
            int32_t actual_end = std::min(end, static_cast<int32_t>(pool_len));
            int32_t actual_out_end = out_start + (actual_end - start);
            
            // Validate output bounds
            if (actual_out_end > pool_len) {
                actual_end = start + (pool_len - out_start);
                actual_out_end = out_start + (actual_end - start);
            }
            
            // Copy tokens from req_to_token to token_pool
            int64_t src_base_offset = req_idx * pool_len;
            int64_t dst_base_offset = req_idx * pool_len;
            
            // Handle copying with simple loop since we removed vectorization
            for (int64_t j = 0; j < (actual_end - start); ++j) {
                int64_t src_pos = src_base_offset + start + j;
                int64_t dst_pos = dst_base_offset + out_start + j;
                
                token_pool_ptr[dst_pos] = req_to_token_ptr[src_pos];
            }
        }
    });
}

} // namespace

void assign_req_to_token_pool_kernel(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    at::Tensor& token_pool,
    int64_t pool_len,
    int64_t bs_upper) {
    
    // Validate input tensor dimensions
    TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be 1D");
    TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2D");
    TORCH_CHECK(start_offset.dim() == 1, "start_offset must be 1D");
    TORCH_CHECK(end_offset.dim() == 1, "end_offset must be 1D");
    TORCH_CHECK(out_cache_loc.dim() == 1, "out_cache_loc must be 1D");
    TORCH_CHECK(token_pool.dim() == 2, "token_pool must be 2D");
    
    int64_t batch_size = req_pool_indices.size(0);
    int64_t num_requests = req_to_token.size(0);
    
    TORCH_CHECK(start_offset.size(0) == batch_size, "start_offset size mismatch");
    TORCH_CHECK(end_offset.size(0) == batch_size, "end_offset size mismatch");
    TORCH_CHECK(out_cache_loc.size(0) == batch_size, "out_cache_loc size mismatch");
    TORCH_CHECK(token_pool.size(0) == num_requests && token_pool.size(1) == pool_len, "token_pool size mismatch");
    
    // Dispatch based on tensor type
    AT_DISPATCH_INTEGRAL_TYPES(req_pool_indices.scalar_type(), "assign_req_to_token_pool_kernel", [&] {
        assign_req_to_token_pool_kernel_impl<scalar_t>(
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            start_offset.data_ptr<int32_t>(),
            end_offset.data_ptr<int32_t>(),
            out_cache_loc.data_ptr<int32_t>(),
            token_pool.data_ptr<int32_t>(),
            batch_size,
            num_requests,
            pool_len,
            bs_upper
        );
    });
}