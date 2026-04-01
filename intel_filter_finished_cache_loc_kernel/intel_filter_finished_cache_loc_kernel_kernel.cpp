// Filename: intel_filter_finished_cache_loc_kernel_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

template <typename scalar_t>
void filter_finished_cache_loc_kernel_impl(
    scalar_t* __restrict__ out_cache_loc,
    const scalar_t* __restrict__ tgt_cache_loc,
    const scalar_t* __restrict__ accept_length,
    const scalar_t* __restrict__ accept_length_filter,
    int64_t bs_upper,
    int64_t num_verify_tokens_upper
) {
    // Process each batch item sequentially (equivalent to different bid values in Triton)
    for (int64_t bid = 0; bid < bs_upper; bid++) {
        // Calculate cumulative offset for original array (old_start)
        int64_t old_start = bid;  // Start with bid offset
        for (int64_t i = 0; i < bid; i++) {
            old_start += static_cast<int64_t>(accept_length[i]);  // Add all previous accept lengths
        }
        
        // Calculate cumulative offset for filtered array (new_start) 
        int64_t new_start = 0;  // Start from beginning of output array
        for (int64_t i = 0; i < bid; i++) {
            new_start += static_cast<int64_t>(accept_length_filter[i]);  // Add all previous filtered lengths
        }
        
        // Get number of elements to copy for current batch item
        int64_t copy_len = static_cast<int64_t>(accept_length_filter[bid]);
        
        // Bound the copy length to prevent out-of-bounds access
        copy_len = std::min(copy_len, num_verify_tokens_upper);
        
        // Copy 'copy_len' elements from source to destination
        for (int64_t copy_offset = 0; copy_offset < copy_len; copy_offset++) {
            // Load value from source position: tgt_cache_loc[old_start + copy_offset]
            scalar_t value = tgt_cache_loc[old_start + copy_offset];
            
            // Store value to destination position: out_cache_loc[new_start + copy_offset]  
            out_cache_loc[new_start + copy_offset] = value;
        }
    }
}

void filter_finished_cache_loc_kernel_cpu(
    at::Tensor out_cache_loc,
    const at::Tensor& tgt_cache_loc,
    const at::Tensor& accept_length,
    const at::Tensor& accept_length_filter,
    int64_t bs_upper,
    int64_t num_verify_tokens_upper
) {
    AT_DISPATCH_ALL_TYPES(tgt_cache_loc.scalar_type(), "filter_finished_cache_loc_kernel", ([&] {
        filter_finished_cache_loc_kernel_impl<scalar_t>(
            out_cache_loc.data_ptr<scalar_t>(),
            tgt_cache_loc.data_ptr<scalar_t>(),
            accept_length.data_ptr<scalar_t>(),
            accept_length_filter.data_ptr<scalar_t>(),
            bs_upper,
            num_verify_tokens_upper
        );
    }));
}