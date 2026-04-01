// Filename: intel_get_target_cache_loc_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

template<typename scalar_t>
void get_target_cache_loc_kernel_impl(
    scalar_t* tgt_cache_loc,
    scalar_t* to_free_slots,
    const scalar_t* accept_length,
    const scalar_t* to_free_num_slots,
    const scalar_t* out_cache_loc,
    int64_t num_verify_tokens,
    int64_t num_verify_tokens_upper,
    int64_t bs_upper
) {
    // Process each batch sequentially since we need prefix sums
    at::parallel_for(0, bs_upper, 1, [&](int64_t start, int64_t end) {
        for (int64_t bid = start; bid < end; bid++) {
            //=== FIRST PART: Write to tgt_cache_loc ===
            
            // Calculate prefix sum of accept_length for all batches before current batch
            scalar_t tgt_cache_loc_start = 0;
            for (int64_t i = 0; i < bid; i++) {
                tgt_cache_loc_start += accept_length[i];
            }
            // Add current batch index to get starting position
            tgt_cache_loc_start += bid;
            
            // Get number of tokens to copy for current batch (+1 for some reason)
            int64_t copy_len = static_cast<int64_t>(accept_length[bid]) + 1;
            
            // Copy from out_cache_loc to tgt_cache_loc
            int64_t offset = 0;
            for (; offset < copy_len && offset < num_verify_tokens_upper; offset++) {
                int64_t src_idx = bid * num_verify_tokens + offset;
                int64_t dst_idx = static_cast<int64_t>(tgt_cache_loc_start) + offset;
                
                tgt_cache_loc[dst_idx] = out_cache_loc[src_idx];
            }
            
            //=== SECOND PART: Write to to_free_slots ===
            
            // Calculate prefix sum of to_free_num_slots for all batches before current batch
            scalar_t to_free_slots_start = 0;
            for (int64_t i = 0; i < bid; i++) {
                to_free_slots_start += to_free_num_slots[i];
            }
            
            // Get current batch's free slot count
            int64_t to_free_num_slots_cur = static_cast<int64_t>(to_free_num_slots[bid]);
            
            // Calculate start offset within the current batch's out_cache_loc section
            int64_t out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur;
            
            // Copy from out_cache_loc (end portion) to to_free_slots
            offset = 0;
            for (; offset < to_free_num_slots_cur && offset < num_verify_tokens_upper; offset++) {
                int64_t src_idx = bid * num_verify_tokens + out_cache_loc_start + offset;
                int64_t dst_idx = static_cast<int64_t>(to_free_slots_start) + offset;
                
                to_free_slots[dst_idx] = out_cache_loc[src_idx];
            }
        }
    });
}

} // namespace

void get_target_cache_loc_kernel(
    at::Tensor& tgt_cache_loc,
    at::Tensor& to_free_slots,
    const at::Tensor& accept_length,
    const at::Tensor& to_free_num_slots,
    const at::Tensor& out_cache_loc,
    int64_t num_verify_tokens,
    int64_t num_verify_tokens_upper,
    int64_t bs_upper
) {
    AT_DISPATCH_ALL_TYPES(tgt_cache_loc.scalar_type(), "get_target_cache_loc", [&] {
        get_target_cache_loc_kernel_impl<scalar_t>(
            tgt_cache_loc.data_ptr<scalar_t>(),
            to_free_slots.data_ptr<scalar_t>(),
            accept_length.data_ptr<scalar_t>(),
            to_free_num_slots.data_ptr<scalar_t>(),
            out_cache_loc.data_ptr<scalar_t>(),
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper
        );
    });
}