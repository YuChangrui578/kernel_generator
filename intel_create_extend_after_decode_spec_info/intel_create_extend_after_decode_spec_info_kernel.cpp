// Filename: intel_create_extend_after_decode_spec_info_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

void create_extend_after_decode_spec_info_kernel_impl(
    const int32_t* __restrict__ verified_id_ptr,
    const int32_t* __restrict__ seq_lens_ptr,
    const int32_t* __restrict__ accept_lens_ptr,
    int32_t* __restrict__ positions_ptr,
    int32_t* __restrict__ new_verified_id_ptr,
    int64_t batch_size,
    int64_t bs_upper) {
    
    // Process each batch in parallel
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
        for (int64_t pid = begin; pid < end; ++pid) {
            // Step 1: Load sequence and acceptance lengths for current batch
            int32_t seq_length = seq_lens_ptr[pid];
            int32_t accept_length = accept_lens_ptr[pid];
            
            // Step 2: Calculate cumulative sum of acceptance lengths from previous batches
            int32_t accept_len_cumsum = 0;
            for (int64_t i = 0; i < pid; i++) {
                accept_len_cumsum += accept_lens_ptr[i];
            }
            
            // Step 3: Fill position values for current batch
            int32_t* positions_ptr_batch = positions_ptr + accept_len_cumsum;
            
            // Fill positions with values: (seq_length - accept_length + index)
            // Only fill up to accept_length or bs_upper positions
            int64_t remaining_positions = std::min(static_cast<int64_t>(accept_length), 
                                                 static_cast<int64_t>(bs_upper));
            
            for (int64_t pos_idx = 0; pos_idx < remaining_positions; ++pos_idx) {
                positions_ptr_batch[pos_idx] = seq_length - accept_length + static_cast<int32_t>(pos_idx);
            }
            
            // Step 4: Store the last accepted token ID for this batch
            int32_t final_cumsum = accept_len_cumsum + accept_length - 1;
            int32_t verified_id_data = verified_id_ptr[final_cumsum];
            new_verified_id_ptr[pid] = verified_id_data;
        }
    });
}

} // anonymous namespace

void create_extend_after_decode_spec_info_kernel(
    const at::Tensor& verified_id,
    const at::Tensor& seq_lens,
    const at::Tensor& accept_lens,
    at::Tensor& positions,
    at::Tensor& new_verified_id,
    int64_t bs_upper) {
    
    create_extend_after_decode_spec_info_kernel_impl(
        verified_id.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        accept_lens.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        new_verified_id.data_ptr<int32_t>(),
        seq_lens.size(0),
        bs_upper
    );
}