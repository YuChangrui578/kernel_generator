// Filename: intel_align_evict_mask_to_page_size_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

template <typename scalar_t>
void align_evict_mask_to_page_size_kernel_impl(
    const int32_t* __restrict__ seq_lens_ptr,
    bool* __restrict__ evict_mask_ptr,
    int64_t batch_size,
    int64_t num_draft_tokens,
    int64_t page_size
) {
    // Process each sequence in parallel
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
        for (int64_t bid = begin; bid < end; ++bid) {
            // Step 1: Load sequence length for current sequence
            int32_t seq_len = seq_lens_ptr[bid];
            
            // Step 2: Count True values in the mask row for this sequence
            int64_t mask_row_offset = bid * num_draft_tokens;
            int64_t num_trues = 0;
            
            // Count True values in the mask row
            for (int64_t t = 0; t < num_draft_tokens; ++t) {
                if (evict_mask_ptr[mask_row_offset + t]) {
                    num_trues++;
                }
            }
            
            // Step 3: Calculate number of False values
            int64_t num_false = num_draft_tokens - num_trues;
            
            // Step 4: Calculate the starting position for clearing
            // This implements: start = (seq_len + num_false - 1) // page_size * page_size - seq_len
            int64_t aligned_pos = ((static_cast<int64_t>(seq_len) + num_false - 1) / page_size) * page_size;
            int64_t start = aligned_pos - static_cast<int64_t>(seq_len);
            
            // Step 5: Clear the eviction mask in the calculated range
            // Loop from max(start, 0) to min(start + page_size, num_draft_tokens)
            int64_t clear_start = std::max(start, static_cast<int64_t>(0));
            int64_t clear_end = std::min(start + page_size, num_draft_tokens);
            
            // Clear the range [clear_start, clear_end)
            for (int64_t i = clear_start; i < clear_end; ++i) {
                evict_mask_ptr[mask_row_offset + i] = false;
            }
        }
    });
}

} // namespace

void align_evict_mask_to_page_size_kernel(
    const at::Tensor& seq_lens,
    at::Tensor& evict_mask,
    int64_t page_size,
    int64_t num_draft_tokens,
    int64_t block_size
) {
    // Use AT_DISPATCH_INTEGRAL_TYPES to avoid duplicate case values
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens.scalar_type(), "align_evict_mask_to_page_size", [&]() {
        scalar_t* seq_lens_ptr = seq_lens.data_ptr<scalar_t>();
        bool* evict_mask_ptr = evict_mask.data_ptr<bool>();
        
        int64_t batch_size = seq_lens.size(0);
        
        align_evict_mask_to_page_size_kernel_impl<scalar_t>(
            reinterpret_cast<const int32_t*>(seq_lens_ptr),
            evict_mask_ptr,
            batch_size,
            num_draft_tokens,
            page_size
        );
    });
}