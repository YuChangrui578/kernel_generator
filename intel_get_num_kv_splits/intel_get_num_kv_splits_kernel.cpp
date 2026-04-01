// Filename: intel_get_num_kv_splits_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

// Helper function to compute ceiling division
inline int64_t cdiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <typename scalar_t>
void get_num_kv_splits_kernel_impl(
    scalar_t* __restrict__ num_kv_splits_ptr,
    const scalar_t* __restrict__ seq_lens_ptr,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count,
    int64_t MAX_NUM_SEQ
) {
    // First, find max and min sequence lengths
    int64_t max_seq_len = 0;
    int64_t min_seq_len = INT64_MAX;
    
    // Sequential pass to find min/max sequence lengths
    for (int64_t i = 0; i < num_seq && i < MAX_NUM_SEQ; ++i) {
        int64_t seq_len = static_cast<int64_t>(seq_lens_ptr[i]);
        if (seq_len > max_seq_len) {
            max_seq_len = seq_len;
        }
        if (seq_len > 0 && seq_len < min_seq_len) {
            min_seq_len = seq_len;
        }
    }
    
    // Handle edge case where all sequences are zero length
    if (min_seq_len == INT64_MAX) {
        min_seq_len = (max_seq_len > 0) ? max_seq_len : 1;
    }
    
    // Apply heuristic adjustment - if max and min seq lens are close enough,
    // treat them as equal to avoid excessive splitting
    if (max_seq_len * 8 < min_seq_len * 10) {
        min_seq_len = max_seq_len;
    }
    
    // Strategy 1: Split based on sequence length ratio
    int64_t max_kv_splits_1 = std::min(cdiv(max_seq_len, min_seq_len), max_kv_splits);
    int64_t kv_chunk_size_1 = cdiv(max_seq_len, max_kv_splits_1);
    
    // Strategy 2: Split based on device core availability and sequence characteristics
    float ext_seq_len = static_cast<float>(max_seq_len) / 64.0f;
    float log_ext_seq_len = std::max(std::log2f(ext_seq_len), 1.0f);
    int64_t ext_device_core_count = device_core_count * static_cast<int64_t>(log_ext_seq_len);
    
    int64_t block_h = 16;
    int64_t num_kv_group = num_head / num_kv_head;  // Assuming integer division
    
    int64_t token_grid;
    if (num_kv_group == 1) {
        token_grid = num_seq * num_group * num_head;
    } else {
        // Adjust block size based on grouping
        block_h = std::min(block_h, num_kv_group);
        token_grid = num_seq * num_group * cdiv(num_head, block_h);
    }
    
    // Calculate second strategy's split parameters
    int64_t max_kv_splits_2 = std::min(cdiv(ext_device_core_count, token_grid), max_kv_splits);
    int64_t kv_chunk_size_2 = cdiv(max_seq_len, max_kv_splits_2);
    
    // Parallel computation of KV splits for each sequence and group
    at::parallel_for(0, num_seq, 0, [&](int64_t start, int64_t end) {
        for (int64_t seq_idx = start; seq_idx < end; ++seq_idx) {
            if (seq_idx >= num_seq || seq_idx >= MAX_NUM_SEQ) continue;
            
            int64_t current_seq_len = static_cast<int64_t>(seq_lens_ptr[seq_idx]);
            
            // Calculate splits needed based on both strategies
            int64_t splits_strategy_1 = cdiv(current_seq_len, kv_chunk_size_1);
            int64_t splits_strategy_2 = cdiv(current_seq_len, kv_chunk_size_2);
            
            int64_t num_kv_splits = std::max(splits_strategy_1, splits_strategy_2);
            // Clamp to max allowed splits
            num_kv_splits = std::min(num_kv_splits, max_kv_splits);
            // Ensure at least 1 split
            num_kv_splits = std::max(num_kv_splits, static_cast<int64_t>(1));
            
            // Store the same number of splits for all groups of this sequence
            for (int64_t group_idx = 0; group_idx < num_group; ++group_idx) {
                int64_t output_idx = seq_idx * num_group + group_idx;
                num_kv_splits_ptr[output_idx] = static_cast<scalar_t>(num_kv_splits);
            }
        }
    });
}

} // anonymous namespace

void get_num_kv_splits_kernel(
    at::Tensor& num_kv_splits,
    const at::Tensor& seq_lens,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count,
    int64_t MAX_NUM_SEQ
) {
    AT_DISPATCH_ALL_TYPES(num_kv_splits.scalar_type(), "get_num_kv_splits", [&] {
        get_num_kv_splits_kernel_impl<scalar_t>(
            num_kv_splits.data_ptr<scalar_t>(),
            seq_lens.data_ptr<scalar_t>(),
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count,
            MAX_NUM_SEQ
        );
    });
}