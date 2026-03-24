#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cmath>

namespace {

inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

inline int64_t floor_div(int64_t a, int64_t b) {
    return a / b;
}

} // anonymous namespace

void get_num_kv_splits_kernel_impl(
    int32_t* num_kv_splits_ptr,
    const int32_t* seq_lens_ptr,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count
) {
    using fVec = at::vec::Vectorized<float>;
    using iVec = at::vec::Vectorized<int32_t>;
    
    constexpr int64_t VecSize = iVec::size();
    
    // Find max and min sequence lengths
    int32_t max_seq_len = 0;
    int32_t min_seq_len = INT32_MAX;
    
    for (int64_t i = 0; i < num_seq; ++i) {
        int32_t seq_len = seq_lens_ptr[i];
        if (seq_len > max_seq_len) {
            max_seq_len = seq_len;
        }
        if (seq_len < min_seq_len) {
            min_seq_len = seq_len;
        }
    }
    
    // Adjust min_seq_len if the difference is too large
    if (static_cast<int64_t>(max_seq_len) * 8 < static_cast<int64_t>(min_seq_len) * 10) {
        min_seq_len = max_seq_len;
    }
    
    // Calculate max_kv_splits_1
    int64_t max_kv_splits_1 = std::min(static_cast<int64_t>(ceil_div(max_seq_len, min_seq_len)), max_kv_splits);
    int64_t kv_chunk_size_1 = ceil_div(max_seq_len, max_kv_splits_1);
    
    // Calculate max_kv_splits_2 based on device core count
    float ext_seq_len = static_cast<float>(max_seq_len) / 64.0f;
    float log_ext_seq_len = std::log2(std::max(ext_seq_len, 1.0f));
    int64_t ext_device_core_count = static_cast<int64_t>(
        static_cast<float>(device_core_count) * std::max(log_ext_seq_len, 1.0f)
    );
    
    int64_t num_kv_group = num_head / num_kv_head;
    int64_t block_h = 16;
    
    int64_t token_grid;
    if (num_kv_group == 1) {
        token_grid = num_seq * num_group * num_head;
    } else {
        block_h = std::min(block_h, static_cast<int64_t>(num_kv_group));
        token_grid = num_seq * num_group * ceil_div(num_head, block_h);
    }
    
    int64_t max_kv_splits_2 = std::min(ceil_div(ext_device_core_count, token_grid), max_kv_splits);
    int64_t kv_chunk_size_2 = ceil_div(max_seq_len, max_kv_splits_2);
    
    // Process each sequence and calculate number of splits
    for (int64_t seq_idx = 0; seq_idx < num_seq; ++seq_idx) {
        int32_t seq_len = seq_lens_ptr[seq_idx];
        
        // Calculate number of splits based on both chunk sizes
        int64_t splits_1 = ceil_div(seq_len, kv_chunk_size_1);
        int64_t splits_2 = ceil_div(seq_len, kv_chunk_size_2);
        int64_t num_kv_splits = std::max(splits_1, splits_2);
        
        // Clamp to max_kv_splits
        num_kv_splits = std::min(num_kv_splits, max_kv_splits);
        
        // Store result for each group
        for (int64_t group_idx = 0; group_idx < num_group; ++group_idx) {
            int64_t output_idx = seq_idx * num_group + group_idx;
            num_kv_splits_ptr[output_idx] = static_cast<int32_t>(num_kv_splits);
        }
    }
}

// Dispatch function that handles different data types if needed
void get_num_kv_splits_kernel(
    at::Tensor& num_kv_splits,
    const at::Tensor& seq_lens,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count
) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Int, at::ScalarType::Long, num_kv_splits.scalar_type(), "get_num_kv_splits_kernel", [&] {
        auto num_kv_splits_accessor = num_kv_splits.accessor<int32_t, 1>();
        auto seq_lens_accessor = seq_lens.accessor<int32_t, 1>();
        
        // Get raw pointers for efficient processing
        int32_t* num_kv_splits_ptr = num_kv_splits.data_ptr<int32_t>();
        const int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
        
        get_num_kv_splits_kernel_impl(
            num_kv_splits_ptr,
            seq_lens_ptr,
            num_seq,
            num_group,
            num_head,
            num_kv_head,
            max_kv_splits,
            device_core_count
        );
    });
}