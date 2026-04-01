// Filename: intel_generate_draft_decode_kv_indices_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

using namespace at;

template<typename T>
static inline void data_index_init(int64_t index, int64_t& bi, int64_t batch_size, 
                                  int64_t& si, int64_t seq_len, int64_t& ni, int64_t heads) {
    bi = index / (seq_len * heads);
    int64_t remaining = index % (seq_len * heads);
    si = remaining / heads;
    ni = remaining % heads;
}

static inline void data_index_step(int64_t& bi, int64_t batch_size, 
                                  int64_t& si, int64_t seq_len, int64_t& ni, int64_t heads) {
    ni++;
    if (ni >= heads) {
        ni = 0;
        si++;
        if (si >= seq_len) {
            si = 0;
            bi++;
        }
    }
}

template<typename scalar_t>
void generate_draft_decode_kv_indices_kernel_impl(
    const Tensor& req_pool_indices,
    const Tensor& req_to_token,
    const Tensor& paged_kernel_lens,
    Tensor& kv_indices,
    Tensor& kv_indptr,
    const Tensor& positions,
    int64_t pool_len,
    int64_t kv_indices_stride,
    int64_t kv_indptr_stride,
    int64_t bs_upper,
    int64_t iter_upper,
    int64_t num_tokens_upper,
    int64_t page_size
) {
    // Get raw data pointers
    auto req_pool_indices_ptr = req_pool_indices.data_ptr<int32_t>();
    auto req_to_token_ptr = req_to_token.data_ptr<int32_t>();
    auto paged_kernel_lens_ptr = paged_kernel_lens.data_ptr<int32_t>();
    auto kv_indices_ptr = kv_indices.mutable_data_ptr<int32_t>();
    auto kv_indptr_ptr = kv_indptr.mutable_data_ptr<int32_t>();
    auto positions_ptr = positions.data_ptr<int32_t>();
    
    int64_t num_seqs = req_pool_indices.size(0);
    int64_t speculative_num_steps = kv_indices.size(0);
    
    // Main parallel processing loop
    at::parallel_for(0, speculative_num_steps * num_seqs, 0, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            int64_t iters = idx / num_seqs;
            int64_t bid = idx % num_seqs;
            
            if (iters >= speculative_num_steps) continue;
            
            // Calculate sequence length for current bid
            int64_t seq_len = paged_kernel_lens_ptr[bid];
            
            // Calculate cumulative sequence length up to bid
            int64_t cum_seq_len = 0;
            for (int64_t i = 0; i < bid && i < bs_upper; ++i) {
                cum_seq_len += paged_kernel_lens_ptr[i];
            }
            
            // Process for different topk values (assuming topk is inferred from output shape)
            int64_t topk = kv_indices.size(2);
            
            for (int64_t topk_id = 0; topk_id < topk; ++topk_id) {
                int64_t current_iter = iters + 1;
                
                // Calculate kv_offset for storing indices
                int64_t kv_offset = cum_seq_len * topk + bid * current_iter * topk + 
                                   topk_id * (seq_len + current_iter);
                
                // Get pointer to req_to_token for current request
                int64_t req_pool_idx = req_pool_indices_ptr[bid];
                int64_t token_pool_start = req_pool_idx * pool_len;
                
                // Process existing tokens in blocks
                int64_t processed = 0;
                const int64_t BLOCK_SIZE = 128;
                
                // Processing of existing tokens
                while (processed < seq_len) {
                    int64_t current_block_size = std::min(BLOCK_SIZE, seq_len - processed);
                    
                    // Handle remaining elements
                    for (int64_t vec_processed = 0; vec_processed < current_block_size; vec_processed++) {
                        int64_t src_idx = token_pool_start + processed + vec_processed;
                        int64_t dst_idx = (kv_indices_stride * iters) + kv_offset + processed + vec_processed;
                        
                        if (src_idx < req_to_token.numel() && dst_idx < kv_indices.numel()) {
                            kv_indices_ptr[dst_idx] = req_to_token_ptr[src_idx];
                        }
                    }
                    
                    processed += current_block_size;
                }
                
                // Handle extended tokens (newly generated ones)
                for (int64_t ext_iter = 0; ext_iter < current_iter && ext_iter < iter_upper; ++ext_iter) {
                    int64_t extend_dst_idx = (kv_indices_stride * iters) + kv_offset + seq_len + ext_iter;
                    
                    int64_t extend_src_idx;
                    if (page_size == 1 || topk == 1) {
                        // Simple case: direct indexing
                        extend_src_idx = token_pool_start + seq_len + 
                                       topk_id * speculative_num_steps + ext_iter;
                    } else {
                        // Complex paged memory layout
                        int64_t prefix_len = seq_len;
                        int64_t last_page_len = prefix_len % page_size;
                        int64_t num_new_pages_per_topk = (last_page_len + speculative_num_steps + 
                                                         page_size - 1) / page_size;
                        int64_t prefix_base = (prefix_len / page_size) * page_size;
                        int64_t start = prefix_base + topk_id * num_new_pages_per_topk * 
                                       page_size + last_page_len;
                        
                        extend_src_idx = token_pool_start + start + ext_iter;
                    }
                    
                    if (extend_src_idx < req_to_token.numel() && 
                        extend_dst_idx < kv_indices.numel()) {
                        kv_indices_ptr[extend_dst_idx] = req_to_token_ptr[extend_src_idx];
                    }
                }
                
                // Update kv_indptr for this sequence-topk combination
                int64_t zid = bid * topk + topk_id;
                if (zid == 0) {
                    zid = num_seqs * topk;
                }
                
                // Calculate base position by summing positions up to zid
                int64_t base_pos = 0;
                for (int64_t pos_i = 0; pos_i < zid && pos_i < num_tokens_upper; ++pos_i) {
                    base_pos += positions_ptr[pos_i];
                }
                
                // Store the computed indptr value
                int64_t indptr_idx = (kv_indptr_stride * iters) + zid;
                if (indptr_idx < kv_indptr.numel()) {
                    kv_indptr_ptr[indptr_idx] = static_cast<int32_t>(base_pos + zid * current_iter);
                }
            }
        }
    });
}

void generate_draft_decode_kv_indices_kernel(
    const Tensor& req_pool_indices,
    const Tensor& req_to_token,
    const Tensor& paged_kernel_lens,
    Tensor& kv_indices,
    Tensor& kv_indptr,
    const Tensor& positions,
    int64_t pool_len,
    int64_t kv_indices_stride,
    int64_t kv_indptr_stride,
    int64_t bs_upper,
    int64_t iter_upper,
    int64_t num_tokens_upper,
    int64_t page_size
) {
    AT_DISPATCH_INTEGRAL_TYPES(req_pool_indices.scalar_type(), "generate_draft_decode_kv_indices", [&] {
        generate_draft_decode_kv_indices_kernel_impl<scalar_t>(
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
    });
}