// Filename: intel_reconstruct_indices_from_tree_mask_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

template <typename scalar_t>
void reconstruct_indices_from_tree_mask_kernel_impl(
    const bool* __restrict__ tree_mask_ptr,
    const int64_t* __restrict__ verified_seq_len_ptr,
    int64_t* __restrict__ positions_ptr,
    int64_t* __restrict__ retrive_index_ptr,
    int64_t* __restrict__ retrive_next_token_ptr,
    int64_t* __restrict__ retrive_next_sibling_ptr,
    int64_t batch_size,
    int64_t draft_token_num) {
  
  at::parallel_for(0, batch_size * draft_token_num, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      // Calculate batch id and token id from linear index
      int64_t bid = idx / draft_token_num;
      int64_t tid = idx % draft_token_num;
      
      if (bid >= batch_size || tid >= draft_token_num) {
        continue;
      }
      
      int64_t base_offset = draft_token_num * draft_token_num;
      // token_idx: [bid * draft_token_num, (bid + 1) * draft_token_num)
      int64_t token_idx = bid * draft_token_num;
      // tree_mask_idx: [bid * base_offset, (bid + 1) * base_offset)
      int64_t tree_mask_offset = bid * base_offset;

      // Step 1: Calculate depth and find parent index
      int64_t depth = 0;
      int64_t parent_idx = -1;

      for (int64_t i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
        if (tree_mask_ptr[start_idx + i]) {
          depth++;
          if (parent_idx == -1) {
            parent_idx = i;
          }
        }
      }
      
      // Step 2: Set retrieve index (identity mapping)
      retrive_index_ptr[token_idx + tid] = token_idx + tid;
      
      // Step 3: Set position based on depth and verified sequence length
      positions_ptr[token_idx + tid] = depth + verified_seq_len_ptr[bid];

      // Step 4: Find next token that depends on current token
      int64_t next_token_idx = -1;
      for (int64_t i = tid + 1; i < draft_token_num; i++) {
        if (tree_mask_ptr[tree_mask_offset + i * draft_token_num + tid]) {
          next_token_idx = i;
          break;
        }
      }
      retrive_next_token_ptr[token_idx + tid] = next_token_idx;

      // Step 5: Find next sibling (tokens that share same parent)
      int64_t next_sibling_idx = -1;
      if (parent_idx != -1) {
        for (int64_t i = tid + 1; i < draft_token_num; i++) {
          int64_t start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
          if (tree_mask_ptr[start_idx]) {
            bool is_sibling = true;
            int64_t end_idx = tree_mask_offset + i * draft_token_num + i;
            for (int64_t j = start_idx + 1; j < end_idx; ++j) {
              if (tree_mask_ptr[j]) {
                is_sibling = false;
                break;
              }
            }
            if (is_sibling) {
              next_sibling_idx = i;
              break;
            }
          }
        }
      }
      retrive_next_sibling_ptr[token_idx + tid] = next_sibling_idx;
    }
  });
}

void reconstruct_indices_from_tree_mask_kernel(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  
  // Convert tree_mask to bool if needed
  auto tree_mask_bool = tree_mask.to(at::ScalarType::Bool);
  
  reconstruct_indices_from_tree_mask_kernel_impl<bool>(
      tree_mask_bool.data_ptr<bool>(),
      verified_seq_len.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      batch_size,
      draft_token_num);
}