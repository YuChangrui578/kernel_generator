// Filename: intel_verify_tree_greedy_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CONTIGUOUS(x); \
  TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")

void verify_tree_greedy_cpu_impl(
    int32_t* predicts,
    int32_t* accept_index,
    int32_t* accept_token_num,
    const int64_t* candidates,
    const int64_t* retrive_index,
    const int64_t* retrive_next_token,
    const int64_t* retrive_next_sibling,
    const int64_t* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  
  // Process each batch sequentially
  for (uint32_t bx = 0; bx < batch_size; ++bx) {
      
    // Calculate offsets for current batch
    uint32_t candidate_offset = bx * num_draft_tokens;
    uint32_t retrive_index_offset = bx * num_draft_tokens;
    uint32_t next_token_offset = bx * num_draft_tokens;
    uint32_t next_sibling_offset = bx * num_draft_tokens;
    uint32_t target_offset = bx * num_draft_tokens;
    uint32_t accept_index_offset = bx * num_speculative_tokens;
    
    // Initialize state for current batch
    int64_t last_accepted_retrive_idx = retrive_index[retrive_index_offset];
    accept_index[accept_index_offset] = static_cast<int32_t>(last_accepted_retrive_idx);
    
    uint32_t num_accepted_tokens = 0;
    int64_t cur_index = 0; // Start from root of tree for current batch
    
    // Traverse through speculative tokens for current batch
    for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
        
      // Move to next token in the tree structure
      cur_index = retrive_next_token[next_token_offset + cur_index];
      
      // Traverse siblings at current level until match is found or all siblings are exhausted
      while (cur_index != -1) {
          
        // Get draft token information for current position
        int64_t draft_index = retrive_index[retrive_index_offset + cur_index];
        int64_t draft_token_id = candidates[candidate_offset + cur_index];
        
        // Use last_accepted_retrive_idx as the target position (as per CUDA reference)
        int64_t target_token_id = target_predict[last_accepted_retrive_idx];
        
        if (draft_token_id == target_token_id) {
          // Token accepted - update prediction and tracking variables
          predicts[last_accepted_retrive_idx] = static_cast<int32_t>(target_token_id);
          ++num_accepted_tokens;
          
          // Record the accepted token's index
          accept_index[accept_index_offset + num_accepted_tokens] = static_cast<int32_t>(draft_index);
          
          // Update the last accepted retrieve index for next iteration
          last_accepted_retrive_idx = draft_index;
          
          // Break out of sibling traversal loop - move to next level
          break;
        } else {
          // No match - try next sibling at current level
          cur_index = retrive_next_sibling[next_sibling_offset + cur_index];
        }
      }
      
      // If no valid sibling found (cur_index == -1), stop processing for this batch
      if (cur_index == -1) {
        break;
      }
    }
    
    // Store total number of accepted tokens for current batch
    accept_token_num[bx] = num_accepted_tokens;
    
    // Set final prediction for the last accepted token
    predicts[last_accepted_retrive_idx] = static_cast<int32_t>(target_predict[last_accepted_retrive_idx]);
  }
}

void verify_tree_greedy(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict) {
  
  CHECK_INPUT(predicts);
  CHECK_INPUT(accept_index);
  CHECK_INPUT(accept_token_num);
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);
  
  auto device = target_predict.device();
  TORCH_CHECK(device.is_cpu(), "All tensors must be on CPU");
  
  TORCH_CHECK(predicts.dim() == 1, "predicts must be 1-dimensional");
  TORCH_CHECK(accept_index.dim() == 2, "accept_index must be 2-dimensional");
  TORCH_CHECK(accept_token_num.dim() == 1, "accept_token_num must be 1-dimensional");
  TORCH_CHECK(candidates.dim() == 2, "candidates must be 2-dimensional");
  TORCH_CHECK(retrive_index.dim() == 2, "retrive_index must be 2-dimensional");
  TORCH_CHECK(retrive_next_token.dim() == 2, "retrive_next_token must be 2-dimensional");
  TORCH_CHECK(retrive_next_sibling.dim() == 2, "retrive_next_sibling must be 2-dimensional");
  TORCH_CHECK(target_predict.dim() == 2, "target_predict must be 2-dimensional");
  
  unsigned int batch_size = candidates.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  
  TORCH_CHECK(batch_size == accept_index.size(0));
  TORCH_CHECK(batch_size == accept_token_num.size(0));
  TORCH_CHECK(batch_size == retrive_index.size(0));
  TORCH_CHECK(batch_size == retrive_next_token.size(0));
  TORCH_CHECK(batch_size == retrive_next_sibling.size(0));
  TORCH_CHECK(batch_size == target_predict.size(0));
  TORCH_CHECK(num_draft_tokens == retrive_index.size(1));
  TORCH_CHECK(num_draft_tokens == retrive_next_token.size(1));
  TORCH_CHECK(num_draft_tokens == retrive_next_sibling.size(1));
  TORCH_CHECK(num_draft_tokens == target_predict.size(1));
  
  TORCH_CHECK(predicts.dtype() == at::kInt, "Expected 'predicts' to be of type int (torch.int32)");
  TORCH_CHECK(accept_index.dtype() == at::kInt, "Expected 'accept_index' to be of type int (torch.int32)");
  TORCH_CHECK(accept_token_num.dtype() == at::kInt, "Expected 'accept_token_num' to be of type int (torch.int32)");
  TORCH_CHECK(candidates.dtype() == at::kLong, "Expected 'candidates' to be of type long (torch.int64)");
  TORCH_CHECK(retrive_index.dtype() == at::kLong, "Expected 'retrive_index' to be of type long (torch.int64)");
  TORCH_CHECK(retrive_next_token.dtype() == at::kLong, "Expected 'retrive_next_token' to be of type long (torch.int64)");
  TORCH_CHECK(retrive_next_sibling.dtype() == at::kLong, "Expected 'retrive_next_sibling' to be of type long (torch.int64)");
  TORCH_CHECK(target_predict.dtype() == at::kLong, "Expected 'target_predict' to be of type long (torch.int64)");

  verify_tree_greedy_cpu_impl(
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int64_t*>(candidates.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<int64_t*>(target_predict.data_ptr()),
      batch_size,
      num_spec_step,
      num_draft_tokens);
}