// Filename: create_flashinfer_kv_indices_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

// Declare the kernel implementation
void create_flashinfer_kv_indices_cpu_kernel(
    at::Tensor req_to_token,
    at::Tensor req_pool_indices,
    at::Tensor page_kernel_lens,
    at::Tensor kv_indptr,
    at::Tensor kv_start_idx,
    at::Tensor kv_indices,
    int64_t req_to_token_stride);

// Binding definition
at::Tensor create_flashinfer_kv_indices(
    at::Tensor req_to_token,
    at::Tensor req_pool_indices,
    at::Tensor page_kernel_lens,
    at::Tensor kv_indptr,
    c10::optional<at::Tensor> kv_start_idx_opt,
    at::Tensor kv_indices,
    int64_t req_to_token_stride) {

  // Handle optional tensor
  at::Tensor kv_start_idx;
  if (kv_start_idx_opt.has_value()) {
      kv_start_idx = kv_start_idx_opt.value();
  } else {
      // Create a dummy undefined tensor or empty tensor to pass None logic
      kv_start_idx = at::Tensor(); 
  }

  // Input checks
  TORCH_CHECK(req_to_token.is_cpu(), "req_to_token must be on CPU");
  TORCH_CHECK(req_pool_indices.is_cpu(), "req_pool_indices must be on CPU");
  TORCH_CHECK(page_kernel_lens.is_cpu(), "page_kernel_lens must be on CPU");
  TORCH_CHECK(kv_indptr.is_cpu(), "kv_indptr must be on CPU");
  TORCH_CHECK(kv_indices.is_cpu(), "kv_indices must be on CPU");
  
  TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kInt, "req_pool_indices must be int32");
  TORCH_CHECK(page_kernel_lens.scalar_type() == at::kInt, "page_kernel_lens must be int32");
  TORCH_CHECK(kv_indptr.scalar_type() == at::kInt, "kv_indptr must be int32");
  TORCH_CHECK(kv_indices.scalar_type() == at::kInt, "kv_indices must be int32");

  // Check tensor dimensions
  TORCH_CHECK(req_to_token.dim() >= 1, "req_to_token must have at least 1 dimension");
  TORCH_CHECK(req_pool_indices.size(0) == page_kernel_lens.size(0), 
              "req_pool_indices and page_kernel_lens must have the same batch size");
  TORCH_CHECK(req_pool_indices.size(0) <= kv_indptr.size(0), 
              "kv_indptr must have at least batch_size elements");
  
  // Check if kv_indices has enough space
  int64_t total_kv_len = kv_indptr[req_pool_indices.size(0)].item<int64_t>();
  TORCH_CHECK(kv_indices.numel() >= total_kv_len, 
              "kv_indices must have enough space for all KV indices");

  // Call the kernel
  create_flashinfer_kv_indices_cpu_kernel(
      req_to_token,
      req_pool_indices,
      page_kernel_lens,
      kv_indptr,
      kv_start_idx,
      kv_indices,
      req_to_token_stride
  );

  return kv_indices;
}

// Register the operator
TORCH_LIBRARY_FRAGMENT(intel_create_flashinfer_kv_indices, m) {
  m.def("create_flashinfer_kv_indices(Tensor req_to_token, Tensor req_pool_indices, Tensor page_kernel_lens, Tensor kv_indptr, Tensor? kv_start_idx, Tensor kv_indices, int req_to_token_stride) -> Tensor");
}

// Register the implementation
TORCH_LIBRARY_IMPL(intel_create_flashinfer_kv_indices, CPU, m) {
  m.impl("create_flashinfer_kv_indices", &create_flashinfer_kv_indices);
}

// Python binding helper for pybind11 (if direct pybind11 is used, but Torch Library API is preferred for modern pytorch)
// However, to ensure it acts like a standard python extension, we define the module:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_flashinfer_kv_indices", &create_flashinfer_kv_indices, "create_flashinfer_kv_indices CPU Implementation");
}