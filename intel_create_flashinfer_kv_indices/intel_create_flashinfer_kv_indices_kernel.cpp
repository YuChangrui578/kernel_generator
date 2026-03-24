// Filename: intel_create_flashinfer_kv_indices_kernel.cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <c10/util/Exception.h>

template <typename scalar_t>
void create_flashinfer_kv_indices_kernel_impl(
    const scalar_t* __restrict__ req_to_token_ptr,
    const scalar_t* __restrict__ req_pool_indices_ptr,
    const scalar_t* __restrict__ page_kernel_lens_ptr,
    const scalar_t* __restrict__ kv_indptr,
    const scalar_t* __restrict__ kv_start_idx,
    scalar_t* __restrict__ kv_indices_ptr,
    int64_t req_to_token_stride,
    int64_t batch_size) {
  
  // Parallelize over the batch dimension
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // 1. Determine the source row in req_to_token
      int64_t req_pool_index = req_pool_indices_ptr[i];

      // 2. Calculate write offset in kv_indices
      int64_t kv_indices_offset = kv_indptr[i];

      // 3. Calculate read start offset (optional)
      int64_t kv_start = 0;
      if (kv_start_idx) {
        kv_start = kv_start_idx[i];
      }

      // 4. Determine the number of elements to copy (length)
      int64_t kv_len = page_kernel_lens_ptr[i];

      // 5. Calculate base pointers
      const scalar_t* src_base = req_to_token_ptr + req_pool_index * req_to_token_stride + kv_start;
      scalar_t* dst_base = kv_indices_ptr + kv_indices_offset;

      // 6. Simple copy loop
      for (int64_t j = 0; j < kv_len; ++j) {
        dst_base[j] = src_base[j];
      }
    }
  });
}

void create_flashinfer_kv_indices_cpu_kernel(
    at::Tensor req_to_token,
    at::Tensor req_pool_indices,
    at::Tensor page_kernel_lens,
    at::Tensor kv_indptr,
    at::Tensor kv_start_idx, // Optional tensor
    at::Tensor kv_indices,
    int64_t req_to_token_stride) {

  int64_t batch_size = req_pool_indices.size(0);

  AT_DISPATCH_INTEGRAL_TYPES(req_to_token.scalar_type(), "create_flashinfer_kv_indices_cpu_kernel", ([&] {
    const scalar_t* req_to_token_ptr = req_to_token.data_ptr<scalar_t>();
    const scalar_t* req_pool_indices_ptr = req_pool_indices.data_ptr<scalar_t>();
    const scalar_t* page_kernel_lens_ptr = page_kernel_lens.data_ptr<scalar_t>();
    const scalar_t* kv_indptr_ptr = kv_indptr.data_ptr<scalar_t>();
    
    const scalar_t* kv_start_idx_ptr = nullptr;
    if (kv_start_idx.defined()) {
        kv_start_idx_ptr = kv_start_idx.data_ptr<scalar_t>();
    }

    scalar_t* kv_indices_ptr = kv_indices.data_ptr<scalar_t>();

    create_flashinfer_kv_indices_kernel_impl<scalar_t>(
        req_to_token_ptr,
        req_pool_indices_ptr,
        page_kernel_lens_ptr,
        kv_indptr_ptr,
        kv_start_idx_ptr,
        kv_indices_ptr,
        req_to_token_stride,
        batch_size
    );
  }));
}