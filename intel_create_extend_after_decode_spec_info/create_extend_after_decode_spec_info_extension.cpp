// Filename: create_extend_after_decode_spec_info_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

void create_extend_after_decode_spec_info_kernel(
    const at::Tensor& verified_id,
    const at::Tensor& seq_lens,
    const at::Tensor& accept_lens,
    at::Tensor& positions,
    at::Tensor& new_verified_id,
    int64_t bs_upper);

void create_extend_after_decode_spec_info(
    const at::Tensor& verified_id,
    const at::Tensor& seq_lens,
    const at::Tensor& accept_lens,
    at::Tensor& positions,
    at::Tensor& new_verified_id,
    int64_t bs_upper) {
    
    TORCH_CHECK(verified_id.is_contiguous(), "verified_id tensor must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens tensor must be contiguous");
    TORCH_CHECK(accept_lens.is_contiguous(), "accept_lens tensor must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions tensor must be contiguous");
    TORCH_CHECK(new_verified_id.is_contiguous(), "new_verified_id tensor must be contiguous");
    
    TORCH_CHECK(verified_id.dtype() == torch::kInt32, "verified_id tensor must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens tensor must be int32");
    TORCH_CHECK(accept_lens.dtype() == torch::kInt32, "accept_lens tensor must be int32");
    TORCH_CHECK(positions.dtype() == torch::kInt32, "positions tensor must be int32");
    TORCH_CHECK(new_verified_id.dtype() == torch::kInt32, "new_verified_id tensor must be int32");
    
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-dimensional");
    TORCH_CHECK(accept_lens.dim() == 1, "accept_lens must be 1-dimensional");
    
    auto batch_size = seq_lens.size(0);
    TORCH_CHECK(accept_lens.size(0) == batch_size, "seq_lens and accept_lens must have same first dimension");
    TORCH_CHECK(new_verified_id.size(0) == batch_size, "new_verified_id must match batch size");
    
    create_extend_after_decode_spec_info_kernel(
        verified_id,
        seq_lens,
        accept_lens,
        positions,
        new_verified_id,
        bs_upper
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_extend_after_decode_spec_info", &create_extend_after_decode_spec_info, 
          "Intel CPU implementation of create_extend_after_decode_spec_info kernel");
}