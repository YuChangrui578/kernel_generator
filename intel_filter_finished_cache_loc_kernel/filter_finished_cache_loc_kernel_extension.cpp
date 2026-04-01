// Filename: filter_finished_cache_loc_kernel_extension.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

void filter_finished_cache_loc_kernel_cpu(
    at::Tensor out_cache_loc,
    const at::Tensor& tgt_cache_loc,
    const at::Tensor& accept_length,
    const at::Tensor& accept_length_filter,
    int64_t bs_upper,
    int64_t num_verify_tokens_upper
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_finished_cache_loc_kernel", &filter_finished_cache_loc_kernel_cpu, 
          "Filter finished cache location kernel for Intel CPU");
}