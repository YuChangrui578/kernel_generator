#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>

void get_num_kv_splits_kernel(
    at::Tensor& num_kv_splits,
    const at::Tensor& seq_lens,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count
);

void get_num_kv_splits(
    at::Tensor& num_kv_splits,
    const at::Tensor& seq_lens,
    int64_t num_seq,
    int64_t num_group,
    int64_t num_head,
    int64_t num_kv_head,
    int64_t max_kv_splits,
    int64_t device_core_count
) {
    get_num_kv_splits_kernel(
        num_kv_splits,
        seq_lens,
        num_seq,
        num_group,
        num_head,
        num_kv_head,
        max_kv_splits,
        device_core_count
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_num_kv_splits", &get_num_kv_splits, "get_num_kv_splits kernel");
}