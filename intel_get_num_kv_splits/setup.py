from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_get_num_kv_splits',
    ext_modules=[
        CppExtension(
            name='intel_get_num_kv_splits',
            sources=['get_num_kv_splits_extension.cpp', 'intel_get_num_kv_splits_kernel.cpp'],
            extra_compile_args=['-O3', '-march=native', '-mavx512f', '-mavx512dq', '-mavx512cd', '-mavx512bw', '-mavx512vl'],
            language='c++'
        )
    ],
    cmdclass={'build_ext': BuildExtension}, 
    zip_safe=False,
)