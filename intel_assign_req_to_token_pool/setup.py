# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_assign_req_to_token_pool',
    ext_modules=[
        CppExtension(
            name='intel_assign_req_to_token_pool',
            sources=[
                'assign_req_to_token_pool_extension.cpp', 
                'intel_assign_req_to_token_pool_kernel.cpp'
            ],
            extra_compile_args=[
                '-O3', 
                '-march=native', 
                '-mavx2', 
                '-ffast-math',
                '-funroll-loops',
                '-fno-semantic-interposition',
                '-flto',
                '-fopenmp'
            ],
            language='c++'
        )
    ],
    cmdclass={'build_ext': BuildExtension}, 
    zip_safe=False,
)