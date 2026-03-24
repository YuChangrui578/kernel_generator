# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_create_flashinfer_kv_indices',
    ext_modules=[
        CppExtension(
            name='intel_create_flashinfer_kv_indices',
            sources=['create_flashinfer_kv_indices_extension.cpp', 'intel_create_flashinfer_kv_indices_kernel.cpp'],
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