# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_filter_finished_cache_loc_kernel',
    ext_modules=[
        CppExtension(
            name='intel_filter_finished_cache_loc_kernel',
            sources=['filter_finished_cache_loc_kernel_extension.cpp', 'intel_filter_finished_cache_loc_kernel_kernel.cpp'],
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