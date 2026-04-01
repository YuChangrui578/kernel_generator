# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_get_target_cache_loc',
    ext_modules=[
        CppExtension(
            name='intel_get_target_cache_loc',
            sources=['get_target_cache_loc_extension.cpp', 'intel_get_target_cache_loc_kernel.cpp'],
            extra_compile_args=[
                '-O3', 
                '-march=native', 
                '-ffast-math',
                '-funroll-loops',
                '-fno-semantic-interposition',
                '-fopenmp'
            ],
            language='c++'
        )
    ],
    cmdclass={'build_ext': BuildExtension}, 
    zip_safe=False,
)