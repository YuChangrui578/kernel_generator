# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_reconstruct_indices_from_tree_mask',
    ext_modules=[
        CppExtension(
            name='intel_reconstruct_indices_from_tree_mask',
            sources=['reconstruct_indices_from_tree_mask_extension.cpp', 'intel_reconstruct_indices_from_tree_mask_kernel.cpp'],
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