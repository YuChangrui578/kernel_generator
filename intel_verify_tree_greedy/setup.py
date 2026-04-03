# Filename: setup.py
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_verify_tree_greedy',
    ext_modules=[
        CppExtension(
            name='intel_verify_tree_greedy',
            sources=['verify_tree_greedy_extension.cpp', 'intel_verify_tree_greedy_kernel.cpp'],
            extra_compile_args=[
                '-O3', 
                '-march=native', 
                '-ffast-math',
                '-funroll-loops',
                '-fno-semantic-interposition',
                '-flto'
            ],
            language='c++'
        )
    ],
    cmdclass={'build_ext': BuildExtension}, 
    zip_safe=False,
)