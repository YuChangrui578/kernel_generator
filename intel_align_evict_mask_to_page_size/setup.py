# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_align_evict_mask_to_page_size',
    ext_modules=[
        CppExtension(
            name='intel_align_evict_mask_to_page_size',
            sources=['align_evict_mask_to_page_size_extension.cpp', 'intel_align_evict_mask_to_page_size_kernel.cpp'],
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