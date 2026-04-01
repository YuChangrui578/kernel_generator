# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_create_extend_after_decode_spec_info',
    ext_modules=[
        CppExtension(
            name='intel_create_extend_after_decode_spec_info',
            sources=[
                'create_extend_after_decode_spec_info_extension.cpp', 
                'intel_create_extend_after_decode_spec_info_kernel.cpp'
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