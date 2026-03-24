# Filename: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='intel_assign_draft_cache_locs',
    ext_modules=[
        CppExtension(
            name='intel_assign_draft_cache_locs',
            sources=[
                'assign_draft_cache_locs_extension.cpp', 
                'intel_assign_draft_cache_locs_kernel.cpp'
            ],
            extra_compile_args=[
                '-O3', 
                '-march=native', 
                # Enable AVX2 and AVX-512 instruction sets
                '-mavx2', 
                '-mfma',
                # AVX-512 flags (uncomment if your CPU supports AVX-512, e.g., Xeon Scalable or newer Core)
                # '-mavx512f', '-mavx512dq', '-mavx512cd', '-mavx512bw', '-mavx512vl'
            ],
            extra_link_args=[],
            language='c++'
        )
    ],
    cmdclass={'build_ext': BuildExtension}, 
    zip_safe=False,
)