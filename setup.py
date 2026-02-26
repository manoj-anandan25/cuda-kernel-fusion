from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn_cuda',
    ext_modules=[
        CUDAExtension('flash_attn_cuda', [
            'csrc/fusion.cpp',
            'csrc/fusion_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)