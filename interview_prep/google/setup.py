from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

# Define the C++ extension modules
ext_modules = [
    CUDAExtension('example_kernels', [
        '/tiledMatMul_binding.cpp',
        '/tiledMatMul.cu',
    ])
]

setup(
    name="cuda_basics",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
