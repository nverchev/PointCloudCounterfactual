from setuptools import setup #type: ignore
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    install_requires=['torch'],
    packages=['emd'],
    package_data={"emd": ["py.typed", "*.pyi"]},
    ext_modules=[
        CUDAExtension(
            name='emd.emd_backend',
            sources=[
                'src/emd.cpp',
                'src/emd_cuda.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
