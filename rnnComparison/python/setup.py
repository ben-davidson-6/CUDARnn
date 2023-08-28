from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rnn",
    ext_modules=[
        CUDAExtension(
            "rnn_cuda",
            [
                "rnn_cuda.cpp",
                "rnn_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
