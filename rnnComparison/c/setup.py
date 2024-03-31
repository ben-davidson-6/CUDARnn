import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["CXX"] = "/usr/bin/g++-12"
os.environ["CC"] = "/usr/bin/gcc-12"

setup(
    name="ben_rnn",
    ext_modules=[
        CUDAExtension(
            "ben_rnn",
            [
                "torch_op.cpp",
                "rnn.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
