# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/jaywalnut310/vits/

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    name="monotonic_align.core",
    sources=["core.pyx"],
    include_dirs=[numpy.get_include()],
    # Define additional arguments if needed
)

setup(
    name="monotonic_align",
    ext_modules=cythonize([extension]),
)
