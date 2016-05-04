# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 23:15:23 2015

Make sure you are in the directory where this setup.py file is placed!
Run this script in the terminal by typing:
"python cythonize.py build_ext --inplace"

If one wants to cythonize the .pyx file or de .c file is unavailable:
python cythonize.py build_ext --inplace --use-cython

This option requires Cython to cythonize the .pyx file.

@author: Raoul Collenteur
"""

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import sys

# Standard use-cython is put to False, so package is independent of Cython.

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False


ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("recharge", ["recharge"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    include_dirs = [np.get_include()],
    cmdclass = {'USE_CYHTON' : 'USE_CYHTON'},
    ext_modules = extensions
)
