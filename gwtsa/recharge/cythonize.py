# -*- coding: utf-8 -*-

"""
Cythonize.py can be used to cythonize and/or compile recharge.pyx and recharge.c
files. Cythonizing the .pyx file should only be done by developers! The c-file
that is shipped with this software can be compiled into an .so file or a .exe file.

Compilation instructions:
-------------------------
[1] Open the command window (Windows) or terminal (MacOS).
[2] Move to the directory where this file is located.
[3] type the following and press enter:
#   >>> python cythonize.py build_ext --inplace

Cythonize instructions:
-----------------------
[1] Open the command window (Windows) or terminal (MacOS).
[2] Move to the directory where this file is located.
[3] type the following and press enter:
#   >>> python cythonize.py build_ext --inplace --use-cython

This option requires the Cython package to be installed.

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

extensions = [Extension("recharge", ["recharge" + ext])]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setup(
    include_dirs=[np.get_include()],
    cmdclass={'USE_CYHTON': 'USE_CYHTON'},
    ext_modules=extensions
)
