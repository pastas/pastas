from setuptools import setup, find_packages

# from os import path
# from codecs import open  # To use a consistent encoding
# here = path.abspath(path.dirname(__file__))

l_d = ''
try:
    import pypandoc

    l_d = pypandoc.convert('README.md', 'rst')
except:
    pass

# Set the version. Possibly this can be converted to another method such as Numpy
#  is using in the future. https://github.com/numpy/numpy/blob/master/setup.py
# DO NOT USE from pastas import __version__ as it causes problems with Travis.

MAJOR = 0
MINOR = 9
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

setup(
    name='pastas',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=VERSION,

    description='Python Applied System TimeSeries AnalysiS',
    long_description=l_d,

    # The project's main homepage.
    url='https://github.com/pastas/pastas',
    author='Mark Bakker, Raoul Collenteur, Ruben Calje, Frans Schaars',
    author_email='markbak@gmail.com, r.collenteur@artesia-water.nl, '
                 'r.calje@artesia-water.nl, f.schaars@artesia-water.nl',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5'
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.9', 'matplotlib>=1.4', 'lmfit>=0.9',
                      'pandas>=0.15', 'scipy>=0.15', 'statsmodels>=0.5'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
