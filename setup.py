from setuptools import setup, find_packages

# from setuptools import find_packages
# from os import path
# from codecs import open  # To use a consistent encoding
# here = path.abspath(path.dirname(__file__))
#
# Get the long description from the relevant file
# with open(path.join(here, 'README'), encoding='utf-8') as f:
#    long_description = f.read()

l_d = ''
try:
    import pypandoc

    l_d = pypandoc.convert('README.md', 'rst')
except:
    pass

# Set the version. Possibly this can be converted to another method such as Numpy
#  is using in the future. https://github.com/numpy/numpy/blob/master/setup.py
# DO NOT USE from pasta import __version__ as it causes problems with Travis.

MAJOR               = 0
MINOR               = 0
MICRO               = 1
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

setup(
    name='pasta',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=VERSION,

    description='Open Source Time Series Analysis',
    long_description=l_d,

    # The project's main homepage.
    url='https://github.com/pastas/pasta',

    # Author details
    author='Mark Bakker',
    author_email='markbak@gmail.com',

    # Choose your license
    license='MIT',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        # 'Intended Audience :: Groundwater Modelers',
        # Pick yor license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7'
        'Programming Language :: Python :: 3.5'
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.9', 'matplotlib>=1.4', 'lmfit>=0.9', 'pandas>=0.15',
                      'scipy>=0.15', 'statsmodels>=0.5', 'requests',
                      'pyproj', "io"],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
