from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the version.
version = {}
with open("pastas/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pastas',
    version=version['__version__'],
    description='Python package to perform time series analysis of '
                'hydrological time series.',
    long_description=long_description,
    url='https://github.com/pastas/pastas',
    author='R.A. Collenteur, M. Bakker, R. Calje, F. Schaars',
    author_email='raoulcollenteur@gmail.com, markbak@gmail.com, '
                 'r.calje@artesia-water.nl',
    project_urls={
        'Source': 'https://github.com/pastas/pastas',
        'Documentation': 'http://pastas.readthedocs.io/en/latest/',
        'Tracker': 'https://github.com/pastas/pastas/issues',
        'Help': 'https://github.com/pastas/pastas/discussions'
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.16.5',
                      'matplotlib>=2.0',
                      'pandas>=1.0',
                      'scipy>=1.1'],
    packages=find_packages(exclude=[]),
)
