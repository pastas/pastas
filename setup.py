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
    description='Python Applied System TimeSeries Analysis Software',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/pastas/pastas',
    author='R.A. Collenteur, M. Bakker, R. Calje, F. Schaars',
    author_email='raoulcollenteur@gmail.com, markbak@gmail.com, '
                 'r.calje@artesia-water.nl',
    project_urls={
        'Source': 'https://github.com/pastas/pastas',
        'Documentation': 'http://pastas.readthedocs.io/en/latest/',
        'Tracker': 'https://github.com/pastas/pastas/issues',
        'Help': 'https://stackoverflow.com/questions/tagged/pastas'
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.15', 'matplotlib>=2.0', 'pandas>=0.23',
                      'scipy>=1.1'],
    packages=find_packages(exclude=[]),
    package_data={"pastas": ["log_config.json"], },
)
