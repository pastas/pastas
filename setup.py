from setuptools import setup, find_packages

l_d = ''
try:
    import pypandoc

    l_d = pypandoc.convert('README.rst', 'rst')
except:
    pass

# Get the version.
version = {}
with open("pastas/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pastas',
    version=version['__version__'],
    description='Python Applied System TimeSeries AnalysiS',
    long_description=l_d,
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
    install_requires=['numpy>=1.9', 'matplotlib>=1.5', 'lmfit>=0.9',
                      'pandas>=0.19', 'scipy>=0.17'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
