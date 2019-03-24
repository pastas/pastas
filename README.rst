PASTAS: HYDROLOGICAL TIME SERIES ANALYSIS
=========================================

.. image:: /doc/_static/logo_small.png
   :width: 200px
   :align: left

.. image:: https://travis-ci.org/pastas/pastas.svg?branch=master
                    :target: https://travis-ci.org/pastas/pastas
.. image:: https://img.shields.io/pypi/v/pastas.svg
                    :target: https://pypi.python.org/pypi/pastas
.. image:: https://img.shields.io/pypi/l/pastas.svg
                    :target: https://mit-license.org/
.. image:: https://img.shields.io/github/release/pastas/pastas.svg
                    :target: https://github.com/pastas/pastas/releases
.. image:: https://api.codacy.com/project/badge/Grade/0e0fad469a3c42a4a5c5d1c5fddd6bee
                    :target: https://app.codacy.com/app/raoulcollenteur/pastas?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastas&utm_campaign=Badge_Grade_Dashboard
.. image:: https://codecov.io/gh/pastas/pastas/branch/master/graph/badge.svg
                    :target: https://codecov.io/gh/pastas/pastas
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1465866.svg
                    :target: https://doi.org/10.5281/zenodo.1465866

Pastas: what is it?
~~~~~~~~~~~~~~~~~~~
Pastas is an open source python package for processing, simulating and analyzing 
hydrological time series (models). The object oriented stucture allows for the 
quick implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in 
optimization, visualisation, and statistical analysis tools.

Documentation & Examples
~~~~~~~~~~~~~~~~~~~~~~~~
- Documentation is provided on a dedicated website: http://pastas.readthedocs.io/
- Examples can be found on the `examples directory on the documentation website <http://pastas.readthedocs.io/en/dev/examples.html>`_.

Get in Touch
~~~~~~~~~~~~
- Questions on Pastas can be asked and answered on `StackOverFlow <https://stackoverflow.com/questions/tagged/pastas>`_.
- Bugs, feature requests and other improvements can be posten as the `Github Issues <https://github.com/pastas/pastas/issues>`_.
- Pull requests will only be accepted on the development branch (dev) of this repository. Please take a look at the `developers section <http://pastas.readthedocs.io/>`_ on the documentation website for more information on how to develop Pastas.

Quick installation guide
~~~~~~~~~~~~~~~~~~~~~~~~
To install Pastas, a working version of Python 3.5 3.6 or 3.7 has to be installed on 
your computer. We recommend using the `Anaconda Distribution <https://www.continuum.io/downloads>`_
with Python 3.7 as it includes most of the python package dependencies and the Jupyter
Notebook software to run the notebooks. However, you are free to install any
Python distribution you want.

Stable version
--------------
To get the latest stable version, use::

  pip install pastas
  
or directly from Github::
  
  pip install https://github.com/pastas/pastas/zipball/master

Update
------
To update pastas, use::

  pip install pastas --upgrade  
  
Developers
----------
To get the latest development version, use::

   pip install https://github.com/pastas/pastas/zipball/dev
  
Dependencies
~~~~~~~~~~~~
Pastas depends on a number of Python packages, of which all of the necessary are 
automatically installed when using the pip install manager. To summarize, the 
following pacakges are necessary for a minimal function installation of Pastas:

- numpy>=1.10
- matplotlib>=2.0
- pandas>=0.23
- scipy>=1.0

Background
~~~~~~~~~~
Work on Pastas started in the spring of 2016 at the Delft University of Technology and `Artesia Water <http://www.artesia-water.nl/>`_. 

How to Cite Pastas?
~~~~~~~~~~~~~~~~~~~
To cite a specific version of Python, you can use the DOI provided for each official release (>0.9.7) through Zenodo. Click on the link to get a specific version and DOI, depending on the Pastas version.

Collenteur, R., Bakker, M., Caljé, R. & Schaars, F. (XXXX). Pastas: open-source software for time series analysis in hydrology (Version X.X.X). Zenodo. http://doi.org/10.5281/zenodo.1465866

License (MIT License)
~~~~~~~
Copyright (c) 2016-2018 R.A. Collenteur, M. Bakker, R. Calje, F. Schaars

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

