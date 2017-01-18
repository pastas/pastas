PASTAS: HYDROLOGICAL TIME SERIES ANALYSIS
=========================================

==============  ==================================================================
Build Status    .. image:: https://travis-ci.org/pastas/pastas.svg?branch=master
                    :target: https://travis-ci.org/pastas/pastas
Pypi            .. image:: https://img.shields.io/pypi/v/pastas.svg
                    :target: https://pypi.python.org/pypi/pastas
License         .. image:: https://img.shields.io/pypi/l/pastas.svg
                    :target: https://pypi.python.org/pypi/pastas
Latest Release  .. image:: https://img.shields.io/github/release/qubyte/pastas.svg
                    :target: https://pypi.python.org/pypi/pastas
==============  ==================================================================

Pastas Introduction
===================
Pastas is an open source python package for processing, simulating and analyzing 
hydrological time series (models). The object oriented stucture allows for the 
quick implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in 
optimization, visualisation, and statistical analysis tools.

Documentation
~~~~~~~~~~~~~
Documentation is provided on a dedicated website: http://pastas.github.io/pastas/

Examples
~~~~~~~~
All examples are provided in the `Examples directory <https://github.com/pastas/pastas/tree/master/examples>`_. 
These include Python scripts and Ipython Notebooks. HTML versions of the IPython 
notebooks can be found on the documentation website.

Quick installation guide
~~~~~~~~~~~~~~~~~~~~~~~~
To install Pastas, a working version of Python 2.7 or 3.5 has to be installed on 
your computer. We recommend using the `Anaconda Distribution <https://www.continuum.io/downloads>`_
as it includes most of the python package dependencies and the Ipython Notebook 
software to run the notebooks. However, you are free to install any Python
distribution you want. 

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
  
Dependencies
~~~~~~~~~~~~
Pastas depends on a number of Python packages, of which all of the necessary are 
automatically installed when using the pip install manager. To summarize, the 
following pacakges are necessary for a minimal function installation of Pasta: 
numpy>=1.9, matplotlib>=1.4, lmfit>=0.9, pandas>=0.15, scipy>=0.15, 
statsmodels>=0.5.
  
Developers
~~~~~~~~~~
To get the latest development version, use::

   pip install https://github.com/pastas/pastas/zipball/dev

If you have found a bug in Pastas, or if you would like to suggest an
improvement or enhancement, please submit a new Issue through the Github Issue
tracker toward the upper-right corner of the Github repository. Pull requests will
only be accepted on the development branch (dev) of this repository.

Please take a look at the `developers section <http://pastas.github.io/pastas/developers.html>`_
on the documentation website for more information on how to develop Pastas.
