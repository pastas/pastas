Introductie
===========

Pastas: What is it?
-------------------
Pastas is an open source python package for processing, simulating and analyzing
hydrological time series (models). The object oriented stucture allows for the
quick implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in
optimization, visualisation, and statistical analysis tools.

Examples
--------
All examples are provided in the `Examples directory <https://github.com/pastas/pastas/tree/master/examples>`_.
These include Python scripts and Ipython Notebooks. HTML versions of the IPython
notebooks can be found on the documentation website.

Quick installation guide
------------------------
To install Pastas, a working version of Python 2.7 or 3.5 has to be installed on
your computer. We recommend using the `Anaconda Distribution <https://www.continuum.io/downloads>`_
as it includes most of the python package dependencies and the Ipython Notebook
software to run the notebooks. However, you are free to install any Python
distribution you want.

Stable version
~~~~~~~~~~~~~~
To get the latest stable version, use::

  pip install pastas

or directly from Github::

  pip install https://github.com/pastas/pastas/zipball/master

Update
~~~~~~
To update pastas, use::

  pip install pastas --upgrade

Dependencies
------------
Pastas depends on a number of Python packages, of which all of the necessary are
automatically installed when using the pip install manager. To summarize, the
following pacakges are necessary for a minimal function installation of Pasta:
numpy>=1.9, matplotlib>=1.4, lmfit>=0.9, pandas>=0.15, scipy>=0.15,
statsmodels>=0.5.

Developers
----------
Please take a look at the `developers section <http://pastas.github.io/pastas/developers.html>`_
on the documentation website for more information on how to develop Pastas.


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
