============
Introduction
============
|Project| is an open source python package for processing, simulating and analyzing
hydrological time series. The object oriented stucture allows for the quick
implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in
optimization, visualisation, and statistical analysis tools.

General Outline
---------------
A time-series model consists of stresses, called StressModel, which together with a Constant and an optional NoiseModel form the simulation.
Most StressModel use a response-function, called rfunc, that transform the stress in its contribution in the simulation.
Examples of response-functions are Gamma, Exponential or One (which is used for the Constant).
Each StressModel has a number of parameters, which are optimized by the Solver.
During optimization the residuals (the difference from the observations, or the innovations when a NoiseModel is used) are minimized.

Examples
--------
Examples can be found on the examples directory on the documentation website.
All examples are provided in the `examples directory on GitHub <https://github.com/pastas/pastas/tree/master/examples>`_.
These include Python scripts and Jupyter Notebooks.

Quick installation guide
------------------------
To install |Project|, a working version of Python 2.7 or 3.5 has to be installed on
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
To update |Project|, use::

  pip install pastas --upgrade

Dependencies
------------
|Project| depends on a number of Python packages, of which all of the necessary are
automatically installed when using the pip install manager. To summarize, the
following packages are necessary for a minimal function installation of |Project|:

* numpy>=1.9
* matplotlib>=1.4
* lmfit>=0.9
* pandas>=0.15
* scipy>=0.15
* statsmodels>=0.5

Pastas Users
------------
We encourage users that have applied |Project| in their research or consultancy
work to share any public reports that are available with other |Project| users.
If you have a report you would like to share please sent an email an we will
add a reference to your report to the list of showscases.

Developers
----------
Since |Project| is an open-source framework, it depends on the Open
Source Community for continuous development of the software. Any help in
maintaining the code, writing or updating documentation is more then
welcome. Please take a look at the :ref:`developers`
on the documentation website for more information on how to develop
|Project|.

.. toctree::
    :maxdepth: 2
    :hidden:

    Getting Started <getting-started>
    Examples <examples>
    Concept of Pastas <concepts>
    Developers <developers>
    API-Docs <modules>
