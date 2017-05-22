Introductie
===========

Pastas: What is it?
-------------------
Pastas is an open source python package for processing, simulating and analyzing
hydrological time series (models). The object oriented stucture allows for the
quick implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in
optimization, visualisation, and statistical analysis tools.

General Outline
---------------
A time-series model consists of stresses, called Tseries, which together with a Constant and an optional NoiseModel form the simulation.
Most Tseries use a response-function, called rfunc, that transform the stress in its contribution in the simulation.
Examples of response-functions are Gamma, Exponential or One (which is used for the Constant).
Each Tseries has a number of parameters, which are optimized by the Solver.
During optimization the residuals (the difference from the observations, or the innovations when a NoiseModel is used) are minimized.

Stress-series
-------------
Most Tseries-classes use one or more stress-series.
Each TimeStamp in the series represents the end of the period that that record describes.
For example, the precipitation of January 1st, has the TimeStamp of January 2nd 0:00 (this can be counter-intuitive).
The stress-series have to be equidistant (at the moment, the observation-series can be non-equidistant).
The user can use Pandas resample-methods to make sure the Series satisfy this condition, before using the Series for Pastas.
The model frequency is set at the highest frequency of all the Tseries. Other frequencies are upscaled by using the bfill()-method.
For these frequency-manipulations, the series need to have a frequency-independent unit. For example, precipitation needs to have the unit L/T, and not L.

Examples
--------
Examples can be found on the `examples directory on the documentation website <http://pastas.github.io/pastas/examples.html>`_.
All examples are provided in the `examples directory on GitHub <https://github.com/pastas/pastas/tree/master/examples>`_.
These include Python scripts and Jupyter Notebooks.

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


.. toctree::
    :maxdepth: 2
    :hidden:

    Introduction <index>
    Examples <examples>
    Developers <developers>
    API-Docs <modules>
