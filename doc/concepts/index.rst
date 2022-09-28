Concepts of Pastas
==================
Pastas is an open source Python package for the analysis of hydrogeological time series.
In Pastas, transfer function noise modeling is applied using predefined response functions.
For example, the head response to rainfall is simulated through the convolution of measured rainfall with a Gamma response function.
A Pastas model can be constructed in seven simple steps:

#. import Pastas
#. read the time series
#. create a model
#. specify the stresses and the types of response functions
#. estimate the model parameters
#. visualize output
#. analyze the results.

A time-series Model consists of one or more StressModels which together with a Constant produce the simulated time series.
Most StressModels use a response function that transform one or more stress-series into its contribution in the simulation.
Examples of response-functions are Exponential and Gamma. Each StressModel has a number of parameters, which are optimized by the Solver.
The Solver minimizes the noise, which a NoiseModel calculates from the residuals (the difference between the simulation and the observations).

This section holds notebooks describing some of the methods underlying Pastas.

.. note::
    This section is still being developed (November, 2020)

.. toctree::
    :maxdepth: 1
    :numbered:
    :glob:

    ./*
