Concepts of Pastas
==================
A time-series model consists of one or more StressModels which together with
a Constant and a NoiseModel form the simulation. Most StressModels use a
response-function from the rfunc-module that transforms the stress into its
contribution in the simulation. Examples of response-functions are Gamma,
Exponential or One (which is used for the Constant). Each StressModel has a
number of parameters, which are optimized by the Solver. During optimization
the noise (or the residuals, the difference between the simulation and the
observations, when no NoiseModel is used) are minimized.

Below is a list of Jupyter Notebooks with code, comments and figures:

.. toctree::
    :maxdepth: 1
    :numbered:
    :glob:

    ./*
