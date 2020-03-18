Concepts of Pastas
==================
A time-series model consists of one or more StressModels which together with
a Constant and a NoiseModel form the simulation. Most StressModels use a
response-function, from the rfunc-module, that transform the stress in its
contribution in the simulation. Examples of response-functions are Gamma,
Exponential or One (which is used for the Constant). Each StressModel has a
number of parameters, which are optimized by the Solver. During optimization
the noise (or the residuals, the difference from the observations, when no
NoiseModel is used) are minimized.

.. toctree::
    :maxdepth: 1

    ./sources.rst
    ./timesteps.rst
    ./stressmodels.rst
    ./solver.rst
    ./statistics.rst
    ./visualization.rst

.. warning::
    This section is unfortunately slightly outdated and needs to be updated.
    Check the code on GitHub for the current implementations.
